# Label smoothing expansion

Written 2026-09-25. No code was changed and nothing here has been run.

Every statement about an experiment was read from its file on 2026-09-25.
Statements about the uncertainty results come from
`qsar_qm_models/PAPER_REVISION_GUIDE_FINAL.md` §R6, drafted the same day.
Statements about censoring and about the cost of rewriting come from
`/Volumes/seagate/local_noise_run/loop_headroom/FINDINGS.md` (2026-09-23, herg_ki
and caco2, 3 folds or fewer). That is a working note, not a result file.

"Smoothing" below means any operation that moves a training label to a new value.
It excludes deleting a molecule and changing its weight.

Contents:

1. What already exists
2. Experiment by experiment
3. The smoothing methods and what each needs swept
4. What the uncertainty results say about this
5. Grid size
6. Which experiments to test every smoother in
7. What epistemic uncertainty can still do

---

## 1. What already exists

### 1a. The eleven cleanups in n07

All eleven live in `n07_remediation.cleaned_training_set`. Every other experiment
that cleans imports them from there. Seven of the eleven move a label.

| cleanup | moves the label toward | which labels move, and how far | setting, and its default |
|---|---|---|---|
| `label_smooth` | mean label of its 10 nearest training molecules, Euclidean | every label. The share of the way it moves is `strength` × its suspicion score rescaled to run from 0 to 1 | `strength`, default 0.5 |
| `knn_smooth` | the same 10-neighbour mean | only the flagged share. Those labels are replaced outright | flag rate |
| `prox_smooth` | mean label of the 10 molecules a random forest most often puts in the same leaf | only the flagged share, replaced outright | flag rate. Refused above 2,000 training molecules |
| `oof_smooth` | a LightGBM prediction trained on the other training molecules, scaffold-group-aware inner split, 5 inner folds | every label, as `label_smooth` | `strength`, default 0.5 |
| `gp_smooth` | a Gaussian process posterior mean, fitted the same way | every label, as `label_smooth` | `strength`, default 0.5. Refused above 1,000 training molecules |
| `js_shrink` | the same LightGBM prediction as `oof_smooth` | every label. Share moved = σ²ᵢ / (σ²ᵢ + τ²). τ² is the variance of the label minus the prediction. σ²ᵢ is the suspicion score rescaled to [0, 1], times τ², times `strength` | `strength`, default 1.0 |
| `quantile_clip` | the nearer edge of a 10th-to-90th percentile range, from a quantile regression forest | only labels outside their range. They are moved to its edge | none |

The other four do not move a label: `remove`, `downweight`, `huber_weight`, `invvar_weight`.

`gp_smooth` numbers recorded before 2026-09-21 are not Gaussian process results.
The kernel's length scale started at 1.0 and never moved. So the target was the
inner fold's mean label (n07 docstring).

`src/kirby/utils/noise_estimation.smooth_by_scores` sets how far a label moves.
It has four rules, all taking one number `alpha`:

- `proportional`: share moved = alpha × score rescaled to [0, 1]. The only rule n07 uses.
- `threshold`: the top alpha share of scores move by alpha. The rest stay.
- `soft_threshold`: share moved = alpha × a logistic curve in the rescaled score, centred at 0.5.
- `confidence`: uses alpha × the raw score. It does not rescale.

Only n11 sweeps all four.

### 1b. Label-moving operations written inside other experiments

These are not in n07's list, but each one moves a label toward a target.

| experiment | operation | target | how far |
|---|---|---|---|
| n19 | `shrunk` | out-of-fold LightGBM prediction | further the larger that molecule's own measurement variance |
| n19 | `calibrated` | the same | as `shrunk`, but the variance comes from a curve against measurement count. That curve is fitted on molecules measured 3 or more times |
| n23 | `move` | the uncertainty model's own out-of-fold prediction | further the wider its predicted spread. Rules `proportional` and `confidence`; strengths 0.25, 0.5, 1.0 |
| n27 | `relabelled` | out-of-fold LightGBM | fixed share 0.25, 0.5 or 1.0. Applied to the **correct** labels, to measure what the move costs |
| n30 | `winsor_c<c>` | out-of-fold LightGBM | only the part of the distance beyond c × s, where s = 1.4826 × median absolute residual. c = 1.0, 1.25, 1.5 |
| n30 | `constant_lift`, `per_molecule_lift`, `truncated_normal_mean` | a truncated-normal mean above the censoring limit | only the labels at the limit |
| n31 | `rewrite` | out-of-fold LightGBM | flagged labels, all the way |
| n31 | `bayes_shrink` | the same | flagged labels, share = σ²ₙ / (σ²ₙ + s²ₘ), taken from the injector. An oracle |
| n32 | `shrink_flagged_one_weight` | the same | flagged labels, one shared share estimated from the residuals |
| n32 | `shrink_flagged_per_row` | the same | flagged labels, each its own share max(r² − s²ₘ, 0) / (max(r² − s²ₘ, 0) + s²ₘ) |
| n33 | `huber_teacher_blend` | out-of-fold LightGBM fitted with a Huber loss, c = 1 | every label, half the way |
| n34 | `group_center` | the label minus its scaffold group's mean out-of-fold residual | every label. `group_center_3plus` only moves groups with 3 or more training molecules |
| n34 | `random_intercept` | the same, with the group offset shrunk toward zero by n / (n + σ²/τ²) | every label |
| n38 | `winsorise_c1.5` | n30's winsorise at c = 1.5 | as n30 |
| n41 | `winsorise_<cal>_q<q>` | n30's winsorise at c = 1.0 | only molecules flagged by Benjamini-Hochberg on conformal p-values, q = 0.05, 0.1, 0.2 |

`n30`'s winsorise is already the dead-zone shape from the design space (§3,
axis 2). n32's per-row shrink is a rising shape with a floor. n34's
`random_intercept` is already the partial-pooled target (axis 1).

---

## 2. Experiment by experiment

Four labels:

- **Uses**: moves labels now.
- **Should**: an axis from §3 answers a question this experiment already asks.
- **Could**: it would run, but the experiment's question does not need it.
- **Does not apply**: no training label is changed, or no model is fitted, or smoothing is degenerate there. The reason is given.

c10 is a complexity experiment and is left out.

### n01, activity cliff framework. Does not apply.
It measures prediction error on cliff molecules against non-cliff molecules. It changes no training label.
n25 is where smoothing meets cliffs.

### n02, sample weighting. Could.
It sets a per-molecule weight from agreement with neighbours in an importance-weighted distance. It compares base representations against hybrids.
The same reliability score could set `label_smooth`'s share instead of a weight. n24 already compares rewriting against weighting, so n02 does not need it.

### n06, BindingDB ground truth. Does not apply.
It ranks noise detectors against real disagreement between labs. It fits no model on moved labels.

### n07, remediation under injected noise. Uses; should.
Runs all eleven cleanups at their defaults. `strength` is never passed, so `label_smooth`, `oof_smooth` and `gp_smooth` run at 0.5 only. `js_shrink` runs at 1.0 only.
The flag rate equals the injected fraction, default 0.1. Condition default is `outlier_p05`.
n07 is the synthetic-noise home for every new axis in §3. Every new smoother is added here first.

### n09, learning-with-noisy-labels survey. Does not apply.
Its 12 methods change the training loop: which molecules are selected, or how they are weighted, inside the fit.
I read `EarlyLearning` and the opening of `DivideMixReg` and found no label rewrite. I have not read all 12 in full.

### n11, estimation × remediation grid. Uses; should.
The only experiment that sweeps the α rule. It sweeps:

- the four `smooth_by_scores` rules;
- α in 0.1, 0.3, 0.5, 0.7;
- three targets: the estimator's own, out-of-fold LightGBM, and Gaussian process;
- 10 base estimators, plus 3 torch estimators and ngboost where installed;
- every distance metric, for the four neighbour estimators;
- `quantile_clip_<alpha>` under `quantile_forest`.

The new α shapes belong here: redescending, dead-zone, agreement-gated, uncertainty-gated, count-scaled.
n11 is also the only experiment that already crosses target with α rule. So the median target and the partial-pooled target belong here too.

### n12, importance-weighted distance. Could.
Its repair is a weight, `inverse` at strength 3.0, not a label move. Its question is which distance finds noisy molecules.
`label_smooth` in the importance-weighted distance would test whether that distance also gives a better neighbour mean. n21 asks the same thing over more distances.

### n13, noise distributions. Should.
It varies the shape of the injected noise at a matched amount. Shapes: gaussian, student_t at ν = 10, 5 and 3, laplace, outlier at 1, 5 and 10%, grouped_wider, grouped_shifted.
That is the question the redescending shape turns on. A redescending shape leaves the largest residuals alone. It should lose under outlier noise, where the largest residuals are the injected ones, and should not matter under gaussian.
Its Track B is written as **sample-weight only**. Adding label moves changes a stated design decision, so that is your call.

### n14, cliff landscape; n16, importance under noise. Does not apply.
Both train no downstream model.

### n15, clean-searched vs noise-searched recipe. Does not apply.
It changes no label. It compares two recipes fitted on the same noisy labels.

### n17, cleaning on real repeat measurements. Uses; should.
Runs 7 cleanups: `remove`, `downweight`, `label_smooth`, `knn_smooth`, `oof_smooth`, `gp_smooth`, `quantile_clip`. Flag share 0.10. Detector `neighbor_residual`, k = 10. Strength is never passed.
Four of n07's eleven are missing: `js_shrink`, `prox_smooth`, `huber_weight`, `invvar_weight`.
It **should** gain two things:

1. The missing smoothers and a strength sweep.
2. One measurement no other experiment can make. For each target, how far is it from the consensus label, against how far the single measurement is? FINDINGS.md §2 made that comparison under injected noise only. It is the break-even for every smoother.

### n18, cleaning the real labels, nothing injected. Uses; should.
Runs the same 7 as n17. Flag rates 0.01, 0.05, 0.20. Strengths: `label_smooth`, `oof_smooth` and `gp_smooth` at 0.25, 0.5, 1.0; `downweight` at 1.0, 3.0, 9.0.
Two detectors, `neighbor_residual` and `cross_val`. Three spaces: base, prediction-built hybrid, similarity hybrid. QM9 runs as the control.
It is missing `js_shrink` and `prox_smooth`.
It **should** gain the preservation axis (§3, axis 3). n18 has no injected truth, so a win or a loss here is the net of two effects. The first is reordering labels. The second is compressing their spread. Running each smoother with and without variance preservation separates the two on real labels.
It is also where the redescending shape tests its own assumption. On real labels, if leaving the largest residuals alone beats smoothing them, the largest residuals are more likely hard molecules than wrong labels.

### n19, measurement count. Uses; should.
`shrunk` and `calibrated` are the count-scaled α from the design space. They move each label further the less precise it is.
They should be expressed through the same α function as n07, so a count-scaled share can be crossed with the other targets. Today they are hard-wired to the out-of-fold LightGBM target.
Runs only on `herg_ic50` and `bindingdb_kd`, the two datasets that keep repeat measurements.

### n20, clean before search. Uses; should.
Runs `remove`, `label_smooth`, `knn_smooth`, `prox_smooth`, `oof_smooth`, `gp_smooth`, `quantile_clip`, `js_shrink` and `downweight`. Each label-moving one has a matching random-flag control.
Flag rates 0.05, 0.10, 0.20. Strength is never passed.
Any new smoother that returns a full cleaned label column can go straight into the recipe search here. So every smoother added to n07 **should** be added here too. Otherwise the question "does cleaning have to come before the search" is answered for the old list only.

### n21, which distance finds the neighbours. Should.
Five distances: plain, importance-weighted, forest proximity, prediction-built hybrid, similarity hybrid. k = 5, 10, 20.
Its repair is a weight, like n12's. The geometry axis (§3, axis 4) is a question about where a kNN target's neighbours come from. That is n21's question, asked of the target instead of the detector.

### n22, curator flag. Does not apply.
It grades detection against ChEMBL's curator flags. It fits nothing on moved labels.

### n23, uncertainty cleaning. Uses; should.
`move` shifts each label toward the uncertainty model's own prediction. It moves further the wider the predicted spread. Rules are `proportional` and `confidence`, at strengths 0.25, 0.5, 1.0.
Sources: `hetero_gp`, `ngboost`, `heteroscedastic_nn`, `gaussian_process`, `activity_cliff_aware`.
This is the opposite of the uncertainty-gated shape in the design space. There, α is divided by epistemic uncertainty, so labels move least where the model is least sure.
§4 gives the paper's evidence that n23's direction cannot work under random noise. The inverted gate **should** run beside it, on the same sources.

### n24, delete or keep. Uses; should.
Its "rewrite" family is only `label_smooth`, `knn_smooth` and `oof_smooth`, all at default strength. Flag rates 0.05, 0.10, 0.20. Five random-deletion seeds.
Its conclusion about rewriting rests on 3 of the 8 label-moving cleanups. `gp_smooth`, `js_shrink`, `prox_smooth` and `quantile_clip` **should** be added before "rewrite" is ranked against delete and weight.

### n25, cliff exemption. Uses; should.
Runs all eleven except `invvar_weight`, which is undefined here. Flag rates 0.01, 0.05, 0.20. Strengths: `label_smooth`, `oof_smooth` and `gp_smooth` at 0.25, 0.5, 1.0; `js_shrink` at 0.5, 1.0, 2.0; `huber_weight` at 0.5, 1.345, 3.0.
Three arms: cleaned as is, cliff molecules put back, and the same number of random molecules put back.
The redescending shape is a cliff exemption chosen by residual size instead of by structure. It **should** run here as a fourth arm. Then "exempt by chemistry" and "exempt by size of the residual" are compared on the same folds.

### n26, noise variance from repeats; n37, noise budget. Do not apply. Both supply a parameter.
Neither changes a label. Both measure the label-noise variance.
n26 uses the pooled within-molecule variance and a regression of squared error on 1 / count. n37 uses the Gamma test.
Either number can be σ² in `js_shrink`'s share, in place of the rescaled suspicion score. That gives a `js_shrink` whose share comes from a measured noise variance instead of from the detector (§3, axis 2).

### n27, what removal and relabelling cost. Uses; should.
Its `relabelled` part moves **correct** labels 0.25, 0.5 or 1.0 of the way to an out-of-fold LightGBM prediction. It then measures how much of that change reaches the held-out error.
That is the cost side of every smoother, and n27 measures it for one target only. It **should** measure it for every target (§3, axis 1) and every preservation variant (axis 3). Then each smoother's cost is known before it is tried on noisy labels.

### n29, document bias. Does not apply.
It removes a fitted offset per ChEMBL document. That is an offset correction, a different operation from smoothing.

### n30, censoring. Uses. Target smoothing is degenerate here.
It already runs the dead-zone shape (`winsor_c`) and the lifts.
Every target in §3 axis 1 is built from the clipped labels: a neighbour mean, any model prediction, any quantile. So none can sit above the limit. A clipped label moved toward any of them moves down or stays.
FINDINGS.md §6 measured this for the out-of-fold prediction. It recovered 0% of what the clip cost, on 4 models.
Rank-space smoothing is degenerate too, because every clipped label is tied at the top rank.
That is a derivation from how the targets are built, and only the out-of-fold case has been run.
One variant is not degenerate by construction: variance preservation. Rescaling the spread back up pushes the top labels up. It has not been tried.

### n31, detection regime and break-even precision. Uses; should.
`rewrite` and `bayes_shrink` run at set flag precisions of 0, 0.25, 0.5, 0.75, 1.0.
At precision 0 every flagged label was already correct. So any loss there is the smoother's own cost.
FINDINGS.md §1 measured −0.0059 rmse for rewriting a tenth of the labels to the out-of-fold prediction at level 0.
The preservation axis **should** run here at precision 0. It says how much of that cost is compression of the spread and how much is reordering.

### n32, iterative and two-learner selection. Uses; should.
It runs ITLM (4 rounds, keep share 0.8, 0.9, 0.95), two learners choosing for each other, and two flagged-only shrinks.
The iteration axis (§3, axis 5) belongs here: iterate to a fixed point, anneal α down across passes, or alternate smoothing with refitting.
Its per-row shrink rises with the residual. The redescending shape is its direct contrast and **should** run beside it.

### n33, robust training. Uses; should.
`huber_teacher_blend` is the "out-of-fold prediction of a deliberately different model" target, at share 0.5. `yeo_johnson` is a target transform.
The scale axis (§3, axis 6) belongs here, beside `yeo_johnson`: raw, residual space, rank space.
Log scale is undefined for logd, which has negative labels. n33 records that as the reason it used Yeo-Johnson.

### n34, group random effects. Uses; could.
`group_center` and `random_intercept` are already the group-mean and partial-pooled targets.
Mean preservation within group (§3, axis 3) could be added to any other smoother as a control: that smoother, forbidden to shift a scaffold group's mean. That tests whether smoothing damage comes from moving group levels. The scaffold group stands in for "market" in the suggestion.

### n35, near-duplicate disagreement. Could.
It downweights both molecules of a close pair whose labels disagree by more than 3√2 σ, to a weight of 0.25.
Moving both labels toward the pair's mean is the smoothing version of the same rule.

### n36, which molecules to remeasure. Does not apply.
The only label change is a real second measurement.

### n38, noise sized by model error. Uses; should.
It runs `winsorise_c1.5` with a random control, on noise sized in units of the out-of-fold residual SD. Sparse shifts are 1, 2 or 4 of those units, on 2, 5 or 10% of molecules.
Here the largest residuals are known to be injected errors. So the redescending shape **should** lose here, by an amount that grows with the shift size.
That makes n38 the control for n18: the same shape, run where the answer is known.

### n39, correlated sources. Does not apply.
Its training label is the mean of simulated sources, and its arms are offset removal and weights.

### n40, model-side regularisation. Does not apply. It is the comparator.
Its premise is that under dense zero-mean noise every label change is a smoother in disguise. So it changes the model instead. Arms: retuned regularisation, seed averaging, stacking, a Gaussian process with its noise term set from the Gamma test, kNN with k chosen by Mallows Cp.
Every label smoother should be read against `gp_nugget_from_gamma` and `knn_k_cp` on the same folds.

### n41, conformal false-flag control. Uses; could.
It winsorises only molecules flagged by Benjamini-Hochberg, at c = 1.0.
Any flagged-only smoother (`knn_smooth`, `prox_smooth`, n32's shrinks) could be applied to the same flag set. Then which molecules are chosen is held fixed while the operation changes.

### Summary

| label | experiments |
|---|---|
| Uses and should expand | n07, n11, n17, n18, n19, n20, n23, n24, n25, n27, n31, n32, n33, n38 |
| Should add, uses none today | n13, n21 |
| Uses; could expand | n34, n41 |
| Could add | n02, n12, n35 |
| Uses; target smoothing degenerate | n30 |
| Supplies a parameter | n26, n37 |
| Comparator | n40 |
| Does not apply | n01, n06, n09, n14, n15, n16, n22, n29, n36, n39 |

---

## 3. The smoothing methods and what each needs swept

Six axes, from the suggestion you pasted. For each option:

- whether code for it exists and where;
- what has to be swept, and the values in use today.

The suggestion was written for house prices. Its "market" is a scaffold group here, and its "asset" is a molecule.
Its argument for the median assumes multiplicative error on a positive label. pKi, pIC50 and logD are already log units, so on them the mean and median arguments do not carry over unchanged.

### Axis 1. What the label moves toward

| target | exists? | parameters to sweep |
|---|---|---|
| kNN mean label, Euclidean | `label_smooth`, `knn_smooth` | k. Hard-coded at 10 in n07. n12 and n21 sweep k = 5, 10, 20 for detection only |
| forest-proximity mean | `prox_smooth` | k (10); forest size (200 trees, from n35/n37) |
| out-of-fold prediction, same model family | `oof_smooth`, `js_shrink`, n19, n27, n30–n32 | inner folds (5). Always LightGBM today |
| out-of-fold prediction, a deliberately different model | n33's Huber teacher only | which model. One option is the downstream model's opposite: kNN target for a forest, forest target for kNN |
| Gaussian process posterior mean | `gp_smooth` | kernel and starting length scale. The 2026-09-21 fix starts it at the median distance. Capped at 1,000 molecules |
| conditional median | **computed and thrown away.** `QuantileForestEstimator` predicts the 0.5 quantile out of fold as `q_med` and does not store it (`noise_estimation.py:1458`) | none beyond the forest's own |
| a central quantile other than the median | not built | the quantile |
| scaffold-group mean | n34 `group_center` | minimum group size (all, or 3+) |
| partial-pooled group estimate | n34 `random_intercept` | EM rounds; floor on τ² |
| near-duplicate pair mean | not built | pair distance cut-off (n35 uses the 1st percentile) |

### Axis 2. How far each label moves (α as a function of the residual r)

| shape | exists? | parameters to sweep |
|---|---|---|
| constant α | n27 `relabelled`; n33 blend at 0.5 | α: 0.25, 0.5, 1.0 |
| proportional to the suspicion score | `smooth_by_scores` `proportional` | α: 0.1, 0.3, 0.5, 0.7 (n11); 0.25, 0.5, 1.0 (n18, n25) |
| threshold | `threshold`, `knn_smooth`, `prox_smooth` | flag rate: 0.01, 0.05, 0.10, 0.20 across experiments |
| logistic | `soft_threshold` | α. Its centre (0.5) and slope (5) are hard-coded |
| variance ratio | `js_shrink`; n31 `bayes_shrink`; n32 shrinks; n19 `shrunk` | where σ² comes from: the rescaled suspicion score (today), the Gamma test (n37), pooled within-molecule variance (n26), measurement count (n19). `strength` 0.5, 1, 2 (n25) |
| dead zone: 0 below a threshold, then linear | n30/n38/n41 winsorise | c: 1.0, 1.25, 1.5 residual-scale units |
| **redescending**: rises, then falls to 0 for very large r | **not built** | where it peaks and where it returns to 0, both in units of s = 1.4826 × median absolute residual. Tukey's biweight constant 4.685 (n33) is one choice for the upper point |
| **uncertainty-gated**: α ÷ epistemic uncertainty | **not built**. n23 does the reverse | which model's epistemic term; the floor on it. See §4 for which models qualify |
| **agreement-gated**: α × agreement between two models | **not built**. n41's `residual_low_disagreement` is the nearest: 5 seeds of one model | which two models; how agreement is measured (sign agreement, or the spread of the two predictions) |
| **count-scaled**: α ÷ √k | n19 `shrunk`/`calibrated` in another form | only on herg_ic50 and bindingdb_kd |

### Axis 3. What is preserved

None of these exists anywhere in the noise suite. Each is applied after any smoother, so each multiplies the grid by one.

| constraint | what it does | parameters |
|---|---|---|
| none | today's behaviour | — |
| variance | rescales residuals about the group mean back to their spread before smoothing | which groups: whole fold, or scaffold group |
| mean within group | subtracts each group's mean change, so no group's level moves | the grouping: scaffold group (`noise_groups_for`), ChEMBL document on herg_ic50 |
| rank | maps the smoothed values back onto the original sorted label values. The order changes; the set of values does not | none |

Per group, variance and mean preservation are undefined for a group of one molecule. n34 found a median of 1 training molecule per injector group on herg_ki. So on herg_ki the whole-fold version is the one that is defined.

### Axis 4. Where the neighbours come from

Applies only to the kNN, proximity and pair-mean targets.

| geometry | exists? | parameters |
|---|---|---|
| Euclidean, base representation | `label_smooth`, `knn_smooth` | k |
| importance-weighted | n12, n21, for detection | exponent (0.5 Euclidean, 1.0 Manhattan); importance method |
| forest proximity | `prox_smooth` | trees; refused above 2,000 molecules |
| prediction-built hybrid | n21, for detection | — |
| similarity hybrid | n21, for detection | — |
| two distances, both must agree | **not built** | which two. The suggestion's attribute and embedding blocks would be, for example, a fingerprint and a learned embedding |
| graph Laplacian solve over the whole fold | **not built** | graph k; the regularisation weight |

### Axis 5. Iteration

| schedule | exists? | parameters |
|---|---|---|
| one pass | everything | — |
| repeat to a fixed point | not built. FINDINGS.md §6 did 3 passes of the out-of-fold rewrite under censoring; it changed nothing | tolerance; maximum passes |
| anneal α down across passes | not built | starting α; decay per pass; passes |
| alternate smoothing with refitting the target | n32's ITLM does this for deletion, not for label moves | passes (n32 uses 4) |

### Axis 6. Scale the smoothing is done in

| scale | exists? | parameters |
|---|---|---|
| raw label | everything | — |
| log | undefined on logd and on any label with negatives | — |
| Yeo-Johnson | n33, as a target transform for the model, not for smoothing | fitted per fold |
| residual space: smooth the residuals, add the prediction back | not built | which base model |
| rank space: smooth ranks, map back through the empirical distribution | not built | none |

Rank space cannot change the set of label values. It overlaps with rank preservation on axis 3.

---

## 4. What the uncertainty results say about this

Source: `PAPER_REVISION_GUIDE_FINAL.md` §R6, the draft of `\subsection{Uncertainty under label noise}`.
Those models are qsar_qm_models' models on QM9, hERG Ki, Caco-2 and logD. They are not KIRBy's.
Of the uncertainty models n23 uses, only `hetero_gp` was ported from there.

**1. A model's out-of-fold uncertainty does not say which labels are noisy.**
The per-molecule Spearman between predicted uncertainty and injected noise is −0.16 to 0.16. That holds for every model, representation and dataset, under six of the seven noise conditions.
So uncertainty cannot choose **which** labels to move. That is n23's `move` and its `drop`.
Whether it can say how much to trust the **target** is a separate question, and it is untested. §7 covers it.

**2. Under censoring, uncertainty points the wrong way.**
Models are least uncertain on the labels that were clipped hardest. On QM9 at half clipped, NGBoost reaches −0.84 to −0.80. On logD at half clipped, every model except the two GPs is at −0.29 or below.
An uncertainty-gated α would therefore move the clipped labels hardest. And §2 n30 gives the reason every target sits at or below the limit.
So under censoring the uncertainty-gated shape moves the most damaged labels furthest, and in the wrong direction. Both halves of that are derived. Neither has been run.

**3. Only some models' epistemic term can be the gate.**
The gate needs an epistemic term that does not grow with label noise.

- Full-BNNs with a variance head: epistemic grows by at most 0.43 of aleatoric growth, on all four datasets, all six representations and every condition.
- Heteroscedastic GP: under about a quarter, on PDV, ChemBERTa and MHG-GNN. On ECFP4 under grouped-shifted, both terms grow by about the same amount.
- QRF: 0.63 to 1.09 on fingerprints, 1.01 to 1.45 on PDV, 1.28 to 2.05 on ChemBERTa and MHG-GNN. Its epistemic term rises with the noise, so as a gate it would stop smoothing as the noise grows.

`quantile_clip` uses a quantile regression forest too. Whether its 10th-to-90th percentile range widens with injected noise, and so clips less, has not been measured.

**4. Under grouped-shifted, added noise goes into the epistemic term.**
This holds for the heteroscedastic GP and QRF, in every pairing of dataset and representation.
So an epistemic gate smooths least under exactly the condition that n34's group offsets target. Uncertainty-gated and group-mean smoothing should both run under `grouped_shifted`.

**5. The aleatoric term cannot supply σ² in absolute units.**
It rises 0.36 to 0.99 units per unit of injected noise, and some models start at 0.31 to 1.26 on clean labels. The paper says it "does not estimate an assay's measurement error".
So `js_shrink`'s σ² should come from n26 or n37, not from a model's aleatoric term.

**6. Uncertainty on the training molecules themselves was never recorded.**
Every per-molecule correlation above is out of fold. A smoother that gates on in-sample uncertainty would have no evidence behind it either way (§R6 TODO list).

---

## 5. Grid size

Crossing every option in §3 is 10 targets × 10 α shapes × 4 preservation × 7 geometries × 4 schedules × 5 scales = 56,000 smoothers.
That is before strengths, flag rates, detectors, spaces, models, folds and conditions.
Some cells are undefined, and I have not counted how many. The geometry axis applies only to the neighbour and pair targets. Log scale is undefined on negative labels. The count-scaled shape needs repeat measurements.

Which axes are crossed and which are run one at a time is your decision.

---

## 6. Which experiments to test every smoother in

The aim is to put the smoothers beside as many other noise mitigation methods as possible.
The candidates are the experiments in §1b, which already run their own mitigation methods.
The table counts each experiment's mitigation methods. Baselines and random controls are left out of the count, and a swept setting counts as one method per value.
Counts were read from each file on 2026-09-25.

| experiment | mitigation methods it runs now | count | noise | graded against |
|---|---|---|---|---|
| n33 | 5 label arms (Yeo-Johnson, NGBoost inverse-variance weights, C-Mixup, target-noise bagging, Huber-teacher blend) + 20 loss swaps (Huber ×4 values of c on LightGBM, xgboost and the DNN; fair; median; Tukey; expectile ×2; xgboost absolute; rf_l1; NGBoost Student-t) | 25 | every regression condition in the registry, injected | clean labels |
| n19 | 3 weights by count or spread, 2 label moves (`shrunk`, `calibrated`), 4 fitted-variance and count-feature arms, 3 model pairs (NGBoost, GP, xgboost AFT) | 12 | real repeat measurements; herg_ic50 and bindingdb_kd only | consensus of the other measurements |
| n30 | 3 lifts + shuffled lift, drop, winsorise ×3, xgboost AFT, rf_l1, LightGBM Huber | 11 | censoring only, injected | clean labels |
| n32 | ITLM ×3 keep shares, two learners, 2 flagged-only shrinks; 6 detectors | 6 | outlier, shift and grouped conditions, injected | clean labels |
| n38 | winsorise, robust loss, ITLM, NGBoost inverse-variance weights | 4 | noise sized by the model's own error: dense, sparse at d′ = 1, 2, 4, and two heteroscedastic; on real and on semi-synthetic labels | clean labels, and the known signal on semi-synthetic |
| n31 | remove, downweight, rewrite, `bayes_shrink`, each at flag precision 0, 0.25, 0.5, 0.75, 1.0 | 4 | outlier-only, shift-only, gaussian, injected | clean labels |
| n34 | `group_center`, `group_center_3plus`, `random_intercept` | 3 | grouped_shifted, grouped_wider, none | clean labels |
| n23 | drop ×3 shares, move ×2 rules ×3 strengths, over 5 uncertainty sources | 2 operations | conditions where detection is defined | clean labels |
| n41 | winsorise and remove on Benjamini-Hochberg flags, q = 0.05, 0.1, 0.2 | 2 operations | injected | clean labels |
| n27 | none. It measures what a label move costs on correct labels | 0 | none and gaussian | the labels as recorded |

**Recommendation: n33, n19, n30 and n38.**

- n33 has the most methods by far (25) and runs every regression noise condition.
- n19 is the only one on real noise with a consensus label to grade against.
- n30 is censoring, the one condition where changing labels has recovered damage. FINDINGS.md §14 has 32 readings, 45 to 98% of the damage recovered.
- n38 separates dense noise from gross errors by the model's own error. It also has semi-synthetic labels, where the true signal is known.

n32 is worth adding if the iteration axis is tested. n31 is worth adding if the preservation axis is tested, because at flag precision 0 any loss is the smoother's own cost.
n27 is not a mitigation experiment, but it remains the baseline for what each smoother costs.

**What adding the smoothers costs in code.** Of these ten, only n23 calls n07's `cleaned_training_set`. Every other experiment would need the call wired in, and each builds its own out-of-fold prediction with `oof_predict`. That prediction can be handed in as the `oof` target, so it is not refitted per smoother.

---

## 7. What epistemic uncertainty can still do

Source: `PAPER_REVISION_GUIDE_FINAL.md` §R6 and its TODO list, read 2026-09-25.

**It cannot pick which label is wrong.** Out of fold, the Spearman between predicted uncertainty and each molecule's injected noise is −0.16 to 0.16. §R6 says "predicted uncertainty". It does not say whether the epistemic term was tested on its own.

**It may say how far to trust the value a label is moved toward.** That is what the uncertainty-gated shape (§3, axis 2) needs. The decomposition result is why it might work.
In the full-BNNs with a variance head, epistemic growth is at most 0.43 of aleatoric growth as noise rises. So their epistemic term still marks where training data is thin, even on noisy labels.
The quantile forest's does not. Its epistemic growth is 0.63 to 2.05 times its aleatoric growth.

**The needed test has not been run.** Does low epistemic uncertainty on a molecule mean its target sits closer to its clean label?
NoiseInject writes `unc_error_rho`, the rank correlation between uncertainty and error. It is computed on total uncertainty. Nothing in the guide reports it for the epistemic term alone.
Any experiment with injected noise can run the test, because the clean labels are known. Per molecule, it is the Spearman between epistemic uncertainty and |target − clean label|. The only extra fits are the uncertainty model's own.

**Aleatoric uncertainty is useful at dataset level.** It rises 0.36 to 0.99 units per unit of injected noise, so it can compare noise between datasets or levels. That could set how hard to smooth per dataset. It cannot set the share for each molecule.

**Under grouped-shifted noise, added noise goes into the epistemic term.** This holds for the heteroscedastic GP and the quantile forest, in every pairing of dataset and representation.
So an epistemic gate would smooth least where the group offsets are. n34's group corrections are the matching tool there.

**Limit in KIRBy.** n23's docstring says `hetero_gp` is the only model in KIRBy's roster whose two uncertainty terms separate. The variance-head BNNs are not in KIRBy. So an epistemic gate here could only be built on `hetero_gp` today.
