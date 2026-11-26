# From literature review to the sweep: what each new dataset needs

Temporary file, 2026-11-17. To be deleted once it is moved somewhere permanent.

## Part 1. The steps, for all 9 datasets

Agreed with Adelaide, 2026-11-17.

### 1. Source

Do we have a proper source? Is it right, and are we following it as closely as we can?
- Where the original code came from, with the published paper and table it reproduces.
- What we change from it, and why. Examples: seeding the random numbers, a different split.

### 2. Train and test

- Train and test are fixed, by files or by a seed.
- No leakage between them. Check the authors' own split too: other authors have leaked.
- A plan for the further splitting Papers 2 and 3 need (they draw 5 folds). Adelaide,
  2026-11-20: "the intake needs to change - I need to have a plan for future splitting".

### 3. Results and room

The published numbers hold up when re-run. Count the test rows the best published model
gets wrong; that count must be well above the test set's resolution,
sqrt(n x p x (1 - p)). Adelaide sees the results and we discuss them before this step
passes.

### 4. Models

Every model the dataset's main papers test, each in the configuration its paper used.
A model KIRBy does not have still goes in.

### 5. Views

The manual research step. Build a starting set of views large enough for the method to
work. Nothing is built until this step is done.
- Views are listed in two groups: full views, and augmented features. Adelaide,
  2026-11-20: "views need to distinguish between full views and augmented features".

## Part 2. Speck (Gohr 2019, agohr/deep_speck)

### 1. Source

- Code: github.com/agohr/deep_speck, commit c3364c0. Paper: Gohr 2019 (ePrint 2019/037),
  Table 2.
- Change: the random numbers are seeded (Gohr uses `os.urandom`). Adelaide approved this,
  2026-11-17: "this is a good change".
- Other differences, none significant. Adelaide, 2026-11-20: "I don't think any of these
  are significant issues".
  1. Gohr's pre-trained networks are run with a numpy copy of their layers, not Keras.
  2. His shipped networks have 1 residual block; his paper's 5-round network has 10. Their
     scores still match the paper within 0.001 (step 3).
  3. Only the real-or-random setting is built, not his real-differences setting.

### 2. Train and test

- Set by the source: every pair is generated fresh from its own random key and message.
  Gohr trains on 10^7 pairs and tests on 10^6.
- Leakage: two pairs are identical only if their 64-bit key and 32-bit message both match.
  Across 10^7 x 10^6 train-test comparisons that is about 1 in 10^16. Nothing else links
  one pair to another.
- Adelaide, 2026-11-20: "I don't think it matters as much for this, no? Isn't this preset?"
- Every later split works the same way: fresh pairs from a new seed. Adelaide, 2026-11-20:
  "Just make sure future splits will work this way and its good".

### 3. Results and room

Passed. Adelaide, 2026-11-20: "Yea that checks out, I'm okay with that."

Dataset-specific case. Gohr's pre-trained networks (`net5_small.h5` to `net8_small.h5`, one
residual block each) are a point of comparison only. They were trained on the raw pair and
cannot take any other input, so they cannot score a view or a hybrid. Adelaide, 2026-11-18:
"the pretrained resnet becomes just a point of comparison ... that needs to be documented,
that's actually a dataset-specific case". For the ResNet in the screen, see step 4.

Gohr's pre-trained networks, scored on 10^6 new pairs per round count:

| rounds | here | Gohr 2019 Table 2 |
|---|---|---|
| 5 | 0.9279 | 0.929 |
| 6 | 0.7873 | 0.788 |
| 7 | 0.6160 | 0.616 |
| 8 | 0.5144 | 0.514 |

### 4. Models

Decided 2026-11-20. Adelaide: "Lets just use my defaults + resnet as a comparison. In the
sweep it may be worth testing out these alternative configurations from the lit review,
might as well give them a try. SO the defaults + the additoinal configurations".
- The screen runs the find_best_hybrid defaults (Adelaide, 2026-11-20: "the defaults for
  the find_best_hybrid search"): the models it uses when none are named, `DEFAULT_MODELS`
  tiers 'fast' and 'full' in src/kirby/find_best_hybrid.py. Read 2026-11-20: rf, xgboost,
  lightgbm, svm, dnn. svm is skipped above 10,000 training rows (`_max_samples`).
- The lit review gives no settings for any model except Gohr's ResNet: it records
  Benamira's and Ebrahimi's scores only. The LightGBM settings in b11 (500 trees, learning
  rate 0.1, 63 leaves) were chosen by Claude, not taken from a paper.
- Gohr's ResNet: comparison score only (below).
- Gohr's paper is the primary source; it tests only the ResNet. The other five models
  below came from the lit review (Benamira 2021, Ebrahimi 2021), not from the source.

Earlier, Adelaide, 2026-11-17: "I need to test all of those 6 models including the resNet in the
screening/sweep ledger. From there 2 models will be chosen".

Adelaide, on the ResNet: "this NEEDS to be in there, with the same configuration as the
original paper."

1. ResNet, Gohr's configuration. A comparison score only, never used with the hybrids.
   It takes only the raw pair, split into four 16-bit words, so it cannot be trained on a
   hybrid. Adelaide, 2026-11-20: "that's a comparison point, it needs to be mentioned in
   the paper 2 outline if the dataset gets used. It won't be used with the hybrids, but
   it'll be used as a comparison score."
   - If Speck is used: add it to PAPER_2_OUTLINE.md as a comparison score.
   - Open: whether a version that reads any input as one row (Gohr's layers otherwise
     unchanged) is built for the screen. Not built; run time not measured. KIRBy's `cnn`
     (one convolution layer, filter widths 3, 7 and 15) exists.
   Configuration read from `train_nets.py` and `train_5_rounds.py` (deep_speck commit
   c3364c0):
   - 10 residual blocks at 5 rounds; 32 filters, kernel size 3; two dense layers of 64
   - L2 penalty 10^-5; Adam; mean squared error loss
   - batch 5,000; cyclic learning rate from 0.002 down to 0.0001 over 10 epochs
   - 10^7 training pairs, 10^6 validation pairs, 200 epochs; best validation loss kept
   - The repository trains only the 5-round network. The settings for 6, 7 and 8 rounds
     come from the paper. Not checked: the paper's text for those three.
   - The research file says the 7-round network was trained on key-averaging targets and
     the 8-round network in stages. Not checked in the paper.
   - Gohr's code needs Keras 2.1.5 with TensorFlow 1.6.0. TensorFlow does not import in
     the local environment.
2. LightGBM (Benamira 2021, Table 9). His settings: not checked.
3. Random forest (Benamira 2021, section 5.1). No settings or scores published.
4. SVM (Benamira 2021, section 5.1). No settings or scores published.
5. dnn, standing in for the MLP of Ebrahimi 2021 and Benamira section 5.1. Ebrahimi's
   settings: not checked (abstract only).
6. Logistic regression, standing in for the "linear regression" of Benamira section 5.1.


### 5. Views

Not started; later. Adelaide, 2026-11-20: "we'll start that later". The research file (`~/Downloads/speck.md`, Table B) lists 21 operations.
The 10 run in the smoke test were a first pass, not the starting set.

## Part 3. LSST (UEA archive; Middlehurst et al. 2026, aeon-toolkit/multiverse)

### 1. Source

- Code: github.com/aeon-toolkit/multiverse, commit adc21c1, which builds every classifier
  with tsml-eval (github.com/time-series-machine-learning/tsml-eval, main at 3ac0e86).
  Paper: Middlehurst et al. 2026, arXiv 2603.20352, Tables 17-18; scores in
  results/multiverse/accuracy_mean.csv.
- Same as the source: the LSST_TRAIN and LSST_TEST files unchanged, seed 0, one CPU core,
  no time limit, the tsml-eval settings for all 18 classifiers.
- Changes, approved by Adelaide 2026-11-20 ("that's fine"):
  1. aeon 1.3.0 here; Multiverse asks for 1.4.0 or newer. The version behind its scores
     is not recorded.
  2. STC is aeon's. The published run probably used tsml-eval's older copy; tsml-eval
     commit b9d31ae says the two give identical predictions. Not checked here.
  3. MiniRocket, MultiRocket and Hydra are added; they are not in Multiverse. MiniRocket's
     published score (0.643) is from RLPAM (Gao et al. 2022).
  4. FreshPRINCE has two scores in Multiverse's own files: 0.7028 (1,733 of 2,466 test
     curves) in accuracy_mean.csv and 0.7040 (1,736) in its own file.
- Script: tests/experiments/benchmarks/b12_lsst_published.py.

### 2. Train and test

- Fixed by the archive's files: LSST_TRAIN.ts (2,459 curves) and LSST_TEST.ts (2,466),
  downloaded from timeseriesclassification.com. Built by Bagnall et al. 2018 from the
  PLAsTiCC training release (4,925 of its 7,848 objects). Adelaide, 2026-11-20: "Okay
  that's fine".
- How the authors chose train against test is not documented. Every class is cut almost
  exactly in half (the largest, class 90: 777 and 777; the smallest, class 53: 7 and 7).
- Leakage: no test curve is identical to a training curve; no curve appears twice.
- Further splitting for Papers 2 and 3, agreed by Adelaide 2026-11-20 ("That sounds good
  for step 2, write that up"): pool all 4,925 curves; within each class, deal the curves
  at random into 5 equal folds. This copies how the files were made (each class cut at
  random), with 5 parts instead of 2. About 985 test curves per fold; the smallest class
  (14 curves) gets 2 or 3 per fold.
- Code: the Papers fold function, cv_split in tests/experiments/_shared.py
  (StratifiedGroupKFold, shuffle, seed 42), with every curve given its own group number
  by LSST's loader. No new splitting code. The loader is not written yet.

### 3. Results and room

- The re-runs of the published classifiers run in the sweep, not on the laptop. Adelaide,
  2026-11-20: "I think this can be pushed to the sweep, you don't need to drain my
  laptops resources". The laptop run was stopped; 4 of 18 had finished.
- Re-run so far, accuracy on 2,466 test curves: majority class 0.3151, Catch22 0.6679,
  QUANT 0.6452, MiniRocket 0.6509.
- Published scores: Multiverse gives two per classifier. accuracy_mean.csv and each
  classifier's own file (results/multiverse/<name>/<name>_accuracy.csv, resample 0)
  disagree for most, e.g. HC2 0.6740 against 0.6752, FreshPRINCE 0.7028 against 0.7040.
  The re-run Catch22 matches its own file exactly (0.6679). Compare against the own files.
- Room: FreshPRINCE 0.7040 gets about 730 of 2,466 test curves wrong; the test set's
  resolution is about 23 curves. Fine-tuned Mantis reports 0.7109.
- Not yet discussed with Adelaide: the re-run scores.

### 4. Models

- Main paper: Multiverse, which scores 32 classifiers on LSST (each classifier's own file,
  resample 0).
- Cut: every classifier below 0.6 accuracy. The sweep runs the 20 at 0.6 or above.
  Adelaide, 2026-11-20: "Run the sweep on the 20 that are <0.6 and I'll select 2 from
  there, potentially the top N=5 or something for metalearners." (Her "<0.6" means the
  20 kept, which are at 0.6 or above.)
- The 20, accuracy: FreshPRINCE 0.7040, HC2 0.6752, Catch22 0.6679, STSF 0.6679,
  Summary 0.6650, PatchMTSC 0.6582, Arsenal 0.6569, QUANT 0.6452, RDST 0.6423,
  ROCKET 0.6363, MRHydra 0.6346, CIF-500 0.6338, DrCIF 0.6322, DrCIF-500 0.6298,
  CIF 0.6294, RIST 0.6285, TSF 0.6253, ConvTran 0.6212, STC 0.6131, MUSE 0.6107.
- Cut (12): TimesURL 0.5900, TS2Vec 0.5766, TimesNet 0.5762, TDE 0.5596, 1NN-DTW 0.5511,
  LiteTIME 0.4777, XCM 0.4517, DisjointCNN-Aeon 0.2705, DisjointCNN 0.2652,
  LITETime-MV 0.2563, H-InceptionTime 0.2230, majority class 0.3151.
- Not said: whether fine-tuned Mantis (0.7109, Mantis paper) and the majority-class
  baseline are run; whether MiniRocket, MultiRocket and Hydra (not in Multiverse) stay.
- Afterwards Adelaide selects 2 for the experiments ledger, possibly the top 5 for the
  meta-learners.
- Not built: the sweep holds 7 other models (from 2026-11-09) and none of these 20.

### 5. Views

Starting list from ~/Downloads/LSST.md sections (d) and (e). Rule (Adelaide, 2026-11-20):
a full view describes the whole light curve and is valid as the only input; augmentation
features describe one aspect. Not yet agreed as the starting list.

- Full views: raw flux (216); tsfresh per band; catch22 per band plus mean and SD (144);
  ROCKET (20,000); MiniRocket (about 10,000); MultiRocket (49,728); Hydra; QUANT; RDST
  shapelets; interval statistics; SFA word counts; MantisV2, MOMENT and Chronos-2 frozen
  embeddings.
- catch22, RDST shapelets and interval statistics are full views. Each is the only input
  to one model in a published classifier (Catch22: random forest, 0.6679; RDST: ridge,
  0.6423; RIST: extra trees, 0.6285). Adelaide, 2026-11-20: "Okay then they belong as
  full views."
- Augmentation features: per-band summaries (54); colours, differences and ratios of
  per-band summaries over 15 band pairs (270); periodogram by position per band (108).
- Left out, need observation times, errors or redshift, which this version lacks:
  Lomb-Scargle periods; Bazin, SALT2 and Gaussian-process fits; redshift features.
- Not set: the number of intervals for interval statistics.

## Part 4. BEANS bats (Hagiwara et al. 2022, earthspecies/beans)

### 1. Source

- Code: github.com/earthspecies/beans, commit b278b1d (`scripts/bats.py`, `beans/datasets.py`,
  `scripts/evaluate.py`, `run_benchmark.py`). Paper: BEANS, arXiv 2210.12300, Table 2.
- Port: `tests/experiments/benchmarks/beans_bats_published.py`.
- Changes, approved by Adelaide 2026-11-20 ("1 and 2 - all seem good"):
  1. random_state 42 on decision tree, gradient boosting and XGBoost; BEANS sets none.
  2. scikit-learn 1.6.1, xgboost 2.1.4, torch and torchaudio 2.2.2, pandas 2.2.0, Python
     3.11; BEANS pins 1.1.1, 1.6.1, 1.12.1 / 0.12.1, 1.4.3, 3.8.
  3. WAVs read with scipy, not `torchaudio.load`.
  4. Features cast to float64 before the scaler; BEANS scales float32.
- Gradient boosting reruns 35 of 2,000 test calls below its published score. Cause not
  tested.

### 2. Train and test

- BEANS split: 6,000 / 2,000 / 2,000 calls by seed 42; md5 matches BEANS's files.
- It leaks: 46 source recordings have calls on both sides; for 222 of 2,000 test calls the
  same bat has a training call within 10 s.
- Grouped split, approved by Adelaide 2026-11-20 ("That sounds good"): calls whose
  recording times are at most 60 s apart, chained into one group. 7,634 groups, largest 16
  calls. 5 folds by `StratifiedGroupKFold`, shuffle, seed 42: 1,991 to 2,015 calls held out.
- Papers 2 and 3: `nested_cv_outer_folds` uses the same splitter and seed 42 when given a
  group per item, and passes the training calls' groups to the inner search. The group of
  each call is saved in `/Volumes/seagate/benchmarks_protocol/beans_bats/calls.csv`
  (`group_60s`).
- No split removes the colony set-up: bats 226 and 228 were recorded in one set-up only.

### 3. Results and room

Passed, Adelaide 2026-11-20: "it passes that check". Best published 0.816 (MetaPerch, not
checked in its paper) leaves about 368 of 2,000 test calls wrong; resolution about 17.

### 4. Models

Adelaide 2026-11-20: "I agree with your proposal in 4".
- In the screen, BEANS's configuration: lr, svm, decisiontree, gbdt, xgboost.
- Points of comparison only, published numbers: ResNet-18/50/152 from scratch and from
  ImageNet, fine-tuned VGGish, attentive and prototype probes, fine-tuned encoders. They take
  a spectrogram, audio or per-frame outputs, so they cannot score a view or a hybrid.
- Open: how the grid value is chosen on the grouped folds, which have no validation calls.

### 5. Views

38 full views drafted from the research file; no augmented features. Not discussed yet.
