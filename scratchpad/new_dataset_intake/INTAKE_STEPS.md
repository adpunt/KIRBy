# From literature review to the sweep: what each new dataset needs

Temporary file, 2026-11-17. To be deleted once it is moved somewhere permanent.

## Part 1. The steps, for all 9 datasets

Agreed with Adelaide, 2026-11-17.

### The goal of intake

Intake exists to prepare a dataset for the sweep. When the five steps are recorded, a new
chat must be able to implement the dataset in the sweep with no further discussion, and
the sweep run on the server must give scores that roughly match the published ones while
working inside KIRBy. Adelaide, 2026-11-26: "that is the entire goal, preparing it for the
sweep to check on the server that the numbers are right."

Preparing a dataset means working out how it maps onto KIRBy correctly: what its modelling
process is and whether it differs from KIRBy's, how its numbers compare with published
research, and what its feature space is. find_best_hybrid needs both proper models and
correct views; one view and an rf give it nothing to find. No shortcuts: stay true to the
source, and bring in new views and featurisations from the field for find_best_hybrid.

So every step ends in decisions a chat can build from, not in discussion:
- Each step's record names exact things: the code and commit, the files, the split
  code, each model with its settings, each view with its settings and group, and the
  published score each sweep result is checked against.
- A point is raised only if it changes what gets built. Each point comes with the
  options, their cost, and a recommendation, and is closed when Adelaide decides.
- A step is done when nothing in it is left for the implementing chat to choose.
- After step 5, the dataset's Part is the implementation brief. The implementing chat
  builds it, runs a short local check, and the server sweep checks the numbers.
- Context only, not part of intake: after the sweep, Adelaide decides which models and
  views go on to the KIRBy experiments.

How a chat works through the steps (Adelaide, 2026-11-26):
- One step at a time, in discussion. No reports, no drafts of the whole dataset.
- Before raising a step, the chat does the reading it needs: the source code, the papers'
  settings, and the KIRBy code that will run the dataset. Adelaide does not research.
- It brings only that step's decisions, each with options, cost and a recommendation, and
  records each decision in the dataset's Part once she makes it.

### Reference: how KIRBy splits the data

The Papers 2 and 3 run splits by ratio at four levels, read from the code on 2026-11-27.
N is the total number of rows; one row of the table is one level, and the percentages
are of N.

| Level | Where it is set | Fit on | Scored on |
|---|---|---|---|
| Outer folds: 5 folds, each 80/20. One search per fold (search_unit.py), on that fold's training rows | `nested_cv_outer_folds`, src/kirby/find_best_hybrid.py | 80% | 20% |
| Inner folds of the search: 5 folds of the outer training set. Every configuration is scored on these | `DEFAULT_N_FOLDS = 5`, src/kirby/find_best_hybrid.py | 64% | 16% |
| Hybrid-building split: 80/20 of the rows the hybrid is built on. The 80% gives the feature importances; the 20% scores each view to split the budget | `internal_holdout_split`, src/kirby/hybrid.py | 51.2% in the search, 64% when the chosen hybrid is rebuilt on the outer training set | 12.8% in the search, 16% at the rebuild |
| DNN early-stopping split: 20% of whatever the DNN is fit on | src/kirby/models/dnn.py | 40.96% as internal evaluator in the search, 51.2% downstream in the search, 64% downstream on the outer fold | 10.24%, 12.8%, 16% |

The smallest set anything trains on is 40.96% of N; the smallest set anything is scored
on is 10.24% of N.
- These experiments run only outer folds 0 to 2 unless the dataset is on a named list:
  c05, c06, c07, c10, c13, c15, c23, c33, n15, n26 (`OUTER_FOLDS_RUN`,
  tests/experiments/_shared.py).
- The meta-learner's inner folds and minimum rows come from RUN_SETTINGS.json, with no
  default in the code. Values not checked.
- Every split is group-aware, and the code refuses to split with no groups. A dataset with
  no natural groups needs groups defined at intake.

The ratios are fixed and N is chosen per dataset, as for QM9. This differs from the
authors' original papers in three ways (Adelaide, 2026-11-27):
- Data size: N must be large enough for this structure.
- Data splits: KIRBy does its own splitting. Where the authors have preset splits, say
  whether they can be replicated, and stay as close to the original method as possible.
- Results: KIRBy's scores are expected to be slightly lower than the published ones,
  because they are trained on less data. That is acceptable.

### 0. Does the problem map onto KIRBy?

Asked first. If the answer is no, the dataset is dropped and steps 1 to 5 are not done.
Steps 1 to 5 then work out how it maps.

KIRBy was built for small-molecule activity prediction: one row per sample, a fixed-length
feature vector per view, tabular models on top, a hybrid chosen from several views'
features. Answer conceptually, not by whether the code would run. Adelaide, 2026-11-26:
"Just because I can in theory adapt these problems to RF doesn't mean they should be
adapted in that way."

Reason from what find_best_hybrid needs: several views that each carry part of the signal,
models that suit those views, and room to improve. Some ways a dataset fails:
- The field's best method reads the raw input directly, and features add nothing.
- One view or model is far ahead of every other, so there is nothing to find.
- The best results are barely above a simple baseline, so there is nothing to improve.

These, and the two below, are illustrations Adelaide gave, not a checklist. A dataset can
fail in a way none of them names; say so when it does.
- Maps well: real estate price prediction (same models, used the same way; settings may
  differ). Maps badly: very long, per-minute, multi-year financial forecasting.
- Say where this dataset sits, and why: what the field's best methods do with the raw
  data, and whether that is "features, then an ordinary model".
- Name each place it does not map, for example views whose signal is spread thinly over
  thousands of features, or whether a fixed feature budget or a per-feature importance
  means anything for a view.

### 1. Source

Do we have a proper source? Is it right, and are we following it as closely as we can?
- Where the original code came from, with the published paper and table it reproduces.
- What we change from it, and why. Examples: seeding the random numbers, a different split.
- How the dataset becomes KIRBy input, decided here: one row, the label, the tasks, the
  train and test sizes, the folds and the metric. Check each against the KIRBy code that
  will run it: the screen contract (tests/experiments/screen/datasets.py) and
  find_best_hybrid at that number of rows.

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

Then three questions, answered conceptually for this dataset (Adelaide, 2026-11-26). The
answer is never "KIRBy's models run on it, so use them":
1. Downstream models. Do KIRBy's defaults (rf, lightgbm, dnn, xgboost, svm), or any of its
   other prebuilt models, make sense for this kind of data? Or are they tossed and the
   field's own models put in? For each model: which paper on this dataset uses it on
   features, and on which kind of view.
2. Internal evaluators. If a model makes no sense downstream, it makes no sense as an
   internal evaluator either (default list: match, rf, lightgbm, dnn). If question 1
   gives defaults plus others, say whether the number of internal evaluators should grow.
3. Published classifiers that cannot take a view (each tree draws its own intervals, an
   ensemble of classifiers, a network fed raw sequences): list them; they run as
   themselves on the raw data.

Settings are chosen per dataset too, for downstream models and internal evaluators alike.
KIRBy's defaults are generic settings, not the answer: where the field uses other settings
(for example fewer epochs, or more trees at a lower learning rate, on large datasets), the
sweep tests the published settings, or their equivalent on the same kind of model, beside
the defaults. There is no cutoff by size; Adelaide, 2026-11-27: "That's up to the user.
I can provide generic settings but they know they're data."

Then how each model's own settings are chosen inside Papers 2 and 3's 5 folds, where the
source chose them another way (for example on a fixed set of validation rows).

### 5. Views

The manual research step. It starts from the literature review,
`~/Downloads/<dataset>.md`. Build a starting set of views large enough for the method to
work. Nothing is built until this step is done.
- Views are listed in two groups: full views, and augmented features. Adelaide,
  2026-11-20: "views need to distinguish between full views and augmented features".

## Part 2. Speck (Gohr 2019, agohr/deep_speck)

### 0. Does the problem map onto KIRBy?

Yes, with two limits. Adelaide, 2026-11-27: "we'll call that a yes and move on".
- Room: Gohr's ResNet on the raw pair (0.929 / 0.788 / 0.616 at 5 / 6 / 7 rounds) is not
  beaten by any published single-pair model. Features close most of the gap for trees
  (Benamira 2021 LightGBM at 5 rounds: 0.7634 on raw bits, 0.923 on mask-probability
  features). A hybrid can at best approach the ResNet.
- Per-feature importance: a single bit carries almost no signal; the signal is in XORs
  between bits. Adelaide, 2026-11-27: "that's why I want to explore alternative features,
  especially aggregation ones".

### 1. Source

- Code: github.com/agohr/deep_speck, commit c3364c0. Paper: Gohr 2019 (ePrint 2019/037),
  Table 2.
- Port: `tests/experiments/benchmarks/b11_speck_published.py`. Matches the Speck test vector
  and Gohr's own encryption output at 7 rounds.
- Change: the random numbers are seeded (Gohr uses `os.urandom`). Adelaide, 2026-11-17:
  "this is a good change".
- Other differences, not significant. Adelaide, 2026-11-20: "I don't think any of these are
  significant issues".
  1. Gohr's pre-trained networks are run with a numpy copy of their layers, not Keras.
  2. His shipped networks have 1 residual block; his paper's 5-round network has 10.
  3. Only the real-or-random setting is built, not his real-differences setting.
- More than one number of rounds. Adelaide, 2026-11-26: "I should be testing this with
  multiple rounds".
- Tasks are 5, 6 and 7 rounds, each run separately. Adelaide, 2026-11-26: "yes 5, 6, and 7
  is good".

### 2. Train and test

- Every pair is generated fresh from its own random key and message. Two pairs are
  identical only if key and message both match: about 1 in 10^16 across all train-test
  comparisons.
- 5 folds, each a fresh train set and test set from its own seed. Adelaide, 2026-11-26:
  "Yes 5 folds like th others"; 2026-11-20: "Just make sure future splits will work this
  way and its good".
- Open: the training and test sizes.

### 3. Results and room

Passed. Adelaide, 2026-11-20: "Yea that checks out, I'm okay with that."

Gohr's pre-trained networks, scored on 10^6 new pairs per number of rounds:

| rounds | here | Gohr 2019 Table 2 |
|---|---|---|
| 5 | 0.9279 | 0.929 |
| 6 | 0.7873 | 0.788 |
| 7 | 0.6160 | 0.616 |
| 8 | 0.5144 | 0.514 |

### 4. Models

Adelaide, 2026-11-20: "Lets just use my defaults + resnet as a comparison ... SO the
defaults + the additoinal configurations". 2026-11-26: "I would like to run all except SVM".

In the screen:
- KIRBy defaults, settings from `DEFAULT_MODELS` in src/kirby/find_best_hybrid.py: rf,
  xgboost, lightgbm, dnn.
- Benamira 2021 LightGBM: binary objective, 1,000 trees, learning rate 0.01, 50 leaves,
  max depth 10, min 10 rows per leaf, max bin 50, L1 0.1, L2 0; features standardised.
- Benamira 2021 random forest: 100 trees, max_features 'auto', max depth 100, min split 5,
  min leaf 2, bootstrap.
- Benamira 2021 MLP: dense 64, 64, each with batch normalisation and ReLU; sigmoid output;
  Adam; mean squared error; 20 epochs; batch 500; cyclic learning rate 0.002 to 0.001 over
  10 epochs.
- Ebrahimi 2021 MLP: dense 32, 64, 32; sigmoid output; Adam; binary cross-entropy plus L2
  10^-5; 10 epochs; Gohr's cyclic learning rate; other settings Keras defaults.
- Sources: Benamira's code, github.com/AnonymousSubmissionEuroCrypt2021/A-Deeper-Look-at-Machine-Learning-Based-Cryptanalysis-v5
  (commit 6715533), `src/classifiers/classifier_all.py` and
  `src/classifiers/nn_classifier_keras.py`; Ebrahimi, ePrint 2021/1479, section 3.
- Published training sizes: Benamira 10^6 train, 10^5 validation; Ebrahimi 10^7 train,
  10^6 validation.

Not in the screen: both SVMs (KIRBy's default, and Benamira's linear SVM, C 0.025).

Comparison score only: Gohr's ResNet, published scores in step 3. It takes only the raw
pair, so it cannot score a view or a hybrid. Adelaide, 2026-11-20: "It won't be used with
the hybrids, but it'll be used as a comparison score." If Speck is used, it goes in
PAPER_2_OUTLINE.md as a comparison score.

Large row counts (Adelaide, 2026-11-27):
- Defaults and published versions both run in the sweep: "the sweep can test defaults +
  the published versions and I'll decide from there. There may have to be cutoffs for time."
- N is not capped to make the defaults fit: "i'd prefer not to do this unless absolutely
  necessary".
- No large-data flag and no cutoff. Adelaide, 2026-11-27: "No flag. I can make these
  decisions on a per-dataset basis." and "I abandon the idea of trying to determine a
  cutoff for large/specific chnages. That's up to the user."
- Model settings are chosen per dataset, for downstream models and internal evaluators:
  "I need to make it easier to modify the internal and external evaluators settings to fit
  those that are common with each dataset." For Speck: "test published settings or the
  equivalent on similar models".

Open: the internal evaluators.

### 5. Views

Not started. Adelaide, 2026-11-20: "we'll start that later".

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
