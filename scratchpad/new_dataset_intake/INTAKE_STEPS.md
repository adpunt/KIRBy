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
- Not yet listed: every other difference from the source.

### 2. Train and test

Not checked.

### 3. Results and room

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

Adelaide, 2026-11-17: "I need to test all of those 6 models including the resNet in the
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

Not set: the configurations of models 2 to 6. Only model 1 has a published configuration
in hand.

### 5. Views

Not started. The research file (`~/Downloads/speck.md`, Table B) lists 21 operations.
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

Not discussed yet.

### 4. Models

Not discussed yet.

### 5. Views

Not started.
