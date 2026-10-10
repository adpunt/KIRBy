# Thesis terminology decisions (2026-10-10)

Decided by the author on 2026-10-10, in the final edit pass before submission.
Applies to `thesis.tex`, `main.tex` (List of Abbreviations) and
`thesis_rework/stage1_takeaways_bullets/additional_files_body.tex`.

## Abbreviations

- Every abbreviation in the list is defined once, at its first use in the thesis
  ("random forest (RF)"), and the abbreviation is used after that. Chapter and
  section titles keep the full words.
- The Additional files are a separate document: each abbreviation is defined once
  at its first use there too.
- **Activity cliff is the one exception:** defined again in every chapter that uses
  it, as "activity cliff (AC)", then AC; plural ACs. Names that contain the word
  keep it: "cliff-aware gap", "cliff put-back", "cliff molecules".
- Every abbreviation used in the text must be in the list, and every list entry
  must be used.

## Models

There are four different networks in the thesis. They must not share a name.

| Name in thesis | What it is | Where |
|---|---|---|
| NN-α, NN-β (and BNN-α/β, VBLL-α/β) | hidden layers 128-64 and 128-128 | Ch3 only |
| DNN | hidden layers 256-128-64, dropout 0.2 (`kirby/models/dnn.py`; `_build_dnn` in `lnl_methods.py` matches it) | Ch4 to Ch8; also the network inside the 12 LNL methods |
| neural network (NN) | the model family in general | Ch2 |
| variance-head network, evidential network | separate models in Ch7 | Ch7 |

- "The network" for the 256-128-64 model is replaced by "DNN" everywhere.
- **SVM** everywhere for the support vector model. In the code it is SVR on regression
  datasets and SVC on classification datasets. Not "SVR", not "support vector
  regressor".

## Similarity and distance (chosen by Claude at the author's request)

- "Similarity" = the idea (similar property principle, Tanimoto-type scores).
- "Distance" = a computed number in a feature space (Euclidean, Manhattan).
- SPP: defined once, then abbreviated.
- Ch6: "distance search" / "distance hybrid" (kNN downstream model) are renamed
  **"kNN search" / "kNN hybrid"**. "Similarity search" / "similarity hybrid" (the
  search scored by how accurately each molecule's label is predicted from its five
  nearest neighbors) keep their names. "Neighbor-label" was rejected because it
  collides with the Ch7 method "neighbor-label gap".

## Noise methods

- The 12 methods from the noisy-labels literature (Ch4, expanded in Ch7) are
  **"the LNL methods"**, and the experiment is **"the LNL sweep"**. Never
  "published methods" or "published sweep".
- Every noise method has one fixed name, taken from the three method tables in Ch7
  (`tab:noise:detectors`, `tab:noise:mitigation`, `tab:noise:robust`). That name is
  used verbatim at every mention, marked with the `\method{}` macro, plain text since 2026-10-10 (defined in
  `preamble.tex`).
- "Detector" is only the group word ("the nine detectors"). A specific method is
  never called "the detector"; the sentence names it.
- The same rule applies to other generic stand-ins for a named thing: "the network",
  "the cleanup", "the forest", "the residual".
- Vague table names were renamed by Claude at the author's request; see the
  rename table below.

## Canonical method names

Chosen by Claude on 2026-10-10 at the author's request ("do what you think is best").
Left column: the name used verbatim everywhere (in running text inside `\method{}`).
Right column: what it replaces.

### Detectors (`tab:noise:detectors`)
| Canonical | Was |
|---|---|
| Forest residual | Out-of-fold forest residual; the residual; the detector based on a random forest's held-out errors |
| LightGBM residual | Out-of-fold LightGBM residual; the residual |
| Neighbor-label gap | neighbor gap |
| Gap over neighbor spread | (unchanged) |
| Cliff-aware gap | (unchanged) |
| GP spread and residual | Gaussian process; the GP; GP spread (clashed with the GP downstream model) |
| Heteroscedastic GP spread | Heteroscedastic GP (clashed with the LNL method of the same name) |
| Variance-head network | (unchanged) |
| NGBoost spread | NGBoost (clashed with NGBoost as a model) |
| Nearest-neighbor Shapley value | the Shapley value |
| Cleanlab score | cleanlab regression score; cleanlab |

### Mitigation (`tab:noise:mitigation`)
| Canonical | Was |
|---|---|
| Delete flagged | Delete (also an action name) |
| Five-detector delete or move | Detectors compared |
| Score down-weight | Down-weight (also an action name) |
| Neighbor-mean smoothing | Label smoothing (clashed with standard label smoothing) |
| Move toward prediction | (unchanged) |
| Score-scaled shrink | (unchanged) |
| Neighbor weights | (unchanged) |
| Exponential down-weight | Detector weights |
| Close-pair weights | (unchanged) |
| Top-share down-weight | Residual flags |
| Cliff put-back | (unchanged) |
| Constant lift; Own lift; Truncated-normal mean; Censored EM; Delete clipped | (own lift and delete clipped were unnamed) |
| Scaffold-family offset | Scaffold-group offset (text said family) |
| Source offset; Source weights; Count and spread weights; Shrink by spread | (unchanged) |
| Huber blend | Huber teacher blend ("teacher" never explained) |
| Neighbor-agreement weights | (not in the table; unchanged) |

### LNL methods (`tab:noise:robust`) and other robust training
The twelve LNL method names are unchanged. Other names: Retuning, Seed averaging,
Stacking, 1,000-tree RF, Huber loss, Two-learner selection (co-teaching), One row per
measurement, Log-count feature, Interval model.
