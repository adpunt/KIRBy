# Vocabulary for Papers 2 and 3

Every document, code comment and paper draft uses these words. The source is `thesis.tex`;
line numbers were read on 2026-11-17. A thing not listed here has no agreed word yet: ask
Adelaide, do not make one up.

## Representation or view

> "Views in the context of molecular data is molecular representations, so for a paper
> talking only about molecules its appropriate to use reps. But for a paper talking about
> ML more generally, it's views." (Adelaide, 2026-11-17)

- Paper 2 is the general one (Matbench, reactions, nucleotide sets, DroneRF): **view**.
- Paper 3 is molecules and antibodies: **representation**.
- The method keeps its name in both: **adaptive hybrid representation** (Adelaide, 2026-11-17).
- `NEW_DATASETS.md` already says view for the non-chemistry datasets.

## The search

| word | what it is | thesis.tex |
|---|---|---|
| hybrid search | the search that builds the adaptive hybrid representation | 1399 |
| configuration | one list of settings for feature selection | 1471 |
| setting | one searched thing (six in total); never "axis" | 1484 |
| importance method | how features are scored; never "feature ranking" | 1484 |
| allocation method | how the budget is shared between representations | 1489 |
| global pool | the allocation method of that name; "pool" means nothing else | 1489 |
| budget | the number of features the hybrid may hold | 1484 |
| representation count | 2, 3, 5 or all; never "number of representations" | 1467 |
| internal evaluator | the model that scores inside the search; never "scorer" | 1576 |
| downstream model | the model fitted on the hybrid | 1484 |
| blueprint | a finished search's saved settings; never "recipe" | 1399 |
| fast tier | the shorter fixed list of settings | 1623 |
| search grid | each dataset's values of each setting; never "decided grid" in prose | 1619 |
| per-fold search | a search on one outer fold's training molecules | 1607 |
| earlier search of the whole dataset | the old search over all molecules; never "precompute" or "stored search" | 1703 |
| general search | the standard hybrid search | 2808 |
| kNN search | the search with a kNN model as the downstream model, budgets up to 500 (see the note below) | 2808 |
| similarity search | the search under the nearest-neighbor objective | 2808 |
| predictive hybrids, kNN hybrid, similarity hybrid | the top hybrids of those three searches | 2812-2814 |
| first hybrid, second hybrid | a model's two best hybrids | 3495 |
| augmentation features, donors | extra features added to a hybrid, and the feature sets they come from; never "extra features" | 1593 |
| decoy donors | the two fake donors put among the real ones in the augmentation experiment: pure random noise, and features copied from the representation with a little noise added; any feature taken from them is junk (Adelaide, 2026-11-18: "fine decoy donors") | |
| adaptive pruning | dropping settings after the first folds | 1603-1609 |
| spread across folds | never "fold spread" | 2188 |

The kNN search searches kNN and nothing else (her decision of 2026-11-04,
`AC_GRID_TEST_PLAN.md`; built by that plan's first prompt).

## Base and single representations

| word | what it is | source |
|---|---|---|
| base representation | one representation scored on its own, as a baseline | thesis.tex 2124; code `top_base_reps` |
| top base | the best-ranked base representation for a model; not the same thing as a base | her words, `AC_GRID_TEST_PLAN.md` |
| `base_1`, `base_2` | code names for a model's best and second-best base representation; in prose, "first" and "second base representation" | `_shared.py:1728` |
| single representation | any one representation, as against a hybrid | thesis.tex 1491 |

The thesis also uses "base" for a second thing: in augmentation, the finished hybrid is the
base and the feature sets that add to it are the donors (1593). Say which one is meant.

## Noise

| word | what it is | thesis.tex |
|---|---|---|
| noise condition | Gaussian, Student-t, Laplace, Outlier, Grouped-wider, Grouped-shifted, Censoring; never "noise kind", "damage kind", "noise type" (Adelaide, 2026-11-17) | 209, 256-268 |
| noise level | a fraction of the SD of the clean training labels; never "dose" | 221 |
| noise draw, draw | one random draw of the injected noise | 342 |
| widening factor | λ = 3 | 229, 267 |
| replicate | one QM9 split and draw | 434-436 |
| seed | one repeat of a random control | 3679, 3885 |
| injected | noise added by us | — |
| damaged | a label carrying injected noise; never "corrupted" | 3606 |
| flag, flagged | a molecule a detector or cleanup marks | 3675 |
| cleanup, cleaning method | what acts on the flagged molecules | 3675 |
| noise mitigation method | the umbrella for cleanups | 3611 |
| noise-handling methods | the umbrella for everything in the noise chapter | 3778 |
| noise-robust training | changing the model instead of the labels | 22, 3709 |
| detector | gives each molecule a score for how likely its label is wrong | 3558 |
| flagging method | hands back a yes-or-no list of molecules instead of a score (`conformal_bh`, `two_learner_small_loss`, flags of known content). It feeds only cleanups that act on a list, and is not graded like a detector (Adelaide, 2026-11-17) | |
| shuffled detector scores | the control that shuffles the detector's scores across molecules, then runs the cleanup on them. Random deletion is this control with a deleting cleanup, not a separate one (Adelaide, 2026-11-17) | 43, 3679 |
| shuffled cleanup results | the control that runs the cleanup on the real scores, then shuffles its result (which molecules were deleted, each molecule's new weight or label) across molecules (Adelaide, 2026-11-17). Code: `controls.py` `shuffled_twin` | |
| flags of known content | flags built from the injected noise; never "oracle flags" in prose | 3951 |
| Cliff put-back | the method, capitalized; never "spares" | 42, 3675 |
| Random put-back | the control for Cliff put-back: puts back the same number of molecules, picked at random from the non-cliffs (Adelaide, 2026-11-18) | header at 4157 |

"Put-back" with a hyphen is the name of a method. "Put back" without one is the verb: "the
molecules are put back". The thesis follows this (six "Cliff put-back"; table headers
"Cliff molecules put back", "Random molecules put back").
| Multistage cleaning | never "trimmed refit", "repeating find-and-delete" | 3730 |
| configuration (noise chapter) | a model on one input, such as "First single", "First hybrid" | 3952 |

"Configuration" therefore means two things in the thesis: one list of search settings (1471),
and a model on one input (3952). Say which one is meant.

## Similarity and activity cliffs

| word | what it is | source |
|---|---|---|
| activity cliff (AC) | | thesis.tex 12 uses |
| coefficient | the formula comparing two feature vectors | thesis.tex 149 |
| cutoff | her word for the similarity limit; the thesis says "threshold" (3675) | her outline |
| cliff share | the share of molecules that are cliffs; never "target share" (Adelaide, 2026-11-17) | |
| neighbor, neighbor count, neighbor weighting | American spelling; never "how the neighbours are counted" | thesis.tex 3594 |
| space | where the neighbors are found; never "neighbour lists" | thesis.tex 3594 |

## Samples and folds

- Molecules in Paper 3; samples in Paper 2 (thesis.tex 1434). Never "rows" for either.
- Folds, outer fold, held-out. Never "rows" for a split.

## Server words

Not in the thesis, used in the runbooks only: ledger, unit, step, claim, gate, worker, job,
and the two runs of the experiment that pairs every detector with every cleanup: the
**standard cleanup** run, then the **variant cleanups** run on the results she picks
(Adelaide, 2026-11-18: "change it to standard cleanup and variant cleanups. if its internal
only I know what that means"). Never in a paper.
One word per thing: "step", never "stage"; a Claude conversation is a "chat".

## Rejected words

arm, pool (except global pool), cell, round, stage (for runs), knob, recipe, scorer, piece,
dose, chain, read-across, SAC, columns (say features), rows (say molecules, samples or folds).

## Not named yet

Each needs Adelaide's word. Until then, documents describe the thing in full.

Nothing at present.

## Not yet applied to the documents

The audit of 2026-11-17 found these in the kept documents. Each is applied once the list above
is settled; code names in backticks, published names, and her quoted words never change.

| documents say | becomes |
|---|---|
| scorer (DECISIONS.md) | internal evaluator |
| scorer (AC_GRID_TEST_PLAN.md, the eight similarity scores) | metric |
| piece | configuration, with its fold, noise level and noise draw |
| noise kind, kind, damage kind, noise type | noise condition |
| dose | noise level |
| version of the labels | noise condition, level and draw |
| recipe | blueprint, or configuration |
| axis | setting |
| feature ranking | importance method |
| number of representations | representation count |
| decided grid, thesis grid | search grid |
| precompute, whole-dataset search, stored search | earlier search of the whole dataset |
| general tier, predictive search | general search |
| similarity-objective search | similarity search |
| distance search, distance tier | what the code fits (see the kNN note) until the kNN-only search is built |
| `base_1`, `base_2` bare in prose | first, second base representation |
| single feature set (PAPER_3_OUTLINE.md) | single representation |
| model input | configuration (noise chapter sense) |
| extra features | augmentation features |
| importance-multiplied | importance weighting |
| how the neighbours are counted, counting scheme | neighbor weighting |
| neighbour lists | space |
| neighbour | neighbor |
| oracle flags | flags of known content |
| random twin, random deletion, random comparison, yardstick | shuffled detector scores |
| shuffled twin | shuffled cleanup results |
| flag detectors | flagging methods |
| draws of a random selection | seeds |
| cliff put-back, spares | Cliff put-back |
| random put-back, random exempt | Random put-back |
| trimmed refit, repeating find-and-delete | Multistage cleaning |
| clean-label check | QM9 test |
| corrupted | damaged |
| changing the model | noise-robust training |
| cliff rule (a cliff definition) | definition of an AC |
| target share | cliff share |
| fold spread | spread across folds |
| training rows, held-out rows | training or held-out molecules (Paper 3), samples (Paper 2) |
| neural net | DNN |
| band (the 1% window) | the rule written out |
| representation (Paper 2 documents) | view |
| stage (in prose) | step |
| decoy blocks, decoys, planted decoys | decoy donors |
| first run, second run (of the detector and cleanup experiment) | standard cleanup run, variant cleanups run |
| chain (what the model is fitted on, paired with what the detector reads) | no word; written out each time, e.g. "model on the base representation, detector on the hybrid" (Adelaide, 2026-11-18: "it's niche enough you can just describe it") |
| arm, pool, cell, round | what each one is, written out |
