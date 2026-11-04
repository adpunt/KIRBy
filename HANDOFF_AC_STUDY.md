# Handoff: the activity cliff study

Version 1, 2026-11-04. Written to Adelaide's own structure, which she set out on
2026-11-04 after rejecting the previous handoff's framing.

**Supersedes `HANDOFF_PAPER3_CLIFFS.md`**, which is kept only for the measurements already
recorded in it. Its section A holds eight finished pieces of work with their numbers, and
its sections E, F and G hold the sweep grid, the places cliffs enter the noise experiments,
and the settled decisions. Nothing in it should be treated as the plan.

`PAPER_3_OUTLINE.md` is the documentation of record. This handoff exists to get the work
**implemented and ready to run inside the ledger, on the available datasets.** Nothing
here replaces the outline; where the two disagree the outline wins and this file is wrong.

---

## 1. What this study is for, and the two ways it ends

Three claims, in Adelaide's order:

1. **That activity cliffs can be detected accurately**, on a criterion more flexible than
   ECFP4 Tanimoto at a fixed cutoff, which does not work on every dataset.
2. **That detecting them makes noise mitigation work better.**
3. **That the study can offer a clearer definition of an activity cliff, stated twice —
   with hybrids and without** (Adelaide, 2026-11-04). One definition that takes the
   existing ones into account and says why it is better. Conditional on 1 and 2.

Claim 3 is the writing goal, and it has a constraint that decides whether it can be made
at all. "Better" cannot be asserted; it has to be the measured outcome of section 4, on
all three counts: closer agreement with the one external cliff set that exists, a cliff
set that survives changing the representation, and a larger gain in section 4.3 than
sparing the same number of molecules at random. A definition that wins on none of those is
a definition with no claim behind it, however much simpler it reads.

The paper needs both statements, because they do different jobs. **Without hybrids** the
definition runs on a fingerprint, so it is comparable with the published definitions and a
reader with none of this machinery can reproduce it. **With hybrids** it uses the searched
feature space, which is where any advantage over the published definitions would come
from. Which of the two is the paper's headline is Adelaide's.

**If 1 and 2 both hold well enough, Paper 3 splits into two papers:** noise methods with
hybrid features, and cliff detection with cliff-aware noise mitigation.

**If they do not hold, or hold weakly,** a smaller part of this study is reported inside
Paper 3, confined to how cliffs connect to noise — including the outcome that cliffs can
be identified but not in a way that helps.

Both endings are reportable. The design below has to produce the number that decides
which, and that number is in section 4.

---

## 2. What this builds, and what it replaces

**This adds a new set of activity cliff experiments, to be built alongside the restructured
noise experiments and run for Paper 3.** The old cliff code is replaced, not audited. It
relied on several definitions, none of which Adelaide considers sufficient, and it computed
every cliff set from ECFP4 — so the hybrids, which were designed for this, were never used
to find a cliff. That is the oversight this study exists to correct and it needs no further
diagnosis.

Lift from the old files for mechanics only: how a pairwise matrix is built and thresholded,
how a cleanup spares a named set, how a random control is drawn. Never for a definition.

**Settled, so not reopened:**

**Scope: small molecules only** (Adelaide, 2026-11-04). Paper 3 as a whole keeps small
molecules and antibodies; the cliff work is small molecules. Searched 2026-11-04: no
published work applies the term to antibodies, biologics or protein variants, and the same
phenomenon in protein engineering is written about as epistasis or a rugged fitness
landscape. No external antibody cliff set exists, so the agreement tests of section 4
cannot run on one. Consequences are recorded in `HANDOFF_PAPER3_CLIFFS.md` section G.

**No dataset outside drug discovery** (Adelaide, 2026-11-04). Kepler and DroneRF have
nothing outside their features to measure similarity with, so their only cliff set would be
built from the representations being compared. The reaction sets are Paper 2 only. Five of
Matbench's six targets are computed and carry no measurement noise.

**A cliff set built from the representations being compared is refused, not flagged**
(Adelaide, 2026-11-04, implemented the same day). Two files each held a copy of the same
fallback and both now raise. One of the two paths was undocumented: a molecular dataset
whose fingerprint calculation threw fell silently into the circular reference.

**Every definition is computed in a hybrid as well as in single representations**
(Adelaide, 2026-11-04). This is the point of the study. A definition that only ever ran on
a fingerprint is not finished, and an experiment that scores the hybrid on cliffs a
fingerprint chose does not answer anything she asked for.

**One definition module, not several.** Every AC experiment reads its definitions from one
place, so two experiments cannot disagree about what a cliff is. `cliff_rules.py` is that
place today and already holds 32 similarity criteria, 10 gap criteria and 10 named
combinations; what it lacks is a hybrid similarity, which is TBD2.

---

## 3. The definitions, as Adelaide set them out

### Which pairs count as similar

**1. A fixed similarity score above a cutoff.**

| Version | What it is | Unknowns |
|---|---|---|
| One coefficient on one representation | The general form. Not a single rule but a set | TBD1 coefficient, TBD2 representation, TBD3 cutoff |
| MoleculeACE, original | A pair is similar if **any one** of three scores clears 0.90: fingerprint overlap, overlap of the core framework with side groups stripped, or how many single-character edits turn one SMILES into the other | TBD4 cutoff |
| MoleculeACE, hybrid | The same shape with **one** overlap score on a hybrid instead of three | TBD5 cutoff, separate from TBD4 |
| `mmpdb` | A pair is similar when one molecule becomes the other by swapping one chemical group. No score and no cutoff | needs installing |

**2. The top share of pairs by any score from (1).** Per dataset, so "similar" means a
different score on every dataset.

### How far apart the labels have to be

1. Twice the median absolute deviation of that dataset's labels.
2. The gap moved until some share of molecules come out as cliffs. TBD6.

**Not an option, and the previous handoff had this wrong:** the spread of repeat
measurements of one compound. Its only job is to say whether a chosen gap is larger than
the assay's own noise. It is a check on choice 2.

### Models that predict cliffs directly

Not definitions. They predict a cliff label somebody else assigned, and their role is in
section 4.

| | What | State |
|---|---|---|
| GraphCliff | Short-long range gating, 2025. Reports cliff-specific error on MoleculeACE's 30 targets | published numbers only |
| SemiMol | Semi-supervised, 2026. Reports on the same 30 | published numbers only |
| The 2025 foundation-model comparison | Tested descriptor-based foundation models on 29 of the 30, and reports them separating cliff from non-cliff on only 4 of 29. One model it tested, Chemeleon, is in this repo's representation pool | published numbers only |
| `opig_activity_cliff_qsar` | Dablander's replication, **already running here** on dopamine D2, factor Xa and SARS-CoV-2 Mpro. Predicts each pair's activity difference from the two molecules' predicted activities and scores the cliff call | built, its cliff-call correlations computed and never reported |
| A model trained here on MoleculeACE's cliff labels | TBD7 | not assessed |

---

## 4. How a definition gets checked. This is the centre of the study.

Adelaide, 2026-11-04: *"How do I confirm if I actually got cliffs? Well, I can't
conclusively but I have some options."*

**What the literature does, searched 2026-11-04.** No published method tests whether a
cliff is real. What exists is curation applied *before* detection, which prevents false
cliffs rather than confirming true ones:

- Equilibrium constants only, never IC50, Ki and Kd pooled, because mixing assay types
  manufactures cliffs out of unit differences.
- Exact relations only; censored values dropped.
- Records carrying ChEMBL's own data-validity flag excluded.
- A potency gap of at least two orders of magnitude for a "high-confidence" cliff.
- **MoleculeACE drops a compound when the standard deviation of its own repeat
  measurements exceeds 1 log unit.** Verified against van Tilborg et al. 2022 on
  2026-11-04; it removed about 4.33 per cent of molecules across their 30 targets. This is
  the only published rule found for discarding a compound whose own measurements disagree
  too much, and it is applied BEFORE cliff detection, as prevention.
  **Measured here the same day:** on hERG IC50 that rule drops 69 of the 1,218 molecules
  measured more than once, which is 5.67 per cent of those and 0.70 per cent of all 9,802.
  The paper does not say why 1 log unit, and this dataset's own median disagreement is
  0.17 pIC50 with a 90th percentile of 1.07, so their threshold sits near this assay's
  90th percentile. Whether Paper 3 adopts it is open; it is a curation decision applied to
  the data, so it changes every number downstream.

The one wet-lab validation in the literature is prospective: the SAR Matrix method
predicted an MMP-1 cliff, the compounds were synthesised and retested, and the predicted
partner came out 60-fold more potent. Not available here.

**So the functional test in 4.3 is not a weaker substitute for a published method. There
is no published method.** That is worth stating in the paper.

### 4.1 Agreement test one: against MoleculeACE's own labels

Run each definition on their 30 targets and count how many of their cliff molecules it
recovers and how many it adds. Already built and run for the ten named rules; it
established that our copy of their rule reproduces theirs almost exactly and the thesis's
own rule does not.

**What is new:** sweep the cutoff rather than scoring one point, and read where agreement
peaks. That turns TBD3, TBD4, TBD5 and TBD6 from judgements into measurements on 30
targets.

**What it cannot do:** their labels come from their rule, so perfect agreement means
reproducing their definition.

### 4.2 Agreement test two: against the published cliff-prediction models

Put our cliff-specific error beside GraphCliff's, SemiMol's and the foundation-model
comparison's, on the same public splits. **And beside plain ECFP4 Tanimoto at the standard
cutoffs**, which the setup already produces and which has to be named explicitly as the
baseline rather than left implicit (Adelaide, 2026-11-04).

### 4.3 The functional test: D, D', D'' — Adelaide's design, 2026-11-04

Her words, kept as the specification:

> Look at the best outlier detection methods, call one M, and then subsequent best label
> handling methods, call one L. Take dataset D. Run M on D, giving you a list of outliers.
> Now we have to determine if the outliers are noise or ACs. Take the AC detection method
> in question and run it, cross-check with the list of outliers. Now first, apply L without
> AC-awareness to get D'. Then apply L with AC-awareness to get D''. Compare the results of
> training on D, D', and D''. Assuming M and L work, for a higher-is-better metric, we
> should see D'' > D' > D. Potentially D > D' if the removal of ACs was actually damaging.
> But the important thing being D'' should be the best. And then the difference between D''
> and D and D'' and D' should indicate the relative performance of different AC detection
> methods.

**One thing has to be added or the comparison cannot be read.** D'' differs from D' in two
ways at once, not one: the molecules the cliff definition names are spared, **and** fewer
molecules are cleaned in total. A gain could come from either. So every D'' needs a twin in
which the same *number* of molecules is spared, chosen at random. Call it D'''.

- D'' better than D''': the gain comes from WHICH molecules were spared, so the definition
  identified something.
- D'' no better than D''': the gain comes from cleaning fewer molecules, and the
  definition contributed nothing.

So the four trained datasets are D (nothing done), D' (cleaned, no cliff-awareness),
D'' (cleaned, the cliff definition's molecules spared) and D''' (cleaned, the same NUMBER
of molecules spared, chosen at random). D''' is a named arm of the experiment, not a check
run afterwards, and a result reported without it cannot be read.

The code for both exists, in a retired file: `n25 put_back` and `n25 random_exempt_mask`.
Lift, never edit.

**The number that decides the paper split** is D'' minus D''', per dataset and per
definition. Claim 1 of section 1 rests on 4.1 and 4.2; claim 2 rests on this.

**Known limit, measured.** The random twin cannot run where most training molecules are
cliff molecules, recorded as impossible on Caco-2, hERG blocking and LogD under twice the
median absolute deviation. Whether a chosen definition fixes that is one of the census
numbers. This is a reason to prefer a definition with a bounded cliff share, and it is a
measurement, not a verdict.

### 4.4 Two more functional tests, cheaper and weaker

**Are the molecules predicted worse?** Split the held-out molecules into the definition's
cliff set and the rest, compare the error. A definition finding real chemistry gives a
bigger gap; one picking arbitrary pairs gives none.

**Does the cliff set survive changing the representation?** Hold the definition fixed, swap
the representation, count how many molecules stay cliffs. The thesis already measures a
version of this on one dataset: holding the gap and the number of closest pairs fixed, the
cliff molecule count runs from 138 to 267 across representations.

---

## 5. The seven unknowns, and where each answer comes from

Two of the seven are answered by reading results that already exist. Four are answered by
one sweep. One is Adelaide's judgement.

**TBD1, which similarity coefficients. A set, not one.** Answered from her own results,
read 2026-11-04.

**Where the numbers are.** `tests/results/framework/f34_full_sweep/` is the file to read,
not `f34_importance_sweep/` as version 1 of this handoff said. The full sweep carries all
14 coefficients and no importance weighting; the importance sweep carries 8 of the 14
crossed with importance weighting. `f34_neighbour_weighting_cross__four_coefficients_2026_09_28/`
carries only manhattan, euclidean, cosine and minkowski p=0.5 on 7 fingerprints. Memory
records that framework local results count as evidence for the nearest-neighbour question
from f34 onwards, which is this question.

**What was scored.** Predicting a held-out molecule's label from its nearest training
molecules, 5 folds, on four of Paper 3's small-molecule datasets: hERG Ki, Caco-2,
MDR1-MDCK and hERG blocking. hERG Ki, Caco-2 and MDR1-MDCK are RMSE in label units, lower
better; hERG blocking is AUC, higher better.

**Table A. One row is one similarity coefficient.** The number is the best score that
coefficient reaches on that dataset, searching over every representation, every neighbour
weighting and every neighbour count, mean over the 5 folds. The last column counts how
many of that dataset's 12 representations the coefficient is defined on at all, in the
order the dataset columns run.

| coefficient | hERG Ki RMSE | Caco-2 RMSE | MDR1-MDCK RMSE | hERG blocking AUC | representations defined on |
|---|---|---|---|---|---|
| manhattan | 0.5604 | 0.3173 | 0.4862 | 0.8475 | 12/12 12/12 12/12 12/12 |
| minkowski p=0.5 | 0.5653 | 0.3190 | 0.4886 | 0.8487 | 8/12 7/12 7/12 7/12 |
| minkowski p=0.3 | 0.5698 | 0.3202 | 0.4890 | 0.8490 | 8/12 7/12 7/12 7/12 |
| euclidean | 0.5700 | 0.3161 | 0.4886 | 0.8423 | 12/12 12/12 12/12 12/12 |
| correlation | 0.5813 | 0.3143 | 0.4877 | 0.8438 | 8/12 7/12 7/12 7/12 |
| cosine | 0.5818 | 0.3137 | 0.4856 | 0.8450 | 12/12 12/12 12/12 12/12 |
| tversky | 0.5981 | 0.3295 | 0.5409 | 0.8246 | 3/12 4/12 4/12 4/12 |
| dice | 0.5997 | 0.3296 | 0.5401 | 0.8247 | 4/12 5/12 5/12 5/12 |
| tanimoto | 0.5997 | 0.3308 | 0.5408 | 0.8247 | 4/12 5/12 5/12 5/12 |
| soergel | 0.6068 | 0.3319 | 0.5560 | 0.8270 | 1/12 1/12 1/12 1/12 |
| chebyshev | 0.6347 | 0.3323 | 0.5308 | 0.7977 | 8/12 7/12 7/12 7/12 |
| mahalanobis | 0.6651 | 0.3306 | 0.6028 | 0.8187 | 6/12 6/12 6/12 7/12 |

**Table A cannot be read as a ranking of the coefficients**, and the last column is why.
The six coefficients at the top of it run on a standardised hybrid; tanimoto, dice,
tversky and soergel do not, so their best score is the best score of a fingerprint. The
table says which coefficient reaches the best number, not which coefficient is the better
measure of similarity.

**Table B. The same quantity on ecfp4 alone**, so the coefficients defined only on 0/1
features are scored on the same features as the others. One row is one coefficient.

| coefficient | hERG Ki RMSE | Caco-2 RMSE | MDR1-MDCK RMSE | hERG blocking AUC |
|---|---|---|---|---|
| tanimoto | 0.6095 | 0.3308 | 0.5823 | 0.8146 |
| dice | 0.6095 | 0.3296 | 0.5823 | 0.8154 |
| tversky | 0.6100 | 0.3295 | 0.5851 | 0.8150 |
| cosine | 0.6109 | 0.3294 | 0.5937 | 0.8147 |
| manhattan | 0.6215 | 0.3321 | 0.6685 | 0.8153 |
| euclidean | 0.6215 | 0.3328 | 0.6685 | 0.8153 |

On the same 2,048 bits the four overlap-style coefficients beat euclidean and manhattan on
three of the four datasets, by 0.0120 to 0.0862 RMSE, and tie on hERG blocking. So the
ordering in Table A is the representation, not the coefficient.

**Dice and tanimoto are the same ordering of pairs.** Over 912 configurations of
representation, weighting and neighbour count where both ran, the two give an identical
fold-mean score to 9 decimal places in 304 of them; where they differ the largest
difference is 0.0203 in the dataset's own metric, which comes from the weighting using the
distance values and not only their order. Treating them as two coefficients buys one
number.

**The coefficient is not a free parameter.** Holding the best representation for each
dataset and varying only the coefficient moves the score by 0.0186 RMSE on Caco-2, 0.1172
on MDR1-MDCK, 0.1350 on hERG Ki and 0.0810 AUC on hERG blocking. Holding that dataset's
best coefficient and varying only the representation moves it by 0.0249, 0.1577, 0.0950
and 0.0477. The two axes are within a factor of 1.3 of each other on every dataset, and on
hERG Ki and hERG blocking the coefficient matters more than the representation. A cliff
definition that fixes the coefficient by convention is fixing something that moves the
answer as much as the representation does.

**What this does not settle.** Every number above scores a coefficient by how well it
finds neighbours with similar labels, which is a different question from how well it finds
cliffs — pairs of neighbours whose labels differ. Section 4.1's sweep scores them on
cliffs. The two can disagree and TBD1 is not closed until both are in front of her.

**TBD2, which representations. A set, and hybrids are the reason the hybrids were built.**
Same files, plus `s02_hybrid_similarity`, `s05_knn_prediction`, `s06_applicability_domain`,
`s08_optimal_dimension` and s10 under `tests/results_server/`.
**Two complications.**

*One is mechanical and has to be solved before anything is built.* An overlap score needs
features at or above zero, and a hybrid's are standardised and go negative. Measured
2026-11-03: between 45 and 66 per cent of a standardised block's entries are below zero,
on four datasets, so this is not a rounding problem.

**Two routes, and one of them is already closed by a measurement in the repo.** Adding a
constant to lift the features to zero is the obvious route and it is dead:
`preprocessing.require_nonnegative`'s docstring records that a shifted Tanimoto is
Manhattan distance under another name, Spearman 0.999995 against plain Manhattan on
MolFormer's 768 features at the smallest shift that reaches zero, rising to 1.000000 by a
shift of +100, and 0.999074 to 1.000000 on the 200 physicochemical descriptors, measured
2026-11-03 on 800 molecules. The same docstring records that on z-scored features the
Tanimoto distance leaves its own range entirely: on 600 molecules of the Mpro set, every
one of the 179,700 pairs lands between 1.617 and 2.587, and no denominator went
non-positive, so nothing looks wrong. So a shift gives either a number outside the
formula's range or Manhattan under a new name, and Manhattan is already swept.

**That leaves the binary hybrid of 6.1b as the route**, which is what makes 6.1b the
mechanical answer to this complication and not a separate idea. The decision left for
Adelaide is the one in 6.1b: whether the binary hybrid takes only 0/1 features or counts
as well.

*One is a reporting rule for one experiment.* AC4 scores each representation by how well
it predicts cliff molecules. Where the cliff set came from the hybrid, AC4's row says so,
so a reader knows the hybrid was scored on a set it chose. It is a label on a row, not a
restriction on which definitions get built.

**TBD3 and TBD4, the cutoff, and whether it is global, per domain or per dataset.** The
sweep in 4.1 answers it: where agreement with MoleculeACE's labels peaks, per target.
Whether that peak is one number across 30 targets or thirty different numbers is itself the
answer to the global-or-per-dataset question, and it is a measurement.
**What is already measured and bears on it:** at ECFP4 Tanimoto 0.90, hERG Ki, Caco-2 and
MDR1-MDCK have 5, 7 and 0 cliff pairs, so a fixed high cutoff is comparable and empty. The
top 1 per cent of pairs gives 655, 6,512 and 12,836 pairs at 30.5, 74.5 and 88.9 per cent
of molecules, so a share is usable and means something different each time. That is the
trade, measured.

**TBD5, the cutoff for the single hybrid overlap.** Separate from TBD4 because one score
replaces three. Measured on one target: of MoleculeACE's 1,222 cliff molecules there, 1,055
came from the string score and 148 from the fingerprint, so their 0.90 is mostly a cutoff on
text similarity and does not transfer.

**TBD6, the share a moved gap aims at.** The same sweep. What is already measured: a share
target can be unreachable, and the gap then goes to zero so a cliff is any similar pair
whose labels differ at all; or it can be reached at 0.009 label units. Both happen on real
datasets and neither raises an error. Any guard has to read the gap in label units, not the
achieved share.

**TBD7, whether to train a cliff-prediction model here.** Adelaide's.
**What is known:** the training data ships with MoleculeACE, a binary cliff label per
molecule on 30 targets, so it costs nothing to assemble, and it is straightforward — a
binary classifier on features this repo already computes.
**What is unclear:** a model trained on their labels and scored on their labels measures
their definition twice. It becomes worth doing if it is scored on something else: our
datasets, where there is no cliff label, with 4.3 as the score.

---

## 6. The experiments, and where they live

Adelaide, 2026-11-04: *"Make this a separate set of AC experiments"*, and the cliff numbers
come **out** of the restructured noise experiments.

### 6.1 A separate family

The restructured noise suite is NA, DC, DG, ST and MS — built as n43 to n47, with n15, n20
and n36 standing. The cliff measurements currently sit inside MS (n47), lifted from n01 and
n14, and they are to be removed from there.

A new family for this study, so the cliff work is one set of files with one definition
module behind it, and the only thing left inside the noise suite is the one arm that needs
the detector-by-cleanup machinery. **AC1 to AC5 below are placeholders written by a
session on 2026-11-04. Adelaide names them.**

| | What it runs | Lifted from | Section |
|---|---|---|---|
| AC1, the definitions and their counts | Every definition on every dataset: cliff pairs, cliff molecules, the achieved share, the gap in label units, the gap against the dataset's own repeat spread where it has one | `cliff_census.py`, and the cliff half of n47 | 3 |
| AC2, agreement with MoleculeACE | Every definition against their labels on 30 targets, **with the cutoff swept** | `opig_moleculeace_cliff_check.py` | 4.1 |
| AC3, against the published models | Our cliff-specific error beside GraphCliff's, SemiMol's and the foundation comparison's, and beside plain ECFP4 Tanimoto at the standard cutoffs | `opig_activity_cliff_qsar.py`, which already computes the cliff-call correlations and reports none | 4.2 |
| AC4, predicted worse or not | Held-out error on cliff against non-cliff molecules, per representation and per hybrid | s04, and the cliff half of n47 | 4.4 |
| AC5, does the set survive the representation | One definition, every representation, how many molecules stay cliffs | `thesis.tex:1123`'s measurement, extended | 4.4 |

### 6.1b The binary hybrid (Adelaide, 2026-11-04)

The AC experiments build **a binary version of the hybrid, per dataset**, so that an
overlap score is defined on it the same way it is on a fingerprint. No change to the
hybrid search: it is the same search asked for a restricted list of representations.

Why nothing else is needed. Building a hybrid is column selection and concatenation —
`hybrid._build_from_selection` takes the chosen feature indices from each representation
and stacks them, with no scaling, weighting or standardising in that step. And
`kirby.scaling.FeatureScaler` already marks every 0/1 feature and sets its mean to 0 and
its scale to 1, so a binary feature passes through untouched (dated 2026-10-05 in that
file). Today's hybrids come out continuous only because the pool mixes binary fingerprints
with embeddings and descriptors.

**Re-read 2026-11-04 and both halves hold.** `_build_from_selection` is two lines of column
indexing and one `hstack`, with no scaler, weight or divisor in it. And neither
`hybrid.py` nor `find_best_hybrid.py` constructs a plain `StandardScaler` anywhere — every
one of the 17 scaling points in `hybrid.py` and the 7 in `find_best_hybrid.py` is a
`FeatureScaler`. So there is no path by which a 0/1 feature entering the search comes out
z-scored.

**Where the list lives:** each dataset's binary representations are recorded in
`scratchpad/precompute_cuts/DECISIONS.md`, beside that dataset's two models, and the AC
experiment reads them from there. Nothing is set at the command line.

**Which of each dataset's representations are 0/1.** `similarity.preprocessing.feature_kind`
decides it from the values rather than the name, returning binary, count or continuous.

**Four of the eight datasets are already measured**, read out of the stored framework
results on 2026-11-04 rather than recomputed: every row of `f34_full_sweep` and
`f34_importance_sweep` carries the kind of the matrix it scored. Over 1,241 result files
there are 48 dataset-and-representation pairs with a measured kind and no pair disagrees
with itself. One row below is one representation; the number is how many features it has
on that dataset.

| dataset | binary | count | continuous |
|---|---|---|---|
| hERG Ki | ecfp4 2048, avalon 2048, atom_pair 2048 | graph_kernel 276 | mordred 1613, mhg_gnn 1024, smited 768, grover 4800, pdv 200, chembert 256 |
| Caco-2 | ecfp4 2048, avalon 2048, atom_pair 2048, top_torsion 2048 | graph_kernel 317 | mhg_gnn 1024, smited 768, grover 4800, pdv 200, molformer 768 |
| MDR1-MDCK | ecfp4 2048, avalon 2048, atom_pair 2048, maccs 167 | graph_kernel 143 | mordred 1613, mhg_gnn 1024, smited 768, grover 4800, pdv 200 |
| hERG blocking | ecfp4 2048, avalon 2048, atom_pair 2048, top_torsion 2048 | graph_kernel 162 | mordred 1613, mhg_gnn 1024, grover 4800, mol2vec 300, molformer 768 |

So three to four of each dataset's ten representations are 0/1, and the graph kernel is the
only count one. The searched hybrid, 100 features wide on all four, comes back continuous
every time, which is the thing 6.1b exists to fix.

**The signs, measured on hERG Ki 2026-11-04:** of its six continuous representations the
lowest value runs from -1.9 to -5,667.6 and the share of entries below zero from 1.7 to
50.4 per cent.

**Two more datasets measured the same day**, by running `feature_kind` on the stored
representation matrices in the local cache rather than rebuilding them. The cache is keyed
on a hash of the dataset's SMILES, so the matrix read is the one the framework used.

| dataset | binary | count | continuous |
|---|---|---|---|
| LogD, 5,028 molecules | ecfp4 2048, avalon 2048, top_torsion 2048 | none in its pool | mordred 1613, mhg_gnn 1024, grover 4800, smited 768, molformer 768, chembert 256, graphmvp 300 |
| human microsomal clearance, 3,086 molecules | avalon 2048, atom_pair 2048 | graph_kernel 141 | mordred 1613, mhg_gnn 1024, smited 768, molformer 768, pdv 200, chembert 256 |

`grover` is not in the cache for human microsomal clearance, so that one of its ten is
unmeasured. The signs on LogD's seven continuous representations: the lowest value runs
from -1.1 to -2,939 and the share of entries below zero from 10.5 to 52.9 per cent.

**`chemberta` is 384 features and `chembert` is 256**, measured on LogD, where both are
cached. Both are continuous and about half their entries are below zero. Memory's warning
that the two names are distinct holds.

**Still unmeasured, and why.**

- **Rat microsomal clearance, 3,053 molecules.** Only `ecfp4`, `maccs` and `pdv` are in the
  cache; the other eight of its ten have to be built before the kinds can be read. ecfp4 is
  binary, pdv is continuous.
- **The three QM9 targets.** The loader cannot run on this machine. It looked for
  `~/.cache/kirby/QM9/raw/qm9.zip`, which is not there, and started downloading
  `deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip` into a path that
  does not exist, raising `FileNotFoundError`. So QM9's pool, which is the only pool carrying
  `selformer`, is unmeasured; `selformer` is cached for no dataset at all.
- **hERG IC50 and Dablander's three targets** have no pool to classify, below.

**One thing seen and then ruled out.** In the run that loaded five datasets one after
another, hERG Ki raised `FileExistsError: /Users/apunt/.cache/kirby` immediately after the
QM9 loader failed. Loaded on its own in a fresh process, hERG Ki reads its 1,415 cached
ChEMBL records and deduplicates to 1,412 molecules with no error. So it is the failed QM9
download leaving the process in a state the next loader trips over, not a fault on the hERG
Ki path. Worth knowing only because it means QM9 cannot share a process with another
dataset until the QM9 download path is fixed.

**hERG IC50 and Dablander's three targets have no representation pool at all** in
`complexity_helpers.DATASET_REPS`, so there is no list of ten to classify for them. Section
7 names all four as Paper 3 small molecules, so either they get a pool or the AC experiments
name theirs.

**One decision for Adelaide:** whether a binary hybrid takes only 0/1 bits, or also counts
such as the graph kernel. Overlap scores are defined on both; only the first keeps the
output 0/1.

### 6.1c Wiring the binary-only coefficients to the binary hybrid

The point of the binary hybrid is that the coefficients defined only on 0/1 features
become available on a searched feature space. Those coefficients have to be reached by
name, from the kind of the features, and never by a default list.

**The three lists already exist**, keyed on the feature kind, in
`f18_distance_metric_sweep.coefficients_for(kind)`, which reads
`simeval.BIT_COEFFS` and `simeval.COUNT_COEFFS`:

| Kind | Coefficients |
|---|---|
| binary | tanimoto, dice, cosine, tversky, euclidean, manhattan |
| count | tanimoto, dice, cosine, soergel, euclidean, manhattan |
| continuous | euclidean, manhattan, minkowski p=0.5, minkowski p=0.3, chebyshev, cosine, correlation, canberra, braycurtis, mahalanobis |

Two of those are on the binary list and on no other: **tversky**, and **soergel** which is
deliberately kept off the binary list and on the count list, for the reason given in
`simeval`'s module docstring.

**What the AC experiment must do**, and the order matters:

1. Build the binary hybrid from the dataset's binary representations.
2. Ask `feature_kind` what the assembled matrix is, rather than assuming it from the
   inputs. A hybrid built from binary inputs comes back binary, and if it does not, that
   is a fault to raise and not to work around.
3. Take the coefficient list from that kind, and record the kind beside every result, so a
   stored number says which list it came from.

**The failure this prevents, already written into the code.**
`preprocessing.require_binary` raises rather than coercing, and its docstring records why:
scipy's `jaccard` does not raise on continuous features, it thresholds at nonzero, so on
z-scored features every row becomes all-True and the distance matrix comes back exactly
all zeros — every pair identical, with no warning. A binary coefficient handed a
continuous hybrid by mistake therefore produces a clean-looking result that means nothing.
Every binary-only coefficient in the AC experiments goes through that guard.

### 6.1d Built 2026-11-04: the restricted lists and the guard

`tests/experiments/cliffs/binary_hybrid.py`. It holds the per-dataset restricted
representation lists, the guard, and the coefficient list read from the assembled matrix.
Nothing in it searches or fits.

**Both lists are built, not one, because both are defined and neither is degenerate.**
`bits` keeps the representations whose features are 0 or 1, and the assembled hybrid is
0/1, so tversky becomes available. `bits+counts` also keeps the count representations, and
the assembled hybrid is counts, so soergel becomes available instead. Soergel is off the
binary list because on 0/1 features it equals tanimoto exactly, which `simeval`'s selftest
proves. So the choice between the two lists is not a cut of either and is settled by
building both; what the paper leads with is still Adelaide's.

**Measured on hERG Ki, 1,412 molecules, 2026-11-04.** Three of its ten representations are
0/1 — `avalon`, `ecfp4`, `atom_pair`, 2,048 features each — and `graph_kernel` is the only
count one at 276 features. `bits` concatenates to 6,144 features and the kind read back
from the concatenation is binary. `bits+counts` concatenates to 6,420 and reads back as
count. The coefficient lists that come out of those kinds are the two in the table above,
with tversky on the first and soergel on the second.

**`tests/experiments/cliffs/_smoke_binary_hybrid.py` passes, 14 checks.** It assembles a
hybrid through the search's own `hybrid.apply_feature_selection` and checks the kind, for
both lists. Three of its checks are the ones worth recording:

- A hybrid assembled from `avalon`, `ecfp4` and `atom_pair` reads back as binary, and
  `FeatureScaler` marks all 36 of its features 0/1 and leaves every one of them unchanged.
- The guard raises when a continuous representation is mixed in, instead of returning a
  number.
- The silent failure in `require_binary`'s docstring is real and was reproduced here:
  scipy's `jaccard` on 64 z-scored `mhg_gnn` features of 60 molecules returns exactly 0.0
  for all 1,770 pairs, largest value 0. Every pair identical, no warning.

**Can a hybrid be a 0/1 feature vector? Yes, and no search is needed to find out**
(Adelaide, 2026-11-04: a full search should not be needed to answer that, and she is
right). Read end to end on 2026-11-04, the whole path from a searched recipe to a matrix is
`hybrid.apply_feature_selection`: for each representation it takes the chosen column
indices, clips to plus or minus 1e10, replaces any non-finite entry with 0, and stacks the
blocks. Clipping at 1e10 cannot touch a 0 or a 1, and replacing a missing value with 0
leaves the matrix 0/1. The result is cast to float32, which changes the storage and not the
values. So a hybrid built from 0/1 representations is a 0/1 feature vector by construction.

**One condition, and it is the only one.** Extra features from a donor dataset are appended
after the base blocks, through `apply_augmentation_selection`, and those come from whatever
the donor holds. A hybrid that admitted them is not 0/1. So the AC experiments pass no
donor features, and the guard catches it if one arrives anyway.

### 6.1e Built 2026-11-04: the overlap scores now run on a hybrid

This is TBD2's mechanical half, closed. `cliff_rules.py` held 32 ways of calling a pair
similar and not one of them computed an overlap score on anything but a fingerprint. It now
has twelve that do.

**What was added to `cliff_rules.py`.**

- Tversky and Soergel as similarity formulas, which the module did not have. Tversky is on
  the 0/1 list and on no other, Soergel on the count list and deliberately off the 0/1 one.
- A count form of Tanimoto and of Dice. The two forms differ in their denominator: the 0/1
  form divides by the row sums, the count form by the inner products. The module's existing
  `tanimoto` and `dice` are the 0/1 form and were left untouched, because every existing
  caller hands them a fingerprint.
- `CliffContext.binary_rep` and `.count_rep`, which refuse a matrix of the wrong kind
  instead of coercing it.
- Twelve named criteria, six on a hybrid built from the dataset's 0/1 representations and
  five on a hybrid that also took its count representation, plus MoleculeACE's shape with
  one overlap score in place of three.

**The kind is read from the matrix and the formula follows from it.** Ask for Tversky on a
count hybrid and it raises, saying Tversky is defined on 0/1 features only. Ask for any
overlap score on a standardised hybrid and it raises, saying to build the hybrid from the
dataset's 0/1 representations. Nothing is assumed from a representation's name.

**Three equalities checked rather than asserted**, on hERG Ki's 250-feature binary hybrid:
the count form of Tanimoto and the 0/1 form agree to exactly 0; the same for Dice; and
Soergel agrees with Tanimoto to 6e-08, which is the reason Soergel is off the 0/1 list.

**One thing to know before reading a Tversky number.** At the weights this suite uses, 0.7
and 0.3, Tversky is close to Dice: on that same hybrid the largest difference between the
two over all pairs is 0.0256 and the median is 0.0018. At weights of 0.5 and 0.5 they are
the same formula. So Tversky is not a second independent score unless the weights move, and
whether they should is open.

**The existing rules are unchanged.** `ecfp4_tanimoto_055` still calls 451 of hERG Ki's
1,412 molecules cliff molecules, which is the number this module's own header recorded
before any of this was added.

**`tests/experiments/cliffs/_smoke_hybrid_similarity.py` passes.** It builds both hybrids by
taking columns, which is what the search's own `apply_feature_selection` does, runs all
twelve criteria against a gap of twice the median absolute deviation, and prints how many
pairs each calls similar and how many molecules come out as cliff molecules. Those counts
are from a hybrid whose columns were drawn at random, not searched, so they are a check that
the path runs and are not results.

### 6.1f Built 2026-11-04: AC1, every definition on every dataset

`tests/experiments/cliffs/ac1_definition_counts.py`, with
`tests/experiments/cliffs/_smoke_ac1.py` beside it. It lifts its census from
`noise/cliff_census.py` and adds the three things AC1's row in the table above asks for and
that file does not have.

**One row is one dataset crossed with one similarity criterion, one gap criterion and one
feature space.** Every row carries: how many pairs the similarity side called similar, how
many of those the gap side also passed, how many molecules are in at least one cliff pair,
that as a share of the dataset's molecules, and the gap in the label's own units. Beside the
gap, two ratios: the gap divided by this dataset's own measured repeat spread where it has
one, and the gap divided by Kramer's 0.54 pKi, which is borrowed on any label that is not a
public Ki and is marked as borrowed in the file.

**What is new against `cliff_census.py`.**

- The two binary hybrids are built and handed in as named representations, so the twelve
  hybrid criteria run. No cliff set in this repository had ever been built from a hybrid.
- The gap in label units is on every row. `cliff_census` had it on the named rules and in
  the threshold sweep only.
- The dataset list is derived, not typed: `papers.plan.PAPER3_DATASETS` filtered to the
  registry's molecular domain gives exactly the fourteen section 7 names — the three QM9
  targets, hERG Ki, hERG IC50, Caco-2, LogD, rat and human microsomal clearance, MDR1-MDCK,
  hERG blocking, and Dablander's three.

**Where the hybrid comes from, and that it is recorded on every row.** The default is the
winning recipe of the restricted search, through `_shared.get_fitted_hybrid`, which resolves
the search on the decided grid in `scratchpad/precompute_cuts/DECISIONS.md` and rebuilds the
winner's features on every row. One hybrid per decided downstream model, because a dataset
has two and neither is a cut. Every row then carries that recipe: the budget in features,
the allocation method, the importance method, the internal evaluator and the search's own
score. The alternative source, `--hybrid-source budget:N`, names one budget instead and is
the debug path; a row says which of the two built its hybrid, so the two are never mixed
without saying so.

**The circularity in the hybrid rows, which no other row has.** A hybrid's features are
chosen using the labels. The gap criterion is a statement about those same labels, over the
same molecules, with no fold held out. So a hybrid row's similarity side is not label-free
and an ECFP4 row's is. Every hybrid row carries `similarity_used_labels: true` and every
other row carries false. **Whether AC1's hybrid should instead be built per fold is
Adelaide's decision.** It is the only thing in AC1 that is not settled. AC4 and AC5 hold a
fold out for this reason; AC1 is a census over all the molecules the loader returns, which
is what `cliff_census` was and what makes its counts comparable with the counts already on
disk.

**It refuses above 12,000 molecules rather than running out of memory.** The limit is
`n35.MAX_PAIR_ROWS`, read from that file rather than set again here, and it carries its own
reason: a square matrix of 32-bit numbers at 12,000 rows is 576 MB, and several matrices are
live at once. hERG IC50 is loaded one row per measurement, 12,008 rows, so it lands just
over and is refused with its row count and the limit printed. Nothing silently shrinks.

**It ran end to end on hERG Ki**: 1,412 molecules, 310 rows, nothing failed to compute,
188.7 s, at `--hybrid-source budget:250` with the four representations of its pool that are
0/1 or counts. **A local run is debug, not a result**, so no count below is a finding about
hERG Ki. Two things about the definitions, though, are properties of the formulas and not of
the run:

**An overlap score on a count hybrid saturates, and Soergel does not.** Measured on that
hybrid, with a gap of twice the median absolute deviation:

| similarity criterion | similar pairs | cliff molecules of 1,412 |
|---|---|---|
| hybrid_counts_soergel_090 | 1,216 | 52 |
| hybrid_counts_tanimoto_090 | 40,922 | 994 |
| hybrid_counts_dice_090 | 441,157 | 1,404 |

The mechanism, measured on the same matrix: of the count hybrid's 250 features, 61 come from
the graph kernel and reach as high as 59, and those 61 hold 58.9 per cent of the matrix's
total mass. Tanimoto and Dice in count form divide by inner products, so those few
large-magnitude features decide the score and almost every pair comes out similar. Soergel
divides each feature's difference by that feature's larger value in the pair, so a feature
that reaches 59 counts no more than a feature that reaches 1. On the 0/1 hybrid, where every
feature's maximum is 1, the same three criteria give 959, 3,950 and 3,951 similar pairs.

So `hybrid_counts_dice_090` at the published 0.90 is near-degenerate on this dataset. It is
kept and reported, not cut: degeneracy at one cutoff on one dataset is a number the cutoff
sweep of section 4.1 is for, and the top-share criteria carry no cutoff at all.

**`_smoke_ac1.py` passes, 18 checks.** It checks the dataset list, both hybrids with the
kind read from the assembled matrix, the coefficient list following from that kind, every
column on every row, that every row has its gap in label units, that a molecule count never
exceeds the dataset and a cliff pair count never exceeds the similar pairs, and that the
hybrid rows and only the hybrid rows say their similarity side used the labels.

**One fix made in the noise suite while doing this.** `cliff_census.own_repeat_spread` called
`load_data` again with no representation list, which widened a restricted pool back to the
dataset's full ten. It now takes the already-loaded data when the caller has it. No other
file calls that function.

### 6.2 The one arm that stays in the noise suite

The D/D'/D'' test is a detector crossed with a cleanup plus a modifier, which is exactly
what DC is. It stays in `n44_detect_and_clean` as:

- the cliff-awareness modifier on a cleanup, which produces D'', and
- D''', sparing the same number of molecules chosen at random.

Both lift from `n25 put_back` and `n25 random_exempt_mask`. This is the only place the
cliff work touches the noise suite, and it is the place the paper split is decided.

### 6.3 Hybrids

AC4 and AC5 are where TBD2 is answered on cliffs rather than on neighbours, and s09's
aggregation is the shape to reuse: per dataset, one hybrid minus one representation per
held-out fold, in the input's own quantity, on all molecules, cliff molecules and
non-cliff molecules separately.

### 6.4b Two mismatches in the old files, found 2026-11-04

Recorded because a session lifting mechanics from these files will meet them. Neither needs
fixing in place; both say that a name in the old code does not describe what ran.

**The rule named `s12` in `cliff_rules.py` is not what s12 runs.** The module defines it as
the top 0.5 per cent of pairs on every representation glued together, with a gap of twice
the median absolute deviation. The experiment uses ECFP4 Tanimoto at 0.55 whenever the
dataset has SMILES, and solves the gap to a target share. Both halves differ.

**The two cliff blocks inside n47 use different gaps and write the same field.** One uses
twice the median absolute deviation, the other a flat 2 log units, and both write
`n_cliff_compounds` into the same result file.

### 6.5 What comes out of the noise suite

The cliff counts and the cliff-split error now in n47 move to AC1 and AC4. n47 keeps the
noise measurements and loses the cliff ones. Note before touching it: n47's two blocks
currently disagree on the gap — one uses twice the median absolute deviation, the other a
flat 2 log units, and both write a field called `n_cliff_compounds` into the same result
file.

---

## 7. Getting it into the ledger

The target is **implemented and ready to run inside the ledger, on the available
datasets**, not a design. `HANDOFF_PAPERS_2_3_LEDGER.md` holds the five steps and the gate;
read it in full with the files it names before changing any of it.

The AC family runs at **Step 5**, with Paper 3's experiments, and depends on Steps 1 to 3
for the hybrids it needs. Two parts of it do not: AC2 and AC3 run on MoleculeACE's and
Dablander's own data and splits, so they can run before the gate.

**What has to be settled for a ledger entry to exist**, in the order it blocks:

1. The definition set. One entry per definition, or one entry sweeping them.
2. Which datasets. Paper 3's small molecules: the three QM9 targets, hERG Ki, hERG IC50,
   Caco-2, LogD, rat and human microsomal clearance, MDR1-MDCK, hERG blocking, and the
   three Dablander targets.
3. Whether each AC experiment is one unit per dataset or one per dataset and definition,
   because the ledger resets and reruns one unit at a time.
4. The memory class per unit. AC1 builds a pairwise matrix, which is the cost: 9,978
   molecules is about 400 MB a matrix and several are held at once, so QM9 needs chunking.

---

## 8. Other things worth considering

Raised here as options with what each needs, not as recommendations.

**Assay type as a filter before detection.** The literature's first curation rule is not to
pool IC50, Ki and Kd, because mixing them manufactures cliffs out of unit differences. Two
datasets here are hERG Ki and hERG IC50 and they are kept separate, which is right. Worth
checking whether any single dataset pools types internally.

**Censored values.** The literature drops them before detecting cliffs. A censored label is
a bound, not a measurement, so a pair involving one has an unknown gap. This repo already
has a censoring noise condition, so it knows what censoring is; whether any dataset ships
censored labels into the cliff calculation is unchecked.

**Same paper, same lab.** Two measurements from different papers differ partly because the
labs differ. hERG IC50 names the document each measurement came from, so a same-document
restriction on which pairs may be called cliffs is one line where the pair list is built.

**Stereoisomer pairs.** Named in the Geometry preprint as the kind of cliff fingerprints
miss entirely. Already added as a criterion, never run as a definition.

**The half-cliff band.** Dablander drops pairs 1 to 2 log units apart from both sides rather
than calling them non-cliffs. That is a third gap option not in the list in section 3, and
it changes what the cliff call is scored against.

**Cliff molecules against the molecules this suite corrupted on purpose.** The detector
grading experiment knows which rows it damaged by construction, so the two masks cross
directly. It is the cheapest version of asking whether a definition separates chemistry from
error, and its limit is that injected noise is not the noise the data came with.

**The curator flag, already run and weak.** Every definition's cliff molecules carry
ChEMBL's suspect-record mark at about the rate the whole dataset does. The "potential
transcription error" text covers 6 molecules in all of hERG IC50, so it cannot separate
anything. Worth one sentence in the paper and no more work.

---

## 9. How to work this document

Written because three sessions failed on this before, in ways that are recorded.

**State the decision, not the evidence for it.** Lead with what has to be decided, what the
situation is, and what the options are. The measurement is the evidence for the options, not
the message.

**Never refer back to a list, option number or name she cannot see.** Restate the options in
full in the message that asks her to choose.

**Say what a thing is before asking about it.** Never name code, a dataset or a score she
has not used first.

**Every number carries its unit and what it is a share of.** Molecules, pairs, folds,
label units. A table needs a line above it saying what one row is.

**No verdict written into any document of hers.** Later it cannot be told from a measured
result. Headings state what was measured.

**Do the measurement rather than offer it.** Only a scope cut, a dependency to install or a
dataset to drop needs her first.

**Nothing lives in a chat reply.** If she raises something, it goes in this file before the
reply is written.

### Before starting

- Repo-root `CLAUDE.md` in full. It sets how to write to her.
- `PAPER_3_OUTLINE.md`. The documentation of record.
- `HANDOFF_PAPERS_2_3_LEDGER.md` and the files it names.
- `HANDOFF_NOISE_RESTRUCTURE.md` sections 1 to 4, 8, 9 and 10. The old experiments n01,
  n14, n25, n35 and n42 are retired: lift from them, never edit them.
- `HANDOFF_PAPER3_CLIFFS.md` sections A, E, F and G, for the measurements already made.
- From `MEMORY.md`: no patient data; one heavy local job at a time; everything on
  `/Volumes/seagate`; no significance tests; never cut a definition, dataset or setting for
  compute; local runs check code and are not results; post a line every few minutes because
  her terminal disconnects; she decides, so options come with numbers attached.
