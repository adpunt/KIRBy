# Published work that trains a classifier on an activity cliff label

Searched 2026-11-28 for TBD7, her outline item "4. Train my own model on MoleculeACE (TBD7)".
Claude's search results, not reviewed by the author. Abstracts and search summaries were read;
the papers' own results tables were NOT opened, so no number below should be cited from this
file without reading the source.

## The headline

No paper was found that trains a classifier on MoleculeACE's per-molecule `cliff_mol` flag.
Every classification paper found predicts whether a PAIR of molecules is a cliff, on matched
molecular pairs, not whether one molecule carries the flag. MoleculeACE's own paper uses the
flag only to split the test molecules in two and report RMSE separately on each half; it does
not train anything to predict the flag.

So `moleculeace_cliff_model.py` has no published counterpart to be scored against, and its
median Matthews correlation of 0.4464 cannot be compared to a published number, because the
published numbers are for a different task.

## One row per paper

| paper | what it predicts | models | representation | split | metric | number |
|---|---|---|---|---|---|---|
| van Tilborg, Alenicheva, Grisoni 2022, J. Chem. Inf. Model. 62, MoleculeACE | nothing; the flag labels test molecules | 24 regression methods | ECFP, descriptors, SMILES, graph | their own, stratified on the flag and similarity-aware | RMSE, and RMSE on cliff molecules | not a classification number |
| Dablander et al. 2023, J. Cheminform., DOI 10.1186/s13321-023-00708-w | whether a PAIR is a cliff, and the direction of the potency difference | 9 combinations of 3 representations and 3 regression techniques | ECFP, physicochemical descriptors, graph isomorphism network features | 2-fold cross-validation, 3 random seeds | Matthews correlation, sensitivity, precision | their words: "a significant portion of our QSAR models exhibit an AC-MCC value greater than 0.5"; per-model values are in their GitHub numerical results, not read |
| arXiv 2411.13688, 2024, same group | whether a PAIR is a cliff | graph neural networks and classical features | ECFP-MLP, GIN-MLP | 2-fold cross-validation, 3 random seeds | Matthews correlation | "reasonably high MCC across MMP sets", no value read |
| J. Comput.-Aided Mol. Des. 2025, DOI 10.1007/s10822-025-00721-1, protein kinase inhibitors | whether a PAIR is a cliff, at a 100-fold potency difference | deep autoencoder feeding SVM and a neural network; names Siamese networks as the natural fit | autoencoder features | not read | not read | not read |
| GraphCliff, arXiv 2511.03170, KDD 2026 | potency, not the flag | their own gated graph network | molecular graph | MoleculeACE's own splits | RMSE, RMSE on cliff molecules | mean RMSE 0.665 overall, 0.619 on non-cliff molecules, best overall on 18 of 30 targets |
| iCliff, bioRxiv 2025.03.09.642269 | how many cliffs a dataset holds | none, not a model | - | - | a dataset-level roughness value | not a classification number |

## What this changes about her own run

Three differences between her run and the closest published work, Dablander et al.

1. Hers is per molecule, theirs is per pair. `cliff_mol` is set on a molecule because some
   other molecule in the same target is similar and differs in potency, so her classifier is
   given one molecule's features and asked for a property of its neighbourhood. Theirs is
   given both molecules of a pair.
2. Hers fits each classifier at its registry defaults: `moleculeace_cliff_model.py:173` calls
   `factory()` and fits, with no search over settings. Theirs reports each representation and
   technique combination in full.
3. Theirs reports Matthews correlation per target and per matched-molecular-pair set. Hers
   pools all 30 targets and all 5 folds into one list per arm and reports its median, so a
   fold of a target where 8.8 per cent of molecules carry the flag sits in the same list as
   one where 54.0 per cent do.

Her instruction of 2026-11-28, her words: "I've never cared about averages. I would only in
practice use the BEST one ... This is a TOOL."

## Sources

- https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.2c01073 and the 2023 correction,
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11683855/ (a splitting bug from incorrectly
  labelled cliff pairs; the authors report the same conclusions)
- https://arxiv.org/pdf/2301.13644 (Dablander), code and numerical results at
  https://github.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments
- https://arxiv.org/html/2411.13688
- https://link.springer.com/article/10.1007/s10822-025-00721-1
- https://arxiv.org/pdf/2511.03170 (GraphCliff), checkpoints for all 30 targets at
  https://zenodo.org/records/21787497
- https://www.biorxiv.org/content/10.1101/2025.03.09.642269v1.full.pdf (iCliff)

---

# How this task is set up in the literature

Searched 2026-11-28, second pass, for her question: "what is the best practice way of using
this data" and "HOW to train the models or what models I should be training or on what data".
Search summaries only; no paper's methods section or results table was opened.

## The unit is a pair, not a molecule

Every classification paper found labels a PAIR of molecules as a cliff or not. The pairs come
from matched molecular pair analysis: two molecules that differ by one chemical change. The
per-molecule flag is used only to split test molecules in two for reporting potency error, as
MoleculeACE's own paper does. Nothing found predicts the per-molecule flag.

## What the classifier is given for a pair

Three constructions appear, and no source read says which wins.

- A twin network, also called Siamese, with shared weights encoding each molecule of the pair
  and trained on the pair label with binary cross-entropy. Dablander's thesis version
  introduces one; a 2024 ICML workshop paper uses one.
- A classical model on a pair representation. Historically a support vector machine with a
  matched molecular pair kernel.
- A difference of the two molecules' vectors. Dablander calls these pairwise difference
  vectors and reports them "consistently inferior to ECFPs and GINs at both QSAR-prediction
  and AC-classification", so the plain difference is the one construction with a published
  verdict against it. A 2026 preprint instead uses the absolute difference of two embeddings
  with logistic regression and XGBoost.

## The split, which is where the published numbers went wrong

- Leakage is compound overlap: the same molecule appearing in a training pair and a test pair.
  A similarity-based model then recognises the molecule rather than the chemistry.
- The old support vector machine results of 80 to 90 per cent accuracy were obtained where
  that overlap was possible. One study's finding, searched not recalled: without leakage,
  "AC prediction accuracy based on chemical structure is low or moderate at best".
- The recommended split is two-stage: split the MOLECULES first, by scaffold or cluster, then
  assign each pair by how many of its two molecules were seen in training, and report the
  pairs with both molecules unseen, one seen, and both seen separately. A stricter variant
  splits on the structural cores of the matched molecular pairs.
- One review's words: there is "no consensus about which data splitting strategy should be
  canonically used".

## Which models, and what a good number is

- Across 100 activity classes, a support vector machine was preferred overall "though by
  relatively small margins", deep learning gave no clear advantage, and median Matthews
  correlation reached about 0.5. XGBoost was reported as poor in that study.
- A 2026 study on bioactivity profiles reports Matthews correlation between 0 and 0.5 for
  random forest, XGBoost and support vector machine.
- Graph isomorphism network features were competitive with or better than classical
  representations for cliff classification; in GraphCliff's own paper a support vector machine
  on ECFP still beats their network on cliff pairs for a subset of the 30 targets.

**So the published ceiling for this task, under leakage control, is a median Matthews
correlation around 0.5 or below.** Her per-molecule run's 0.4464 sits just under that, on an
easier-to-leak but harder-to-define task, and the two numbers are not comparable.

## What her own run would have to change to be the published task

1. Build matched molecular pairs per target. `mmpdb` is already named in her outline under
   "Which pairs count as similar", so the pair source exists in her plan.
2. Label each pair a cliff by potency difference, not read the per-molecule flag.
3. Split molecules first, then assign pairs by how many of their two molecules were seen, and
   report those groups separately. Her folds are already scaffold-grouped on Bemis-Murcko
   groups, which is the molecule-level half; the pair assignment does not exist.
4. Choose the pair input: twin network, a kernel on the pair, or an embedding difference. The
   plain difference has a published verdict against it.
5. Search each model's settings. `moleculeace_cliff_model.py:173` fits at registry defaults.

None of that is written. Each of the five is hers to decide and none is implied by the others.

## Sources, second pass

- https://link.springer.com/article/10.1186/s13321-022-00676-7 (100 activity classes, support
  vector machine preferred, median Matthews correlation about 0.5, XGBoost poor)
- https://pubmed.ncbi.nlm.nih.gov/42612264/ (leakage-controlled kinase analysis)
- https://link.springer.com/article/10.1186/s13321-026-01210-9 (bioactivity profiles, Matthews
  correlation 0 to 0.5)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8401777/ (matched molecular pair kernel)
- https://research.latinxinai.org/papers/icml/2024/pdf/Cesar_Valdez.pdf (twin network on pairs)
- https://arxiv.org/pdf/2404.09738 (AMPCliff, a different setting, XGBoost best there)
- https://en.wikipedia.org/wiki/Matched_molecular_pair_analysis

---

# How to train, which models, on what data

Searched 2026-11-28, third pass, for her complaint: "You have not done enough research to
explain HOW to train the models or what models I should be training or on what data." Search
summaries only; no paper's methods section was opened, so every number below carries its
source and none should be cited without reading that source.

## On what data

Two sources of labelled pairs exist, with different sizes.

ACNet, Zhang et al. 2023, https://arxiv.org/pdf/2302.07541, is the benchmark built for this
task. ACNet holds matched molecular pairs against 190 targets. One target's pairs number 36 at
the smallest and 26,376 at the largest. ACNet also ships a pooled set of 278,367 pairs across
all targets, and five named subsets for imbalance, low data and out-of-distribution. ACNet's
own splits are published splits and this lab does not use published or random splits, so ACNet
would be a source of pairs and labels only.

MoleculeACE's 30 targets can also supply pairs, built with mmpdb from the same molecules
already on `/Volumes/seagate/moleculeace_benchmark`. MoleculeACE's targets hold 615 to 3,657
molecules each, and 12 of the 30 hold under 1,000 molecules, which the MTPNet repository
describes as a low-data setting (https://arxiv.org/pdf/2506.05427).

## One model per target, or one model across targets

Not settled in the literature. ACNet treats each of its 190 targets as a separate task and
also ships the pooled set, so both are provided for. MTPNet's authors state that previous
models were limited to one target at a time and give that as their reason for a unified model.

Tamura, Miyao and Bajorath 2023 report no significant correlation between training set size
and prediction accuracy across their nine methods. Poor models did come from small training
sets, but their best accuracies for different methods came from training sets of varying size,
small ones included.

## Which models, and how they were assessed

Tamura, Miyao and Bajorath 2023,
https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00676-7, compared nine methods
over 100 activity classes: pair-based nearest neighbour classifiers, decision tree methods,
kernel methods, and deep neural networks. A support vector machine was best overall by small
margins and deep learning showed no clear advantage. They scored with balanced accuracy and
Matthews correlation. Their main result set used random splits, which they themselves label
"data-leakage possibly included", and a separate leakage analysis covers 42 of the 100 classes.

Abe, Miyao and Bajorath 2026, J. Cheminform. 18, 72,
https://link.springer.com/article/10.1186/s13321-026-01210-9, is the leakage-controlled
follow-up; random forest, XGBoost and support vector machine land between 0 and 0.5 Matthews
correlation there.

## The support vector machine, concretely

Her decision of 2026-11-28: "Let's definitely test SVM, maybe with some hyperparameter tuning".

The published kernel is the matched molecular pair kernel, described in Molecules 26 (2021)
4916, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8401777/. The kernel is a product of two
Tanimoto kernels: one on the pair's shared core and one on the two substituents that differ.
It was used on thrombin and carbonic anhydrase II.

Two consequences for tuning. A Tanimoto kernel has no gamma, so gamma only exists if a radial
basis function is wrapped around the kernel. The penalty C is the parameter the published work
selects, in inner cross-validation folds. Separately, the standard Tanimoto support vector
machine's feature weighting cannot be applied to this kernel, because the kernel does not
decompose into a sum of feature weights; the Molecules paper works around that with its own
decomposition. So an interpretable version of this model is not free.

## The cliff label on a pair

The matched molecular pair literature calls a pair a cliff at a potency difference above
2 log units (Molecules 26 (2021) 4916, and MMP-Cliffs, Hu, Stumpfe and Bajorath). MoleculeACE
uses a 10-fold difference, which is 1 log unit, together with three similarity criteria. The
2025 kinase paper used 100-fold, which is 2 log units. The cutoff is a setting and the three
sources disagree on it.

## What none of these sources state

- How class imbalance was handled in any of them.
- Any concrete C value.
- A minimum number of pairs per target for a usable model.
- Any head-to-head test of one model per target against one model across targets.
