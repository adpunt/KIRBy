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
