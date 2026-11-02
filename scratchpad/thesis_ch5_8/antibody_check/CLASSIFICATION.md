# Antibody sanity check: the two classification datasets

Written 2026-10-09 by Claude. Recommendations only; the decision is hers.

Scores are re-read from
`/Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/<dataset>.json`,
key `standard_selection_summary`. "Best single" is `base1`: the single representation ranked
first on the training antibodies of that fold. "Hybrid" is `hybrid1`. 5 outer folds. SD is the
sample SD over the 5 folds. Metric is AUROC, higher is better, 0.5 is chance.

The fold sizes in the result files match the folds rebuilt locally from the current loaders
(humanness 125/104/101/130/104 antibodies, cdr_flexibility 69/61/63/70/65 rows), so the
split facts below describe what the server ran.

---

## 1. humanness_therapeutic

### What it is

- One row is one therapeutic antibody from IMGT/mAb-DB, as packaged by the BioPhi paper.
  Source files: `Merck/BioPhi-2021-publication/data/tasks/humanness/imgt_mab_db.{tsv,fa}`.
- Input is the heavy chain variable region (VH) only. The light chain is set to empty in
  `src/kirby/datasets/humanness.py` (`_load_therapeutic`).
- Label: 1 if the IMGT species field is exactly "Human", else 0. n = 564: 198 human, 366 not.
- The 0 class holds 229 humanized, 63 chimeric, 41 humanized/chimeric, 13 mouse, 6 caninized,
  3 felinized, 7 "Human/Humanized" and 4 "Human/Chimeric" antibodies.
- So the task is mostly human against humanized. A humanized VH has human framework and
  grafted mouse CDRs.
- Folds: antibodies with at least 95% VH sequence identity are grouped
  (`tests/antibody_baseline_survey.py`, `_identity_groups`). This gives 461 groups; 404 of them
  hold one antibody; the largest holds 13. So the split is close to one over distinct
  antibodies. Folds hold 32 to 50 human antibodies each.
- Decided models: xgboost and rf. No gp.

### Is the task useful

A humanness score is used to flag an antibody candidate whose sequence looks non-human, as an
immunogenicity risk. That is a real use. But the clinical endpoint is anti-drug antibody
response, not the species label. The species label is an administrative category, and a
score of percent identity to the nearest human germline already separates it (0.968 AUROC,
below). The dataset is a recognised benchmark: BioPhi used it, and Ucar et al. 2024 reused it.

### Her scores (AUROC)

| model | best single, mean | single per fold | single SD | hybrid, mean | hybrid per fold | hybrid SD | hybrid minus single | folds hybrid better |
|---|---|---|---|---|---|---|---|---|
| xgboost | 0.9511 | 0.9352, 0.9701, 0.9670, 0.9368, 0.9466 | 0.0165 | 0.9555 | 0.9139, 0.9878, 0.9652, 0.9524, 0.9580 | 0.0269 | +0.0043 | 3 of 5 |
| rf | 0.9659 | 0.9564, 0.9735, 0.9773, 0.9681, 0.9542 | 0.0102 | 0.9614 | 0.9311, 0.9889, 0.9701, 0.9643, 0.9527 | 0.0214 | -0.0045 | 1 of 5 |

Single representation picked per fold: xgboost igt5, antiberty_cdr, igt5, abroberta,
antiberty_cdr; rf antiberty_cdr on 4 folds, igt5 on 1.

The hybrid minus single difference is 0.004 to 0.005 AUROC either way. The fold SD is 0.010
to 0.027. The difference is below the fold SD with both models.

### Literature

Every literature number below scores all the antibodies in the set at once. None of them used
cross-validation folds, and only Hu-mAb was trained on this label.

Source A: Prihoda et al., BioPhi, bioRxiv 2021 preprint doi 10.1101/2021.08.08.455394,
published in mAbs 2022, doi 10.1080/19420862.2021.2020203. OPENED (preprint PDF read this
session). Their set is 198 human against 355 non-human therapeutics (229 humanized,
41 humanized/chimeric, 63 chimeric, 13 mouse, 6 caninized, 3 felinized), 553 antibodies, with
each antibody's score averaged over its chains. AUROC, no split:

| method | AUROC | trained on this label |
|---|---|---|
| Hu-mAb (random forest) | 0.977 | yes |
| germline content (percent identity to nearest human germline) | 0.968 | no |
| IgReconstruct | 0.967 | no |
| OASis medium identity | 0.966 | no |
| MG score | 0.959 | no |
| AbLSTM | 0.937 | no |
| T20 score | 0.896 | no |
| Z-score | 0.837 | no |

Source B: Ucar et al., "Improving Antibody Humanness Prediction using Patent Data",
arXiv:2401.14442v3, ICML 2024 workshop listing. OPENED (HTML full text read this session).
Their Table 3 re-scores the same 553 therapeutics, but puts humanized in the POSITIVE class:
449 paired antibodies of which 423 are called human. That is a different label from hers, so
the numbers are not comparable to hers. Recorded so the pair is not mistaken for the same task:
FT-SelfPAD 91.84, AbNatiV 86.91, OASis 85.42, Hu-mAb 63.90 AUROC on single chains.

Her number against source A: her 0.9511 to 0.9659 AUROC sits with the unsupervised scores
(0.959 to 0.968) and below supervised Hu-mAb (0.977). The comparison is not exact: she scores
on 5 held-out folds of 101 to 130 antibodies each, source A scores all 553 at once, and she
uses VH only while source A averages both chains. So her number is in line with the
literature, on a harder read-out than the literature used.

### Red flags

- 0.95 to 0.97 AUROC means the task is nearly solved before her method starts; the room above
  the best single representation is about 0.03 AUROC.
- The hybrid gain is +0.0043 (xgboost) and -0.0045 (rf) AUROC, both under the fold SD of 0.010
  to 0.027. No direction can be read from it.
- VH only. Every literature number averages heavy and light. State it if the comparison is
  printed.
- A plain percent-identity-to-nearest-human-germline score reaches 0.968 in source A, above
  both of her models. The dataset cannot show that a representation matters.
- n = 564, so a fold holds 101 to 130 antibodies and 32 to 50 of the human class. A 0.01 AUROC
  difference on that is a handful of antibody pairs reordering.
- Label is the IMGT species field, an administrative category, not a measured immunogenicity.
- Source A's own count is 553 antibodies and hers is 564: her loader keeps every row whose
  species field is non-empty and whose VH is in the FASTA, including 7 "Human/Humanized" and
  4 "Human/Chimeric" rows that it labels 0. Source A's negative list does not mention those 11.
  Unverified whether source A dropped them or merged them elsewhere.

### Verdict

REPORT WITH CAVEAT. It is a recognised benchmark and her score is in line with published
numbers, so it works as a check that the pipeline is sound on a classification antibody task.
It cannot carry a claim about the hybrid: the gain is smaller than the fold spread with both
models, and a germline-identity score already scores higher than she does.

---

## 2. cdr_flexibility

### What it is

- Source: ITsFlexible, Spoendlin et al., "Predicting the conformational flexibility of antibody
  and T cell receptor complementarity-determining regions", Nature Machine Intelligence 7,
  1755-1767 (2025), doi 10.1038/s42256-025-01131-6. OPENED (Europe PMC full text and the
  Springer supplementary PDF read this session).
- Her loader (`src/kirby/datasets/itsflexible.py`, `_load_cdr`) reads two files from the
  ITsFlexible GitHub `data/` folder: `CDRH3_test_align_Fv.csv` and `CDRL3_test_align_Fv.csv`.
  Those are the paper's CDR test sets.
- One raw row is one crystal structure of one loop. The files hold 1,627 CDRH3 rows and 1,227
  CDRL3 rows. The loader keeps the first row per distinct CDR3 sequence, which leaves 216 CDRH3
  and 112 CDRL3, and concatenates them: n = 328 rows, 237 flexible (72.3%), 91 rigid.
- Label: 1 if the loop was seen in more than one conformational cluster across its crystal
  structures (Ca RMSD above 1.25 A, complete-linkage clustering); 0 if it was seen in one
  conformation in more than five separate PDB files. Loops meeting neither rule are dropped
  upstream by the paper.
- Features come from the whole Fv sequence (heavy and light, split on the underscore in the
  `Fv` column), not from the loop alone.
- Folds: the CDR3 loop sequences are clustered at 80% identity
  (`tests/experiments/_shared.py`, the `cdr_flexibility` branch, `sequence_identity_clusters`).
  301 groups over 328 rows; 281 groups hold one row; the largest holds 5. Fold sizes 69, 61,
  63, 70, 65. Flexible fraction per fold 0.689 to 0.746, so the class mix is even across folds.
- Decided models: xgboost and lightgbm. No gp.

### Is the task useful

Knowing whether a CDR3 loop has one shape or several tells you whether a single predicted
structure can be trusted for docking or for a structure-based affinity calculation. That is a
real decision an antibody engineer makes. The paper's own motivation is exactly that. But the
label is read off how many crystal structures happen to exist for that antibody, so it is a
statement about the PDB as much as about the molecule, and the paper says so: a loop counts as
rigid only if five or more structures agree, and the authors show that one classifier
(ABB2 RMSPE) was partly predicting how often an antibody appeared in its training set rather
than flexibility.

### Her scores (AUROC)

| model | best single, mean | single per fold | single SD | hybrid, mean | hybrid per fold | hybrid SD | hybrid minus single | folds hybrid better |
|---|---|---|---|---|---|---|---|---|
| xgboost | 0.8527 | 0.7168, 0.8985, 0.8670, 0.8210, 0.9600 | 0.0912 | 0.8689 | 0.7863, 0.9173, 0.8803, 0.8809, 0.8799 | 0.0489 | +0.0163 | 4 of 5 |
| lightgbm | 0.8713 | 0.7411, 0.8960, 0.8963, 0.8691, 0.9542 | 0.0792 | 0.8669 | 0.7905, 0.9010, 0.8763, 0.8948, 0.8719 | 0.0444 | -0.0044 | 3 of 5 |

Single representation picked per fold: xgboost abroberta, antiberty, antiberty, esm2_concat,
onehot; lightgbm esmc, ablang2, ablang2, esm2_concat, ablang2.

Per-fold hybrid minus single, xgboost: +0.0695, +0.0188, +0.0133, +0.0598, -0.0801.
lightgbm: +0.0495, +0.0050, -0.0199, +0.0256, -0.0824. Fold 4 is the loss with both models;
it is also the fold where the single representation scores highest (0.96, 0.95).

Her fold spread is 0.72 to 0.96 AUROC. The gain, +0.016 or -0.004, is far below that.

### Literature

Spoendlin et al. 2025, supplementary Table S10 (ROC AUC) and Table S9 (PR AUC). OPENED. Their
CDRH3 test set is n = 147 loops and CDRL3 is n = 84; both are scored separately, never pooled.
ITsFlexible is a graph neural network trained on general PDB loops, with every loop above 80%
identity to any CDR3 in these test sets removed from training, so the CDR sets are zero-shot
for it.

| method | input | CDRH3 (n=147) AUROC | CDRL3 (n=84) AUROC | trained on CDRs |
|---|---|---|---|---|
| ITsFlexible, crystal structure | 3D | 0.71 | 0.87 | no, general PDB loops |
| ITsFlexible, ImmuneBuilder model | 3D predicted | 0.69 | 0.88 | no |
| ITsFlexible, AlphaFold2 model | 3D predicted | 0.61 | 0.81 | no |
| AlphaFold2 MSA subsampling | 3D predicted | 0.68 | 0.81 | no, zero-shot |
| ABB2 RMSPE (retrained) | 3D predicted | 0.61 | 0.81 | no, zero-shot |
| AlphaFold2 pLDDT | 3D predicted | 0.63 | 0.75 | no, zero-shot |
| loop length | scalar | 0.60 | 0.75 | fitted on PDB loops |
| length + solvent exposure | scalars | 0.63 | 0.72 | fitted on PDB loops |
| solvent exposure | scalar | 0.53 | 0.53 | fitted on PDB loops |
| random | - | 0.50 | 0.50 | - |

Her 0.85 to 0.87 AUROC is above every number in that table, including the paper's own model on
CDRH3 (0.71). The comparison is not on the same task. Four differences, not one:

1. She pools CDRH3 and CDRL3 into one set of 328 rows and reports one AUROC. The paper scores
   them separately, and its two scores differ by 0.16 AUROC (0.71 against 0.87). Pooling two
   sets with different flexible fractions (81% flexible for her CDRH3 rows, 55% for her CDRL3
   rows) lets a model score by telling heavy from light. The sequences make that trivial: the
   heavy and light chains are both in the features.
2. Her n is 328 rows against the paper's 147 plus 84 = 231 loops. The paper's supplementary
   Table S5 counts 152 CDRH3 and 84 CDRL3 loops with a determined flexibility label. Her
   loader keeps 216 CDRH3 rows. Unverified where the extra 64 come from; one candidate is
   that she dedups on the CDR3 sequence while the paper groups on the whole Fv.
3. Her model is supervised on this label, with 258 to 267 training rows per fold. Every
   number in the table is either zero-shot on these loops or fitted on general PDB loops.
4. She predicts from sequence features only. Every number in the table uses 3D structure,
   crystal or predicted.

So: no comparable number exists. Her 0.87 and their 0.71 are not the same measurement.

### Red flags

- Heavy and light pooled in one score. 216 of 328 rows are CDRH3 and 81% of them are flexible;
  112 are CDRL3 and 55% of them are flexible. The chain type alone carries signal, and the
  features contain the chain. One AUROC over the pool cannot be compared with the paper's two.
- 81 Fv sequences appear twice in her 328 rows, once as their CDRH3 and once as their CDRL3.
  Of those 81 pairs, 67 are split across two different folds, 58 of them carrying the same
  label. The grouping is on the CDR3 loop sequence, so it does not keep an antibody's two rows
  together. That is near-duplicate information straddling the train/test boundary.
- n = 328 rows, 91 of them rigid. A fold holds 61 to 70 rows and roughly 17 rigid ones.
- Fold spread 0.72 to 0.96 AUROC, SD 0.079 to 0.091. The hybrid gain, +0.016, is a fifth of it.
- The label counts crystal structures, so it is partly a statement about how often an antibody
  was crystallised. The paper demonstrates that bias on one of its own baselines.
- The dedup keeps the first row per CDR3 sequence. 3 CDRH3 and 2 CDRL3 sequences carry both
  labels in the raw files; which label survives depends on row order.
- `DATASET_STATUS_2026_07_07.md` section 2.2 cut this dataset as "random-split-in-disguise",
  citing an assertion in `opig_cdr_flexibility.py` that every group had size 1. That claim is
  dated and is no longer true of the code that produced these numbers: the c02 path groups on
  the CDR3 loop at 80% identity and the result file records 301 groups with a largest of 5,
  so 47 rows sit in a group with another row. The split is still 281 singletons out of 301
  groups, which is close to a random split over distinct loops, and the Fv-pair leakage above
  is the sharper problem.

### Verdict

DROP, or re-run first. Two things have to be fixed before the number means anything: the
heavy and light loops need scoring separately, as the paper does, and an antibody's two rows
need to stay in the same fold. Until then the 0.87 AUROC cannot be read against the paper's
0.71, and the hybrid's +0.016 is a fifth of the fold spread. If she wants a flexibility
result, the cheaper route is to report CDRH3 alone with the pairs grouped by Fv.

---

## One line each

- humanness_therapeutic: REPORT WITH CAVEAT. 0.9511 (xgboost) and 0.9659 (rf) AUROC for the
  best single representation, in line with the published unsupervised scores of 0.959 to 0.968,
  but the hybrid moves it by under 0.005 against a fold SD of 0.010 to 0.027.
- cdr_flexibility: DROP or re-run. Heavy and light loops are pooled into one AUROC where the
  paper reports two that differ by 0.16, and 67 of 81 antibodies have their heavy row and
  light row in different folds.
