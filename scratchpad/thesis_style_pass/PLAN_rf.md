# Plan: random forest rank statements, Chapter 3 (chap:robustness)

Source text: /Users/apunt/repos/KIRBy/thesis.tex as of 2026-10-10. Each OLD block below was checked with python3 -I str.count to occur exactly once.

Data: /Users/apunt/repos/qsar_qm_models/results/decisions_arc_20261008/auc_norm_qm9.csv and auc_norm_assay.csv (median AUC_norm per dataset, model, representation and noise condition; 10 replicates on QM9, 5 scaffold folds on the assay datasets). decisions_arc/ holds an identical d9_rank_transfer.csv. No SOURCE comment sits near any of these sentences. The author says some Chapter 3 results are being fixed, so these ranks must be re-checked once the new results land.

Method: the 13 base models are the 20 in the files minus the seven in VARIANT_MODELS (scripts/figlib_config.py): rf300, dnn_bnn_full_mve, mlp_bnn_full_mve, dnn_vbll_hetero, mlp_vbll_hetero, het_gp_rbf, gauche. Rank 1 is the highest AUC_norm among the base models present in one dataset, representation and noise condition. RF has no ties in any of them.

## What the data say

Rank of RF among the base models, by AUC_norm:

| scope | QM9 | LogD | hERG K_i | Caco-2 | all four |
|---|---|---|---|---|---|
| Gaussian, 6 representations | 2-3 | 1-2 | 2-4 | 3-7 | 1-7 |
| grouped-shifted, 6 representations | 1-2 | 1-2 | 1-4 | 3-8 | 1-8 |
| grouped-wider, 6 representations | 1-3 | 1-2 | 1-7 | 2-5 | 1-7 |
| Laplace, Student-t, outlier (ECFP4, PDV, ChemBERTa; 4 base models) | 1-2 | 1-2 | 1-2 | 1-2 | 1-2 |
| censoring (ECFP4 only; RF is the only base model) | 1 | 1 | 1 | 1 | 1 |
| all seven noise conditions | 1-3 | 1-2 | 1-7 | 1-8 | 1-8 |

The Gaussian seventh places are Caco-2 with Avalon and with Sort & Slice. The one eighth place is Caco-2 with PDV under grouped-shifted noise.

Verdict on the two sentences named in the task:
- Results, "among the 13 base models under Gaussian noise it ranks between first and seventh ... second or third on QM9 and first on logD at five of six": correct (1-7; QM9 2-3; LogD first at five of six, second at Avalon).
- Conclusions, "ranking between first and eighth for \aucnorm{} on every dataset, representation and noise condition": correct as a range. The scopes differ because the eighth place comes from grouped-shifted noise, which the Results sentence leaves out. The sentence is loose in one way. Laplace, Student-t, outlier and censoring did not run on every representation, and ranked RF against 4 base models or none. F1 below is optional.

Other random-forest rank statements:
- Chapter 1 (line ~101, "random forests were the most consistent model across datasets, representations, and noise conditions") gives no rank. No change.
- Abstract in main.tex and Chapter 8 give no RF rank. No change.
- Chapter 3 Results, grouped-shifted paragraph: wrong for hERG K_i (F2).
- Chapter 3 Results, forests paragraph: wrong (F3).

Overlap: PLAN_ch3.md also edits the paragraphs of F2 and F1 (for example "group-shifted" -> "grouped-shifted", "logD" -> "LogD", "data you trust"). The OLD blocks below avoid those words, so they apply before or after PLAN_ch3.md. Put the todo_line above the whole paragraph; if PLAN_ch3.md already added an "edited your text" TODO there, append this reason to it.

## Edits

### F1 — (optional) say which noise conditions the Conclusions rank range covers
- file: thesis.tex
- kind: author-text
- why: The range 1-8 is right. Laplace, Student-t and outlier noise ranked RF against 4 base models on three representations, and censoring ran RF on ECFP4 only, so "every representation and noise condition" among 13 models does not hold for them. Every number below is already in thesis.tex.
- todo_line: % TODO: manual review -- edited your text: the 1st-8th range holds among the 13 base models under Gaussian, grouped-wider and grouped-shifted noise; the other four conditions ran on fewer models and representations, so the sentence now names the three. Optional; the old sentence is not wrong as a range.
OLD:
```latex
ranking between first and eighth for \aucnorm{} on every dataset, representation and noise condition.
```
NEW:
```latex
ranking between first and eighth of the 13 base models for \aucnorm{} on every dataset and representation under Gaussian, grouped-wider and grouped-shifted noise.
```

### F2 — RF is not first or second on hERG K_i under grouped-shifted noise at every representation
- file: thesis.tex
- kind: author-text
- why: Under grouped-shifted noise RF ranks first or second on LogD at all six representations. On hERG K_i it is first (PDV), second (ECFP4, ChemBERTa, Sort & Slice), third (MHG-GNN) and fourth (Avalon). "fourth" is a new number from the data file above, not yet in thesis.tex.
- todo_line: % TODO: manual review -- edited your text: under grouped-shifted noise RF is first or second on LogD at every representation, but third with MHG-GNN and fourth with Avalon on hERG K_i (auc_norm_assay.csv, decisions_arc_20261008); sentence split to say so.
OLD (if PLAN_ch3.md has already run, "logD" reads "LogD"):
```latex
RF is the dominant model on hERG $K_i$ and logD, where it ranks first or second for \aucnorm{} with every representation.
```
NEW:
```latex
RF is the dominant model on LogD, where it ranks first or second for \aucnorm{} with every representation. On hERG $K_i$ it ranks first to fourth.
```

### F3 — The forests do not rank first or second outside Caco-2, and QRF is not third to seventh on Caco-2
- file: thesis.tex
- kind: author-text
- why: Under Gaussian noise RF ranks 3-7 on Caco-2, but QRF ranks 1-4 there. Outside Caco-2, RF is 2-4 on hERG K_i and QRF is 3-8 on hERG K_i, so "first or second" holds only on LogD and QM9 for RF. The NEW sentence keeps to RF and uses numbers already in thesis.tex. The author may prefer to restate QRF herself.
- todo_line: % TODO: manual review -- edited your text: under Gaussian noise RF ranks 3rd-7th on Caco-2 but QRF ranks 1st-4th, and outside Caco-2 neither forest is first or second on hERG K_i (RF 2nd-4th, QRF 3rd-8th); sentence now states RF only. Check QRF wording.
OLD:
```latex
Caco-2 is the exception, where they rank third to seventh rather than first or second.
```
NEW:
```latex
Caco-2 is the exception, where RF ranks third to seventh under Gaussian noise, against second or third on QM9 and first or second on LogD.
```
