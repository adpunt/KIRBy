# \paragraph headings in thesis.tex (2026-10-10)

45 found in thesis.tex. 43 became \subsubsection, 2 were removed. None left.
No subsection that held them already had a \subsubsection.
Line numbers are from before the edit.

## Became \subsubsection (heading kept, trailing full stop dropped)

| Line | Subsection | New heading |
|---|---|---|
| 517 | The NoiseInject package | Regression noise |
| 569 | The NoiseInject package | Classification noise |
| 576 | The NoiseInject package | Robustness metrics |
| 592 | The NoiseInject package | Uncertainty metrics |
| 1528 | Flaws in similarity and other pairwise computations | Activity cliffs |
| 1532 | same | Distance and label difference |
| 1538 | same | Agreement on neighbors |
| 1540 | same | Agreement with shared structure |
| 2406 | Datasets and splitting | Pre-processing |
| 2411 | same | Splitting rules |
| 2414 | same | Groups by domain |
| 2431 | same | Consequences of the grouping |
| 2453 | Representations by domain | Small molecules |
| 2455 | same | Chemical reactions |
| 2458 | same | Antibodies |
| 2742 | Comparators, controls and ablations | Best single representation and PCA |
| 2746 | same | Ensembles |
| 2750 | same | Many or few representations |
| 2754 | same | Easy and hard halves |
| 2758 | same | Overfitting with the budget |
| 2762 | same | One fold fewer |
| 2766 | same | Feature reduction at the same width |
| 2770 | same | Internal evaluator |
| 2774 | same | Importance during training |
| 2787 | same | Fewer features or a tuned model |
| 2791 | same | Adding a new embedding |
| 2795 | same | Calibration |
| 2799 | same | Stability and blueprints |
| 3550 | Experiments (similarity chapter) | Distance against label difference |
| 3557 | same | Prediction from nearest neighbors |
| 3565 | same | Importance weighting in four models |
| 3573 | same | Activity cliffs |
| 3583 | same | Distance against prediction error |
| 3591 | same | Retrieval and clustering |
| 3597 | same | Three smaller checks |
| 4116 | Noise mitigation | Selected by a detector |
| 4136 | same | Selected by a known flag (\label{sec:noise:censoring} kept on the same line) |
| 4167 | same | Every molecule |
| 4178 | Noise-robust training | Changing the model instead of the labels |
| 4182 | same | The expanded published sweep |
| 4214 | same | Robust losses |
| 4218 | same | Two learners |
| 4223 | same | Repeat measurements |

Where the heading ran into the paragraph text on one line, the text now starts on the next line. No words changed.

## Removed (the only run-in heading in its subsection)

| Line | Subsection | Heading | Paragraph | Text change |
|---|---|---|---|---|
| 1956 | From base representations to a hybrid | Blueprints. | hers (no review marker in the backup) | none; her first sentence left as written |
| 3485 | Nearest-neighbor settings | Importance weighting. | AI-written (% TODO: manual review) | none needed; first sentence starts "Importance weighting is feature-weighted nearest neighbors" |

## Note

main.tex uses the report class, which numbers down to subsections only. The new \subsubsection headings are unnumbered, so \label{sec:noise:censoring} resolves to its subsection's number, as it did under \paragraph. No \ref to it exists in thesis.tex.

## Other files (not edited)

- main.tex: no \paragraph.
- preamble.tex: no \paragraph.
- thesis_rework/stage1_takeaways_bullets/additional_files_body.tex: 19 \paragraph, at lines 1461, 1466, 1502, 1510, 1516 (search settings, importance methods, allocation methods, matched internal evaluator, fast tier) and 1696, 1700, 1705, 1710, 1715, 1720, 1763, 1767, 1787, 1792, 1798, 1803, 1815, 1819 (one per dataset: hERG Ki, LogD and Caco-2, RLM and MDR1-MDCK, QM9, Lhasa hERG sites, hERG IC50, Buchwald-Hartwig yield and plates, Koenig, nanobody thermostability, humanness, trastuzumab-HER2, H3K4me3, TATA promoter).
