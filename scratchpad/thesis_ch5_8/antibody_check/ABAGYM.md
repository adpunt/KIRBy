# AbAgym sanity check: abagym_EGFR, abagym_VEGF, abagym_Ang2

Written 2026-10-09. Read this session: `src/kirby/datasets/abagym.py`,
`tests/antibody_baseline_survey.py` (registry + `_assign_groups`),
`tests/experiments/dataset_policy.py`, the cached source table
`~/.cache/kirby/abagym/AbAgym_data_non-redundant_interface.csv`, the three result
files in `/Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/`,
and `/Volumes/seagate/claude_analysis_2026-10-08/answers/FAULTS.md`.

Every number below was read in this session unless the line says otherwise.

---

## Common to all three

**Source.** One table, `AbAgym_data_non-redundant_interface.csv`, downloaded by
`src/kirby/datasets/abagym.py` from the 3BioCompBio/AbAgym GitHub. It holds 36,541
mutations at antibody-antigen interfaces across 68 deep mutational scans. 63 of the
68 scans mutate the antigen. 5 mutate the antibody. Those 5 are the only ones the
loader keeps. Three of them are these datasets.

**Label.** The loader uses the column `MinMax_normalized_DMS_score`, rescaled per
scan to [0, 1]. The paper's sign convention is that a positive score means binding
affinity went down. So a high label is a mutation that loses binding, not one that
gains it. Her metric is Spearman, so the direction does not change the score, but it
changes the sentence in the thesis.

**What one row is.** One single amino-acid substitution in one chain of one parent
antibody. Heavy and light are both mutated. The rest of both chains is the parent
sequence read out of the PDB file.

**Folds.** `tests/antibody_baseline_survey.py` line 75-77 registers all three as
`('abagym', 'position')`. `_assign_groups` builds one group per (chain, residue
position) and refuses any variant carrying more than one mutation. So all ~19
substitutions at one position sit on the same side of every split. 5 outer folds.

**A curation flag on two of the three (paper, OPENED).** The AbAgym paper says that
for ANG2-G6, VEGF-G6, HIV-VRC01 and HER2-Trastuzumab "tabular data was not available
for download, but graphical heatmaps of the DMS scores were included", and that the
authors "processed the heatmap image and mapped the RGB color of each cell to numeric
values". The labels in abagym_Ang2 and abagym_VEGF are therefore colours read off a
published figure, not numbers from a supplementary table. The EGFR scan is not in
that list.

**The literature the paper gives (OPENED).** Cia, Li, Poblete, Rooman, Pucci,
"AbAgym: a well-curated dataset for the mutational analysis of antibody-antigen
complexes", mAbs 17(1), 2025, DOI 10.1080/19420862.2025.2592421, read at
https://pmc.ncbi.nlm.nih.gov/articles/PMC12645894/. Table 1, average per-scan
performance over the 37,361 interface mutations, all antigen and antibody scans
pooled into one average:

| Method | Type | Spearman | ROC-AUC |
|---|---|---|---|
| RSA (relative solvent accessibility, baseline) | baseline | 0.28 | 0.69 |
| FoldX5.1 | energy | 0.25 | 0.72 |
| BeAtMuSiC | energy | 0.20 | 0.63 |
| SAAMBE-3D | machine learning | 0.11 | 0.58 |
| mCSM-AB | machine learning | 0.01 | 0.49 |
| SaProt | protein language model | 0.01 | 0.54 |
| CI (conservation index) | evolutionary | 0.01 | 0.56 |
| LOR (log-odds ratio) | evolutionary | -0.05 | 0.52 |
| korpm | energy | -0.04 | 0.48 |

Every method in that table is applied without training on AbAgym: they are existing
tools scored zero-shot on a structure and a mutation. The only trained model in the
paper is a linear fit, evaluated leave-one-complex-out. The page gives no per-scan
number for any antibody-side scan; it points to its Supplementary Tables 1 and 2,
which I did not open. So **there is no published supervised, cross-validated Spearman
on any one of these three scans.** Her numbers cannot be put next to a like-for-like
literature number. They can only be put next to a zero-shot average over a different
mixture of scans.

Earlier note to correct: `antibody_datasets_research.md` says "Best method Spearman =
0.28 (RSA, a trivial structural feature)". That is right and now re-checked, but it is
an average over 37,361 interface mutations dominated by antigen-side scans, not a
number for these three.

---

## 1. abagym_EGFR

**What it is.** Cetuximab against EGFR, PDB 1yy9, DMS scan source given by the AbAgym
paper as Forsyth et al. 2013, mAbs (reference 25 of the paper; I did not open that
paper). Endpoint: the scan's own score, min-max rescaled to [0, 1], unitless. One row
is one single substitution in cetuximab. The source table holds 659 interface
mutations for this scan; the loader keeps all 659 (375 heavy, 284 light). 36 position
groups, 14 to 19 rows each. Label mean 0.641, standard deviation 0.265, range 0.015 to
1.000. All 659 sequences are distinct.

**Is the task useful.** It ranks which single substitutions in a marketed antibody
lose EGFR binding. That is the measurement an engineer makes before mutating a lead
for developability or humanisation: you want to know which interface residues you may
touch. The decision it would support is real. What limits it is that it is one
antibody: a model fitted here tells you nothing about a second antibody, and the
thesis claim has to be "which representation ranks substitutions within one antibody".

**Her scores.** c02, `standard_selection_summary`, Spearman, 5 outer folds, models svm
and dnn, none dropped.

| model | arm | mean | per fold | spread (min to max) |
|---|---|---|---|---|
| svm | best single representation | 0.2821 | 0.1114, 0.2142, 0.5164, 0.3491, 0.2197 | 0.1114 to 0.5164 |
| svm | hybrid | 0.3138 | 0.2343, 0.3076, 0.5069, 0.3036, 0.2164 | 0.2164 to 0.5069 |
| dnn | best single representation | 0.1810 | 0.1694, 0.1988, 0.5201, 0.1893, -0.1727 | -0.1727 to 0.5201 |
| dnn | hybrid | 0.2160 | 0.1309, 0.1215, 0.5783, 0.1501, 0.0994 | 0.0994 to 0.5783 |

The single representation picked per fold changes fold to fold: mutation_features,
abroberta_cdr, esm2_early, abroberta_cdr, abroberta_cdr for svm.

**Literature.** Table above. No per-scan number for this scan exists on the page I
opened; Forsyth et al. 2013 NOT OPENED.

Her svm hybrid, 0.3138 Spearman mean over 5 position-grouped folds, is above the
zero-shot pooled averages the AbAgym paper reports (RSA 0.28, FoldX 0.25), but those
are a different quantity on a different mixture of scans, so **no comparable number
exists**. Say that in the thesis rather than claiming to beat FoldX.

**Red flags.**
- No published supervised number on this scan to compare against.
- Fold 2 carries the score: svm goes 0.1114, 0.2142, 0.5164, 0.3491, 0.2197. The mean
  is not a stable estimate with 5 folds.
- dnn has one negative fold on the single representation, -0.1727.
- One parent antibody, so nothing here speaks to unseen antibodies.
- 36 position groups over 659 rows; a held-out fold is roughly 7 positions.

**Verdict.** REPORT. 659 rows, no duplicate sequences, no site dropped, labels from
the scan's own table rather than a figure, both models positive on every fold except
one dnn fold, and the hybrid is above the single representation on both models.

---

## 2. abagym_VEGF

**What it is.** The G6 antibody against VEGF, PDB 4zff, DMS scan source given by the
AbAgym paper as Koenig et al. 2017, PNAS (reference 31; NOT OPENED). Endpoint the same
rescaled scan score, unitless. The source table holds 513 interface mutations. The
loader keeps 418 (285 heavy, 133 light). 21 position groups. Label mean 0.680,
standard deviation 0.173.

**95 rows are dropped by the loader and the reason is residue numbering.** The loader
compares the parent residue named in the AbAgym table against the residue it finds in
the PDB file at that number, and skips the row when they disagree. Five sites
disagree: H100 P, H100A Y, H52 T, L28 F, L29 L, 19 rows each. In 4zff the heavy chain
is numbered with insertion codes (…51I 52T 52AP 53A…, …99L 100P 100AY 100BA…) and the
light chain starts 26S 27Q 27AF 28L 29S. `_parse_site` strips the letter, so AbAgym
site "100A" is looked up as residue 100 and the parent residue does not match. Those
five sites never reach the model.

**19 pairs of rows share a sequence and carry two different labels.** Same cause: the
light chain has both site 30 (S) and site 30A (S), both become position 30, both
produce the same mutant sequence from `apply_mutation`, and the two scan scores differ
by 0.111 on average. Those 19 pairs also land in one 38-row position group, twice the
size of the others.

**Is the task useful.** Same shape as EGFR: which single substitutions in one
anti-VEGF antibody lose binding. The G6 scan is the measurement behind a real
affinity-maturation campaign, so the endpoint is meaningful.

**It largely duplicates a dataset she already runs.** `flab_binding_koenig` is the same
G6 antibody against VEGF, 4,275 single mutants, label the binding Kd as -log10. Matched
this session on sequence: 285 of the 418 abagym_VEGF rows (every heavy-chain row) are
also rows of `flab_binding_koenig`, and on those 285 the abagym label and the FLAb Kd
label correlate at Spearman 0.698. The same experiment reaches her pipeline twice, once
with 418 rows read off a heatmap and once with 4,275 rows from a table. (On
abagym_Ang2 the same match gives Spearman 0.055 on 169 heavy rows, which is the control:
different antigen, so no agreement expected.)

**Her scores.** c02, Spearman, 5 folds, models gp and svm.

| model | arm | mean | per fold | spread |
|---|---|---|---|---|
| gp | best single representation | 0.1981 | 0.5109, 0.3402, 0.3033, -0.1312, -0.0326 | -0.1312 to 0.5109 |
| gp | hybrid | 0.3017 | 0.3089, 0.5166, 0.3459, 0.1322, 0.2048 | 0.1322 to 0.5166 |
| svm | best single representation | 0.4317 | 0.6279, 0.4915, 0.7467, 0.2435, 0.0489 | 0.0489 to 0.7467 |
| svm | hybrid | 0.4737 | 0.3710, 0.6817, 0.7639, 0.1274, 0.4242 | 0.1274 to 0.7639 |

**Literature.** Table above; no per-scan published number; Koenig 2017 NOT OPENED. **No
comparable number exists.**

**Red flags.**
- 95 of 513 rows silently dropped on insertion-code numbering, five whole positions.
- 19 sequences carry two labels each, from merging light sites 30 and 30A.
- Labels were read off a published heatmap image by the AbAgym authors.
- 285 of 418 rows are also rows of `flab_binding_koenig`, Spearman 0.698 between the
  two labels: reporting both as separate results double-counts one experiment.
- gp is in the five datasets of fault A2 in `FAULTS.md` (kernel length scale started at
  1; searches ran 6 Oct, the fix ffcd254 is 8 Oct), so its two negative folds and its
  mean 0.1981 are due for replacement. Not re-checked by me.
- svm fold 4 at 0.0489 single, 0.4242 hybrid; fold 3 the other way, 0.2435 single,
  0.1274 hybrid.

**Verdict.** REPORT WITH CAVEAT, or DROP if `flab_binding_koenig` is already in the
chapter. The caveat is three lines long: 95 dropped rows, 19 double-labelled
sequences, heatmap-derived labels, and an overlap with `flab_binding_koenig` at
Spearman 0.698 on 285 shared mutants. Its svm numbers are the strongest of the three
datasets, which is the argument for keeping it; the overlap is the argument against.

---

## 3. abagym_Ang2

**What it is.** The same G6 antibody against angiopoietin-2, PDB 4zfg, DMS scan source
given by the AbAgym paper as Koenig et al. 2015, J Biol Chem (reference 23; NOT
OPENED). Same rescaled score. The source table holds 471 interface mutations; the
loader keeps 397 (169 heavy, 228 light). 20 position groups. Label mean 0.583,
standard deviation 0.231, range 0.000 to 1.000.

**74 rows dropped, same cause as VEGF:** H100A Y (17 rows), H52 T, L28 F, L29 L
(19 each) fail the parent-residue check because of insertion-code numbering in 4zfg.
**19 pairs share a sequence with two labels**, light sites 30 and 30A again, mean label
difference 0.405 — four times the VEGF gap, on a label whose standard deviation is
0.231. One position group holds 38 rows.

**One more thing specific to this scan (paper, OPENED).** The AbAgym rank-quartile
normalisation uses "the distribution of the enrichment ratio from the angiopoietin
2-G6 antibody complex" as its reference distribution. She uses the min-max column, not
the rank-quartile one, so this does not touch her labels; it does mean this scan is
the paper's own yardstick.

**Is the task useful.** Which single substitutions in G6 lose angiopoietin-2 binding.
G6 is a dual-specificity engineering target — the same antibody is scanned against
VEGF and against Ang2 — so the pair is the interesting object: a model would have to
rank the same substitutions differently for two antigens. Measured here: on the 169
heavy mutants that appear in both this scan and the VEGF Kd data, the two labels
correlate at Spearman 0.055. The two endpoints really are different. As a single
dataset, it is one antibody and one antigen.

**Her scores.** c02, Spearman, 5 folds, models gp and svm.

| model | arm | mean | per fold | spread |
|---|---|---|---|---|
| gp | best single representation | 0.4550 | 0.2073, 0.6126, 0.7541, 0.6403, 0.0608 | 0.0608 to 0.7541 |
| gp | hybrid | 0.5049 | 0.2410, 0.7776, 0.6380, 0.4933, 0.3748 | 0.2410 to 0.7776 |
| svm | best single representation | 0.4543 | 0.1980, 0.6713, 0.5940, 0.6101, 0.1979 | 0.1980 to 0.6713 |
| svm | hybrid | 0.5216 | 0.2690, 0.7047, 0.5785, 0.4767, 0.5792 | 0.2690 to 0.7047 |

These are the highest antibody numbers in the c02 table.

**Literature.** Table above; no per-scan published number; Koenig 2015 NOT OPENED. **No
comparable number exists.**

**Red flags.**
- 74 of 471 rows silently dropped on insertion-code numbering, four whole positions.
- 19 sequences carry two labels each, mean difference 0.405 against a label standard
  deviation of 0.231. That is the worst of the three.
- Labels were read off a published heatmap image by the AbAgym authors.
- gp is in fault A2 in `FAULTS.md`, so its mean 0.4550 single and 0.5049 hybrid are due
  for replacement. Not re-checked by me.
- 397 rows in 20 position groups; a held-out fold is 4 positions.
- `CH5_RESULTS.md` item 4 says abagym_Ang2 is named as a win in two outline bullets and
  has no c11 or c17 file. Re-checked: no `abagym_Ang2.json` under c11 or c17 in the
  10-09 copy.

**Verdict.** REPORT WITH CAVEAT. The strongest antibody scores she has and the hybrid
beats the single representation on both models, but the caveat has to state the 74
dropped rows, the 19 double-labelled sequences, the heatmap-derived labels, and that
the gp column is due for replacement.

---

## If she wants the loader defect fixed first

Both faults are one function. `_parse_site` in `src/kirby/datasets/abagym.py` strips the
insertion-code letter from a site label, so "100A" and "100", and "30A" and "30", become
the same residue. The first collapse makes the parent-residue check fail and the row is
skipped; the second makes two different mutants produce the same sequence with two
different labels. Fixing it needs the PDB insertion code carried through the lookup, and
would change n on abagym_VEGF from 418 to 513 and on abagym_Ang2 from 397 to 471, so
every cached representation and every search on those two datasets would have to be
rebuilt. abagym_EGFR is unaffected: 1yy9 has no insertion codes at the scanned sites and
all 659 rows load.
