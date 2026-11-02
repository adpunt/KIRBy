# Sanity check: ddg_combined and abcov_kd

Written 2026-10-09. Every number below was read in this session from
`/Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/<dataset>.json`
(key `standard_selection_summary`, written 2026-10-06), from the loaders in `src/kirby/datasets/`,
or from a paper page I opened. Spearman is the metric on both datasets, higher is better, 5 outer
folds. `base1` is the best single representation picked on the training molecules of that fold;
`hybrid1` is the top hybrid of that fold. Both are scored on that fold's held-out rows.

A note used twice below. Spearman over a held-out block of n rows wobbles even when the model knows
nothing: with permuted labels its standard deviation is about 1/sqrt(n-1). That is 0.11 for a
86-row block and 0.19 for a 28-row block. Neither number is a significance test; it is the width of
the ruler.

---

## ddg_combined

### 1. What it is

The endpoint is DDG, the change in antibody-antigen binding free energy on mutating the antibody,
in kcal/mol. Positive means the mutation weakens binding.

One row is one mutant of one antibody-antigen complex: a parent PDB structure plus 1 to 9 amino-acid
substitutions on the antibody side only (298 of 429 rows are single substitutions; the rest carry 2
to 9). Mutations on the antigen side are dropped (`ddg_combined.py` lines 359-369).

n = 429 rows over 30 PDB complexes, measured in this session by running the loader. Two sources are
merged: the antibody-antigen subset of SKEMPI 2.0 (267 rows survive parsing out of 1,231 candidate
entries) and AB-Bind (230 rows). 35 rows are in both and AB-Bind's value is kept. 25 rows are then
dropped as outliers, |DDG| > 8 kcal/mol or more than 3 standard deviations out. Labels run -4.06 to
6.00 kcal/mol, mean 0.47, median 0.27. 32% of rows sit within 0.5 kcal/mol of zero.

The folds are leave-one-PDB-complex-out, `assign_complex_groups` in the same file, 30 groups into 5
folds. Held-out blocks are 85 or 86 rows. The complexes are very uneven: 1MHP alone holds 69 of the
429 rows, 1VFB 36, 3L5X 35, and three complexes hold one row each.

Two things about the labels are worth writing down because they are not the same measurement.
AB-Bind rows are DDG measured directly. SKEMPI rows that had no DDG column are computed from the
mutant and wild-type Kd by RT ln(Kd_mut/Kd_wt) (`_compute_ddg`, line 212). So one target column mixes
two quantities with different noise.

### 2. Is the task useful

Yes, and it is the one antibody task here that maps onto a decision somebody actually makes. An
affinity-maturation campaign has a parent antibody and a list of candidate point mutations, and has
to pick which ones to make. A model that ranks mutations by DDG is exactly that filter. It is also
the task the field has agreed to measure on: AB-Bind and SKEMPI are the two standard sets, and the
leave-one-complex-out split is the strict one.

### 3. Her scores

Decided models: svm and dnn (`scratchpad/precompute_cuts/DECISIONS.md` line 23).

| model | arm | mean Spearman | per fold (0-4) | fold spread (sd) |
|---|---|---|---|---|
| svm | best single representation | 0.0819 | 0.1254, 0.2372, 0.1881, -0.3775, 0.2364 | 0.2609 |
| svm | hybrid | 0.2178 | 0.1270, 0.1166, 0.2861, 0.3217, 0.2374 | 0.0926 |
| dnn | best single representation | 0.1317 | 0.3427, 0.0680, 0.0760, 0.2151, -0.0434 | 0.1494 |
| dnn | hybrid | 0.3542 | 0.1961, 0.2050, 0.4120, 0.5834, 0.3744 | 0.1609 |

The single representation picked differs by fold. svm: esm2_wt_delta, mutation_features,
mutation_features, antiberty, mutation_features. dnn: esm2_wt_delta on folds 0-2, onehot on folds 3
and 4.

The svm gain is one fold. The per-fold hybrid-minus-single differences are 0.0017, -0.1205, 0.0980,
0.6992, 0.0010; fold 3 is where the single representation scored -0.3775. The dnn gain is spread
over four of five folds: -0.1466, 0.1369, 0.3360, 0.3683, 0.4178.

### 4. Literature

One row of this table is one published result on antibody-antigen DDG. Pearson, not Spearman, in
every case; no paper I opened reports Spearman on the experimental sets.

| source | number | metric | dataset | split | supervised? | opened |
|---|---|---|---|---|---|---|
| Hummer, Schneider, Chinery, Deane 2025, Nat Comput Sci 5:635-647, doi 10.1038/s43588-025-00823-8 (Graphinity) | 0.87 | Pearson | AB-Bind, 645 single-point mutations, 29 complexes | random 10-fold | supervised, structure-based | OPENED (pmc.ncbi.nlm.nih.gov/articles/PMC12374841) |
| same paper, Fig. 2b | drops 63% on average from the random-split value | Pearson | same | 100% length-matched CDR identity cutoff | supervised, structure-based | OPENED |
| same paper, Methods | 0.34 | Pearson | AB-Bind | none, FoldX is not trained | not supervised, structure-based | OPENED |
| same paper, Methods | 0.20 | Pearson | SKEMPI 2.0 subset, 608 single-point mutations, 44 complexes | none | not supervised, structure-based | OPENED |
| same paper, Methods | 0.46 | Pearson | SKEMPI 2.0 subset, 608 mutations | none | not supervised, Rosetta Flex ddG | OPENED |
| same paper, citing ref. 17 (TopNetTree) | 0.17 average | Pearson | AB-Bind | leave-one-complex-out | supervised, structure-based | number read in Graphinity's text; TopNetTree itself NOT OPENED |
| same paper, Supplementary Table 2 (DGCddG, RDE-PPI retrained) | 0.26 and 0.22 | Pearson | AB-Bind | 90% CDR identity cutoff | supervised, structure-based | numbers read in Graphinity's text; the two papers NOT OPENED |

Her number is in line with the literature on a comparable split. The only published
leave-one-complex-out number I found is Pearson 0.17 and the only identity-cutoff retrained numbers
are 0.22 and 0.26, all from structure-based models; her dnn hybrid is Spearman 0.3542 and her svm
hybrid 0.2178 from sequence alone. Two cautions on that sentence: Pearson and Spearman are not the
same statistic, and her 429 rows are a different row set from the 645 and 608 used in those papers.
The 0.87 is a random split and is not the comparison.

### 5. Red flags

- 30 complexes, and one of them (1MHP) holds 69 of 429 rows, 16%. A fold's score depends heavily on
  which complexes landed in it.
- Each held-out block is 85 or 86 rows, so a fold's Spearman wobbles by about 0.11 on its own.
- The svm result is one fold. Drop fold 3 and the svm gain is about +0.01.
- The target mixes directly measured DDG (AB-Bind) with DDG computed from two Kd values (SKEMPI).
- The splits are by PDB code, and the loader's own header (lines 24-30) says structurally similar
  interfaces recur across distinct PDB codes, so some leakage remains. Graphinity measured that same
  effect as the 63% drop under a CDR identity cutoff; her split does not apply an identity cutoff.
- The objective `similarity` (predicting each mutant's DDG from its five nearest neighbours) is
  undefined on outer folds 1-4: 1MHP is a held-out block of its own inside the search and has no
  neighbour outside it (`dataset_policy.py` lines 669-672). So any claim about that objective rests
  on fold 0 alone.
- Published sequence-only DDG performance has a ceiling. The loader's header puts it at Pearson
  0.25-0.50 against about 0.73 for structure-based methods; that is a repo claim, dated and not
  re-verified here.
- No gp in the decided models, so the gp-replacement question does not apply.

### 6. Verdict

REPORT WITH CAVEAT. The task is the one real antibody engineering decision in this set, the split is
the strict one, and the dnn hybrid gain is spread over four of five folds. The caveat has two parts
to state in the text: the svm gain is one fold, and the single-representation arms score near zero
(0.08 and 0.13), so the hybrid's gain is measured against a weak reference, not against a strong one.

---

## abcov_kd

### 1. What it is

The endpoint is -log10(Kd in M) for an antibody binding a coronavirus spike protein. 5.22 to 11.22,
mean 8.14, standard deviation 1.18, so about 6 orders of magnitude of Kd.

One row is one whole antibody, heavy plus light chain. There is no parent and no mutant: these are
different antibodies from different publications, not a mutational library (`abcov.py` sets
`is_intra_family: False` and `mutations` to empty lists).

n = 141 rows, read in this session from `/Users/apunt/.cache/kirby/flab/binding_rawat2022abcov_kd.csv`.
136 distinct heavy chains; 109 rows have a light chain over 10 residues. The docstring at
`abcov.py` lines 8 and 41 says about 1,780 antibodies. That is the size of the Ab-CoV database, not
of this file. I re-downloaded the file from FLAb in this session
(`raw.githubusercontent.com/Graylab/FLAb/main/data/binding/rawat2022abcov_kd.csv`): 141 rows,
byte-identical to the local cache. So the file is not a truncated download; the 1,780 is a wrong
docstring. FLAb's own data index lists the Ab-CoV binding entry at n = 568, which is the Kd count in
the source paper, not this file's 141.

The folds are 95%-identity single-linkage clusters on heavy+light concatenated
(`_identity_groups` in `tests/antibody_baseline_survey.py`, strategy `identity`). Measured here:
112 clusters over 141 antibodies, largest 5, 93 of 112 singletons. Held-out blocks are 28 or 29 rows.

Source: Rawat, Sharma, Prabakaran, Ridha, Mohkhedkar, Janakiraman, Gromiha 2022, Bioinformatics
38(16):4051-4052, doi 10.1093/bioinformatics/btac439, reached through FLAb. OPENED. That paper
curates 568 Kd values over 1,780 antibodies against SARS-CoV, SARS-CoV-2 and MERS-CoV spike, pulled
from the literature; it names no assay type and reports no prediction model on Kd.

### 2. Is the task useful

The decision it would support is weak. Each row is a different antibody against a spike protein that
is not recorded in the file, with a Kd taken from a different publication and a different assay. So
the model is asked to rank antibodies that were never measured against each other. That is not a
decision anyone makes in a campaign: nobody chooses between two unrelated antibodies on affinity
numbers from two papers. Compare ddg_combined, where every row shares a parent complex and the
ranking is the decision.

There is a measurable consequence of that. Four pairs or triples of rows have identical heavy and
light sequences but different labels, spanning up to 1.26 in -log10 Kd, which is 18-fold in Kd. The
same antibody, two reported numbers. That is the floor: no model can resolve differences below it.

### 3. Her scores

Decided models: lightgbm and xgboost (`DECISIONS.md` line 14).

| model | arm | mean Spearman | per fold (0-4) | fold spread (sd) |
|---|---|---|---|---|
| lightgbm | best single representation | 0.2494 | 0.3409, -0.1547, 0.2167, 0.5362, 0.3078 | 0.2542 |
| lightgbm | hybrid | 0.1654 | 0.0498, 0.1947, 0.4001, 0.4700, -0.2875 | 0.3030 |
| xgboost | best single representation | 0.2925 | 0.3212, 0.3578, 0.3919, 0.2838, 0.1079 | 0.1108 |
| xgboost | hybrid | 0.3050 | 0.3394, 0.3098, 0.1708, 0.5745, 0.1303 | 0.1749 |

Representations picked per fold. lightgbm: abroberta, onehot, abroberta, abroberta_cdr, abroberta.
xgboost: blosum, onehot, esmc_cdr, abroberta, esm3.

The hybrid loses with lightgbm, -0.0840 on the mean, and the per-fold differences change sign four
times: -0.2911, 0.3494, 0.1834, -0.0662, -0.5953. With xgboost it is +0.0125, which is a tenth of
that model's own fold spread.

### 4. Literature

No comparable number exists. I could not find any paper that reports a supervised,
identity-clustered Spearman for predicting Kd across unrelated anti-coronavirus antibodies from
sequence. The closest anchors, none of them on this split and task:

| source | number | metric | dataset | split | supervised? | opened |
|---|---|---|---|---|---|---|
| Chungyoun and Gray 2025, FLAb2 preprint | best few-shot binding result, rho 0.61 | Spearman | one of FLAb's binding datasets, not named in the text I read | random 80/10/10 | supervised, embeddings into a 2-layer network | OPENED (pmc.ncbi.nlm.nih.gov/articles/PMC12767642) |
| same | significant correlations on 20% of binding datasets zero-shot, 9% few-shot | Spearman | FLAb binding datasets | as above | both | OPENED |
| same | zero-shot per-dataset binding values -0.53 to 0.57 | Spearman | named FLAb binding sets, Ab-CoV not among those listed | n/a | zero-shot | OPENED |
| Zhang and Tsuda 2025, NbBench, arXiv 2505.02022 | affinity regression stays hard for all 11 models; the repo's earlier note records best rho under 0.20 | Spearman | nanobody affinity tasks | frozen embeddings | supervised head | abstract OPENED; the per-model affinity table was NOT OPENED, so the under-0.20 figure is unverified here |
| SimBinder-IF preprint, Dec 2025, arXiv 2512.17815 | ESM-IF mean rho 0.264, their model 0.410 | Spearman | AbBiBench 11-assay subset | zero-shot likelihood against measured affinity, within one parent antibody per assay | zero-shot | NOT OPENED, from a search snippet |

FLAb2 cites Ab-CoV (reference 88), so this file is probably among FLAb's 132 binding sets, but I
could not find a per-dataset number for it in the text I read. Her 0.25 to 0.31 is inside the band
these anchors occupy, but none of them is this task on this split, so it is not a comparison.

### 5. Red flags

- n = 141, held-out blocks of 28 or 29 rows. A fold's Spearman wobbles by about 0.19 on its own,
  which is larger than every mean gain in the table above.
- The docstring says about 1,780 antibodies and the dataset is 141. Any earlier write-up that quoted
  the larger number is wrong; `DATASET_STATUS_2026_07_07.md` line 47 calls it the strongest antibody
  card on the strength of about 1,780.
- Negative folds: lightgbm's best single representation scores -0.1547 on fold 1 and its hybrid
  -0.2875 on fold 4.
- The hybrid loses with lightgbm (-0.0840) and ties with xgboost (+0.0125 against a 0.1108 fold
  spread).
- Four groups of rows share an identical heavy+light sequence with labels differing by up to 1.26 in
  -log10 Kd. The labels come from different publications.
- The antigen is not in the file. Kd values against SARS-CoV, SARS-CoV-2 and MERS spike, and against
  different variants of each, sit in one target column.
- 93 of 112 identity clusters are singletons, so at 95% identity this split is close to splitting on
  distinct antibodies. That is the strict end, not a defect, but it is what the folds hold out and
  the write-up should say so rather than say "homology-clustered".
- No gp in the decided models.

### 6. Verdict

DROP, or report only as a named null. Two reasons, in order: the hybrid does not win on it (-0.0840
with lightgbm, +0.0125 with xgboost, both inside the fold spread), and at 28 rows per held-out block
no result on it can be read anyway. The underlying task is also the weakest of the two here, since
one column holds Kd values measured against different antigens in different papers. If you want an
antibody null in the chapter, abcov_kd is a defensible one to name, with the n and the mixed-antigen
target stated in the same sentence. It should not carry a figure of its own.

---

## Note on the figure script

`tests/analysis/thesis_figures.py` picks its datasets from `--datasets` or from the promoted rows of
`<results root>/ledger.json`, and orders them by `DATASET_DOMAIN`. Neither dataset is named in that
file, so dropping abcov_kd means leaving it out of `--datasets`; nothing in the script needs editing.
