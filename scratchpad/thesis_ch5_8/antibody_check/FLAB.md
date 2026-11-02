# FLAb antibody datasets: sanity check before reporting

Written 2026-10-09. Three datasets: `flab_expression_koenig`, `flab_binding_koenig`,
`flab_thermo_tresanco`.

Everything below was read in this session: the CSV in `~/.cache/kirby/flab/`, the loader
`src/kirby/datasets/flab.py`, the grouping code in `tests/antibody_baseline_survey.py`, and the
raw result files under
`/Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/`.
Literature numbers are marked OPENED (I read the number on the page) or NOT OPENED.

Metric throughout: Spearman rank correlation between predicted and measured label, higher is
better, computed on held-out molecules per outer fold. 5 outer folds.

---

## 1. flab_expression_koenig

### What it is

The file is `expression/koenig2017mutational_er_g6.csv`, 4,275 rows, four columns: heavy, light,
`enrichment ratio`, `fitness`. The label is the enrichment ratio from a phage-display selection,
a dimensionless ratio. Range 0.001 to 3.178, mean 1.005, standard deviation 0.571, 4,186 distinct
values in 4,275 rows.

One row is one single amino-acid substitution of one parent antibody: the anti-VEGF Fab G6.31.
I counted the substitutions against the modal heavy and modal light sequence: all 4,275 rows carry
exactly one substitution. 225 distinct positions, 19 substitutions at each, across both chains.

Folds: `tests/antibody_baseline_survey.py` registers this dataset as `('flab', 'position')`, which
makes one group per (chain, position). That gives 225 groups of 19 rows each, and the 5-fold split
holds out whole positions. The result file's fingerprint says `n_samples 4275`, `n_groups 225`, so
the run used that grouping.

Representations in the pool: 10 protein language model representations, 480 to 3,072 features wide
(ablang2, esm2, esm2_concat, esm2_early, esm2_midlayers, esm3_cdr, esmc, igt5, prot_t5, saprot).
Feature budgets searched: 250, 500, 1,000, 2,000.

### Is the task useful

The measurement is how well a Fab variant is displayed and folded, which is the first filter in
antibody engineering: a variant that does not express cannot be taken forward whatever its
affinity. A model on this task answers one question — for this one antibody, which of the 4,275
possible single substitutions can I make without losing expression. It does not transfer to a
different antibody, because every sequence in the file is one mutation away from the same parent.
Koenig 2017 PNAS is the source and is a recognised dataset: it is in the FLAb benchmark collection
and used as a benchmark by later papers (DASM, below).

### Her scores

svm, Spearman on held-out positions:

| arm | per fold (0..4) | mean | min | max |
|---|---|---|---|---|
| best single representation | 0.6983, 0.7227, 0.6820, 0.7373, 0.6741 | 0.7029 | 0.6741 | 0.7373 |
| hybrid | 0.7954, 0.7968, 0.7579, 0.7526, 0.7619 | 0.7729 | 0.7526 | 0.7968 |

The single representation picked on training molecules was esm3_cdr on folds 0, 1, 3 and prot_t5 on
folds 2, 4. Hybrid higher on 5 of 5 folds, mean gain 0.0700 Spearman. The spread across folds is
0.063 Spearman for the single representation and 0.044 for the hybrid.

dnn, Spearman:

| arm | per fold (0..4) | mean | min | max |
|---|---|---|---|---|
| best single representation | 0.7103, 0.6935, 0.6715, 0.7343, 0.7719 | 0.7163 | 0.6715 | 0.7719 |
| hybrid | 0.7564, 0.7417, 0.6946, 0.7481, 0.7354 | 0.7352 | 0.6946 | 0.7564 |

Hybrid higher on 4 of 5 folds, mean gain 0.0189 Spearman.

### Literature

| source | number | metric | split | supervised or zero-shot |
|---|---|---|---|---|
| DASM, Sung et al., eLife reviewed preprint 109644, 2025, Table 1 (OPENED, pmc.ncbi.nlm.nih.gov/articles/PMC12633574/) | 0.688 heavy chain, 0.674 light chain | Pearson | none; every mutation scored | zero-shot |
| same table, ESM2 | 0.418 heavy, 0.524 light | Pearson | none | zero-shot |
| same table, ProGen2 | 0.559 heavy, 0.568 light | Pearson | none | zero-shot |
| same table, AbLang2 | 0.344 heavy, 0.084 light | Pearson | none | zero-shot |
| FLAb2, Chungyoun et al., bioRxiv 2025 (OPENED, ncbi.nlm.nih.gov/pmc/articles/PMC12767642/) | 0.22 average over models, 0.44 best (ESM2 150M) | Spearman | none | zero-shot, expression datasets as a group |
| FLAb2, same paper, embeddings plus a two-layer network | 0.56 best (ESM2 35M, ISM 3B) | Spearman | random 80/10/10 | supervised, expression datasets as a group |

No number exists on the same metric and the same split. The nearest comparison is FLAb2's
supervised 0.56 Spearman under a random 80/10/10 split over its expression datasets, which is below
her 0.7729, and her split is the stricter of the two because it holds out whole positions. The DASM
numbers are Pearson, not Spearman, and are scored on every mutation with nothing held out, so her
number and 0.688 cannot be put side by side.

### Red flags

- One parent antibody. Every row is a single substitution of G6.31, so a result here says nothing
  about a second antibody.
- The hybrid chose the largest budget, 2,000 features, on all 5 folds and both models. The best
  number of features may be above the grid.
- The label is an enrichment ratio from display, not a titer in mg/L. It mixes expression, folding
  and assembly; the FLAb loader docstring and the Koenig patent family both say so.
- The same 4,275 sequences carry the binding label too (next section). Rank correlation between the
  two labels is -0.499 over the 4,275 rows. The two tasks are not independent evidence.
- dnn gain 0.0189 Spearman is smaller than the fold spread of 0.100 Spearman for that model; the
  svm gain 0.0700 is larger than its fold spread of 0.063.

### Verdict

REPORT. 4,275 rows, whole positions held out, hybrid higher on 5 of 5 folds with svm, and the gain
exceeds the fold spread. State that it is one antibody and that no literature number shares the
metric and split.

---

## 2. flab_binding_koenig

### What it is

The file is `binding/koenig2017mutational_kd_g6.csv`, 4,275 rows. The `fitness` column is the
`-log( KD (M) )` column: minus log base 10 of the dissociation constant in molar. Range 7.221 to
12.158, mean 9.368, standard deviation 0.556, 4,132 distinct values.

The heavy and light columns are byte-identical, row for row, to the expression file — I checked.
So one row is again one single substitution of the G6.31 anti-VEGF Fab, 225 positions by 19
substitutions, and the registry gives it the same `('flab', 'position')` grouping, 225 groups.
The fold-0 payload in the partial file reports 3,420 training and 855 held-out rows, which matches
a 5-fold split of 4,275.

### Is the task useful

How tightly a variant binds VEGF, in molar Kd on a log scale. This is the endpoint an affinity
maturation campaign optimises, so a model that ranks substitutions by it supports a real decision:
which positions to mutate next. Same single-antibody limit as the expression set.

### Her scores

There is no finished result file. Only `flab_binding_koenig.partial.jsonl` exists, with the
identity line and fold 0. One of 5 folds, so there is no mean and no spread.

Fold 0, Spearman on held-out rows, models rf and svm:

| model | best single representation | hybrid | difference |
|---|---|---|---|
| rf | esm3, 0.4220 | 0.4611 | +0.0391 |
| svm | ablang2, 0.4690 | 0.5339 | +0.0649 |

Both hybrids used 1,000 features drawn from all 10 representations.

### Literature

| source | number | metric | split | supervised or zero-shot |
|---|---|---|---|---|
| DASM, eLife 109644, Table 1 (OPENED, PMC12633574) | 0.335 heavy, 0.316 light | Pearson | none | zero-shot |
| same table, ESM2 | -0.001 heavy, 0.308 light | Pearson | none | zero-shot |
| same table, ProGen2 | 0.156 heavy, 0.276 light | Pearson | none | zero-shot |
| same table, AbLang2 | -0.032 heavy, 0.049 light | Pearson | none | zero-shot |
| FLAb2 (OPENED, PMC12767642) | 0.61 best (ISM 3B) | Spearman | random 80/10/10 | supervised, binding datasets as a group |
| NbBench, Zhang and Tsuda, arXiv 2505.02022v2, Table (OPENED, arxiv.org/html/2505.02022v2) | 0.184 best (IgBert) | Spearman | MMseqs2 70% identity clusters | supervised on frozen embeddings, nanobody affinity, different data |

No number exists on the same metric and the same split. Her 0.5339 on one fold sits between
FLAb2's supervised 0.61 under a random split and NbBench's 0.184 under 70%-identity clusters on
different (nanobody) data. The DASM 0.335 is Pearson with nothing held out.

### Red flags

- 1 of 5 folds exists. Every number above rests on one held-out set of 855 rows.
- Models differ from the expression run: this file searched rf and svm, the expression file searched
  svm and dnn. A side-by-side of the two Koenig tasks would be comparing different models.
- The representation pool also differs: this run carried esm3, esmc_cdr and esm2_entropy_top20,
  the expression run carried esm3_cdr, esmc and esm2_midlayers. Four of ten entries are not the
  same.
- One parent antibody, as above.
- Label correlated with the expression label at -0.499 Spearman over the same 4,275 rows.
- It is in `thesis_figure_style.py`'s dataset maps (label `Anti-VEGF scan binding`, domain
  Antibodies), so any figure that reads c02 will want 5 folds it does not have.

### Verdict

DROP from the figures for now, or REPORT WITH CAVEAT as a single-fold line in text. The reason is
coverage, not the task: 1 of 5 folds, and a different model pair and representation pool from its
sibling. If the run finishes with 5 folds and the same two models as the expression set, it becomes
the stronger pair of the two Koenig tasks, because binding is the endpoint the field cares about.

---

## 3. flab_thermo_tresanco

### What it is

The file is `thermostability/tresanco2023nbthermo_tm.csv`, 672 rows, 556 distinct heavy sequences,
no light chain. The label is the melting temperature in degrees Celsius. Range 39.0 to 98.2, mean
67.5, standard deviation 10.1, 228 distinct values.

One row is one nanobody (a VHH single-domain antibody, no light chain) with one measured Tm, taken
from the literature by the NbThermo curators. Sequence lengths 104 to 151 residues. Two further
columns carry the assay and the species: `Tm Type` is circular dichroism 307 rows, DSF (SYPRO) 254,
nanoDSF 76, differential scanning calorimetry 30, other 5; `Origin` is llama 395, camel 189,
alpaca 36, shark 30, unknown 20, human 2.

116 heavy sequences appear more than once with different Tm values. Within a repeated sequence the
mean standard deviation of Tm is 1.92 °C and the largest gap between two values is 8.8 °C.

Folds: the registry entry is `('flab', 'identity')`, which runs
`sequence_identity_clusters` at 95% identity, single-linkage, on the heavy chain. The result file's
fingerprint says `n_groups 309` over 672 rows, so 309 clusters. Near-duplicates stay on one side of
every fold.

Representations in the pool: 10, including onehot at 5,600 features and `esm2_wt_delta`. Budgets
searched: 50, 100, 250, 500, 1,000, 2,000.

### Is the task useful

Melting temperature decides whether a nanobody survives formulation and storage, and unlike binding
it is an intrinsic property of the fold, so a model trained on one set of nanobodies can score a
new one. This is the one task of the three that answers a question about an antibody the model has
never seen, which is also what makes it the right one for a claim about representations.

### Her scores

dnn, Spearman on held-out clusters:

| arm | per fold (0..4) | mean | min | max |
|---|---|---|---|---|
| best single representation (saprot, all 5 folds) | 0.7700, 0.7542, 0.5636, 0.6867, 0.6951 | 0.6939 | 0.5636 | 0.7700 |
| hybrid | 0.6998, 0.7819, 0.5174, 0.6672, 0.7157 | 0.6764 | 0.5174 | 0.7819 |

Hybrid lower by 0.0175 Spearman on the mean, higher on 2 of 5 folds.

svm, Spearman:

| arm | per fold (0..4) | mean | min | max |
|---|---|---|---|---|
| best single representation (saprot, all 5 folds) | 0.7550, 0.7214, 0.6572, 0.6687, 0.6482 | 0.6901 | 0.6482 | 0.7550 |
| hybrid | 0.6604, 0.7688, 0.6475, 0.6684, 0.7044 | 0.6899 | 0.6475 | 0.7688 |

Hybrid lower by 0.0002 Spearman on the mean, higher on 2 of 5 folds. The fold spread is 0.107
Spearman for the single representation and 0.121 for the hybrid.

### Literature

The same NbThermo source, so these are the closest comparisons of the three datasets.

| source | number | metric | split | supervised or zero-shot |
|---|---|---|---|---|
| NanoMelt, Ramon et al., mAbs 2025 (OPENED, pmc.ncbi.nlm.nih.gov/articles/PMC11730357/), 640 unique nanobody Tm curated from NbThermo | 0.832 | Spearman | nested stratified 3-fold CV, strata = assay method and k-medoids sequence cluster (13 clusters) | supervised ensemble |
| same paper, best single model, ESM-1b plus SVR | 0.816 | Spearman | same | supervised |
| same paper, one-hot plus GPR | 0.784 | Spearman | same | supervised |
| same paper, external set of 80 sequences | 0.751 | Pearson | held out entirely | supervised |
| same paper, ESM-2 sequence likelihood | 0.337 | Pearson | none | zero-shot |
| NbBench, arXiv 2505.02022v2 (OPENED), Thermo-seq, 761 rows from NbThermo, TEMPRO and NanoMelt | 0.587 best (AntiBERTa2) | Spearman | MMseqs2 70% identity clusters, 20% of low-frequency clusters held out | supervised on frozen embeddings |
| same paper, Thermo-tm, 567 rows | 0.593 best (AntiBERTa2-CSSP) | Spearman | stratified sampling by binned Tm | supervised on frozen embeddings |
| TEMPRO, Alvarez and Dean, Scientific Reports 2024 | mean absolute error 4.03 °C, root mean squared error 5.66 °C | error in °C, no correlation found | not established | supervised; NOT OPENED, from a search snippet only |
| FLAb2 (OPENED, PMC12767642), thermostability datasets as a group | 0.61 average, 0.75 best (ISM uc30 650M) | Spearman | random 80/10/10 | supervised on embeddings |

Her 0.69 Spearman is below NanoMelt's 0.832 on essentially the same source data, and above
NbBench's 0.587 and 0.593. The splits are not the same in any of the three: hers holds out
95%-identity clusters, NanoMelt stratifies by cluster but does not hold clusters out, NbBench holds
out 70%-identity clusters and is therefore stricter than hers. The one number that should worry her
is NanoMelt's one-hot plus Gaussian process at 0.784 Spearman, which is above every arm she reports
here, under a looser split.

### Red flags

- The hybrid loses with both models: dnn -0.0175, svm -0.0002 Spearman.
- 672 rows, 556 distinct sequences, 309 clusters. Small.
- Five different assays in one label column, and they are not a covariate or a group in her code.
  Circular dichroism, SYPRO DSF, nanoDSF and DSC do not return the same Tm for the same protein.
- 116 sequences carry more than one Tm, disagreeing by up to 8.8 °C. That is a ceiling on any
  correlation measured here, and the duplicate rows are 232 of 672.
- `esm2_wt_delta` is in the pool. It is computed as a difference from the modal sequence, and the
  modal heavy sequence here appears twice, so the reference is an arbitrary row. The loader's own
  comment says the modal sequence is meaningless for an inter-family panel. It ranked 5th of 10 for
  dnn and 4th for svm on training rows, so it is in the feature pool the hybrid draws from.
- saprot is the best single representation on all 5 folds and both models, and the hybrid does not
  beat it. Any story here is about saprot, not about the hybrid.
- Her 0.69 is below a one-hot baseline published on the same source data (0.784 Spearman,
  NanoMelt), under a looser split.

### Verdict

REPORT WITH CAVEAT, as the antibody case where the hybrid does not help. It is the only one of the
three that generalises to an unseen antibody, and the literature comparison is real rather than
metric-mismatched, so dropping it would remove the honest negative. The caveats to state: five
assays pooled, 8.8 °C disagreement between repeat measurements of the same sequence, and the
hybrid behind the best single representation on the mean of 5 folds.

---

## Notes for the figure script

- All three datasets are already in `tests/analysis/thesis_figure_style.py`: labels at lines 317 to
  319 (`Nanobody thermostability`, `Anti-VEGF scan binding`, `Anti-VEGF scan expression`) and the
  Antibodies domain at lines 338 to 342.
- `flab_binding_koenig` has 1 of 5 folds. Any figure that reads c02 per-fold values will plot a
  single point for it, or drop it, depending on how the reader handles a partial file. Decide which
  before the figure is generated.
- `flab_expression_koenig` was searched with svm and dnn, `flab_binding_koenig` with rf and svm. A
  figure that puts one marker per dataset per model will have no shared model across the two
  Koenig panels except svm.
- `tests/analysis/thesis_figures.py` itself names no antibody dataset; it reads the domain map, so
  whichever datasets are passed in will be drawn.
