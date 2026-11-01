# NEW_DATASETS — standard benchmarks outside drug discovery, and the noise suite's datasets

Written 2026-10-26. Four benchmarks with a public source, a published score, a fixed
protocol, and more than one feature set measured on the same rows: Matbench v0.1,
ProteinGym, Kepler and DroneRF. Nothing here is wired into KIRBy yet.

The plan for each benchmark:
1. Reproduce the published score under the published protocol (files below).
2. Run the hybrid search on the benchmark's feature sets, on training rows only.
3. Rerun the same published protocol on the hybrid's features and compare.

The reruns below only check that the benchmark files follow the published protocol.
The baseline the hybrid is compared against is the one run on the server, with the same
code and library versions as the hybrid runs.

Everything downloaded is under `/Volumes/seagate/benchmarks_protocol/`. Results of the
benchmark files go to `/Volumes/seagate/benchmarks_protocol/results/`.

Choosing a dataset's views: every candidate view is scored on its own, under the models
the dataset's papers used, and its overlap with the other views is measured. These
screens run through the dataset screening ledger (since 2026-11-01): one row per
dataset, one unit per view x model x task, queued jobs and sruns recording into one
file. Runbook `tests/slurm_scripts/runbook/SCREEN.md`; the datasets and their units are
in `tests/experiments/screen/datasets.py`. Each section below has a "Screen" subsection
naming its views, its models and where its `summary.md` lands. Rows today: Matbench
(section 3), DroneRF (section 6) and the three reaction-yield datasets (section 7).

---

## Small molecules for Papers 2 and 3

Chosen by Adelaide 2026-10-31 and 2026-11-01.

Which paper uses what (Adelaide, 2026-11-01):
- Paper 2: the complexity and OPIG experiments, on the benchmarks in this file and on the
  small-molecule and antibody datasets below.
- Paper 3: the similarity experiments (not settled) and the noise experiments, on Paper
  2's small-molecule and antibody datasets.
- Every dataset below runs every experiment: complexity, OPIG, similarity and noise. Which
  ones each paper shows is chosen later.

This section tracks what each dataset needs before it runs. Checked against the code
2026-11-01.

What the wiring columns mean:
- Registry: has an entry in `tests/experiments/dataset_policy.py`, giving task and metric.
- Representations: has a list in `complexity_helpers.DATASET_REPS`.
- Grid: has an entry in `tests/experiments/decided_grid.json`.
- Run list: returned by `orchestrate/plan.py::dataset_roster()`, so a ledger plans it.
- Loader: `_shared.load_data` loads it by this name.

| Dataset | What it is | Registry | Representations | Grid | Run list | Loader |
|---|---|---|---|---|---|---|
| qm9 | QM9 HOMO-LUMO gap, 10,000 molecules, computed labels | yes, rmse | yes | yes | yes | yes |
| qm9_mu | QM9 dipole moment (debye), same 10,000 molecules | yes, rmse | yes, qm9's | yes | yes | yes |
| qm9_alpha | QM9 polarizability (bohr^3), same 10,000 molecules | yes, rmse | yes, qm9's | yes | yes | yes |
| herg_ki | hERG Ki | yes, rmse | yes | yes | yes | yes |
| herg_ic50 | hERG IC50, repeat measurements per molecule | NO: no entry; `plan.py` falls back to regression | herg_ki's (`REP_LIST_FROM`) | yes, herg_ki's | own row, `HERG_IC50_EXPERIMENTS` | yes |
| caco2 | Caco-2 | yes, rmse | yes | yes | yes | yes |
| logd | LogD | yes, rmse | yes | yes | yes | yes |
| adme_rlm | Biogen rat liver microsomal clearance | yes, rmse | yes | yes | yes | yes |
| adme_hlm | Biogen human liver microsomal clearance | yes, rmse | yes | yes | NO: in `CUT_DATASETS`, cut 2026-10-09 for time; back in 2026-11-01 | yes |
| adme_mdr1 | Biogen MDR1-MDCK efflux | yes, rmse | yes | yes | yes | yes |
| herg | hERG blocking, classification | yes, auc | yes | yes | yes | yes |
| chembl_qsar | Dablander et al. 2023: dopamine D2 (6,333 molecules, Ki), factor Xa (3,605, Ki), SARS-CoV-2 Mpro (1,924, IC50); pKi or pIC50; ships matched molecular pairs | parent entry only, mae; to be rmse | yes, borrowed from six pruned datasets, never pruned | yes, herg_ki's settings | NO: in `EXCLUDED_DATASETS` | NO: only `opig_activity_cliff_qsar.py` loads it |

Still to do, one line each:
- [ ] herg_ic50's row lists the old experiment numbers (n17, n19, n22, n26, n29, n36). Add
      the new noise experiments that read repeat measurements.
- [ ] herg_ic50: has no registry entry. Code that needs its metric gets none.
- [ ] adme_hlm: take it out of `CUT_DATASETS`. It was cut 2026-10-09 for time, not for a
      fault in the data (Adelaide, 2026-11-01).
- [ ] chembl_qsar: one name covers three targets. The run list plans one row per dataset
      name, so each target needs a name of its own (for example chembl_qsar_d2).
- [ ] chembl_qsar: add a branch to `find_best_hybrid_molecular.load_dataset`, with scaffold
      groups like the other small-molecule datasets.
- [ ] chembl_qsar: change its registry metric from mae to rmse, like every other
      regression set above (Adelaide, 2026-11-01).
- [ ] chembl_qsar: take it out of `EXCLUDED_DATASETS`, so the complexity, similarity and
      noise experiments all run on it, not only `opig_activity_cliff_qsar.py`.
- [ ] chembl_qsar: its representation list is borrowed from six pruned datasets. It has
      never been through the pruning step.

### Antibodies for Papers 2 and 3

Chosen by Adelaide 2026-11-01. Every one runs every experiment, except antibody humanness,
which skips the noise experiments (one representation already solves it). Sizes and split
groups are from the thesis dataset table; abcov_kd and the trastuzumab sets from
DATASET_STATUS.md and a load on 2026-11-01.

| Dataset | What it is | Rows | Split groups | Run list |
|---|---|---|---|---|
| flab_expression_koenig | anti-VEGF scan, expression | 4,275 | 225 mutated positions | yes |
| flab_binding_koenig | anti-VEGF scan, binding | 4,275 | 225 mutated positions | yes |
| abagym_EGFR | EGFR binding scan | 659 | 36 mutated positions | yes |
| ddg_combined | binding free energy | 429 | 30 complexes | yes |
| abagym_Ang2 | angiopoietin-2 binding scan | 397 | 20 mutated positions | yes |
| ginkgo_acsins | AC-SINS self-association | 242 | 5 competition folds | yes |
| ginkgo_colloidal | SMAC colloidal stability | 242 | 5 competition folds | yes |
| ginkgo_hic | HIC retention | 242 | 5 competition folds | yes |
| ginkgo_titer | expression titer | 239 | 5 competition folds | yes |
| humanness_therapeutic | human or not; no noise experiments | 564 | 461 identity clusters | yes |
| cdr_flexibility | CDR3 loop flexible or rigid | 328 | split fixed 2026-11-01, see below | NO: in `CUT_DATASETS` |
| abcov_kd | anti-coronavirus binding, labels pooled from papers | 141 | identity | NO: in `CUT_DATASETS` |
| tz_her2_binary | trastuzumab CDRH3 library, binder or not | 5,000 of 524,346 | 10 edit-distance groups; to fix | NO: in `CUT_DATASETS` |
| tz_her2_multiclass | same library, three classes | 5,000 of 524,346 | 10 edit-distance groups; to fix | NO: in `CUT_DATASETS` |

Out (Adelaide, 2026-11-01): flab_thermo_tresanco, ginkgo_pr_cho, abagym_VEGF. Waiting:
flab_binding_adams and flab_expression_adams, until the server logs show why their
searches ran 1 to 36 hours without finishing.

Still to do, one line each:
- [x] cdr_flexibility split. One antibody can give two rows (its CDRH3 and its CDRL3), and
      the features are computed from the whole heavy and light chain, so the two rows have
      the same features. 81 of 247 antibodies have two rows; grouping on the loop alone put
      the two rows of all 81 in different groups. Fixed 2026-11-01 in
      `_shared._load_data_impl` (`_join_groups_by_key`): rows with the same heavy and light
      chain share a group. Check after the fix: 0 antibodies on both sides of any fold.
- [ ] cdr_flexibility: the two rows of one antibody have the same features but ask about
      different loops, and 10 of the 81 have one loop flexible and the other rigid. No
      feature says which loop a row is about. Adelaide to decide.
- [ ] tz_her2 split. The 10 groups are the number of mutations from the parent (1 to 10).
      Rows per group: 2, 6, 13, 60, 271, 824, 1,529, 1,440, 668, 187. The 5 folds held out
      271, 187, 1,494, 3,035 and 13 rows, and the binder share falls from 0.92 at 3
      mutations to 0.16 at 10, so each fold has a different binder share. Adelaide to
      decide the new grouping.
- [ ] Take cdr_flexibility, abcov_kd, tz_her2_binary and tz_her2_multiclass out of
      `CUT_DATASETS`.
- [ ] humanness_therapeutic: leave it out of the noise experiments' run list only.

---

## 3. Matbench v0.1 (six regression tasks)

### Source

| | |
|---|---|
| Data | `https://ml.materialsproject.org/projects/<task>.json.gz`, the files matminer's `load_dataset` downloads |
| Downloaded | 2026-10-23 to `/Volumes/seagate/matbench/data/`. sha256 of all six equals matminer's own metadata |
| Leaderboard | github.com/materialsproject/matbench, `benchmarks/matbench_v0.1_rf/` and `benchmarks/matbench_v0.1_modnet_v0.1.12/` (main at commit `fbb04eeb`), copied to `/Volumes/seagate/benchmarks_protocol/matbench/rf/` and `.../modnet_v0.1.12/` |
| MODNet code | modnet-matbench v0.4.0, Zenodo record 6368562, `modnet-matbench-v0.4.0.zip` md5 `d7c2b6e8eb35f09453824f6efc70911b`, unzipped in `.../modnet_matbench_v0.4.0/`. The GitHub tag the leaderboard links to no longer exists |
| Paper | Dunn, Wang, Ganose, Dopp, Jain, "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm", npj Comput. Mater. 6, 138 (2020) |

md5 of the data files and the two published results files:

| File | md5 |
|---|---|
| matbench_jdft2d.json.gz | `ed28bfc923fc85fa065430484a7344eb` |
| matbench_phonons.json.gz | `e3ceed6bd88fbbd5422fa621705cfc94` |
| matbench_expt_gap.json.gz | `7e6d1d1a110cf4b91c9ceaeb403c0735` |
| matbench_dielectric.json.gz | `e99b44cc39008af148da225b93e7b7ad` |
| matbench_log_gvrh.json.gz | `29b57cf9952f5980cd440f900eafc271` |
| matbench_perovskites.json.gz | `5e12df6ad727dd650986f570c479d2a5` |
| rf/results.json.gz | `c4d1e9a86bf8d1e188a1028b676803fd` |
| modnet_v0.1.12/results.json.gz | `925f166684e9963df839c9d0584dadaf` |

### What is in it

One row is one compound. The target is one number per compound. Metric: MAE, in the
target's own unit.

| Task | Compounds | Input | Target (unit) |
|---|---|---|---|
| matbench_jdft2d | 636 | structure | exfoliation energy (meV/atom) |
| matbench_phonons | 1,265 | structure | last phonon DOS peak (cm⁻¹) |
| matbench_expt_gap | 4,604 | composition only | measured band gap (eV) |
| matbench_dielectric | 4,764 | structure | refractive index (no unit) |
| matbench_log_gvrh | 10,987 | structure | log10 shear modulus (log10 GPa) |
| matbench_perovskites | 18,928 | structure | formation energy (eV/unit cell) |

"Structure" means a crystal structure per compound, from which every representation
below can be computed. "Composition only" means just the chemical formula, so only the
composition representations exist for expt_gap.

The split: Matbench's own 5 test folds, `KFold(5, shuffle=True, random_state=18012019)`
over the file's row order. Random, not grouped. You approved them for Matbench only on
2026-10-23. `src/kirby/datasets/matbench.py::load_matbench` already returns each row's
Matbench fold (`matbench_fold`) and one group per compound (`groups`).

### Preprocessing

None outside the two published pipelines below. The published scores are per fold, so
every comparison here is per fold.

### Published protocol: random forest ("RF-SCM/Magpie", leaderboard `matbench_v0.1_rf`)

From `run.py` and `info.json`, run inside automatminer 1.0.3:
- Features: magpie (132 features from the elements' properties) on every task, plus the
  sine Coulomb matrix (sorted eigenvalues, zero-padded) on structure tasks. Both fitted
  on the training compounds.
- Cleaning: automatminer `DataCleaner(max_na_frac=0.01, feature_na_method='mean',
  na_method_fit='drop', na_method_transform='mean')`.
- No feature reduction.
- Model: `RandomForestRegressor(n_estimators=500)`, every other setting sklearn's
  default. No seed recorded.

One place where the text and the code disagree:
- `info.json` says features missing in more than 1% of compounds were dropped.
  automatminer's code fills them with the training mean instead. The benchmark file
  follows the code. On all six tasks no feature was dropped and no training compound
  was dropped, so this does not change any score below.

What the cleaning does, read from automatminer's `DataCleaner.handle_na`:
- Infinite values become missing.
- A feature missing in every training compound is dropped.
- A feature missing in more than 1% of training compounds is filled with its training
  mean.
- A training compound still missing any value is dropped.
- A test value still missing is filled with that feature's mean over the test
  compounds. A feature missing in every test compound gets the training mean.

### Published protocol: MODNet v0.1.12 (leaderboard `matbench_v0.1_modnet_v0.1.12`)

From `run_benchmark.py` in the Zenodo archive, which pins `modnet==0.1.12`:
- Features: MODNet's own matminer set, `DeBreuck2020Featurizer(fast_oxid=True)` on
  structure tasks, `CompositionOnlyFeaturizer()` on expt_gap.
- Feature selection per training fold: `feature_selection(n=-1,
  use_precomputed_cross_nmi=True)`. Features are ranked by mutual information with the
  target minus redundancy with features already ranked. The redundancy between two
  features is read from a file MODNet ships, computed on Materials Project data, not
  from the fold.
- Model: `EnsembleMODNetModel`, 5 networks.
- Tuning: `fit_preset` with `nested=5`. MODNet's built-in list of settings (layer
  sizes, number of features, learning rate, ...), each scored by 5-fold CV inside the
  training fold.
- Folds: `matbench_kfold_splits`, the same 5 folds as above.

One place where the text and the code disagree:
- The leaderboard text says a genetic search was used on the four larger tasks. The
  archived code calls `fit_preset` on every task and never the genetic search. The
  benchmark file follows the code.

The published code needs the old versions. With current matminer (0.9) MODNet's
featurizer warns and computes 0 features. The benchmark file runs in its own
environment, `/Volumes/seagate/matbench/env_modnet_0112`: python 3.8, modnet 0.1.12,
matminer 0.6.5, pymatgen 2020.4.29, scikit-learn 0.23.2, tensorflow 2.4.4. The archive
pins only modnet. modnet 0.1.12 allows pymatgen 2020.0 to 2020.8 and any matminer from
0.6.2; matminer 0.6.5 is the newest that allows that pymatgen. Which versions the
published run actually had is not recorded. The build commands are in the file's
docstring.

### Published scores

One row is one task. MAE per fold, fold 0 to 4, in the target's unit.

| Task | Random forest | MODNet v0.1.12 |
|---|---|---|
| jdft2d | 42.7473 45.7510 66.2421 44.0340 51.4457 | 25.5515 30.4506 45.1925 26.9801 37.7845 |
| phonons | 82.3863 72.8871 59.2712 58.6036 64.9149 | 34.7662 36.3582 36.5373 32.1725 31.5413 |
| expt_gap | 0.4360 0.4387 0.4812 0.4345 0.4400 | 0.3121 0.3388 0.3763 0.3121 0.3241 |
| dielectric | 0.3042 0.4079 0.5220 0.3879 0.4760 | 0.1691 0.2410 0.3899 0.2775 0.2781 |
| log_gvrh | 0.1046 0.1024 0.1025 0.1037 0.1067 | 0.0731 0.0738 0.0731 0.0738 0.0718 |
| perovskites | 0.2357 0.2367 0.2365 0.2395 0.2291 | 0.0932 0.0939 0.0861 0.0892 0.0914 |

### Reproduction: random forest

One row is one task. MAE per fold, fold 0 to 4, published then rerun.

| Task | Published per fold | Rerun per fold | Published mean | Rerun mean |
|---|---|---|---|---|
| jdft2d | 42.7473 45.7510 66.2421 44.0340 51.4457 | 43.5556 45.8665 65.7865 44.1896 51.8824 | 50.0440 | 50.2561 |
| phonons | 82.3863 72.8871 59.2712 58.6036 64.9149 | 82.9955 72.5171 59.0194 58.6847 65.3558 | 67.6126 | 67.7145 |
| expt_gap | 0.4360 0.4387 0.4812 0.4345 0.4400 | 0.4351 0.4389 0.4814 0.4333 0.4386 | 0.4461 | 0.4455 |
| dielectric | 0.3042 0.4079 0.5220 0.3879 0.4760 | 0.3015 0.4181 0.5240 0.3943 0.5043 | 0.4196 | 0.4284 |
| log_gvrh | 0.1046 0.1024 0.1025 0.1037 0.1067 | 0.1051 0.1019 0.1024 0.1039 0.1065 | 0.1040 | 0.1039 |
| perovskites | 0.2357 0.2367 0.2365 0.2395 0.2291 | 0.2361 0.2367 0.2368 0.2401 0.2293 | 0.2355 | 0.2358 |

Run on 2026-10-26 with scikit-learn 1.6.1 and matminer 0.9, seed 42. The cleaning is
reimplemented from the four settings above, not imported from automatminer, whose
pins do not install beside current matminer.

### Reproduction: MODNet

Only a check that the file runs so far: jdft2d, fold 0, in MODNet's `--fast` mode
(fewer settings tried, fewer epochs, no inner CV). That is not the published protocol,
so the score is not comparable.

| Task | Fold | Mode | Published MAE | Rerun MAE | MODNet features | Seconds |
|---|---|---|---|---|---|---|
| jdft2d | 0 | `--fast` | 25.5515 | 34.6863 | 1,698 | 5,281 to compute features, 1,797 to select and train |

Run on 2026-10-26 on the laptop. Features are cached after the first run, under
`/Volumes/seagate/benchmarks_protocol/matbench/modnet_features/`. The full protocol
(`nested=5`, the full list of settings) has not been run on any task.

### Feature sets

The representations already built by `src/kirby/representations/materials.py`, cached
at `/Volumes/seagate/matbench/reps/<task>/<rep>.npy`. One row is one representation;
each cell is its width in features. "missing" means not built yet.

| Representation | jdft2d | phonons | expt_gap | dielectric | log_gvrh | perovskites |
|---|---|---|---|---|---|---|
| magpie | 132 | 132 | 132 | 132 | 132 | 132 |
| stoichiometry | 6 | 6 | 6 | 6 | 6 | 6 |
| valence_orbital | 8 | 8 | 8 | 8 | 8 | 8 |
| ion_property | 3 | 3 | missing | missing | missing | 3 |
| sine_coulomb | 35 | 48 | — | 288 | 160 | 5 |
| soap_avg | 13,860 | 11,520 | — | 14,400 | 15,120 | 10,080 |
| site_stats_fingerprint | 122 | 122 | — | 122 | missing | missing |
| density_features | 3 | 3 | — | 3 | 3 | 3 |
| rdf | 200 | 200 | — | 200 | 200 | missing |
| xrd | 128 | 128 | — | 128 | 128 | 128 |
| global_symmetry | 4 | 4 | — | 4 | 4 | 4 |
| chgnet | 64 | 64 | — | 64 | 64 | 64 |
| mace_mp | 256 | 256 | — | missing | missing | missing |

"—": expt_gap has no structures. 9 caches are missing.

Two differences between these caches and the published random forest's features:
- The caches set missing values to 0. The published cleaning fills them with means.
  The benchmark file recomputes magpie and the sine Coulomb matrix with missing values
  kept, so it can clean them the published way.
- The cached sine Coulomb matrix is fitted on all compounds; the published one on the
  training compounds of each fold. Its width is the most atoms in any one structure,
  so the two can differ: dielectric fold 0 has 160 features fitted on training
  compounds, against 288 in the cache.

MEGNet is left out. The reasons are in the docstring of `materials.py`.

### Benchmark files

`tests/experiments/benchmarks/b03_matbench_rf_published.py`

```
/Volumes/seagate/matbench/venv/bin/python tests/experiments/benchmarks/b03_matbench_rf_published.py
```

Writes `results/b03_matbench_rf.json` after each task: published and rerun MAE per
fold, features and training compounds dropped by the cleaning. `--tasks` and `--folds`
run a subset. `published_rf(X_tr, y_tr, X_te)` is the published protocol on any feature
matrix, so it can score a hybrid's features.

`tests/experiments/benchmarks/b04_matbench_modnet_published.py`

```
/Volumes/seagate/matbench/env_modnet_0112/bin/python tests/experiments/benchmarks/b04_matbench_modnet_published.py
```

Writes `results/b04_matbench_modnet.json` after each fold. `--fast` is MODNet's own
debug mode: fewer settings tried, fewer epochs, no inner CV. `modnet_published_fold(X_tr,
y_tr, X_te)` is the published protocol on any feature matrix. On features that are not
MODNet's own it computes the redundancy from the fold, because MODNet's precomputed
file only covers its own feature names.


### Screen (dataset screening ledger, row `matbench`)

Script `tests/experiments/benchmarks/b03_matbench_views.py`; caches built by
`tests/tools/matbench_build_reps.py --worker TASK REP`. One unit per task x view x model,
each scoring the 5 Matbench test folds; summary at `<run>/matbench/b03_views/summary.md`
(mean MAE per view and model beside the published means, Spearman correlation of
absolute errors between views, linear CKA between views).

- Views: the 13 representations in the Feature sets table above, per task; expt_gap
  has the four composition ones. 69 task x view pairs.
- Models: `rf`, the leaderboard random forest (`published_rf`: 500 trees, the
  automatminer cleaning, seed 42). `modnet` (the leaderboard MODNet,
  `modnet_published_fold`) is planned but not in the row: it needs its own python 3.8
  environment, which is not on the server. It is added with `ledger.py extend
  --datasets matbench --models matbench=modnet`.
- Units: 75 build (60 caches exist and are marked done at init; 9 to build, plus one
  targets file per task), 69 evaluate, 1 summary.
- Laptop check, 2026-11-01: magpie with `rf` on jdft2d gave 50.0047 mean MAE against
  the published 50.0440 (the published random forest uses magpie plus the sine Coulomb
  matrix); density_features gave 59.1080. Both are a check that the unit runs, not
  results.

---

## 4. ProteinGym v1.3 (deep mutational scanning, substitutions)

### Source

| | |
|---|---|
| Data | `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/`, four archives: assay scores, cross-validation folds, zero-shot scores, AF2 structures |
| Downloaded | 2026-10-23 to `/Volumes/seagate/proteingym_raw/` |
| Published scores | `github.com/OATML-Markslab/ProteinGym`, `benchmarks/DMS_supervised/substitutions/Spearman/`, downloaded 2026-10-28 to `/Volumes/seagate/benchmarks_protocol/proteingym/` |
| Baseline code | ProteinNPT, `github.com/OATML-Markslab/ProteinNPT`, cloned to `/Volumes/seagate/ProteinNPT/` |
| Paper | Notin, Kollasch, Ritter, van Niekerk, Paul, Spinner, Rollins, Shaw, Orenbuch, Weitzman, Frazer, Dias, Franceschi, Gal, Marks, "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design", NeurIPS 2023 |

md5 of the four published Spearman files:

| File | md5 |
|---|---|
| DMS_substitutions_Spearman_DMS_level.csv | `d7baa7ce4bdba9663627b82b1757c83d` |
| DMS_substitutions_Spearman_DMS_level_fold_modulo_5.csv | `081af56c2397d4c370af50bbc0b4c102` |
| DMS_substitutions_Spearman_DMS_level_fold_contiguous_5.csv | `9294d336a6c12232cae572ce1c89cd7b` |
| DMS_substitutions_Spearman_DMS_level_fold_random_5.csv | `f3134f2b1cad5cf490ec87d45ab4155c` |

### What is in it

217 assays, each a separate supervised task. One row is one protein variant: a full-length
copy of one protein with one residue substituted, and a measured fitness score. The
reference file `DMS_substitutions_reference.csv` holds 217 rows, the fold archive 217
files, and each published Spearman file 217 rows, all keyed by `DMS_id`.

| | min | median | max |
|---|---|---|---|
| Residues per protein | 37 | 245 | 3,423 |
| Single mutants per assay | 33 | 1,340 | 16,897 |

Each assay ships with three fold assignments: `fold_random_5`, `fold_modulo_5`,
`fold_contiguous_5`. Modulo and contiguous assign positions to folds, so a held-out
position is never mutated in training. Random shares positions between training and
held-out rows. Only modulo and contiguous are used here.

The reference file also carries each assay's `taxon`, `coarse_selection_type`, `seq_len`
and `DMS_number_single_mutants`, which is what the assay choice below is made from.

### Preprocessing

None on the assay files. Each is used as shipped, through
`src/kirby/datasets/proteingym.py::load_proteingym_dataset`, which reads the fold files
because they carry the scores and the three fold assignments in one place.

Per-position encodings do not work under modulo or contiguous folds. Measured on
`ESTA_BACSU_Nutschel_2020`, 2026-10-23: 0 held-out positions are also mutated in
training under either scheme, so one-hot and blosum collapse to a single identical
pattern across the 417 to 470 held-out rows, and Spearman comes back nan. Their own
published One-Hot Encodings scores agree: on the 13 assays chosen below it runs from
-0.076 to 0.179 under modulo.

### Published protocol: the supervised baselines

Read from ProteinNPT's configs (`proteinnpt/baselines/model_configs/`) and
`proteinnpt/baselines/model.py` on 2026-10-28. The three embedding baselines take the
per-residue embeddings of the mutated sequence, shape (variants, residues, features):

1. layer norm, then dropout 0.1
2. `Conv1d(features, features, kernel_size, padding='same')` over the residue axis,
   kernel 7 for ESM-1v and Tranception, 9 for MSA Transformer, then dropout and ReLU
3. mean over residues
4. `Linear(features, 1)`
5. plus the assay's zero-shot score times a learned scalar, `Linear(1, 1, bias=False)`
   initialised to 1.0, added outside the network

Training: 10,000 steps, 64 variants per batch, 100 warmup steps, weight decay 5e-2,
seed 2023.

| Baseline | Embedding model | Features | Kernel | Zero-shot covariate |
|---|---|---|---|---|
| ESM-1v Embeddings | `esm1v_t33_650M_UR90S_1`, layer 33 | 1,280 | 7 | yes |
| MSA Transformer Embeddings | `esm_msa1b_t12_100M_UR50S`, layer 12, 384 alignment rows | 768 | 9 | yes |
| Tranception Embeddings | Tranception Large, last layer | 1,280 | 7 | yes |
| One-Hot Encodings | none | (residues + 2) × 25 | — | no, `augmentation = None`, linear head |

Two consequences for any comparison:

- The convolution runs over residues before the mean, so the published number cannot be
  reproduced from mean-pooled features. Mean-pooling first and fitting a tree or an SVM
  is a different protocol, not a looser version of theirs.
- Only the One-Hot Encodings column is a single representation. The other three each get
  a second input the representation does not contain. Their ProteinNPT, Kermut and the
  five "+ One-Hot Encodings" columns are not single representations either.

### Published scores

One row is one assay. Spearman under `fold_modulo_5`, one number per assay per method.
These are the 13 assays chosen below.

| Assay | ESM-1v | MSA Transformer | Tranception | One-Hot | ProteinNPT |
|---|---|---|---|---|---|
| SDA_BACSU_Tsuboyama_2023_1PV0 | 0.726 | 0.776 | 0.652 | -0.068 | 0.784 |
| AMFR_HUMAN_Tsuboyama_2023_4G3O | 0.732 | 0.789 | 0.632 | -0.031 | 0.829 |
| SR43C_ARATH_Tsuboyama_2023_2N88 | 0.685 | 0.732 | 0.637 | 0.093 | 0.751 |
| POLG_PESV_Tsuboyama_2023_2MXD | 0.484 | 0.661 | 0.540 | 0.037 | 0.678 |
| ENVZ_ECOLI_Ghose_2023 | 0.170 | 0.285 | 0.127 | -0.076 | 0.230 |
| IF1_ECOLI_Kelsic_2016 | 0.556 | 0.474 | 0.583 | 0.141 | 0.460 |
| TAT_HV1BR_Fernandes_2016 | 0.493 | 0.491 | 0.438 | -0.002 | 0.436 |
| RL40A_YEAST_Roscoe_2013 | 0.670 | 0.736 | 0.643 | 0.039 | 0.724 |
| SUMO1_HUMAN_Weile_2017 | 0.529 | 0.550 | 0.474 | 0.038 | 0.495 |
| RL40A_YEAST_Roscoe_2014 | 0.397 | 0.500 | 0.427 | 0.179 | 0.436 |
| KCNE1_HUMAN_Muhammad_2023_function | 0.680 | 0.701 | 0.679 | 0.032 | 0.680 |
| KCNE1_HUMAN_Muhammad_2023_expression | 0.560 | 0.493 | 0.564 | 0.060 | 0.569 |
| SPG1_STRSG_Olson_2014 | 0.617 | 0.320 | 0.608 | -0.042 | 0.479 |

The same files hold `fold_contiguous_5` and `fold_random_5`, and five further methods per
assay.

### Which assays

Not all 217. The rule, applied to the reference file:

1. at least 800 single mutants, so each of the five folds holds at least 160 held-out rows
2. estimated ESM-1v cost at most 3 hours on the laptop
3. of what survives, the cheapest assay in each combination of `coarse_selection_type`
   and `taxon`

92 assays pass the first two conditions; they cover 13 combinations of selection type and
taxon, so the rule gives the 13 assays in the table above. 10.9 hours of ESM-1v in total.

The cost is an estimate, not a measurement: ESM-1v ran at 4.08 s per variant on 212
residues on this laptop, and the estimate scales that linearly with residues. Only
`ESTA_BACSU_Nutschel_2020` has been measured.

### Reproduction

Not done. What exists is one assay outside the 13 above, under a protocol that differs
from theirs.

`ESTA_BACSU_Nutschel_2020`, ESM-1v embeddings, layer 33, mean over tokens, 1,280
features, 2,172 variants, 4.08 s per variant, 8,870 s in total. Scored on the five
`fold_modulo_5` folds, held-out sizes 470, 435, 429, 421, 417. Spearman:

| Model | Features | Mean over the five folds | Per fold |
|---|---|---|---|
| LightGBM | 1,280 | 0.3318 | 0.4112 0.2029 0.3915 0.2851 0.3684 |
| SVM | 1,280 | 0.3895 | 0.4836 0.2902 0.4730 0.3134 0.3873 |
| LightGBM | 1,281, with the paired zero-shot score | 0.4018 | 0.4604 0.3384 0.4606 0.3529 0.3966 |
| SVM | 1,281, with the paired zero-shot score | 0.3956 | 0.4892 0.2989 0.4765 0.3188 0.3945 |

The SVM rows repeat exactly when `b08` is re-run. The LightGBM rows move in the fourth
decimal: an earlier run of the same matrix gave 0.3326 and 0.4014.

Published for this assay under `fold_modulo_5`: 0.408.

Both deviate from the published protocol in the same way: the features are mean-pooled
before the model, so there is no convolution over residues, and the model is a tree or an
SVM rather than a trained linear head. The last two rows add the zero-shot score as a
feature, which is not their mechanism either: theirs adds it outside the network with a
learned scalar starting at 1.0, which a tree cannot express.

Run on 2026-10-26 and 2026-10-28, scikit-learn 1.6.1, transformers via `EsmModel`.

### What it costs to embed

Measured on this laptop, Intel i5-8257U, no usable GPU, MPS unavailable:

| Model | Seconds per variant | Hours for 2,172 variants of 212 residues |
|---|---|---|
| ESM-1v | 4.08 | 2.5 |
| MSA Transformer | 625 | 377 |
| Tranception | not measured | not measured |

MSA Transformer is expensive because each variant is its own 385-row alignment batch:
384 sampled alignment rows plus the variant. Their own precompute script runs
`batch_size=1` and does not pass `--fast_MSA_mode`, so that is their mechanism, not a
setting. Row attention makes the whole batch depend on the variant, so the 384 identical
rows cannot be computed once and reused.

### Feature sets

Any representation used here must not index by position, for the reason in Preprocessing.
What exists in `src/kirby/representations/`:

| Representation | Features | Where |
|---|---|---|
| `aac` | 20 | `protein.py::create_protein_aac` |
| protein descriptors | not counted yet | `protein.py::create_protein_descriptors` |
| `mutation_features` | ~48 | `mutation.py::create_mutation_features` |
| ESM-2 embeddings | per model | `protein.py::create_esm2_protein_embeddings` |
| ESM-1v embeddings | 1,280 | built by `pgym_esm1v_embed.py`, one assay only |
| MSA Transformer embeddings | 768 | script written, not run |
| Tranception embeddings | 1,280 | script written, blocked |

The 95 per-variant zero-shot scores ProteinGym ships are each another model's prediction
of the fitness being predicted, so using them as a representation is stacking rather than
describing the variant. On `ESTA_BACSU_Nutschel_2020` all 95 as one representation scored
0.5958 with LightGBM and 0.6330 with SVM under modulo, above every embedding set here.

### Benchmark files

Four files. The first three compute a representation; the fourth scores one. None of
them reproduces a published number, because of the convolution over residues.

`tests/experiments/benchmarks/b05_proteingym_esm1v_embeddings.py`

```
python tests/experiments/benchmarks/b05_proteingym_esm1v_embeddings.py \
    ESTA_BACSU_Nutschel_2020 --batch 16 --checkpoint-every 160
```

Runs, checkpoints every 160 variants, resumes from the same command. 4.08 s per variant.

`tests/experiments/benchmarks/b06_proteingym_msat_embeddings.py`

```
python tests/experiments/benchmarks/b06_proteingym_msat_embeddings.py \
    ESTA_BACSU_Nutschel_2020 --check-alignment-only
```

`--check-alignment-only` prints the surviving-sequence count against their shipped
weights file and stops, computing nothing. Without it, the script embeds, at 625 s per
variant. Run past 2 variants only on a GPU.

`tests/experiments/benchmarks/b07_proteingym_tranception_embeddings.py`

Written and not run. Loading the checkpoint executes ProteinNPT's vendored Tranception
code, which the auto-mode sandbox refuses without a permission rule for this script.

`tests/experiments/benchmarks/b08_proteingym_score_representation.py`

```
python tests/experiments/benchmarks/b08_proteingym_score_representation.py \
    ESTA_BACSU_Nutschel_2020 esm1v_mean_l33 --with-zero-shot
```

Scores a saved matrix on the assay's own five folds, writes
`results/b08_<assay>__<tag>__<scheme>.json`, and prints their published number for the
matching method beside it with the protocol difference stated. Re-orders rows to the
loader's order rather than assuming they match, and refuses an incomplete matrix.
`fold_scores(X, y, g, model_name)` is the scoring loop on any feature matrix, so it can
score a hybrid's features.

All checkpoints and outputs are on seagate. The first versions of these scripts were
written to the session scratchpad under `/private/tmp`, which was cleared overnight on
2026-10-29 and took them with it.

---
## 5. Kepler DR24 TCEs (AstroNet and Exonet: planet or not, from a light curve)

### Source

| | |
|---|---|
| Data | The pre-computed training set linked from the AstroNet README: Google Drive folder `1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE`, 10 TFRecord files dated 2019-10-24 (`train-0000[0-7]-of-00008`, `val-00000-of-00001`, `test-00000-of-00001`, 14.5 MB each) |
| Downloaded | 2026-10-29 with gdown 5.2.0 to `/Volumes/seagate/benchmarks_protocol/kepler_astronet/tfrecord/`. md5: test `5343c166a82999946d6d010fcbd164f8`, val `db2211faba39d2fb3c31e8f10084f002`, train-00000 `cac0977ab6ab30b346b7453c69c9d6b7`, 00001 `7d892b8efd3f031de73fa7da530e54da`, 00002 `d2bde24557d68a7ae1f8ea7968922e1d`, 00003 `1799a13a34427a88ba50d19f0ba66731`, 00004 `ae5cfe5aafad10ce72fb72b642a15853`, 00005 `7de67216e87d1733c5c6c1e30beed1a0`, 00006 `3a0fbdd871aaf6bda9e9f78fc7db3d1f`, 00007 `d68a0baa2c531ad4c7d4ec0f201a1029` |
| TCE table | NASA Exoplanet Archive table `q1_q17_dr24_tce`, the 7 columns the README names, via TAP: `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+rowid,kepid,tce_plnt_num,tce_period,tce_time0bk,tce_duration,av_training_set+from+q1_q17_dr24_tce&format=csv`. Downloaded 2026-10-29 to `.../kepler_astronet/dr24_tce.csv`, 20,367 rows, md5 `39885f5a6dabce727377c194c279ca00`. Only used to check the split |
| Exonet inputs | Google Drive file `1N6bA2rahvV5kcOGmJnTA5gl_Thno7hTh` linked from the Exonet README: `exonet_inputs.zip`, 234 MB, dated 2021-03-10, md5 `9c317a86cd5ac4b7c3f0dceee6d9b3aa`. Unzipped and untarred under `.../kepler_astronet/exonet/unzipped/exonet_inputs/` (nine archives: `{train,val,test}_{lightcurves,centroids,info}`) |
| Code | AstroNet: https://github.com/google-research/exoplanet-ml, directory `exoplanet-ml/astronet` (Apache-2.0, repository archived 2023, TensorFlow 1 + Bazel). Exonet: https://gitlab.com/frontierdevelopmentlab/exoplanets/exonet-pytorch, one file `exonet.py` (PyTorch, no licence stated, last change 2021-05-14) |
| Papers | Shallue & Vanderburg, "Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90", AJ 155:94, 2018 (arXiv 1712.05044). Ansdell et al., "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning", ApJL 869:L7, 2018 (arXiv 1810.13434) |

### What is in it

- One row is one TCE (threshold crossing event): a periodic dip the Kepler pipeline found
  in one star's brightness. The label `av_training_set` from the DR24 table is PC (planet
  candidate), AFP (astrophysical false positive) or NTP (non-transiting phenomenon); UNK
  rows (4,630) are dropped. Binary target as in the code: PC = 1, AFP and NTP = 0.
- 15,737 labelled TCEs: PC 3,600, AFP 9,596, NTP 2,541. The paper, the shipped files and
  the archive table today all give these counts.
- Per TCE the shipped files carry `global_view` (2,001 bins: the whole orbital period,
  phase folded) and `local_view` (201 bins: the transit and 4 transit durations either
  side), each normalised to median 0 and minimum -1 (checked: every row's minimum is -1.0
  and median 0.0), plus 7 catalogue values the published models do not use (`tce_period`,
  `tce_duration`, `tce_depth`, `tce_impact`, `tce_max_mult_ev`, `tce_model_snr`,
  `tce_prad`).
- The Exonet bundle carries, for 15,084 of the 15,737 TCEs, the same two flux views plus
  a global and a local view of the star's centroid motion (2,001 and 201 bins, file suffix
  `_cen_w`, "whitened") and 10 standardised stellar values per TCE. The paper names 6
  (effective temperature, surface gravity, metallicity, radius, mass, density, from the
  Kepler DR25 stellar table) and says Gaia proper motions and parallax were tried and not
  used; the bundle does not say which of its 10 columns is which.

The split is fixed by the published code (`astronet/data/generate_input_records.py`): the
labelled rows in archive order, shuffled once with numpy seed 123, first 80% train, next
10% validation, last 10% test. Replayed on the TAP download of 2026-10-29, all three
splits match the shipped files exactly (15,737 of 15,737 TCE ids in the same split).

| Split | TCEs | PC | AFP | NTP | In the Exonet bundle |
|---|---|---|---|---|---|
| train | 12,589 | 2,885 | 7,643 | 2,061 | 11,937 |
| validation | 1,574 | 355 | 983 | 236 | 1,574 |
| test | 1,574 | 360 | 970 | 244 | 1,573 |

Every Exonet TCE is in the same split as in AstroNet and carries the same label (15,084 of
15,084). The 653 TCEs the bundle lacks are not explained in its README.

The split is random. The no-random-splits rule is for molecular data (owner, 2026-10-29);
here the published split is used as published.

### Preprocessing

None beyond what the shipped files already carry. The published pipeline (spline
detrending, phase folding, binning, normalisation; paper section 3) was run by the
authors to make them. Rebuilding from raw light curves needs about 90 GB of Kepler FITS
files and the TensorFlow 1 code. Not done.

### Published protocol (AstroNet paper, sections 4 and 5)

- Three model families, each on global, local, and global + local: linear ("zero hidden
  layers, which is equivalent to a linear logistic regression model"), fully connected,
  convolutional.
- Trained with Adam; best convolutional model: learning rate 1e-5, batch 64, 50 epochs,
  no dropout, random reflection of the views during training. Hyperparameters chosen by
  Google Vizier to maximise validation AUC. Each published number is the average of 10
  models with different random initialisations, scored once on the test set.
- Metrics: accuracy at 0.5, and AUC.

| Model | Inputs | Accuracy | AUC |
|---|---|---|---|
| linear | global | 0.869 | 0.922 |
| linear | local | 0.879 | 0.933 |
| linear | global + local | 0.917 | 0.963 |
| fully connected | global + local | 0.941 | 0.977 |
| convolutional | global + local | 0.960 | 0.988 |

Exonet (paper Table 1, ensembled, test set): accuracy / average precision, AstroNet
0.958 / 0.955, Exonet 0.975 / 0.980, AstroNet-XS 0.953 / 0.936, Exonet-XS 0.966 / 0.963.
Adam, learning rate 1e-5, batch 64, no hyperparameter search. AUC is not reported.

Not stated, so chosen in the benchmark file:
- The linear model's hyperparameters (Vizier values are not published). Used:
  scikit-learn `LogisticRegression`, C from 1e-5 to 100 (15 values) chosen on validation
  AUC, inputs as shipped.
- Exonet's epoch count and ensemble size.

Four places where the papers and the files disagree:
- The AstroNet paper says the test set has 1,523 TCEs. The code's split gives 1,574, and
  the shipped test file holds 1,574.
- The AstroNet paper gives the convolutional model 0.960 accuracy; the Exonet paper's
  retrained AstroNet gives 0.958.
- The Exonet bundle holds 15,084 TCEs, the AstroNet files 15,737; 652 of the 653 missing
  are training rows.
- The Exonet paper names 6 stellar parameters; the bundle carries 10.

### Reproduction

One row is one published linear row. Accuracy and AUC on the 1,574 test TCEs.

| Inputs | Chosen C | Published accuracy | Rerun accuracy | Published AUC | Rerun AUC |
|---|---|---|---|---|---|
| global | 0.003 | 0.869 | 0.8259 | 0.922 | 0.8935 |
| local | 0.03 | 0.879 | 0.9003 | 0.933 | 0.9336 |
| global + local | 0.01 | 0.917 | 0.9053 | 0.963 | 0.9559 |
| Exonet, all five inputs (4,414 features, 1,573 test TCEs) | 0.01 | none | 0.9434 | none | 0.9799 |

Run on 2026-10-29 with scikit-learn 1.6.1. The global-only row is 0.04 accuracy and 0.03
AUC under the published number; the other two are within 0.02. The paper's linear model
was trained with Adam under Vizier-chosen settings on reflected copies of the views, none
of which this logistic regression does. The last row has no published counterpart.

The fully connected and convolutional rows are not reproduced here: TensorFlow 1 estimators
trained for 50 epochs and averaged over 10 initialisations, and Exonet's CNN has `.cuda()`
hard-coded. Both are server runs with the published code. Not done.

### Feature sets

Six representations of one TCE. The first three cover all 15,737 TCEs; all six cover the
15,084 in the Exonet bundle.

| Feature set | Features | From | What it is |
|---|---|---|---|
| global_view | 2,001 | AstroNet files | brightness over the whole orbital period, phase folded |
| local_view | 201 | AstroNet files | brightness around the transit |
| tce_scalars | 7 | AstroNet files | catalogue values of the detection; unused by the published models |
| global_cen_w | 2,001 | Exonet bundle | centroid motion over the whole period |
| local_cen_w | 201 | Exonet bundle | centroid motion around the transit |
| stellar | 10 | Exonet bundle | standardised star parameters, columns unnamed |

The global and local views are two binnings of the same folded light curve, so they are
correlated. The published models use both as separate inputs.

### Benchmark file

`tests/experiments/benchmarks/b05_kepler_astronet_published.py`

```
python tests/experiments/benchmarks/b05_kepler_astronet_published.py \
    --tce-csv /Volumes/seagate/benchmarks_protocol/kepler_astronet/dr24_tce.csv --exonet
```

Reads the TFRecord files with its own reader (no TensorFlow), replays the split against
the TCE table, fits the three linear rows, loads the Exonet bundle and checks its ids and
labels against the AstroNet split, and fits the same logistic regression on all five Exonet
inputs (no published linear number exists for that). Writes
`results/b05_kepler_astronet.json`. About 13 minutes on the laptop, 7 of them reading the
Exonet bundle (45,000 small files on the external drive).

---


## 6. DroneRF (which drone and which flight mode, from a 2.4 GHz RF recording)

### Source

| | |
|---|---|
| Data | Mendeley Data v1, doi 10.17632/f4c2b4n755.1, published 2019-03-02, CC BY 4.0. 23 RAR archives, 4.03 GB, holding 454 CSV files. Downloaded 2026-10-30 to `/Volumes/seagate/benchmarks_protocol/dronerf/rar/` (the Mendeley file list is at `public-api/datasets/f4c2b4n755/folders/1` and `.../files?folder_id=<id>&version=1`; the page itself lists nothing without JavaScript), extracted to `dronerf/raw/` |
| Code | https://github.com/Al-Sad/DroneRF, Apache-2.0. `Matlab/Main_1_Data_aggregation.m` (chunking and spectra), `Matlab/Main_2_Data_labeling.m` (normalisation, labels, writes `RF_Data.csv`), `Python/Classification.py` (the published classifier and split). LabVIEW recording programs are also in the repository |
| Papers | Al-Sa'd, Al-Ali, Mohamed, Khattab, Erbad, "RF-based drone detection and identification using deep learning approaches: an initiative towards a large open source drone database", Future Generation Computer Systems 100:86–97, 2019, doi 10.1016/j.future.2019.05.007 (the numbers). Allahham, Al-Sa'd, Al-Ali, Mohamed, Khattab, Erbad, "DroneRF dataset: A dataset of drones for RF-based detection, classification, and identification", Data in Brief 26:104313, 2019, doi 10.1016/j.dib.2019.104313 (the data) |
| Extra representations | Swinney and Woods, "Unmanned Aerial Vehicle Operating Mode Classification Using Deep Residual Learning Feature Extraction", Aerospace 8(3):79, 2021, doi 10.3390/aerospace8030079, CC BY. Abstract read 2026-10-29; the full text refuses automated download, so its DroneRF-specific settings are unverified |

### What is in it

- One recording is one `{BUI}{L|H}_{n}.csv` pair: the lower and the upper 40 MHz half of
  the 2.4 GHz band, captured at the same time by two NI USRP-2943R receivers. The
  published code reads each file as one real-valued column (`csvread`) and keeps only
  the upper half of the FFT, so the samples are treated as real, not I/Q.
- BUI is a 5-character code for the class: `00000` background (no drone), `10000`
  to `10011` Parrot Bebop in flight modes 1 to 4, `10100` to `10111` Parrot AR in modes
  1 to 4, `11000` DJI Phantom 3 in mode 1. Modes: 1 on and connected, 2 hovering, 3 flying
  without video, 4 flying with video.
- Recordings per class, from the file loop in `Main_1_Data_aggregation.m`: background 41,
  each Bebop mode 21, AR modes 1 to 3 21 each, AR mode 4 18, Phantom 21. Total 227, which
  is the Data in Brief count (Bebop 84, AR 81, Phantom 21, background 41).
- Samples per file: 10 million, checked on `11000H_0.csv` 2026-10-30 (the Data in Brief
  text's "1 million" is wrong). Values are integers between -776 and 776 in that file,
  written as `0.000000`. 100 chunks per file, 22,700 in all.
- Shulman (arXiv:2607.01025, 2026; code github.com/shulm/spectrahawk, MIT) says the files
  come from 4 independent flights each for AR and Bebop and 1 each for Phantom and
  background, and that grouping by file still leaks. Its grouping is by the BUI code,
  which is the drone-plus-mode code, so "4 flights" may only be the 4 mode codes. The
  CSVs carry no timestamps, so this cannot be checked from the data. Recorded, unverified.

### Preprocessing (the published pipeline, `Main_1` and `Main_2`)

1. Each file is cut into chunks of L = 100,000 samples (100 per file at 10 million).
2. Per chunk and per band: subtract the chunk mean, 2,048-point FFT, magnitude, keep the
   upper 1,024 bins. MATLAB `fft(x, 2048)` truncates the 100,000-sample chunk to its first
   2,048 samples, so 97,952 of every chunk's samples are not used by the published pipeline. The high band is scaled so its first 10 bins continue the low band's last
   10 bins. The two are stacked (2,048 bins) and squared.
3. In `Main_2`, every class file is divided by its own maximum over all its chunks, then
   all classes are concatenated into `RF_Data.csv` with three label rows.
4. `Classification.py` uses rows 0 to 2,046 as features (2,047 of the 2,048 bins) and
   row 2,050 as the 10-class label.

Step 3 scales each class by a number computed from all chunks of that class, test chunks
included. That is a fact about the published pipeline, recorded here, not fixed.

### Published protocol (FGCS paper; `Classification.py`)

- Three tasks on the same chunks: drone present or not (2 classes), background / Bebop /
  AR / Phantom (4 classes), background plus 9 drone-mode pairs (10 classes).
- Classifier: three hidden layers of 128 ReLU units, sigmoid output, mean squared error
  loss, Adam, 200 epochs, batch 10, one-hot targets. Decision by arg max.
- Split: `StratifiedKFold(n_splits=10, shuffle=True, random_state=1)` over the 22,700
  chunks. Chunks of one recording fall in both the training and the test fold. Every
  test chunk has about 99 same-recording chunks in training.
- Metric: accuracy, averaged over the 10 folds.

| Task | Classes | Published accuracy |
|---|---|---|
| detection | 2 | 99.7 |
| drone type | 4 | 84.5 |
| drone and mode | 10 | 46.8 |

As committed, `Classification.py` runs only the 10-class label; the 2- and 4-class runs
need the label row changed (2,048 and 2,049).

Swinney and Woods 2021 (abstract): PSD, spectrogram, amplitude histogram and I/Q
constellation, each drawn as an image and passed through ImageNet ResNet50; logistic
regression on the features; 5-fold cross-validation; "over 91%" on the 10 classes with
the PSD image. Whether their folds are by chunk or by recording is unverified.

### Two splits (owner, 2026-10-29)

- **Published split, comparison only.** The seeded 10-fold over chunks above. Used once,
  to put the hybrid search next to the published 99.7 / 84.5 / 46.8 under the same
  conditions. Both numbers carry the same recording leak.
- **Grouped split, for everything else.** 10-fold grouped by recording (227 groups, the
  `{BUI}_{n}` file pair), stratified on the 10-class label (`StratifiedGroupKFold`,
  shuffle, seed 1). Phantom and AR mode 4 have 21 and 18 recordings, so about 2 per test
  fold. No published number exists under this split.

### Reproduction

Not run yet (data downloaded 2026-10-30). The benchmark file cuts the chunks and
computes the 2,048-bin spectra as in `Main_1`, applies the per-class scaling as in
`Main_2`, fits the published network under the published split for all three tasks,
and reports the 10-fold mean next to 99.7 / 84.5 / 46.8. Then the same under the
grouped split.

### Object, models and candidate views (owner, 2026-10-30)

- Object: the 100,000-sample chunk, the object of the published split. Every view is
  computed on it. Kılıç 2022 (whole 0.25 s file) and Huynh-The 2022 (10,000-sample
  frame) use other objects, so their numbers have no reproduction here.
- Almasri 2021 deletes chunks whose Welch PSD falls below a threshold before training
  (526 to 4,100 chunks per class remain). Not done: it changes the row set, so nothing
  would be comparable to the published 46.8.
- Models: every vector model any DroneRF paper used. The same six on every view.

| Model | From | Settings |
|---|---|---|
| dense | Al-Sa'd 2019 code | 3 x 128 ReLU, sigmoid, MSE, Adam, 200 epochs, batch 10 (b06) |
| xgboost | Medaiyese 2020 (arXiv:2003.02656) | not stated; xgboost defaults |
| svm | Kılıç 2022, doi 10.1016/j.jestch.2021.06.008 | Gaussian kernel, C and gamma over 2^-15 to 2^15; step not stated, 2^2 used; inner 3-fold |
| logreg | Swinney and Woods 2021 | L2, LBFGS |
| rf | Shulman 2026 code | 300 trees, class_weight balanced |
| knn | Nemer 2021, doi 10.3390/s21061947 | k not stated; k = 5 |

- Views: all candidates below are computed and scored under all six models, both
  splits, three tasks, with out-of-fold predictions kept; the owner then picks up to
  ten by measured accuracy and by correlation between views. "Code" means ported
  from the named repository; "paper" means written from the paper's text.

| View | Features | From | Code or paper | What it is |
|---|---|---|---|---|
| spectrum | 2,047 | Al-Sa'd 2019 | code (Apache-2) | squared magnitude spectrum, per-class scaled, as `Classification.py` reads it |
| spectrum_low | 1,024 | Medaiyese 2020 | the low half of `spectrum` | Medaiyese reports the low band alone at 99.96 / 90.73 / 70.09 with XGBoost |
| spectrum_high | 1,023 | | the high half of `spectrum` | |
| welch_psd | 2,052 | Almasri 2021, doi 10.5220/0010530302080214 | paper (free PDF, read 2026-10-30) | per band Welch PSD, Hamming, 2,050-sample segments, 50 % overlap, one-sided (1,026 bins), both bands, min-max per chunk. The paper's input has 2,050 values; the two-bin gap is our reading of its "2050" |
| spectral_scalars | 16 | Shulman 2026 | code (MIT) | per band: total power, centroid, flatness, 99 % bandwidth, four quarter-band powers (the paper names four; the code computes eight) |
| stats | 28 | no DroneRF paper | the measure list of Anguita et al., ESANN 2013 | per band: mean, std, median absolute deviation, max, min, energy, interquartile range, histogram entropy, four AR coefficients, skewness, kurtosis |
| psd_resnet50 | 2,048 | Swinney and Woods 2021 | paper (settings from the survey; full text refuses download) | Welch PSD (1,024-point FFT, 120 overlap, Hann) drawn as a line plot, 224 x 224, ImageNet ResNet50, pooled output. The paper uses the 100,352-value flattened map; pooled or flattened is still the owner's call |
| spectrogram_resnet50 | 2,048 | Swinney and Woods 2021 | paper | STFT spectrogram (1,024-point FFT) as an image, both bands |
| histogram_resnet50 | 2,048 | Swinney and Woods 2021 | paper | 500-bin amplitude histogram as an image |
| gaf_resnet50 | 2,048 | Fu and He 2024, doi 10.3390/drones8090511 | paper | STFT 1,024 Hann, 128 overlap, averaged to 513 bins, Gramian angular field 513 x 513, bands averaged, through the same ResNet50 (they train a 2-D CNN on it) |
| rfuavnet_emb | 320 | Huynh-The 2022, doi 10.1109/ACCESS.2022.3172787 | code (MATLAB, no licence file) | RF-UAVNet 1-D CNN on 10,000-sample frames, trained inside each fold; the 320-value concatenation of its five pooled outputs, averaged over a chunk's 10 frames. The code trains 5 epochs, the paper says 90; the code is ported |
| alemadi_emb | 256 | Al-Emadi 2020, ICIoT | code (GPL-3.0) | 1-D CNN on the 2,047-bin spectrum, trained inside each fold; its 256-unit dense layer |

Left out, with the reason: Kılıç's MFCC and LFCC (no stated size, whole-file object);
Kumbasar's persistence and percentile spectrum images (MATLAB `pspectrum` outputs with no
definition in the paper); the CEUR wavelet features (no stated recipe); Allahham's 8 x
2,048 channelised spectrum (a reshaping of `spectrum` for a multi-channel CNN, not a new
vector); the raw 20,000 samples as a vector (no paper uses them with a vector model).

The two fold-trained views exist per split, task and fold, since the network sees only
that fold's training chunks. Local runs use `--debug` (fewer epochs and folds, a 4 x 4
SVM grid chosen on 3,000 chunks, one frame per chunk for RF-UAVNet) and are marked debug
in every output file; they are not results.

### Benchmark file

`tests/experiments/benchmarks/b06_dronerf_published.py`

```
python tests/experiments/benchmarks/b06_dronerf_published.py --split published
python tests/experiments/benchmarks/b06_dronerf_published.py --split grouped
```

A line-for-line port of the three published scripts to numpy and PyTorch: chunking and
spectra as `Main_1`, per-class scaling and the three labels as `Main_2`, the network and
the seeded 10-fold as `Classification.py`. `--split grouped` is the one addition. Reads
the raw CSVs from `<KIRBY_DRONERF_DIR>/raw/`, caches the spectra under `spectra/`, and
writes `results/b06_dronerf_<split>.json` with the per-fold accuracies. Each fold's log
line counts the recordings present in both its training and test chunks (227 under the
published split, 0 under the grouped one). Keras to PyTorch: Glorot init and zero biases
are kept; Adam's epsilon differs (1e-7 against 1e-8). Smoke-tested on fake files of the
published shape; not run on the data.

`tests/experiments/benchmarks/b06_dronerf_views.py`

```
python tests/experiments/benchmarks/b06_dronerf_views.py --stage extract
python tests/experiments/benchmarks/b06_dronerf_views.py --stage all --debug
```

The view and model screen of the table above. Stages: `extract` (one pass over the 454
CSVs: the six hand-computed views and the int16 frames, under `dronerf/views/`),
`images` (the four image views through ResNet50; `--views` picks one), `embed` (the two
fold-trained CNN views, built only; `--folds` picks folds), `evaluate` (every view x model
x task x split, the CNN views included, one JSON and one out-of-fold prediction file each
under `results/b06_views/<split>/<task>/`, resuming at the next fold), `summarise` (`results/b06_views/summary.md`: accuracy and
macro-F1 tables, error overlap between views per model, linear CKA between views).
Finished units are skipped on restart. Smoke-tested on real chunks 2026-10-30: every
view function, renderer, ResNet50 and both CNN ports run and give the stated sizes;
RF-UAVNet has 10,442 parameters (the paper says 11K).


### Screen (dataset screening ledger, row `dronerf`)

Script `tests/experiments/benchmarks/b06_dronerf_views.py`. One unit per view x model x
task x split, each scoring the 10 folds of that split; summary at
`<run>/dronerf/b06_views/summary.md` (accuracy and macro-F1 per view and model, error
overlap between views per model, linear CKA between views).

- Views: the 12 in the table above. The six hand-computed ones come from one `extract`
  unit; each image view is one build unit; each CNN view is one build unit per task,
  split and fold (60 each), since it is trained on that fold's training chunks.
- Models in the row: knn, rf, logreg, xgboost, dense. `svm` is planned but not in the
  row: Kılıç's grid over C and gamma with inner CV is about 460,000 fits, and no smaller
  grid has been chosen. It is added with `ledger.py extend --datasets dronerf --models
  dronerf=svm`.
- Units: 125 build, 360 evaluate, 1 summary. The server needs only `dronerf/views/`
  (9.0 GB), so `extract` is done at init.
- Non-torch models run in a subprocess that never imports torch
  (`tests/experiments/benchmarks/_xgb_worker.py`): on the laptop, xgboost and
  scikit-learn's parallel grid search died silently in a process that had loaded torch.

Laptop run, 2026-10-30, `--debug` settings: 3 folds instead of 10, dense at 20 epochs
instead of 200, the SVM grid 4 x 4 chosen on 3,000 chunks. 155 units, all marked debug.
Not results; the ledger does not count them. Files under
`/Volumes/seagate/benchmarks_protocol/results/b06_views/`.

One row is one view. A cell is 10-class accuracy / macro-F1 in %, mean over 3 folds,
published split.

| view | features | knn | rf | logreg | xgboost |
|---|---|---|---|---|---|
| spectrum | 2,047 | 45.2 / 35.5 | 69.4 / 65.6 | 41.6 / 32.3 | 72.7 / 69.9 |
| spectrum_low | 1,024 | 44.3 / 33.9 | 71.5 / 68.3 | 41.7 / 32.5 | 72.3 / 69.5 |
| spectrum_high | 1,023 | 47.0 / 41.4 | 52.0 / 46.1 | 33.6 / 22.7 | 53.6 / 49.0 |
| welch_psd | 2,052 | 66.8 / 64.1 | 76.2 / 74.3 | 71.6 / 69.3 | 83.0 / 81.6 |
| spectral_scalars | 16 | 36.5 / 31.3 | 66.3 / 63.5 | 43.8 / 38.3 | 68.0 / 65.5 |
| stats | 28 | 40.3 / 36.0 | 74.7 / 72.7 | 51.0 / 46.8 | 76.6 / 74.8 |

Same, grouped split.

| view | features | knn | rf | logreg | xgboost |
|---|---|---|---|---|---|
| spectrum | 2,047 | 45.2 / 35.2 | 68.5 / 64.0 | 39.6 / 29.1 | 72.1 / 69.2 |
| spectrum_low | 1,024 | 44.2 / 33.6 | 69.9 / 65.8 | 40.0 / 29.3 | 72.1 / 69.3 |
| spectrum_high | 1,023 | 47.0 / 41.4 | 51.2 / 45.0 | 33.2 / 20.0 | 53.5 / 48.9 |
| welch_psd | 2,052 | 63.7 / 60.7 | 73.5 / 71.3 | 70.7 / 68.3 | 80.9 / 79.3 |
| spectral_scalars | 16 | 35.9 / 30.6 | 63.8 / 60.8 | 43.0 / 37.5 | 65.4 / 62.6 |
| stats | 28 | 39.3 / 34.8 | 72.5 / 70.4 | 49.3 / 44.8 | 74.1 / 72.2 |

Also run on the spectrum only, 10-class: dense 39.9 (published split) and 39.8
(grouped) at 20 epochs; svm 66.0 (published) with the small grid.

Linear CKA between the six views' standardised matrices, 3,000 chunks (up to 300 per
10-class label). 1 = the same information up to a linear map, 0 = unrelated.

| | spectrum | spectrum_low | spectrum_high | welch_psd | spectral_scalars | stats |
|---|---|---|---|---|---|---|
| spectrum | 1.00 | 0.69 | 0.85 | 0.07 | 0.09 | 0.12 |
| spectrum_low | 0.69 | 1.00 | 0.20 | 0.09 | 0.09 | 0.13 |
| spectrum_high | 0.85 | 0.20 | 1.00 | 0.02 | 0.06 | 0.07 |
| welch_psd | 0.07 | 0.09 | 0.02 | 1.00 | 0.52 | 0.42 |
| spectral_scalars | 0.09 | 0.09 | 0.06 | 0.52 | 1.00 | 0.64 |
| stats | 0.12 | 0.13 | 0.07 | 0.42 | 0.64 | 1.00 |

The spectrum uses the first 2,048 of each chunk's 100,000 samples (MATLAB `fft(x, 2048)`
truncates); the Welch PSD and the four scalars and stats use all of them.

---


## 7. Reaction yields: the CGR view screen (buchwald_hartwig, bh_doyle_plates, suzuki)

Added 2026-11-01. Not a new benchmark: the three reaction-yield datasets KIRBy already
uses, screened for which condensed-graph-of-reaction (CGR) representations to carry.
`opig_cgr_comparison.py` waits for this screen's summary and the author's choice of its
`with_cgr` group.


### Screen (dataset screening ledger, row `reactions`)

Script `tests/experiments/benchmarks/b10_reaction_views.py`. The three datasets are the
row's tasks. Folds: KIRBy's own outer folds (`nested_cv_outer_folds`, 5 folds, seed 42,
groups from the loader: additive on the two Buchwald-Hartwig sets, ligand on suzuki).
Metric: the dataset's own (`get_domain_metric`), RMSE in yield % on all three. Summary at
`<run>/reactions/b10_views/summary.md`; on the server
`/data/stat-ecr/scat9264/screen/run_2026_11_01/reactions/b10_views/summary.md`. It holds
per dataset and model each view's 5 fold scores, mean and SD over folds; each view's
fold scores alone and with cgr_bond_changes appended; Spearman correlation of absolute
errors between views; linear CKA between views on each fold's held-out reactions; and
chemprop's own held-out score per fold, for reference.

- Views, per dataset: its pool of 10 (`complexity_helpers.DATASET_REPS[ds]['general']`,
  cgr_morgan among them) and eight CGR views:

| view | features | where it comes from | built |
|---|---|---|---|
| cgr_morgan | 2,048 | existing, `create_cgr_morgan` (in every pool) | once per dataset |
| cgr_smiles_lm | 384 | existing, `create_cgr_smiles_lm`; refuses without the CGR-pretrained checkpoint | once |
| chytorch_graphormer | 1,024 | existing, `create_chytorch_graphormer`; now raises instead of writing zero rows | once |
| cgr_atompair | 2,048 | new, hand-written port of RDKit's atom pair to the CGR | once |
| cgr_torsion | 2,048 | new, hand-written port of RDKit's topological torsion to the CGR | once |
| cgr_bond_changes | 30 | new, hand-written counts of bonds formed, broken and changed | once |
| chemprop_cgr | 300 | chemprop 1.6.1 reaction mode (Heid and Green 2022), defaults; the MPN encoder output | per outer fold |
| cgr_circus, cgr_chyline | set by the fold's fragments | DOPtools ChythonCircus(1, 3), ChythonLinear(2, 4) (Sidorov et al. 2025) | per outer fold |

- Every view but cgr_bond_changes is scored twice: alone, and with the 30
  cgr_bond_changes counts appended, on the same folds and model.
- Models: `decided_models(ds, 'general')`, rf and lightgbm on all three (2026-11-01).
- Units: 90 build (per dataset 15 once, 3 x 5 per fold), 210 evaluate (per dataset 18
  views alone and 17 with cgr_bond_changes, x 2 models), 1 summary.
- Not runnable yet: cgr_circus and cgr_chyline. doptools 1.3.9 requires chython>=1.78
  (3.x on PyPI); chytorch 1.53 runs on the installed chython 1.68. doptools was not
  installed; the 30 units fail with that message until the author decides.

Laptop check, 2026-11-01, buchwald_hartwig, normal settings, each unit time-boxed
(one thread: `OMP_NUM_THREADS=1`, `KIRBY_LGBM_THREADS=1`). These say whether a unit runs
and how long it takes, nothing about which view is better.

| unit | outcome | time |
|---|---|---|
| atom mapping (rxnmapper), once per dataset, kept in `$KIRBY_CACHE_DIR/rxn_mapped/` | done | 22 min for 3,955 reactions |
| cgr_bond_changes build | built, 30 features | 1,344 s, of which the mapping is nearly all |
| cgr_atompair build | built, 2,048 features | 106 s (mapping cached) |
| cgr_torsion build | built, 2,048 features | 57 s (mapping cached) |
| cgr_smiles_lm build | refused: no CGR-pretrained checkpoint on the laptop | 20 s |
| chytorch_graphormer build | stopped at 10 min: 200 of 3,955 reactions embedded, no error | about 100 reactions per 2.5 to 3 min after about 5 min of parsing, so about 2 h for the dataset |
| chemprop_cgr build, fold 0 | stopped in epoch 0 (2,338 training, 719 validation, 898 held-out reactions; 378,601 parameters) | one epoch about 4 min (47 batches of about 5.3 s), so about 2 h for chemprop's default 30 epochs per fold. Over the 1 h per fold the brief allowed; not shortened |
| cgr_circus, cgr_chyline builds | not run: doptools not installed (see above) | |
| evaluate cgr_atompair with cgr_bond_changes appended, rf | ran, 5 folds, result file written | about 1 s per fold |
| summarise | ran | 15 s |

On buchwald_hartwig, 27 of the 30 cgr_bond_changes features are the same for every
reaction: each reaction forms one single C-N bond and breaks one single bond; only which
carbon-halogen bond breaks (C-Cl, C-Br or C-I) differs. Reagent atoms do not appear as
broken bonds.

---


## Not used, and why

The multi-view `.mat` files loaded by `src/kirby/datasets/multiview.py` (Caltech101-7,
Caltech101-20, NUS-WIDE-Object, Reuters) have no published protocol: no fixed folds and
no code from the papers that report on them.

mfeat on OpenML (sections 2 and "For later" removed by the owner on 2026-10-30): the best
single feature set, karhunen, is at 98.55 % published accuracy, 29 of 2,000 digits wrong
against a resolution of about 5, so the headroom condition fails. The benchmark file
`tests/experiments/benchmarks/b02_mfeat_openml_published.py` and the mfeat row of
`tests/experiments/decided_grid.json` are left in place.

OPS-SAT-AD (Ruszczak et al., Scientific Data 2025, doi 10.1038/s41597-025-05035-3;
Zenodo 10.5281/zenodo.15108715) was added on 2026-10-30 and removed the same day by the
owner: no room to improve. The fixed test split has 529 segments, 113 anomalous. The
paper's best row gets 12 wrong; a MultiRocket view with XGBOD gets 5 wrong (0.991
accuracy, 0.972 MCC); the test set cannot separate views closer than about 3 segments.
The screen that showed this is `tests/experiments/benchmarks/b09_opssat_views.py`, with
its results under `/Volumes/seagate/benchmarks_protocol/results/b09_opssat_views/` and the
views under `/Volumes/seagate/benchmarks_protocol/opssat_ad/views/`. The AdaBoost,
logistic regression and XGBOD rows of the paper's Table 3 were reproduced exactly on the
18 shipped features. New condition from this (owner, 2026-10-30): a dataset must leave
room above its published best, measured in test rows, well above the test set's
resolution.

---

## For later: the hybrid runs

Nothing in this section is needed for the baselines above.

### Matbench

- The test folds are Matbench's 5. The search runs 5 times per task, once per fold, on
  that fold's training compounds; the hybrid is then scored on the held-out compounds
  with `published_rf` and `modnet_published_fold`. Same loop as mfeat, in the
  experiment file.
- The search's own folds inside the training compounds use one group per compound,
  which `load_matbench` already returns.
- The cached representations have missing values set to 0 (see Feature sets above);
  the published random forest fills them with means instead.
- Widths: 3 to 15,120 features per representation. soap_avg alone exceeds every budget.
- 9 representation caches are missing (table above).
- A 500-tree random forest (the published model, without the cleaning) on soap_avg
  took 729 s for 509 training compounds of jdft2d. perovskites has 15,142 or 15,143
  training compounds per fold.

### ProteinGym

- The folds are theirs, five per assay, and both modulo and contiguous are position-
  disjoint. The search runs once per fold on that fold's training variants; the hybrid is
  then scored on the held-out variants. Same loop as mfeat and Matbench.
- The search needs groups for its own folds inside the training variants. Position is the
  grouping their own folds use, so position is the one that keeps the inner folds
  position-disjoint too.
- Which model runs on the hybrid's features is open. Their published number comes from a
  convolution over residues and a trained linear head, which takes per-residue
  embeddings; the hybrid's features are one vector per variant. So either their head runs
  on a length-1 sequence, or the comparison names a different model and says so.
- Widths: 20 to 1,280 features per representation. A budget above about 1,400 features
  would exceed everything available except the embeddings.
- `decided_grid.json` has no row for any ProteinGym assay, and `c11` refuses a dataset
  without one.
- Two of the three embedding representations do not exist yet: MSA Transformer needs the
  GPU server at 377 hours per assay on the laptop, Tranception needs permission to run
  ProteinNPT's model code.

### Not yet in KIRBy

- Registry entries. `register_datasets` only accepts the domains `molecular, antibody,
  dta, tcga, reaction, nucleotide` (`src/kirby/registry.py:30`). Neither benchmark fits
  one. Without an entry, the metric falls back to a guess from the task type instead of
  accuracy. `accuracy` is already a metric KIRBy knows.
- A `decided_grid.json` row for the six Matbench tasks. c11 refuses any
  dataset without one.
- Matbench needs a registry domain too; none of the six allowed ones fits.
- c11 has no option for fixed outer folds (it calls `nested_cv_outer_folds`), which
  mfeat and Matbench both need.
- A Kepler loader (TFRecord reader is in `b05`), a registry domain, a `decided_grid.json`
  row, and a c11 option for the fixed published split (train on train + validation, score
  on test).
- DroneRF: the download, a chunking + spectrum routine that matches `Main_1`/`Main_2`,
  the five extra representation routines above (Welch PSD, three ResNet50 image
  embeddings, statistics), a loader, a registry domain, a `decided_grid.json` row, and
  a c11 option for grouped folds by recording.
- The hybrid experiment files.

