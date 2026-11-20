# NEW_DATASETS — standard benchmarks outside drug discovery, and the noise suite's datasets

Written 2026-10-26. Three benchmarks with a public source, a published score, a fixed
protocol, and more than one feature set measured on the same samples: Matbench v0.1,
ProteinGym and DroneRF. None of them is wired into KIRBy yet. Kepler was the fourth
until Adelaide cut it on 2026-11-04; see "Not used, and why". Sections added on
2026-11-09 cover UCR Crop, RML22, LSST, WiSig and BEANS bats.

The plan for each benchmark:
1. Reproduce the published score under the published protocol (files below).
2. Run the hybrid search on the benchmark's feature sets, on training samples only.
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
naming its views, its models and where its `summary.md` lands. Rows in that file today:
matbench (section 3), dronerf (section 6), reactions (section 7), ucr_crop, lsst, wisig and
beans_bats. The first three were written for the server run of 2026-11-01 (the server
was not checked); the other four were written on 2026-11-09.

## The route from this file into a paper

Seven steps, in order. A dataset is at exactly one of them. Nothing moves to the next
step until Adelaide says so, and each step names the one file it writes.

1. **Proposed.** A section in this file: source, what is in it, the published protocol,
   the published score, the candidate views. It meets the 11-point checklist
   (2026-10-30), including room above the published best measured in test samples.
2. **Reproduced.** The published score rerun under the published protocol. The table goes
   in this file and in `scratchpad/precompute_cuts/DECISIONS.md`, section "New datasets",
   subsection "Reproductions of the published scores".
3. **Screened.** Every candidate view scored on its own under the models the dataset's
   papers used, through the dataset screening ledger (`tests/experiments/screen/`,
   runbook `tests/slurm_scripts/runbook/SCREEN.md`). The `summary.md` lands where the
   section's "Screen" subsection says. Rows today: matbench, dronerf, reactions, ucr_crop,
   lsst, wisig, beans_bats.
4. **Decided by Adelaide.** In or out. If in, she also picks the views (at most ten), the
   two downstream models, the fold grouping, and whether the similarity experiments run
   on it. All four go in DECISIONS.md: the views and models under "New datasets", the
   model pair under "Step 1", the similarity call under "Step 10".
5. **Written into the papers.** The dataset is added to the Datasets paragraph of
   `PAPER_2_OUTLINE.md`, `PAPER_3_OUTLINE.md`, or both. **A dataset reaches an outline
   only after step 4.** An outline naming a dataset that has not been decided is a fault,
   not a plan.
6. **Wired into the code.** Six things, each a file to open:
   1. Registry entry: `DATASET_REGISTRY_DATA` in `tests/experiments/dataset_policy.py`,
      giving the task and the metric.
   2. Loader: `_shared.load_data` loads it under that exact name.
   3. Feature sets: `DATASET_REPS[<dataset>]['general']` in
      `tests/experiments/complexity/complexity_helpers.py`. At most ten, except the
      reaction datasets (Adelaide, 2026-11-02).
   4. Two downstream models: recorded in DECISIONS.md "Step 1". The code reads them only
      from `tests/experiments/decided_grid.json`, which `build_decided_grid.py` builds
      from DECISIONS.md. Nothing else in the code reads DECISIONS.md.
   5. Fold grouping. Antibodies: `DATASET_REGISTRY` in `tests/antibody_baseline_survey.py`
      maps the name to a grouping strategy, built further down the same file. Molecules:
      `_shared.load_data` assigns scaffold groups.
   6. Planned by a ledger: the name is in `PAPER2_DATASETS` or `PAPER3_DATASETS`
      (step 7). That ledger does not read `CUT_DATASETS` in
      `tests/experiments/orchestrate/plan.py`.
7. **Planned by the ledger.** Its name is in `PAPER2_DATASETS` or `PAPER3_DATASETS` in
   `tests/experiments/papers/plan.py`, and `ledger.py extend` adds its row to a run.
   The ledger holds every step but `reps` for a dataset with no row in
   `decided_grid.json` (`tests/experiments/papers/ledger.py:210`).

Gaps in this route, checked in the code on 2026-11-17:
- cdr_flexibility_h3 and cdr_flexibility_l3 have no `decided_grid.json` row of their own.
  Both reach the row of `cdr_flexibility` only through `ROW_ALIASES`
  (`tests/experiments/decided_grid.py:53-56`). The Paper 2 and 3 ledger plans both.
- None of the domains in `_VALID_DOMAINS` (`src/kirby/registry.py:30`) fits Matbench,
  ProteinGym or DroneRF.
- DroneRF goes into Paper 2 after its view screen (Adelaide, 2026-11-02). It is in
  neither outline yet, which is expected, not a fault.

Open questions for Adelaide (2026-11-03, not answered):
- The only path from DECISIONS.md into the code is `decided_grid.json`. Should the two
  downstream models stay there, or move beside the registry entry in `dataset_policy.py`?
- A dataset's paper membership is written in three places: `PAPER_2_OUTLINE.md`,
  `PAPER_3_OUTLINE.md` and `tests/experiments/papers/plan.py`. DECISIONS.md repeats it in
  prose. Which one is the record?
- c11 is in Paper 2's experiment list (`PAPER2_EXPERIMENTS`, `papers/plan.py:99`). It
  raises for a dataset with no `decided_grid.json` row. The two cdr_flexibility loops
  pass only through the alias above. Should they get rows of their own?

Where each candidate stands, 2026-11-04. One row is one dataset.

| dataset | at step | what it is waiting for |
|---|---|---|
| Matbench v0.1 (6 tasks) | 3, being screened | the screen to finish, then Adelaide's decision on the views |
| DroneRF | 3, being screened | the screen to finish, then Adelaide's decision on the views |
| reactions (3 yield tasks) | 3, CGR views being screened. buchwald_hartwig and bh_doyle_plates are already planned in Paper 2 (`PAPER2_ONLY` in `tests/experiments/papers/plan.py`); suzuki is out (Adelaide, 2026-11-01) | the screen to finish, then Adelaide's decision on which CGR views they carry |
| ProteinGym v1.3 | 2, reproduced | a view screen |
| RML22 | 1, proposed (2026-11-09) | the download, then the honest held-out score |
| UCR Crop | 1, proposed (2026-11-09); screen script and ledger row written, not run | the Seagate drive, for the data download and the reproduction |
| LSST (UEA) | 1, proposed (2026-11-09); reproduction script, screen script and ledger row written, not run on the data | the Seagate drive, for the download and the reproduction |
| BEANS bats | 1, proposed (2026-11-09); published classical scores rerun; screen script and ledger row written, not run on the server | the screen models and whether to group the split (both open), then the server run |

Matbench and ProteinGym are already named in `PAPER_2_OUTLINE.md` although neither has
reached step 4. That is the fault step 5 describes; it is recorded here rather than fixed
by deleting them, because whether they are in is Adelaide's call.

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
| qm9 | QM9 HOMO-LUMO gap, 5,000 molecules (Adelaide, 2026-11-04; was 10,000), computed labels | yes, rmse | yes | yes | yes | yes |
| qm9_mu | QM9 dipole moment (debye), same 5,000 molecules | yes, rmse | yes, qm9's | yes | yes | yes |
| qm9_alpha | QM9 polarizability (bohr^3), same 5,000 molecules | yes, rmse | yes, qm9's | yes | yes | yes |

The nine other QM9 properties (qm9_homo, qm9_lumo, qm9_r2, qm9_zpve, qm9_u0, qm9_u, qm9_h,
qm9_g, qm9_cv) were added 2026-10-30 for the dataset-to-dataset transfer only. They are in
`plan.CUT_DATASETS` since 2026-11-02, so no ledger plans them; the transfer pairs that
use them are unchanged.

| herg_ki | hERG Ki | yes, rmse | yes | yes | yes | yes |
| herg_ic50 | hERG IC50, repeat measurements per molecule | yes, rmse, 12,008 measurements (added 2026-11-01) | herg_ki's (`REP_LIST_FROM`) | yes, herg_ki's | own row, `HERG_IC50_EXPERIMENTS` | yes |
| caco2 | Caco-2 | yes, rmse | yes | yes | yes | yes |
| logd | LogD | yes, rmse | yes | yes | yes | yes |
| adme_rlm | Biogen rat liver microsomal clearance | yes, rmse | yes | yes | yes | yes |
| adme_hlm | Biogen human liver microsomal clearance | yes, rmse | yes | yes | yes, out of `CUT_DATASETS` 2026-11-01 | yes |
| adme_mdr1 | Biogen MDR1-MDCK efflux | yes, rmse | yes | yes | yes | yes |
| herg | hERG blocking, classification | yes, auc | yes | yes | yes | yes |
| chembl_qsar_d2, chembl_qsar_fxa, chembl_qsar_mpro | Dablander et al. 2023: dopamine D2 (6,333 molecules, 2,876 scaffold groups, Ki), factor Xa (3,605, 1,462, Ki), SARS-CoV-2 Mpro (1,924, 1,087, IC50); pKi or pIC50; ships matched molecular pairs | yes, rmse, one entry per target (parent `chembl_qsar` also rmse) | yes, chembl_qsar's borrowed representation list, never pruned | yes, one row per target with its own two models (2026-11-04); the other settings are chembl_qsar's (herg_ki's settings) | yes, one row per target; the parent name stays in `EXCLUDED_DATASETS` | yes, `find_best_hybrid_molecular.load_dataset`, scaffold groups |

Still to do, one line each:
- [x] herg_ic50's row runs n36, n45 and n46 (`HERG_IC50_EXPERIMENTS`,
      `tests/experiments/orchestrate/plan.py:119`). n45 replaced n22 and n46 replaced n17,
      n19, n26 and n29 (2026-11-02).
- [x] herg_ic50: registry entry added 2026-11-01, rmse, 12,008 measurements.
- [x] adme_hlm: out of `CUT_DATASETS` 2026-11-01.
- [x] chembl_qsar: one name per target, chembl_qsar_d2, chembl_qsar_fxa, chembl_qsar_mpro
      (2026-11-01). Registry entries carry chembl_qsar's task and metric; representation entries are copies of chembl_qsar's. The grid entries copy chembl_qsar's except the two downstream models: d2 dnn+svm, fxa svm+lightgbm, mpro lightgbm+rf (2026-11-04).
- [x] chembl_qsar: branch in `find_best_hybrid_molecular.load_dataset` (prefix
      `chembl_qsar_`); `_shared.load_data` assigns scaffold groups as for every molecular set.
- [x] chembl_qsar: registry metric rmse, parent and the three targets (2026-11-01).
      `opig_activity_cliff_qsar.py` reads the parent, so its searches now rank on rmse.
- [x] chembl_qsar: the three target names are in the run list. The parent name stays in
      `EXCLUDED_DATASETS`; only `opig_activity_cliff_qsar.py` uses it.
- [ ] chembl_qsar: its representation list is borrowed from six pruned datasets. It has
      never been through the pruning step.

### Antibodies for Papers 2 and 3

Chosen by Adelaide 2026-11-01. Every one runs every experiment, except antibody humanness,
which is in Paper 2 only (`PAPER2_ONLY` in `tests/experiments/papers/plan.py`), so it runs no Paper 3 experiment. Adelaide, 2026-11-01: "Leave it out of paper 3 there's no use doing noise mitigation on this". Sizes and split
groups are from the thesis dataset table; abcov_kd and the trastuzumab sets from
DATASET_STATUS.md and a load on 2026-11-01.

| Dataset | What it is | Samples | Split groups | Run list |
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
| cdr_flexibility_h3 | CDRH3 loop flexible or rigid | 216 | loop clusters at 80% identity, 203 groups | yes, from 2026-11-01 |
| cdr_flexibility_l3 | CDRL3 loop flexible or rigid | 112 | loop clusters at 80% identity, 98 groups | yes, from 2026-11-01 |
| abcov_kd | anti-coronavirus binding, labels collected from papers | 141 | identity | yes, from 2026-11-01 |
| tz_her2_binary | trastuzumab CDRH3 library, binder or not | 5,000 of 524,346 | 3,039 CDRH3-loop clusters at 80% identity, fixed 2026-11-02 | yes, from 2026-11-02 |
| tz_her2_multiclass | same library, three classes | 5,000 of 524,346 | 3,043 clusters, same rule | yes, from 2026-11-02 |

Out (Adelaide, 2026-11-01): flab_thermo_tresanco, ginkgo_pr_cho, abagym_VEGF. All three are in
`plan.CUT_DATASETS` (the first two added 2026-11-02), so no ledger plans them, and the
17 dataset-to-dataset transfer pairs that used them are commented out in
`tests/experiments/blueprint_transfer_pairs.txt`.

KEPT (Adelaide, 2026-11-01): flab_binding_adams and flab_expression_adams. Both are still
in `CUT_DATASETS` in `tests/experiments/orchestrate/plan.py`. The Paper 2 and 3 ledger does
not read that list and plans both in Paper 2 (`PAPER2_ONLY`). The 2026-10-08 cut said their per-fold searches
ran 1 to 36 hours without finishing. The server logs of run_2026_10_06 say why. Folds 0
and 3 of both datasets ran on the GPU machine htc-g011, where every search was stopped and
restarted about every four hours: four starts for fold 0 of each, and expression fold 0
printed 111 of its 840 configurations in 21 hours. Folds 1 and 2 ran on ordinary compute
nodes and kept moving: expression fold 1 reached 802 of 840 configurations in 22.1 hours,
and binding fold 1 reached 1,003 of 1,960 in the same time. Those two were not killed by
the cluster; they stopped because the datasets were cut while they ran. A search writes its
scores to the store only after every 25 configurations
(`_DB_FLUSH_INTERVAL`, `src/kirby/find_best_hybrid.py`), so a run that ends before its
first write keeps nothing — a likely reason fold 0 hardly moved, not yet verified.
LEDGER: send the two Adams datasets to ordinary compute nodes, never to htc-g011.

Still to do, one line each:
- [x] cdr_flexibility split. One antibody can give two samples (its CDRH3 and its CDRL3), and
      the features are computed from the whole heavy and light chain, so the two samples have
      the same features. 81 of 247 antibodies have two samples; grouping on the loop alone put
      the two samples of all 81 in different groups. Fixed 2026-11-01 in
      `_shared._load_data_impl` (`_join_groups_by_key`): samples with the same heavy and light
      chain share a group. Check after the fix: 0 antibodies on both sides of any fold.
- [x] cdr_flexibility: the two samples of one antibody had the same features but asked about
      different loops, and 10 of the 81 had one loop flexible and the other rigid. No
      feature said which loop a sample was about. DECIDED 2026-11-01 (Adelaide): one loop per
      dataset, which is how the authors handle it. Their repository ships
      CDRH3_test_align_Fv.csv and CDRL3_test_align_Fv.csv as two test sets, and their model
      is told which loop it is asked about by the chain and the loop's residue range (read
      from oxpig/ITsFlexible, 2026-11-01: 236 unique Fv in the CDRH3 file, 127 in the CDRL3
      file, 93 Fv sequences in both and 14 of those labelled differently in the two).
      Built: `cdr_flexibility_h3` and `cdr_flexibility_l3` in `_shared._load_data_impl`,
      `dataset_policy` (classification, AUC, antibody), `complexity_helpers.DATASET_REPS`
      (both read the representation list pruned for the parent name), `orchestrate/plan.py` (both in SMALL_FIRST)
      and `_shared.py` (both in ALL_FOLDS_DATASETS). The bare name `cdr_flexibility` stays the parent
      registry entry and representation list, is in `EXCLUDED_DATASETS` so no run plans it,
      and raises if the loader is asked for it. Within one loop type the Fv contains the
      loop, so no two samples share features.
      Loaded 2026-11-01, both names, 8 of the 10 representations (esmc and esm3 need a
      module this machine does not have):
      cdr_flexibility_h3 has 216 loops, 175 of them flexible, in 203 groups;
      cdr_flexibility_l3 has 112 loops, 62 flexible, in 98 groups. The counts are lower
      than the samples in their files because the loader keeps one sample per unique CDR3
      sequence. Five folds hold out 43 to 44 loops on h3 and 22 to 23 on l3, and no input
      is on both sides of any fold.
- [x] cdr_flexibility grouping holds out loops, not families, and did so before the loops
      were separated. 192 of h3's 203 groups hold one loop, and 89 of l3's 98 do; the
      largest holds 3 on h3 and 5 on l3. The CDR3 loops here are 5 to 26 residues on h3 and 5 to 19 on l3. At 80% identity, two
      loops fall in one group if they differ by at most one edit per five residues of the
      longer loop. That is one edit on a 5-residue loop and five on a 26-residue loop. Kept as it is (2026-11-02):
      lowering the threshold on a 9-residue loop merges loops differing at one position,
      which is not a homology family either, and grouping on the whole Fv was rejected
      2026-07-19 because antibody frameworks are about 98% alike. Say in the paper that
      the folds hold out loops rather than families.
      The labels are uneven: 175 of 216 loops are flexible on h3, against 62 of 112 on l3.
      Separately, the OPIG walk-through of 2026-11-01 keeps cdr_flexibility out of the
      THESIS on data grounds — 41 rigid loops of 216, and MCC 0.0000 on 11 of 25 rows —
      while both paper outlines list it.
- [x] tz_her2 split, fixed 2026-11-02 (Adelaide). It grouped on the number of mutations
      from the parent (1 to 10): samples per group 2, 6, 13, 60, 271, 824, 1,529, 1,440, 668,
      187; the 5 folds held out 271, 187, 1,494, 3,035 and 13 samples, and the binder share
      fell from 0.92 at 3 mutations to 0.16 at 10.
      It now groups the CDRH3 loops at 80% identity, single linkage, in
      `antibody_baseline_survey` under the strategy name `cdrh3_identity`; the loader
      returns the loops as `cdrh3_seqs`. This follows the clonotype split of Chinery et al.
      2024 (`Chinery2024`, bioRxiv 10.1101/2024.03.26.586756, read 2026-11-02): their main
      result is a random 70:15:15 split, and beside it "Results are also presented with
      train, validation, and test sets split by clonotype (see Methods and SI)", clustered
      by V and J gene and by 70% identity across the CDRH3, every member of a clonotype in
      one set. Every variant here carries trastuzumab's V and J genes, so only the CDRH3
      rule applies. Their random split is not an option here.
      80% rather than their 70%, measured 2026-11-02 on the 5,000-variant subsample: at 70%
      (3 of 10 loop positions) the groups merge into one group of 3,950 of 5,000 variants
      and the folds hold out 1, 242, 266, 278 and 4,213 samples; at 80% there are 3,039 groups,
      the largest 284 samples, folds hold out 740 to 1,354 samples and the binder share runs 0.29
      to 0.39; at 90% 4,646 of 4,809 groups hold one sample, which is a random split in all but
      name. 80% is also what the cdr_flexibility sets use.
      Only `remove_overlap/random_split/0.7_0.15_0.15` is in the repository
      (lewis-chinery/Tz_her2_affinity_and_beyond): 367,042 training, 78,652 validation and
      78,652 test sequences of 524,346, binder share 32.83% in all three. The clonotype
      split is in the paper text, not as a folder. What "remove_overlap" removes is not
      stated and the code that built that folder is not in the repository. The paper's
      Methods section was read through a page summary, not verbatim, so the V/J-and-70%
      wording is unconfirmed against the PDF.
      Their code reports by mutation count rather than splitting on it:
      `plot_PR_AUC_by_edit_distance` in `src/classifiers/evaluate.py` draws one
      precision-recall curve per mutation count on the test set. `_shared` carries
      `edit_distances` out of the antibody loader (2026-11-01), so a score per mutation
      count can still be reported under the new grouping.
      Both paper outlines carry this with the citation.
- [x] cdr_flexibility and abcov_kd out of `CUT_DATASETS` (2026-11-01).
- [x] tz_her2_binary and tz_her2_multiclass out of `CUT_DATASETS` (2026-11-02).
- [x] humanness_therapeutic: its row's noise step is empty (`plan.NOISE_SKIP_DATASETS`,
      2026-11-01); every other step runs.

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

One sample is one compound. The target is one number per compound. Metric: MAE, in the
target's own unit.

| Task | Compounds | Input | Target (unit) |
|---|---|---|---|
| matbench_jdft2d | 636 | structure | exfoliation energy (meV/atom) |
| matbench_phonons | 1,265 | structure | last phonon DOS peak (cm⁻¹) |
| matbench_expt_gap | 4,604 | composition only | measured band gap (eV) |
| matbench_dielectric | 4,764 | structure | refractive index (no unit) |
| matbench_log_gvrh | 10,987 | structure | log10 shear modulus (log10 GPa) |
| matbench_perovskites | 18,928 | structure | formation energy (eV/unit cell) |

"Structure" means a crystal structure per compound, from which every view
below can be computed. "Composition only" means just the chemical formula, so only the
composition views exist for expt_gap.

The split: Matbench's own 5 test folds, `KFold(5, shuffle=True, random_state=18012019)`
over the order of the compounds in the file. Random, not grouped. You approved them for Matbench only on
2026-10-23. `src/kirby/datasets/matbench.py::load_matbench` already returns each sample's
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
- A feature the file does not list is dropped before ranking, with only a warning
  (modnet 0.1.12 `preprocessing.get_features_dyn`, which drops every feature missing
  from the file). On jdft2d, 627 of MODNet's 1,698 features are missing from the file
  (`features_cross`, md5 `b83e0bd43f71ec53c4d69ee0764acfbe`): BondFractions 514,
  XRDPowderPattern 63, CoulombMatrix 24, SineCoulombMatrix 24, Miedema 2. Counted
  2026-11-02 on features computed with matminer 0.6.5. The published run's matminer
  version is not recorded, so whether it dropped the same 627 is not known.
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

Run on 2026-10-26 with scikit-learn 1.6.1 and matminer 0.10.1, seed 42. The cleaning is
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

The views already built by `src/kirby/representations/materials.py`, cached
at `/Volumes/seagate/matbench/reps/<task>/<rep>.npy`. One row is one view;
each entry is its width in features. "missing" means not built yet.

| View | jdft2d | phonons | expt_gap | dielectric | log_gvrh | perovskites |
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

- Views: the 13 views in the Feature sets table above, per task; expt_gap
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

217 assays, each a separate supervised task. One sample is one protein variant: a full-length
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
held-out samples. Only modulo and contiguous are used here.

The reference file also carries each assay's `taxon`, `coarse_selection_type`, `seq_len`
and `DMS_number_single_mutants`, which is what the assay choice below is made from.

### Preprocessing

None on the assay files. Each is used as shipped, through
`src/kirby/datasets/proteingym.py::load_proteingym_dataset`, which reads the fold files
because they carry the scores and the three fold assignments in one place.

Per-position encodings do not work under modulo or contiguous folds. Measured on
`ESTA_BACSU_Nutschel_2020`, 2026-10-23: 0 held-out positions are also mutated in
training under either scheme, so one-hot and blosum collapse to a single identical
pattern across the held-out samples (417 to 470 per fold under modulo, 308 to 516 under contiguous), and Spearman comes back nan. Their own
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

Training: 10,000 steps, 64 variants per batch, 100 warmup steps, weight decay 5e-2 (5e-3 for Tranception),
seed 2023.

| Baseline | Embedding model | Features | Kernel | Zero-shot covariate |
|---|---|---|---|---|
| ESM-1v Embeddings | `esm1v_t33_650M_UR90S_1`, layer 33 | 1,280 | 7 | yes |
| MSA Transformer Embeddings | `esm_msa1b_t12_100M_UR50S`, layer 12, 384 alignment rows | 768 | 9 | yes |
| Tranception Embeddings | Tranception Large, last layer | 1,280 | 7 | yes |
| One-Hot Encodings | none | (residues + 2) × 33 | — | no, `augmentation = None`, linear head |

Two consequences for any comparison:

- The convolution runs over residues before the mean, so the published number cannot be
  reproduced from mean-pooled features. Mean-pooling first and fitting a tree or an SVM
  is a different protocol, not a looser version of theirs.
- Only the One-Hot Encodings column is a single view. The other three each get
  a second input the view does not contain. Their ProteinNPT, Kermut and the
  five "+ One-Hot Encodings" columns are not single views either.

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

The same files hold `fold_contiguous_5` and `fold_random_5`, and six further methods per
assay.

### Which assays

Not all 217. The rule, applied to the reference file:

1. at least 800 single mutants, so each of the five folds holds at least 160 held-out samples
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

Any view used here must not index by position, for the reason in Preprocessing.
What exists, in `src/kirby/representations/` (first four rows) and in `tests/experiments/benchmarks/` (last three rows):

| View | Features | Where |
|---|---|---|
| `aac` | 20 | `protein.py::create_protein_aac` |
| protein descriptors | 30 | `protein.py::create_protein_descriptors` |
| `mutation_features` | ~48 | `mutation.py::create_mutation_features` |
| ESM-2 embeddings | per model | `protein.py::create_esm2_protein_embeddings` |
| ESM-1v embeddings | 1,280 | built by `b05_proteingym_esm1v_embeddings.py`, one assay only |
| MSA Transformer embeddings | 768 | script written, not run |
| Tranception embeddings | 1,280 | script written, blocked |

The 95 per-variant zero-shot scores ProteinGym ships are each another model's prediction
of the fitness being predicted, so using them as a view is stacking rather than
describing the variant. On `ESTA_BACSU_Nutschel_2020` all 95 as one view scored
0.5958 with LightGBM and 0.6330 with SVM under modulo, above every embedding set here.

### Benchmark files

Four files. The first three compute a view; the fourth scores one. None of
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
matching method beside it with the protocol difference stated. Re-orders the saved matrix's variants to the
loader's order rather than assuming they match, and refuses an incomplete matrix.
`fold_scores(X, y, g, model_name)` is the scoring loop on any feature matrix, so it can
score a hybrid's features.

All checkpoints and outputs are on seagate. The first versions of these scripts were
written to the session scratchpad under `/private/tmp`, which was cleared overnight on
2026-10-29 and took them with it.

---
## 6. DroneRF (which drone and which flight mode, from a 2.4 GHz RF recording)

### Source

| | |
|---|---|
| Data | Mendeley Data v1, doi 10.17632/f4c2b4n755.1, published 2019-03-02, CC BY 4.0. 23 RAR archives, 4.03 GB, holding 454 CSV files. Downloaded 2026-10-30 to `/Volumes/seagate/benchmarks_protocol/dronerf/rar/` (the Mendeley file list is at `public-api/datasets/f4c2b4n755/folders/1` and `.../files?folder_id=<id>&version=1`; the page itself lists nothing without JavaScript), extracted to `dronerf/raw/` |
| Code | https://github.com/Al-Sad/DroneRF, Apache-2.0. `Matlab/Main_1_Data_aggregation.m` (chunking and spectra), `Matlab/Main_2_Data_labeling.m` (normalisation, labels, writes `RF_Data.csv`), `Python/Classification.py` (the published classifier and split). LabVIEW recording programs are also in the repository |
| Papers | Al-Sa'd, Al-Ali, Mohamed, Khattab, Erbad, "RF-based drone detection and identification using deep learning approaches: an initiative towards a large open source drone database", Future Generation Computer Systems 100:86–97, 2019, doi 10.1016/j.future.2019.05.007 (the numbers). Allahham, Al-Sa'd, Al-Ali, Mohamed, Khattab, Erbad, "DroneRF dataset: A dataset of drones for RF-based detection, classification, and identification", Data in Brief 26:104313, 2019, doi 10.1016/j.dib.2019.104313 (the data) |
| Extra views | Swinney and Woods, "Unmanned Aerial Vehicle Operating Mode Classification Using Deep Residual Learning Feature Extraction", Aerospace 8(3):79, 2021, doi 10.3390/aerospace8030079, CC BY. Abstract read 2026-10-29; the full text refuses automated download, so its DroneRF-specific settings are unverified |

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
  text's "1 million" is wrong). Values are integers between -740 and 776 in that file,
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
  test chunk has about 89 same-recording chunks in training.
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
  `{BUI}_{n}` file pair), stratified on each task's own label (`StratifiedGroupKFold`,
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
  (526 to 4,100 chunks per class remain). Not done: it changes the set of chunks, so nothing
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
definition in the paper); the CEUR wavelet features (the paper does not say how they are computed); Allahham's 8 x
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

The view and model screen of the table above. Steps: `extract` (one pass over the 454
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
  row: Kılıç's grid over C and gamma with inner CV is about 550,000 fits, and no smaller
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

One row is one view. Each entry is 10-class accuracy / macro-F1 in %, mean over 3 folds,
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
truncates); stats uses all of them; the Welch PSD uses the first 99,425 and spectral_scalars the first 99,840 (Welch keeps whole segments only).

---


## 7. Reaction yields: the CGR view screen (buchwald_hartwig, bh_doyle_plates, suzuki)

Added 2026-11-01. Not a new benchmark: the three reaction-yield datasets KIRBy already
uses, screened for which condensed-graph-of-reaction (CGR) views to carry.
`opig_cgr_comparison.py` waits for this screen's summary and the author's choice of its
`with_cgr` group.

Where each CGR view comes from (refs.bib keys):
- `cgr_morgan`: built with CGRtools / Chython (`Nugmanov2019`). Not a published fingerprint.
- `chemprop_cgr`: Chemprop's CGR mode (`HeidGreen2022`; software `Heid2024`).
- CGR-SMILES language model: `Gerhaher2025`, a poster, results on one task only.
- `chytorch_graphormer`: the Bidirectional Graphormer pretrained for atom mapping (`Nugmanov2022`; Graphormer `Ying2021`).
- DOPtools CircuS and ChyLine (`Byadi2025`): dropped, see below.

Tools that cannot take a CGR, so are not candidates:
- RDKit and Mordred have no dynamic bonds, so compute nothing on a CGR.
- ChemBERTa and MolFormer never saw CGR-SMILES bond tokens in pretraining.
- rxnfp reads `reactants>>products`, not CGR notation.
- Chython's `CGRContainer` has no Morgan fingerprint method.


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

- Views, per dataset: its 10 views (`complexity_helpers.DATASET_REPS[ds]['general']`,
  cgr_morgan among them) and six CGR views:

| view | features | where it comes from | built |
|---|---|---|---|
| cgr_morgan | 2,048 | existing, `create_cgr_morgan` (among the 10 views of all three datasets) | once per dataset |
| cgr_smiles_lm | 384 | existing, `create_cgr_smiles_lm`; refuses without the CGR-pretrained checkpoint | once |
| chytorch_graphormer | 1,024 | existing, `create_chytorch_graphormer`; now raises instead of writing all-zero features for reactions it fails on | once |
| cgr_atompair | 2,048 | new, hand-written port of RDKit's atom pair to the CGR | once |
| cgr_torsion | 2,048 | new, hand-written port of RDKit's topological torsion to the CGR | once |
| cgr_bond_changes | 30 | new, hand-written counts of bonds formed, broken and changed | once |
| chemprop_cgr | 300 | chemprop 1.6.1 reaction mode (Heid and Green 2022), defaults; the MPN encoder output | per outer fold |

- Every view but cgr_bond_changes is scored twice: alone, and with the 30
  cgr_bond_changes counts appended, on the same folds and model.
- Models: `decided_models(ds, 'general')`, rf and lightgbm on all three (2026-11-01).
- Units: 60 build (per dataset 15 once, chemprop_cgr x 5 folds), 186 evaluate (per
  dataset 16 views alone and 15 with cgr_bond_changes, x 2 models), 1 summary.
- Dropped by the author, 2026-11-02: DOPtools CircuS and ChyLine (cgr_circus,
  cgr_chyline). doptools 1.3.9 requires chython>=1.78 (3.x on PyPI); chytorch 1.53
  needs the installed chython 1.68, so the two cannot share an environment. Three
  hand-written CGR fingerprints are in the screen already.
- chemprop on the server: RDKit 2023.09.6 with pandas 2.2.0 fails on `import chemprop`
  (RDKit patches a pandas function that moved). The chemprop build restores the old
  name before the import; no library is changed.

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
| evaluate cgr_atompair with cgr_bond_changes appended, rf | ran, 5 folds, result file written | about 1 s per fold |
| summarise | ran | 15 s |

On buchwald_hartwig, 27 of the 30 cgr_bond_changes features are the same for every
reaction: each reaction forms one single C-N bond and breaks one single bond; only which
carbon-halogen bond breaks (C-Cl, C-Br or C-I) differs. Reagent atoms do not appear as
broken bonds.

---


## UCR Crop (which crop a pixel grows, from its NDVI series)

Added 2026-11-09 from the research file `~/Downloads/UCRCrop.md`. At step 1.

### Source

| | |
|---|---|
| Data | UCR 2018 archive, Crop, donated by F. Petitjean. Downloaded by aeon's `load_classification('Crop')` from timeseriesclassification.com; Zenodo record 11186344 |
| What one sample is | one pixel (64 m²): its NDVI at 46 FORMOSAT-2 dates. 24 crop classes |
| Split | the archive's fixed TRAIN / TEST files: 7,200 training and 16,800 test samples. tsml-eval calls it resample 0 |
| Published code | aeon (tsml-eval's classifiers); github.com/hfawaz/InceptionTime |
| Published scores | tsml-eval `results/classification/Univariate/<classifier>_accuracy.csv`, one column per resample, column 0 = the fixed split. Read 2026-11-09 |

### Published scores, fixed split

One row is one classifier. Test accuracy on the 16,800 test samples. Read from tsml-eval
2026-11-09; the 30-resample mean is beside it.

| classifier | resample 0 | 30-resample mean |
|---|---|---|
| InceptionTime | 0.7985 | 0.7933 |
| H-InceptionTime | 0.7938 | 0.7927 |
| DrCIF | 0.7799 | 0.7796 |
| MultiRocket (MR) | 0.7796 | 0.7772 |
| MR-Hydra | 0.7787 | 0.7782 |
| QUANT | 0.7769 | 0.7759 |
| R-STSF | 0.7760 | 0.7736 |
| RDST | 0.7713 | 0.7681 |
| FreshPRINCE | 0.7704 | 0.7679 |
| MiniROCKET | 0.7663 | 0.7641 |
| ResNet | 0.7633 | 0.7638 |
| WEASEL-2 | 0.7610 | 0.7587 |
| HC2 | 0.7608 | 0.7620 |
| TSFresh | 0.7566 | 0.7563 |
| Hydra | 0.7366 | 0.7357 |
| RSF | 0.7268 | 0.7233 |
| 1NN-DTW | 0.7105 | 0.7107 |
| Catch22 | 0.6916 | 0.6917 |

The research file says 0.7938 for H-InceptionTime is "almost certainly a mean over 30
resamples". The tsml-eval file says otherwise: 0.7938 is resample 0, and the mean is 0.7927.
It also says the fixed split has no published number above about 0.76. InceptionTime's
resample 0 is 0.7985.

### The 11 conditions

1. No people: yes, pixels of satellite images.
2. Per-object score: accuracy per test sample.
3. Label readable from one sample: one crop per pixel. The best published row is 0.7985.
4. Room: the best published classifier gets about 3,385 of 16,800 test samples wrong
   (0.7985). The test set's resolution is about 52 samples (sqrt(16,800 x 0.8 x 0.2)).
5. Code that trains and scores on the published split: aeon and tsml-eval, and the
   InceptionTime repository.
6. Train and test fixed by files: yes.
7. Leaks: not known. Pixels from one field may sit on both sides of the split. The archive
   files carry no field number, so this cannot be checked from the data. The `--stage data` command
   counts test samples identical to a training sample.
8. Re-runs: not yet run (below).
9. Raw signal: the 46 NDVI values per pixel.
10. One label per sample: yes.
11. Free download: yes.

### Reproduction (step 2)

Not run. The data, the views and every output go on Seagate, and the drive was not mounted
on 2026-11-09. The `published` units of the screen are the reproduction: each published
classifier run end to end with aeon 1.3 on the fixed split, its accuracy beside tsml-eval's
resample 0 in the summary. InceptionTime is a PyTorch port of the published Keras code
(the laptop's tensorflow does not import).

### Screen (dataset screening ledger, row `ucr_crop`)

Script `tests/experiments/benchmarks/b11_ucr_crop_views.py`; spec `UCRCrop` in
`tests/experiments/screen/datasets.py`. One fit per unit, on the fixed split. Summary at
`<run>/ucr_crop/b11_views/full/summary.md`.

- Views (24): raw, diff1, diff2, periodogram, harmonic2, harmonic3, catch24, tsfresh,
  rocket, minirocket, multirocket, hydra, quant, rdst, weasel2, random_intervals,
  supervised_intervals, ed_medoid, dtw_medoid, moment, mantis, chronos2, ts2vec,
  inceptiontime. What each one is and where it comes from is in the script's docstring.
- phenology (13 season metrics, TIMESAT-style) is written but not in the row. Three of its
  settings have no value: the smoother, the smoothing window in time steps, and the
  start/end threshold as a fraction of the amplitude.
- Appended blocks: every view is scored alone, then with harmonic2, harmonic3 and catch24
  appended, one at a time.
- Models (7), each with the settings of the code that used it on Crop: ridge_cv (aeon
  MultiRocket), extra_trees (aeon QUANT), rf (aeon Catch22), rotation_forest (aeon
  FreshPRINCE), logreg (the probe of arXiv 2610.28602), svm (TS2Vec's `fit_svm`), knn1
  (the archive's 1-nearest-neighbor baseline).
- Published classifiers (16 units): multirocket, minirocket, rocket, hydra,
  multirocket_hydra, quant, rdst, weasel2, catch22, freshprince, drcif, rstsf, tsf,
  knn1_ed, knn1_dtw, inceptiontime.
- Units: 25 build, 667 evaluate (16 published, 651 view x model), 1 summary.
- Added to a live run with `ledger.py extend --datasets ucr_crop`, after copying the
  data folder (`KIRBY_CROP_DIR`) to the server. ts2vec needs the TS2Vec repository at
  `KIRBY_TS2VEC_DIR`.

---


## 8. RML22 (modulation class of one 2 x 128 I/Q radio frame)

Added 2026-11-09. Adelaide: "I'm fine proceeding with this dataset, but let's test the
numbers quickly with proper splits (random is fine in this case, but it has to be done
correctly)."

### Source

| | |
|---|---|
| Data | Google Drive folder linked from the repository README (`1dEv6gPwPahUfFFRYYxvp3i5M34D7KI9J`): `RML22` (pickle), `Podcast.wav`, `ReadMe`. No form. Not downloaded yet |
| Code | github.com/venkateshsathya/RML22, no licence file. `code/RML22.ipynb` (cell 5 trains, cell 7 tests), `code/CNNArchitecture.py`, `code/TrainHelper.py`, trained weights `code/RML22_model.pt`. Read 2026-11-09 |
| Paper | Sathyanarayanan et al., IEEE Trans. Wireless Commun. 22(11):7663-7675, 2023. Not read (paywalled) |
| Literature | `~/Downloads/RML22.md` (research file, 2026-11-09). Best published: STF-GCN 70.95% overall, 47.60% mean over SNR levels at or below 0 dB, 95.59% at or above, 6:2:2 split, no code, states 462,000 frames |

### What is in it

10 modulations (QAM16, QAM64, 8PSK, WBFM, BPSK, CPFSK, AM-DSB, GFSK, PAM4, QPSK) x 21 SNR
levels (-20 to 20 dB, step 2) x 2,000 frames = 420,000 frames, each 2 x 128 (I and Q).
One label per frame. Simulated in GNU Radio; no data from people.

### The fault in the published test

- Cell 5 splits all 420,000 frames once with `train_test_split(test_size=0.2,
  random_state=1)`, then splits the 80% again the same way for validation (early stopping
  and keeping the best weights). The 84,000 held-out frames are never scored.
- Cell 7 calls `train_test_split(test_size=0.2, random_state=1)` again inside every
  (modulation, SNR) block of 2,000 frames. That draw is unrelated to cell 5's, so about
  64% of the frames it scores were trained on and about 16% were validation frames
  (fractions of the split; counted on the data by `--stage released`).
- The confusion matrix uses a third such split, over -4 to 20 dB.
- The repository's 0.6023 is validation accuracy at epoch 45, not a test score.

If RML22 goes into Paper 2, its outline needs a note: no comparison with the original
paper, because its splits leak, and the held-out frames are the harder case (Adelaide,
2026-11-09, conditional on it going in).

### Benchmark file

`tests/experiments/benchmarks/b11_rml22_published.py`

```
export KIRBY_RML22_DIR=/Volumes/seagate/benchmarks_protocol/rml22
python tests/experiments/benchmarks/b11_rml22_published.py --stage released
python tests/experiments/benchmarks/b11_rml22_published.py --stage train --test-size 0.2 --seed 1
```

- `released`: the authors' weights, scored on cell 5's held-out 20% and on cell 7's
  re-split, with the count of scored frames that were in training and in validation.
- `train`: cell 5's CNN and training loop ported line for line (Adam 1e-3, weight decay
  5e-4, batch 32, StepLR 8 / 0.1, up to 200 epochs, early stop 16, best weights reloaded
  on every epoch without improvement), on a random split given by `--test-size` and
  `--seed`, scored on that split's held-out frames. The one addition: numpy and torch
  seeded with `--seed`.
- Output per run: overall accuracy, macro-F1, accuracy per SNR level, means over SNR
  levels at or below and at or above 0 dB (STF-GCN's columns), per modulation, confusion
  matrix.
- Smoke-tested 2026-11-09 on fake data of the published layout (50 frames per block):
  both `--stage` values run; cell 7's re-split scored 1,369 training and 327 validation frames of
  2,100. Not run on the data.

---


## LSST (UEA archive): which kind of astronomical object, from six passbands of flux

Added 2026-11-09 from the research file `~/Downloads/LSST.md`. At step 1.

### Source

| | |
|---|---|
| Data | UEA multivariate archive, LSST: `LSST_TRAIN.ts`, `LSST_TEST.ts`, downloaded by aeon's `load_classification('LSST')` from timeseriesclassification.com. Built from the PLAsTiCC training set (Bagnall et al. 2018, arXiv 1811.00075) |
| Published reference | Middlehurst et al. 2026, "The Multiverse of Time Series Machine Learning", arXiv 2603.20352, Tables 17-18; `github.com/aeon-toolkit/multiverse`, `results/multiverse/accuracy_mean.csv` |
| Code | Multiverse's driver calls tsml-eval's `get_classifier_by_name(name, random_state=0)`, n_jobs 1, no time limit. The constructor calls were read from tsml-eval main, `tsml_eval/experiments/_get_classifier.py`, on 2026-11-09 |

### What is in it

- One sample is one simulated light curve. 6 passbands (u g r i z y) x 36 flux values.
- 2,459 training and 2,466 test curves, 14 classes. The largest class is 31.5 % of the test
  file (Multiverse's Dummy classifier: 0.3151).
- The archive keeps flux only. Observation times, flux errors and redshift are gone, so
  index j of one band and index j of another are not the same moment. Physical periods,
  same-moment colours and light-curve model fits cannot be computed from it.
- How the archive cut the PLAsTiCC curves to 36 values per band is not documented beyond
  one sentence in Bagnall et al. 2018.

### Published protocol and scores

Fit on the training file, score once on the test file, seed 0, accuracy (Multiverse
section 6). One row is one classifier; accuracy on the 2,466 test curves.

| classifier | Multiverse accuracy |
|---|---|
| FreshPRINCE (tsfresh features + rotation forest) | 0.7028 |
| HIVE-COTE 2 | 0.6740 |
| Catch22 (catch24 + 200-tree random forest) | 0.6695 |
| Arsenal | 0.6577 |
| RDST | 0.6492 |
| QUANT | 0.6452 |
| MultiRocket-Hydra | 0.6427 |
| CIF, 500 trees | 0.6367 |
| ROCKET | 0.6363 |
| DrCIF, 500 trees | 0.6354 |
| RIST | 0.6273 |
| STC | 0.6131 |
| TDE | 0.5576 |
| 1NN-DTW | 0.5511 |
| Dummy | 0.3151 |

Three numbers disagree with each other:
- FreshPRINCE's own Multiverse file, `results/multiverse/FreshPRINCE/FreshPRINCE_accuracy.csv`,
  says 0.7040 (1,736 of 2,466 curves). The combined table says 0.7028 (1,733). Both are on main.
- HIVE-COTE 2's published number is 0.6427 (timeseriesclassification.com, via ConvTran
  Table A2). Multiverse's rerun gives 0.6740.
- ROCKET is 0.5251 in ConvTran Table A1 and 0.6363 in Multiverse.

Not on this split from Multiverse: fine-tuned Mantis, 0.7109 over 3 seeds (Mantis paper);
frozen MantisV2 embeddings with a 200-tree random forest, 0.660 (TS2Vec-FM paper, Table 10).

The 30-resample means of Ruiz et al. 2021 (Table 11) are a different protocol. They are
not comparable with the table above.

### Checklist (2026-10-30), checked 2026-11-09

1. No data from people: simulated astronomical light curves. Passes.
2. Per-object score: accuracy per curve. Passes.
3. Label readable from one object: one of 14 classes, nothing masked. Passes.
4. Room above the published best. FreshPRINCE leaves 733 of 2,466 test curves wrong. The
   test set resolves about 23 curves (sqrt(2,466 x 0.70 x 0.30)). Passes.
5. Code that trains and scores the published model on the published test set: aeon and
   tsml-eval. Passes.
6. Training and test fixed by files. Passes.
7. No leak between training and test: not checked. The archive carries no object ids.
8. Published number not contradicted by reruns: three disagreements, listed above. The
   reproduction decides this.
9. Raw signal shipped: flux values, yes. Times and errors, no.
10. One label per object. Passes.
11. Free download: not checked this session. aeon fetches it with no login.

### Reproduction

`tests/experiments/benchmarks/b12_lsst_published.py` runs the published classifiers end to
end with tsml-eval's constructor calls, seed 0, on the default split. It writes one JSON per
classifier, with accuracy, balanced accuracy, correct curves, seconds and the aeon version.
First four run on the laptop, 2026-11-09, aeon 1.3.0, n_jobs 1. One row is one
classifier; accuracy on the 2,466 test curves.

| classifier | published | rerun | test curves correct | seconds |
|---|---|---|---|---|
| Dummy | 0.3151 | 0.3151 | 777 | 2 |
| Catch22 | 0.6695 | 0.6679 | 1,647 | 1,910 |
| QUANT | 0.6452 | 0.6452 | 1,591 | 34 |
| MiniRocket | 0.643 (RLPAM table, not Multiverse) | 0.6509 | 1,605 | 9 |

Catch22 is 4 test curves below the published number. The other classifiers are not run yet.

Two differences from the published code:
- aeon 1.3.0 is installed here. Multiverse's pyproject asks for aeon 1.4.0 or newer. The
  version behind its results is not recorded.
- STC: tsml-eval uses its own work-in-progress STC. The file uses aeon's.

```
python tests/experiments/benchmarks/b12_lsst_published.py --classifiers dummy,catch22,quant,minirocket,rocket
python tests/experiments/benchmarks/b12_lsst_published.py
```

### Screen (dataset screening ledger, row `lsst`)

Script `tests/experiments/benchmarks/b12_lsst_views.py`, spec `LSST` in
`tests/experiments/screen/datasets.py`. One unit per view x model x split; summary at
`<run>/lsst/b12_lsst_views/summary.md`. It holds accuracy and balanced accuracy per view
and model, seconds per unit, error overlap between views per model, and linear CKA
between views on the test curves.

Views (17). The full definitions are in the script's docstring. Views marked * are fitted
on the training curves only.

| view | features | source |
|---|---|---|
| raw | 216 | none (1NN on raw values is Bagnall 2018's baseline) |
| band_stats | 54 | none: 9 statistics per band |
| band_colours | 270 | paper: the 9 statistics, differences and normalised differences between every pair of bands. Summaries are taken before crossing bands, as Boone 2019 does for flux ratios |
| index_periodogram | 108 | paper: power spectrum per band in index units (DrCIF's periodogram) |
| catch24 | 144 | code: aeon Catch22, tsml-eval's settings |
| tsfresh | 4,698 | code: aeon TSFresh comprehensive, FreshPRINCE's transform |
| rocket* | 20,000 | code: aeon Rocket |
| minirocket* | 9,996 | code: aeon MiniRocket |
| multirocket* | 49,728 | code: aeon MultiRocket |
| hydra* | 3,072 | code: aeon HydraTransformer |
| quant* | 2,610 | code: aeon QUANTTransformer |
| rdst* | 3,270 on random arrays; set by the data | code: aeon RandomDilatedShapeletTransform, 10,000 shapelets at most |
| mantis | 1,536 expected | code: Mantis-8M frozen, each band interpolated to 512 values |
| mantisv2 | 1,536 expected | code: MantisV2 frozen, same handling |
| moment | 6,144 expected | code: MOMENT-1-large frozen, one embedding per band |
| chronos2 | 4,608 expected | code: Chronos-2 frozen encoder, one embedding per band |
| ts2vec* | 320 | code: TS2Vec, trained on the training curves |

Models (7). Each is the classifier a paper used on one of these views. All 7 run on every
view.

| model | settings | from |
|---|---|---|
| ridgecv | scaling without centring, RidgeClassifierCV over 10 alphas from 1e-3 to 1e3 | aeon's ROCKET, MiniRocket and MultiRocket classifiers |
| rotf | aeon RotationForestClassifier, 200 trees | FreshPRINCE |
| rf | random forest, 200 trees | aeon's Catch22 classifier; the TS2Vec-FM probe |
| extratrees | 200 trees, max_features 0.1, entropy | QUANT |
| 1nn | 1 nearest neighbor, Euclidean | Bagnall 2018 |
| logreg | scaled logistic regression, sklearn defaults except max_iter 1000 | TS2Vec-FM linear probe; its settings are not stated |
| svm | RBF SVM, C by 5-fold grid search, 1e-4 to 1e4 and infinity | TS2Vec's protocol, written from memory of its code, not re-read |

Units: 18 build (1 download, 17 views), 119 evaluate, 1 summary. Split in the row: the
default split only. The script also runs resamples 1 to 29 of Ruiz et al. 2021
(`--splits r1,...`). They are not in the row.

Laptop check, 2026-11-09, on 120 random arrays of LSST's shape, nothing written to disk.
It shows that each view and model it tried runs, not how well. Seconds on 120 curves: catch24 43, tsfresh 158,
rocket 28, minirocket 31, multirocket 81, rdst 120, hydra and quant under 1. All 7 models
ran on catch24. One process that fitted rotf and then the next models stopped silently
after 1nn, with a warning that two OpenMP libraries were loaded. logreg and svm ran in a
fresh process. Each ledger unit is its own process. The four pretrained embeddings and
ts2vec were not run: they need model downloads and the TS2Vec repository on Seagate.

---


## WiSig (which WiFi transmitter sent one packet)

Added 2026-11-09 from the research file `~/Downloads/WiSig.md`. At step 1. The
reproduction (step 2) waits for the data: the download to Seagate is running, and the
drive dropped twice on 2026-11-09.

### Source

| | |
|---|---|
| Data | cores.ee.ucla.edu/downloads/datasets/wisig/, CC BY-NC-SA 4.0. Four compact subsets on Google Drive, no form: ManyRx 1.2 GB (file id `1TtdydJCuhkvDQ1RWb3PkxakkWo2-X5Uv`), ManyTx 2.5 GB (`17EnvGFoflJEh1xhFC8wx5fhCuPYhWt2l`), ManySig 1.4 GB (`1szuns8MhcYocdbipK9t9TM9MLgEMklxk`), SingleDay 1 GB (`1lWf9BuUZTSNcABVFWYoBT_-EH8ctXEcZ`). Downloaded 2026-11-09 to `/Volumes/seagate/benchmarks_protocol/wisig/zip/`. SingleDay is not downloaded: none of the authors' notebooks uses it |
| Code | github.com/WiSig-dataset/wisig-examples, BSD-3-Clause, read 2026-11-09: `data_utilities.py` and notebooks d003 to d008 and d011. The committed notebooks keep their printed outputs |
| Paper | Hanna, Karunaratne, Cabric, "WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting", IEEE Access 10:22808–22818, 2022, doi 10.1109/ACCESS.2022.3154790 |

### What is in it

- One object is one packet: the first 256 complex samples at 25 Msps after packet
  detection, stored as 256 x 2 (I and Q). The label is the transmitter.
- 174 off-the-shelf WiFi cards sending, 41 software radios receiving, four capture days
  in March 2021. Each subset comes in two versions: as received ("non-equalised") and with
  the channel removed using the preamble ("equalised").
- ManyTx: 150 transmitters, 18 receivers, 4 days, up to 50 packets per transmitter,
  receiver and day. ManyRx: 10 transmitters, 32 receivers. ManySig: 6 transmitters,
  12 receivers, 1,000 packets each.
- Packet counts are from the paper's table, not counted yet. The research file reports
  511,515 for ManyTx from a third-party repository.

### Published protocols and scores

The paper gives its results only as figures. The committed notebooks print the numbers,
read 2026-11-09. One row is one protocol setting. Accuracy on the test packets, the
authors' network.

| setting | subset | train and test | non-equalised | equalised |
|---|---|---|---|---|
| ntx_10 | ManyTx | first 10 transmitters, all receivers and days combined, packets split 80/10/10 at random (seed 0) | 0.8155 | not run |
| ntx_45 | ManyTx | same, 45 transmitters | 0.6959 | not run |
| ntx_80 | ManyTx | same, 80 | 0.5663 | not run |
| ntx_115 | ManyTx | same, 115 | 0.5492 | not run |
| ntx_150 | ManyTx | same, 150 | 0.5310 | not run |
| nrx_0 | ManyRx | day 1; train on 1 receiver, test on 5 receivers never seen; 5 receiver orders | 0.1465 | 0.3234 |
| nrx_5 | ManyRx | same, 6 training receivers | 0.2380 | 0.5425 |
| nrx_10 | ManyRx | 11 | 0.3719 | 0.6730 |
| nrx_15 | ManyRx | 16 | 0.4339 | 0.7467 |
| nrx_20 | ManyRx | 21 | 0.5256 | 0.7871 |
| nrx_25 | ManyRx | 26 | 0.7150 | 0.7909 |
| ndays_0 | ManySig | receiver 0 only; train day 1, test day 4 | 0.4935 | 0.9585 |
| ndays_1 | ManySig | train days 1-2, test day 4 | 0.7407 | 0.9740 |
| ndays_2 | ManySig | train days 1-3, test day 4 | 0.8292 | 0.9987 |
| nsig_5 | ManySig | 5 training packets per transmitter, receiver and day; test packets from the same captures | 0.7347 | 0.9147 |
| nsig_800 | ManySig | 800 training packets each | 0.9898 | 0.9926 |

nsig also has 10, 25, 50, 100 and 400 (in `b13_wisig_published.py`). The nrx numbers are
the mean over the five receiver orders; the five per order are in the script.

The network: five convolution layers (8, 16, 16, 32, 16 filters), dense layers of 100 and
80, Adam 5e-4, early stop on validation loss with patience 5, up to 100 epochs, class
weights. The port has 39,697 parameters at 5 transmitters, the same as the notebook's own
model summary.

### Two facts about the authors' code

- The receiver study (d003, d011) shuffles the receiver list in place, then looks
  receivers up in the shuffled list while the packets stay in the original order. So the
  receivers named in its printed lists are not the receivers whose packets were used.
  Training and test receivers are still disjoint, so the study is still cross-receiver.
  The port reproduces the same packets.
- The packet split puts packets from one capture (one transmitter, receiver and day) on
  both sides. That is why same-day accuracy is near 1. ntx and nsig use this split; nrx and
  ndays test on receivers or a day not trained on.

### The 11 conditions

1. No people: yes, WiFi cards and radios.
2. Per-object score: accuracy per packet.
3. Label readable from one object: one transmitter per packet.
4. Room. In test packets, from the printed accuracies; test sizes are derived from the
   paper's table, not counted. ntx_150: about 51,000 test packets, 47% wrong. nrx_25:
   up to 10,000 test packets, 29% wrong non-equalised and 21% equalised. ndays_2: 6,000
   test packets, 17% wrong non-equalised; equalised, 0.13% wrong (about 8 packets), so
   ndays equalised has no room. nsig_800: about 1% wrong.
5. Code that trains and scores on the published split: yes, the notebooks above.
6. Train and test fixed by seed: yes, seed 0 in `data_utilities.shuffle` and in the
   receiver orders.
7. Leaks: the packet split, above. nrx and ndays have none of that kind.
8. Re-runs: not yet (step 2 below).
9. Raw signal: the 256 I/Q samples.
10. One label per object: yes.
11. Free download: yes.

### Reproduction (step 2)

`b13_wisig_published.py --protocol <setting> --equalized <0|1>` reruns any row of the
table above. Data downloaded 2026-11-09 to `/Volumes/seagate/benchmarks_protocol/wisig/`.

One row is one run of the authors' network on the laptop, accuracy on the test packets.

| setting | fold | published | rerun | packets (train / validation / test) | time |
|---|---|---|---|---|---|
| nrx_25, as received | receiver order 0 | 0.7212 | 0.7290 | 40,296 / 5,037 / 10,000 | 529 s, stopped at epoch 21 |
| ntx_10, as received | the one split | 0.8155 | 0.8158 | 26,762 / 3,345 / 3,345 | 324 s, stopped at epoch 23 |

The nrx_25 training and validation counts equal the ones the authors' notebook prints.
For ManyTx the notebook prints counts at 45, 80, 115 and 150 transmitters, and the port's
split gives the same four: 121,513 / 15,189, 217,939 / 27,242, 314,033 / 39,254 and
409,213 / 51,151 training / validation packets. So ntx_150 tests on 51,151 packets. The five receiver orders equal the printed ones. ManyRx holds 249,666 packets as
received and 247,684 with the channel removed, so the two versions are not the same
packets and cannot be paired as two views of one packet.

### Screen (dataset screening ledger, row `wisig`)

Script `tests/experiments/benchmarks/b13_wisig_views.py`; spec `WiSig` in
`tests/experiments/screen/datasets.py`. One evaluate unit is one view x model x protocol
setting, every fold of that setting (5 for nrx, 1 otherwise), scored on the authors' test
packets. Summary at `<run>/wisig/b13_wisig_views/summary.md`: accuracy and macro-F1 per
view and model beside the authors' number, error overlap between views, linear CKA
between views.

- Views (15): iq, polar, phase_increment, cfo_compensated_iq, impairments,
  fft_magnitude, welch_psd, spectral_stats, stft_spectrogram,
  channel_independent_spectrogram, stf_response, preamble_error, dctf, hos_cumulants,
  cnn_emb. What each is, its width and its source are in the script's docstring.
  cnn_emb is the authors' network trained on each fold's training packets.
- Left out: wavelet scalogram, phase noise and transient (no source says how to compute them), the
  bispectrum image (no size given), DoLoS (needs a whole L-LTF symbol; the packet holds 56
  of its samples), Transformer embeddings (no code).
- Models (candidates): wisig_cnn (the authors' network, on the three views that keep the
  256 x 2 shape), knn, rf, logreg, xgboost (the DroneRF screen's settings; no WiSig paper
  uses a vector model).
- Two open settings with no default. Init refuses without them: which protocol settings
  run (`KIRBY_WISIG_PROTOCOLS`), and which models (`--models wisig=...`).
- Units for one setting: 15 builds per subset and flag, plus one cnn_emb build per fold;
  evaluate units are 15 views x the vector models, plus 3 for wisig_cnn.
- Laptop check on a made-up file: every unit kind runs and is checked. Not results.

---


## BEANS bats (which of 10 Egyptian fruit bats made a call)

Added 2026-11-09 from the research file `~/Downloads/BEANS.md`. At step 1, proposed; the
published classical scores are rerun (step 2), and the screen script and ledger row are
written, not run on the server.

### Source

| | |
|---|---|
| Data | `https://storage.googleapis.com/ml-bioacoustics-datasets/egyptian_fruit_bats.zip`, 5,246,087,171 bytes, no login. Downloaded 2026-11-09 to the session scratchpad, because the Seagate drive was not connected; it belongs under `/Volumes/seagate/benchmarks_protocol/beans_bats/` |
| Code | github.com/earthspecies/beans at commit `b278b1d9` (MIT): `scripts/bats.py` (the split), `beans/datasets.py` (the features), `scripts/evaluate.py` and `run_benchmark.py` (models, grids, selection) |
| Paper | Hagiwara, Hoffman, Liu, Cusimano, Effenberger, Zacarian, "BEANS: The Benchmark of Animal Sounds", arXiv 2210.12300 (2022). Table 2 read 2026-11-09 |
| Recordings | Prat, Taub, Pratt, Yovel, "An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny", Scientific Data 4:170143 (2017). Their `Annotations.csv` and `FileInfo.csv` (figshare collection 3666502) were downloaded to read where and when each call was recorded |

### What is in it

- 10,000 calls, 1,000 per bat. Bats (`datasets.yml`): 111, 210, 211, 215, 216, 220, 226,
  228, 230, 231. Each call is a float32 WAV at 250 kHz; 18.8 GB unpacked.
- BEANS's `annotations.csv` has two columns, Emitter and File Name. `N.wav` is row N
  (counting from 0) of Prat's `Annotations.csv`, cut from its start to its end sample.
  Checked on `46690.wav`: emitter 216 in both, 467,791 samples against Prat's 467,792.
- Through that row, Prat's metadata gives each call its source recording, treatment
  (colony set-up), recording channel and recording time: 9,921 source recordings, 13
  treatments, 10 channels, 296 days from 2012-06-07 to 2014-02-19.
- The BEANS loader pads every call with zeros, or cuts it, to 5 s (`max_duration: 5`).

### Split

One fixed split of calls, not folds: `train_test_split(test_size=0.4, random_state=42,
stratify=label)`, then the 40% halved the same way. 6,000 training, 2,000 validation,
2,000 test calls. Rebuilt here with scikit-learn 1.6.1 and pandas 2.2.0; the md5 of all
three CSVs equals the md5 BEANS ships in `data/file_hashes`. The research file could not
find this code and marked the split unverified; it is verified now.

### The checklist (2026-10-30)

1. No data from people: bats. Pass.
2. Score per object: accuracy per call. Pass.
3. Label readable from one object: the emitter of the call. Pass.
4. Room above the published best: 0.816 (MetaPerch; from the research file, not checked
   against the paper) leaves about 368 of 2,000 test calls wrong. The test set resolves
   about 17 calls (sqrt(2,000 x 0.816 x 0.184)). Pass.
5. Authors' code trains and scores the published model on the test calls: `evaluate.py`
   does, for the classical models. Pass, rerun below.
6. Training and test fixed by seed: seed 42, md5 matched. Pass.
7. No leaks: partly fails. Next subsection.
8. Published number not contradicted by a rerun: matched, below.
9. Raw signal shipped: the WAVs. Pass.
10. One label per object. Pass.
11. Free download. Pass.

### Calls of one recording session on both sides (condition 7)

BEANS splits calls, not recordings.
- 46 source recordings have calls on more than one side of the split: 97 calls.
- For 222 of the 2,000 test calls (11.1%), the same bat has a training call recorded
  within 10 s of it. Within 60 s: 387 (19.4%). Median gap to the nearest same-bat
  training call: 1,454 s.

Where and when a call was recorded predicts the bat without any audio. Test accuracy,
chance 0.100:
- a decision tree on the treatment and channel numbers: 0.401
- one nearest neighbor on recording time alone: 0.4195

Bats 226 and 228 were recorded only in treatment 9, and bat 111 only in treatments 2, 6
and 9. No split removes this, because those bats have no calls anywhere else. Every
published number carries it.

A grouped split is not built: how to group is open. Candidate rules, measured
2026-11-09. One row is one rule; the last column is the calls held out by each of 5
folds (`StratifiedGroupKFold`, shuffle, seed 42, used only to count).

| rule | groups | largest group, calls | groups holding more than one bat | calls held out per fold |
|---|---|---|---|---|
| source recording | 9,921 | 3 | 12 | 1,997 to 2,003 |
| calls less than 10 s apart, any channel | 8,939 | 6 | 171 | 1,980 to 2,009 |
| calls less than 60 s apart | 7,634 | 16 | 607 | 1,991 to 2,015 |
| calls less than 600 s apart | 4,480 | 43 | 1,283 | 1,857 to 2,105 |
| recording day | 296 | 166 | 244 | 1,719 to 2,350 |

No published number exists under any of these.

### Published protocol (BEANS classical models; `evaluate.py`)

- Features: torchaudio MFCC at 250 kHz, 20 coefficients, 128 mel bands, 12,500-sample
  window (50 ms), 2,500-sample hop (10 ms), power 2. Frames past 5 s cut; shorter calls
  padded with zero frames to 500, and the zero frames enter the statistics. Mean, std,
  min and max of each coefficient over the frames: 80 features.
- StandardScaler fitted on the training calls. One fit per grid value on the training
  calls, the value with the best validation accuracy scored once on the test calls.
- Grids (`run_benchmark.py`): logistic regression C 0.1, 1, 10 (max_iter 1,000); SVM C
  0.1, 1, 10; decision tree max_depth None, 5, 10, 20, 30; gradient boosting
  n_estimators 10, 50, 100, 200; XGBoost n_estimators 10, 50, 100, 200 (n_jobs 4).
- No seed on the models in the published code. The rerun sets random_state 42 on the
  three that take one.

### Reproduction

One row is one model. Test accuracy on the 2,000 BEANS test calls, mfcc_stats features.

| model | published | rerun | grid value chosen on the validation calls | test calls wrong | seconds |
|---|---|---|---|---|---|
| logistic regression | 0.661 | 0.6605 | C 1 | 679 | 17 |
| SVM | 0.720 | 0.7200 | C 10 | 560 | 41 |
| decision tree | 0.474 | 0.4675 | max_depth 20 | 1,065 | 13 |
| gradient boosting | 0.674 | 0.6565 | n_estimators 100 | 687 | 1,401 |
| XGBoost | 0.692 | 0.6950 | n_estimators 200 | 610 | 56 |

Gradient boosting is 0.0175 below its published number: 35 test calls. Its published
run had no seed and scikit-learn 1.1.1; neither difference has been tested as the cause.

Run on 2026-11-09 on the laptop, scikit-learn 1.6.1, xgboost 2.1.4, torch and
torchaudio 2.2.2. The published code pins scikit-learn 1.1.1 and xgboost 1.6.1. The WAVs
are read with scipy instead of `torchaudio.load`; both return the stored float32 values.

The research file's other numbers (Perch 2.0 0.815, MetaPerch 0.816, BirdMAE attentive
0.7515 and the rest of its Table 2) come from frozen encoders with trained probes and
are not rerun here. Sarkar et al. 2026's Table 3 bats row was read 2026-11-09 and is in
the views table below.

### Candidate views

One row is one view: one fixed-length vector per call. Every embedding is the encoder's
final layer averaged over time, the protocol of Miron et al. (arXiv 2508.11845) and
Sarkar et al. (arXiv 2604.27936). Three input modes, all after the 5 s pad or cut:
- `bb`, baseband: resampled to the encoder's rate. Everything above half that rate is lost
  (8 kHz for a 16 kHz encoder).
- `te`, time-expansion (Sarkar et al., Sec. 3.2): the 250 kHz samples read as if recorded
  at the encoder's rate, so 0 to 125 kHz lands in the encoder's band. 5 s becomes 78 s at
  16 kHz.
- `mb`, multi-band (Sarkar et al., Sec. 2, their package `multiband-audio`): the call
  split into 8 kHz bands, each band shifted down to the encoder's band and embedded, the
  band embeddings concatenated. The paper fuses the bands with a trained module; here the
  hybrid search chooses among them instead. Concatenating is this file's choice.

Published accuracy is on the BEANS test calls, with the protocol in brackets.

One row is one encoder. Features are per view: bb / te / mb. Published bats accuracy:
Sarkar et al. Table 3 (read 2026-11-09) is a linear head trained 20 epochs, for bb, te,
and their best fusion module (MoE), which is trained and so is not this file's mb. Other
published numbers are from the research file and not checked against their papers.
Seconds per call: this laptop's CPU, one or two calls, while other jobs ran.

| encoder | views | features | rate | loaded through | published bats accuracy | seconds per call bb / te / mb |
|---|---|---|---|---|---|---|
| mfcc_stats | mfcc_stats | 80 | 250 kHz | BEANS code | SVM 0.720 (BEANS Table 2) | 0.15 |
| effnet_all | bb, te, mb | 1,280 / 1,280 / 20,480 | 16 kHz | avex `esp_aves2_effnetb0_all` | 0.6085 / 0.7025 / MoE 0.6915 | 1.13 / 1.93 / 37.8 |
| effnet_bio | bb, te, mb | same | 16 kHz | avex `esp_aves2_effnetb0_bio` | 0.6075 / 0.7095 / MoE 0.679 | 0.33 / not measured / not measured |
| effnet_as | bb, te, mb | same | 16 kHz | avex `esp_aves2_effnetb0_audioset` | 0.513 / 0.662 / MoE 0.608 | 0.37 / not measured / not measured |
| beats_all | bb, te, mb | 768 / 768 / 12,288 | 16 kHz | avex `esp_aves2_sl_beats_all` | 0.6515 / 0.737 / MoE 0.6865 | 1.04 / not measured / not measured |
| beats_bio | bb, te, mb | same | 16 kHz | avex `esp_aves2_sl_beats_bio` | 0.644 / 0.7335 / MoE 0.683 | 1.40 / not measured / not measured |
| beats_nlm | bb, te, mb | same | 16 kHz | avex `esp_aves2_naturelm_audio_v1_beats` | 0.6545 / 0.7405 / MoE 0.680 | 0.79 / 30.7 / 53.2 |
| eat_all | bb, te, mb | same | 16 kHz | avex `esp_aves2_eat_all` | 0.5075 / 0.6055 / MoE 0.5275 | 2.70 / 2.20 / not measured |
| eat_bio | bb, te, mb | same | 16 kHz | avex `esp_aves2_eat_bio` | 0.6285 / 0.7075 / MoE 0.663 | 1.13 / not measured / not measured |
| birdnet | bb, te, mb | 1,024 / 1,024 / 16,384 | 48 kHz | avex, ONNX export of BirdNET 2.4 by a third party (justinchuby/BirdNET-onnx) | 0.702 / 0.740 / MoE 0.712 | 0.44 / 0.72 / 35.6 |
| sl_eat_all | bb | 768 | 16 kHz | avex `esp_aves2_sl_eat_all_ssl_all` | Miron Table 8 lists EAT variants at 0.635 to 0.655; which row is this checkpoint is not checked | 0.92 |
| sl_eat_bio | bb | 768 | 16 kHz | avex `esp_aves2_sl_eat_bio_ssl_all` | same | 0.81 |
| birdaves | bb | 768 | 16 kHz | avex `aves_bio`, which loads birdaves-biox-base | 0.4895 linear, 0.6847 attentive (Schwinger) | 1.21 |
| aves_bio | bb | 768 | 16 kHz | Earth Species' torchaudio port, aves-base-bio | 0.468 linear, 0.678 attentive (Schwinger); 0.748 fine-tuned (AVES paper) | 0.63 |
| birdmae_large | bb | 1,024 | 32 kHz | avex `birdmae`, DBD-research-group/Bird-MAE-Large | 0.6003 linear, 0.7515 attentive (Schwinger) | 2.89 |
| perch2 | bb | 1,536 | 32 kHz | avex, ONNX export by a third party (justinchuby/Perch-onnx) | 0.766 linear, 0.804 prototypical, random window (Perch 2.0 paper) | 0.27 |
| biolingual | bb | 512 | 48 kHz | transformers ClapModel, davidrrobinson/BioLingual | 0.542 linear (Schwinger); 0.766 fine-tuned | 0.44 |
| hubert | bb | 768 | 16 kHz | transformers, facebook/hubert-base-ls960, final layer | 0.62, time-averaged linear probe on their best layer (Cauzinille) | 0.90 |
| wavlm | bb | 768 | 16 kHz | transformers, microsoft/wavlm-base-plus, final layer | 0.68, same probe | 0.99 |
| vggish | bb | 128 | 16 kHz | torch.hub harritaylor/torchvggish, BEANS's own | 0.743 fine-tuned (BEANS Table 2) | 0.24 |

38 views. Every encoder in the table loaded and embedded on this laptop on 2026-11-09.

EAT pads or cuts its spectrogram to 1,024 frames, 10.24 s at 16 kHz (avex `eat_hf`,
`target_length`). So eat_all_te and eat_bio_te see the first 10.24 s of the 78 s
time-expanded input, which is the first 0.66 s of the call. Sarkar et al.'s package
holds no time-expansion code, so what they did is not known. Not fixed: any fix chooses
a window rule.

Not built, beyond the list below: BEATs pretrained on AudioSet (Miron 0.705 linear;
Schwinger 0.7327 attentive). avex reads its weights from
`gs://representation-learning/pretrained/BEATs_iter3_plus_AS2M.pt`, which refuses
download (HTTP 403). Microsoft's own copy is a OneDrive link.

Not built, with the reason:
- Perch 1.0 and SurfPerch: TensorFlow Hub models; avex marks them deprecated and needs
  TensorFlow 2.21, which is not in either environment.
- XEUS (Cauzinille et al. 0.77 with their time-weighted probe on its best layer): needs
  ESPnet.
- ProtoCLR: its Hugging Face page refuses download without a login (HTTP 401).
- ConvNeXt-BS (Schwinger 0.6475 linear): public on Hugging Face, but needs BirdSet's
  preprocessing, not read.
- AudioMAE, ViT-INS (Schwinger Table 10): not looked for.
- Hand-written call measures (F0, aperiodicity, loudness, spectral centroid, Wiener
  entropy, call length; Rose 2021 dissertation): the measures are named, the settings
  are not published.
- MFCC or mel statistics restricted to 0 to 60 kHz at 250 kHz: suggested by the research
  file, in no paper.
- Prat et al. 2016's 64 MFCCs per 20 ms frame with a GMM-UBM: the classifier works on the
  frame sequence, not a vector.

Models in the papers that are not in the screen: the trained linear head (Sarkar: 20
epochs; Schwinger: on the CLS token), Perch 2.0's prototypical probe, and the attentive
probe (Schwinger), which reads the encoder's token sequence and so cannot run on a vector
view. Fine-tuned networks (VGGish 0.743, AVES 0.748, BioLingual 0.766) train the encoder
itself.

### Environment

The embedding views run through avex (github.com/earthspecies/avex, MIT) 1.4.0, which
declares torch 2.5 or newer. The server's py311-kirby pins torch 2.2.2, so the builds use
a second interpreter, `KIRBY_SCREEN_PY_BEANS_EMBED`. On this laptop (Intel Mac, where
PyTorch stops at 2.2.2) avex was installed without its torch requirement, with
transformers 4.45.1 and numpy 1.26.4; the encoders in the table loaded and embedded two
calls each. The evaluate and summarise units run in py311-kirby on the saved views.

Cost on this laptop's CPU is the last column of the views table. At 30.7 s per call
(beats_nlm_te) a view takes about 85 hours for 10,000 calls; at 53.2 s (beats_nlm_mb)
about 148 hours. The server is not measured. A build unit saves every 200 calls and
resumes from there.

Versions on the laptop: avex 1.4.0, multiband-audio 0.1.0, torch and torchaudio 2.2.2,
transformers 4.45.1, timm 1.0.31, onnxruntime 1.23.2. On the server avex installs with
its own torch (2.5 or newer), so the two will not match.

### Screen (dataset screening ledger, row `beans_bats`)

Script `tests/experiments/benchmarks/beans_bats_views.py`; published-protocol file
`tests/experiments/benchmarks/beans_bats_published.py`. Summary at
`<run>/beans_bats/beans_bats_views/summary.md`: test accuracy per view and model, error
overlap between views per model (shared wrong test calls over the wrong calls of either),
linear CKA between views on the test calls.

- Views: the 38 of the views table.
- Split: `published` only.
- Models: an open setting with no default. Candidates built: the BEANS five.
- Units: 38 build (`extract` for mfcc_stats and the call table, then one per embedding
  view), 38 evaluate per model, 1 summary.

Open settings, each with its candidate values:
- Models for the screen: lr, svm, decisiontree, gbdt, xgboost (all five built; per-view
  fit times not measured beyond mfcc_stats, where they are in the reproduction table).
- Grouped split: the five rules in the grouping table, or none.

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

Kepler DR24 threshold crossing events (AstroNet and Exonet), section 5 until it was
removed on 2026-11-04: out of both papers, Adelaide's decision, no fault in the data. It
had been downloaded, its published accuracies reproduced on linear rows (0.869 / 0.879 /
0.917 published against 0.8259 / 0.9003 / 0.9053), and no view screen had been run. The
benchmark file `tests/experiments/benchmarks/b05_kepler_astronet_published.py` and the
downloads under `/Volumes/seagate/benchmarks_protocol/kepler_astronet/` are left in
place. Do not re-propose it.

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
room above its published best, measured in test samples, well above the test set's
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
- The cached views have missing values set to 0 (see Feature sets above);
  the published random forest fills them with means instead.
- Widths: 3 to 15,120 features per view. soap_avg alone exceeds every budget.
- 9 view caches are missing (table above).
- A 500-tree random forest (the published model, without the cleaning) on soap_avg
  took 729 s for 508 training compounds of jdft2d (fold 0). perovskites has 15,142 or 15,143
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
- Widths: 20 to 1,280 features per view. A budget above about 1,400 features
  would exceed everything available except the embeddings.
- `decided_grid.json` has no row for any ProteinGym assay, and `c11` refuses a dataset
  without one.
- Two of the three embedding views do not exist yet: MSA Transformer needs the
  GPU server at 377 hours per assay on the laptop, Tranception needs permission to run
  ProteinNPT's model code.

### Not yet in KIRBy

- Registry entries. None of the domains `register_datasets` accepts
  (`_VALID_DOMAINS`, `src/kirby/registry.py:30`) fits Matbench, ProteinGym or DroneRF.
  Without an entry, the metric falls back to a guess from the task type instead of
  accuracy. `accuracy` is already a metric KIRBy knows.
- A `decided_grid.json` row for the six Matbench tasks. c11 refuses any
  dataset without one.
- c11 has no option for fixed outer folds (it calls `nested_cv_outer_folds`), which
  mfeat and Matbench both need.
- DroneRF: the download, a chunking + spectrum routine that matches `Main_1`/`Main_2`,
  the five extra view routines above (Welch PSD, three ResNet50 image
  embeddings, statistics), a loader, a registry domain, a `decided_grid.json` row, and
  a c11 option for grouped folds by recording.
- The hybrid experiment files.

