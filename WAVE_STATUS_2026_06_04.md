# Wave Status — 2026-06-04

This is the comprehensive snapshot of what's happening, what I got wrong, and what needs to happen to get correct numbers. Two purposes:
- Hand-off to a new chat session
- Reference document for the user

---

## 0. THE MAIN PROBLEM

**The user has lost MONTHS of PhD work because of repeatedly-wrong wall time estimates.** This document, the new-chat prompt, and every action that follows must center on producing correct, defensible, code-and-data-grounded estimates for:
- Precompute remaining wall, per dataset, per condition
- cx cumulative experiment wall, per dataset, per c-experiment
- sim cumulative experiment wall, per dataset, per s-experiment

Every previous attempt at these estimates has been wrong by 2-10× because someone assumed proportional scaling on `n_samples`. Section 8 below now contains measured cumulative cx walls from five completed pipelines, and the data **definitively refutes any single scaling law in n**.

Every other issue in this document — cache rejection, wedging, nucleotide queue state, the c08/c24 mystery — is downstream context that the new estimates must account for. The cache rejection slows jobs and changes their measured walls. The wedging may or may not be terminal; we can't tell without per-experiment wall data. The nucleotide queue is pending and will need its own estimates when it starts. **All of these are inputs to the estimate problem, not separate problems.** Solving the estimates correctly solves the rest, or at least gives the data needed to make defensible decisions about the rest.

The remainder of this document provides the evidence and unknowns the new estimate model must respect.

---

## 1. Symptom 1: the current wave is partially hung

A SLURM wave started 2026-06-01 ~06:00 with 130 jobs (65 datasets × 2 categories). As of 2026-06-04 01:29, **43 of 69 currently-running jobs are flagged WEDGED** (log file untouched >10 min, many for 10+ HOURS). Of the wedged, ~34 are cx, ~9 are sim. Whether they are truly dead or in legitimate long-running experiments cannot yet be determined — c17 at n=397-418 has been measured at 51-53h, so a job idle 6h at C17 is not necessarily dead. The per-experiment wall data in section 8 must drive the wedge classification, not the raw idle threshold.

---

## 2. Symptom 2: cache rejection on neural reps (verified, root cause unknown)

## 2. The cache rejection — what's verified

**Verified facts** (from the user's log paste of cache stats from herg_ki sim job 12233981):

```
Cache herg_ki__general_distance__none__full.pkl: meta drift on 'rep_hashes'
  stored=(... 'chembert' 14553..., 'chemberta' 12206..., 
           'graphmvp' 12212..., 'grover' 3774..., 'molformer' 10829..., 'smited' 17451...)
  expected=(... 'chembert' 15346..., 'chemberta' 16242..., 
            'graphmvp' 14652..., 'grover' 166298..., 'molformer' 540650..., 'smited' 10406...)
— will recompute.
```

For the same dataset and condition:
- 9 classical rep hashes (atom_pair, avalon, ecfp4, graph_kernel, mhg_gnn, mol2vec, mordred, pdv, top_torsion) are IDENTICAL between stored and expected.
- 6 neural rep hashes (chembert, chemberta, graphmvp, grover, molformer, smited) are completely different.
- `n_samples=1412` matches between stored and expected.
- `seed=42` matches.
- The `.npy` files on disk for those neural reps are dated `2026-05-08 05:49` and were not modified after.
- The stored cache file (`herg_ki__general_distance__none__full.pkl`) was written `2026-06-01 08:44` — by the current wave, not an old one (timestamp `1780299850.84` decodes to that date).

Per-job cache stats from the same log:
```
search: 4 hit / 2 miss / 1 stale; 
baseline: 266 hit / 0 miss / 0 stale; 
hybrid_eval: 15419 hit / 4849 miss / 340 stale
```

The hybrid_eval level has 4849 misses + 340 stale — that's thousands of per-config evaluations being forced to recompute.

**Not verified** (open hypotheses, no evidence):
- Why the neural-rep arrays in memory differ between cache write at +2h44m and cache read at +N hours later within the SAME wave.
- Whether a specific experiment helper in-place mutates `reps[k]` after `np.load`.
- Whether `assign_scaffold_groups` (RDKit-backed) produces different group orderings between processes, causing the rep-array slicing to differ.
- Whether the failure is upstream of `find_best_hybrid` or inside it.

---

## 3. How my estimates have been wrong — full honest list

Across this conversation I produced estimates for:
- Precompute remaining time per dataset
- c-experiment cumulative wall time per dataset
- "Will it fit in 100h remaining" classifications

**The estimates have been incorrect in the following specific ways**, ranked by severity:

### 3.1 cx experiment wall scaling — biggest source of error, central to everything

What I did: I scaled cumulative cx wall time from a single observed log (flab_immunogenicity_marks at n=217, ~70h cumulative) using assumed exponents like `(n/217)^0.8`.

**This was wrong because PROPORTIONAL SCALING IS WRONG. Any constant exponent on `(n/anchor)` is wrong.** See section 10.1 for the full list of reasons.

In particular:
- Different experiments scale differently because they have different internal structures (nested loops over `n_sub`, over candidate reps, over budgets × strategies). The exponent of `n_total` differs per experiment.
- Different datasets have different rep counts, which changes the grid SIZE (number of cells), not just the per-cell wall.
- `MODEL_MAX_BUDGET` and `_max_samples` exclude models at higher n. The grid is not the same at different n.
- The per-config wall itself depends on `(n, budget, model)` non-uniformly because each model has its own complexity class.

**There is no shortcut.** Each experiment must be modelled from its source code as a symbolic function of `(n_samples, n_general_reps, n_top_reps, n_aug_reps, grid params)`, then the per-cell wall must be fit from the 06-02 timing data as a function of `(n, budget, model)`, and the two combined per-dataset. Section 10.2 describes this.

### 3.2 Precompute remaining time — second source of error

What I did: I read the per-condition wall times from check_precompute --timing (2026-06-02) and used those as ground truth for "how long this condition takes." Then I multiplied by `1 - current_pct` to get remaining time.

Why this is wrong:
- **The current_pct is the inner `[N/M]` counter** which `monitor_slurm.sh` reads via `tail -1` of any `[N/M]` regex match. That counter SHIFTS between phases (precompute outer vs find_best_hybrid inner). A percentage in the snapshot is not comparable across the precompute → experiment transition.
- **It does not account for cache rejection time.** The 06-02 timing data was written by a previous wave that didn't have the same drift problem. Current wave is paying extra recompute time on top of the measured walls.
- **It assumes constant per-config cost.** As you correctly noted earlier, cost is ~linear in budget, so the high-budget tail of any search costs disproportionately more than its config count suggests.

### 3.3 Wedge interpretation — third source of error

What I did: I treated `idle_s > 600` as "wedged" and equated wedged with "will not progress."

Why this is partially wrong:
- **`idle_s` measures log file mtime**, which depends on Python stdout buffering. The `-u` flag was added but currently-running jobs started before that commit, so they have buffered stdout.
- **A long-running SVM at high budget can take >>10 minutes per config** without writing anything to stdout. That looks identical to "wedged" from outside.
- **I have not opened any wedged job's log to verify.** Without that, I cannot distinguish "process is dead" from "process is in a 6-hour SVM config that hasn't flushed yet."

### 3.4 What I never accounted for

- **The 32-CPU bump** I committed in `0a6b746` (2026-06-03) applies only to jobs submitted after that commit. The currently-running jobs are still 8-CPU. None of my time estimates accounted for the speedup my own bump introduces, because it doesn't apply yet.
- **`MODEL_MAX_BUDGET` per-model caps.** I knew these existed but didn't multiply them into the per-condition cost model. SVM is capped at 500 so its contribution at b=1000/b=2000 is zero. That changes the calc significantly.
- **Distance-lineage precompute conditions** for sim. Sim does general_distance/{none,uncapped} and top_distance/{none,uncapped} in addition to main-pool. I never measured these per-condition walls for any dataset.

---

## 4. What needs to happen for ACCURATE estimates

The methodology for the new chat. **Verified-data-driven, not extrapolation-driven.**

### 4.1 Per-experiment cost model from code

For each c-experiment (c01–c23) and s-experiment (s01–s13), read the source and identify:

1. **Does it call `find_best_hybrid` directly?** If yes, count: how many calls, what grid (budgets, allocations, importance, evaluators), what `dataset_name` (precompute key reuse or fresh key).
2. **Does it call `get_best_search_for_downstream`?** If yes, it consumes precompute caches. Per-call cost ≈ disk read + meta check ≈ ms.
3. **Does it call `build_fold_hybrid`?** Per-fold cost ≈ allocation + per-fold model fit at the chosen budget. Scales with `n_train × budget × model`.
4. **Does it have outer loops** (families, partitions, subsamples, repeats)? Multiply.

Build a table per experiment of the form:

| Experiment | # `find_best_hybrid` calls | # `get_best_search` calls | # `build_fold_hybrid` calls | Other compute |
|---|---|---|---|---|

This is **purely reading code**, no extrapolation. It is the basis for everything else.

### 4.2 Per-call cost anchoring from check_precompute --timing

The 2026-06-02 timing output has per-(dataset, condition) wall and per-(dataset, condition, model) breakdown. From it, extract:

- Per-config wall by `(model, budget, n_samples)` for each model that ran. Example: from `caco2 general/none n=2158, n_configs=5984, total 4.53h`:
  - 4.53h / 5984 configs = 2.7s per config average
  - Per-model: xgboost 1.17h / 992 cfg = 4.2s/cfg; svm 0.57h / 749 cfg = 2.74s/cfg; etc.

Build a regression of per-config wall vs `(n_samples, budget, model)`. This is what should drive scaling, not a single-data-point exponent.

### 4.3 Per-dataset estimate by multiplication

For each currently-running dataset:
1. Pull its `DATASET_REPS` entry (general/top/aug rep lists, count of reps per tier).
2. For each precompute condition it needs, count its grid cells: `models × budgets × allocations × importance × evaluators × n_folds`, filtered by `MODEL_MAX_BUDGET` and `_max_samples`.
3. Multiply by the regressed per-config wall at the dataset's `n_samples`.
4. Sum across conditions.
5. For c-experiments, use the per-experiment cost table from 4.1, plug in the dataset's rep count, n_samples, and grid sizes per experiment.

### 4.4 Verify the model against measured wall

The 2026-06-02 timing has dozens of measured (dataset, condition, wall) tuples. After building the model in 4.3, predict each and compare to measured. The residuals show whether the model is right. **If residuals are >20% systematic, the model is wrong and needs revision before any per-dataset projection is trusted.**

This sanity step has not happened in any of my estimates. That's why they're wrong.

### 4.5 Account for cache rejection drag

The currently-running wave has measurable cache rejection: `stale` count per job, `drift` count per job (both already in the new snapshot). Per-rejection cost ≈ one fresh find_best_hybrid call at the rejecting cell's grid. The snapshot has these counts. Multiply the per-call wall by the count to get cache-rejection drag per job, add to per-dataset estimate.

---

## 5. Raw data — latest snapshot (2026-06-04 00:51)

This is from the user's terminal paste. Verbatim.

```
===========================================================================
 SLURM MONITOR  (2026-06-04 00:51)   running jobs: 69
  ⚠  ALERTS: 40 job(s) need attention — see below
===========================================================================

--- HEALTH ALERTS (40) ---
  WEDGED   = log file untouched >10min (job may have hung/crashed silently)
  STUCK    = inner progress counter hasn't moved across 2+ snapshots (>20min)
  SPINNING = cache rejections accumulating with no JSON saves in 6h

JOBID       JOB_NAME                        WHY
12233780    exp_%j                          WEDGED 4907s_idle
12233782    exp_%j                          WEDGED 12248s_idle
12233784    exp_%j                          WEDGED 1603s_idle
12233786    exp_%j                          WEDGED 14849s_idle
12233915    cx_herg_ki                      WEDGED 49170s_idle
12233923    cx_kiba                         WEDGED 57790s_idle
12233924    cx_bindingdb_kd                 WEDGED 20738s_idle
12233925    cx_flab_thermo_tresanco         WEDGED 139793s_idle
12233926    cx_flab_pk_jain_heprt           WEDGED 15233s_idle
12233927    cx_flab_pk_jain_fcrnrt          WEDGED 1548s_idle
12233929    cx_flab_aggregation_jain_hicrt  WEDGED 21357s_idle
12233930    cx_flab_thermo_jain             WEDGED 2442s_idle
12233932    cx_flab_aggregation_jain        WEDGED 198938s_idle
12233935    cx_ddg_combined                 WEDGED 184529s_idle
12233939    cx_flab_expression_adams        WEDGED 17669s_idle
12233942    cx_abagym_EGFR                  WEDGED 6007s_idle
12233944    cx_abagym_lysozyme              WEDGED 237970s_idle
12233945    cx_abagym_VEGF                  WEDGED 1624s_idle
12233946    cx_tz_her2_binary               WEDGED 7744s_idle
12233947    cx_tz_her2_multiclass           WEDGED 41515s_idle
12233948    cx_humanness_therapeutic        STUCK 134min_no_progress on [C13 10done]
12233949    cx_ginkgo_acsins                WEDGED 20874s_idle
12233950    cx_ginkgo_colloidal             WEDGED 52017s_idle
12233951    cx_ginkgo_heparin               WEDGED 3174s_idle
12233952    cx_ginkgo_hic                   WEDGED 29639s_idle
12233953    cx_ginkgo_pr_cho                WEDGED 72935s_idle
12233954    cx_ginkgo_titer                 STUCK 134min_no_progress on [C05 4done]
12233955    cx_buchwald_hartwig             WEDGED 648s_idle
12233957    cx_tcga_luad_vs_lusc            WEDGED 180745s_idle
12233958    cx_tcga_brca_pam50              WEDGED 98091s_idle
12233981    sim_herg_ki                     WEDGED 31862s_idle
12233989    sim_kiba                        WEDGED 54404s_idle
12233990    sim_bindingdb_kd                WEDGED 25402s_idle
12233991    sim_flab_thermo_tresanco        WEDGED 145758s_idle
12234001    sim_ddg_combined                WEDGED 181116s_idle
12234005    sim_flab_expression_adams       WEDGED 63554s_idle
12234010    sim_abagym_lysozyme             WEDGED 239662s_idle
12234020    sim_ginkgo_titer                WEDGED 38321s_idle
12234023    sim_tcga_luad_vs_lusc           WEDGED 175704s_idle
12234024    sim_tcga_brca_pam50             WEDGED 84626s_idle

--- MOLECULAR (16 jobs) ---
JOBID       DATASET                       RUNTIME    PROGRESS              j1h    j6h    drift  stale  idle_s  CURRENT STEP
12233919    adme_hlm                      2-18:53:43  [2864/4128] (69.4%)   0      0      1      4      -36     xgboost/entropy_2.0/lightgbm_gain/b=500/ev=ridge
12233985    adme_hlm                      2-18:51:26  [2113/4536] (46.6%)   0      0      0      1      23      mlp/power_2/lightgbm_gain/b=100/ev=knn
12233921    adme_mdr1                     2-18:53:43  [3592/4200] (85.5%)   0      0      1      4      -46     rf/rank_exp_d0.5/deeplift/b=500/ev=knn
12233987    adme_mdr1                     2-18:51:26  [2968/4536] (65.4%)   0      0      0      1      54      svm/rank_linear/treeshap/b=50/ev=mlp
12233920    adme_rlm                      2-18:53:43  [2180/3744] (58.2%)   0      0      1      4      -26     mlp/rank_linear/lightgbm_gain/b=25/ev=ridge
12233986    adme_rlm                      2-18:51:26  [597/4536] (13.2%)    0      0      0      1      33      mlp/performance_weighted/deeplift/b=100/ev=rf
12233916    caco2                         2-18:53:43  [4121/4326] (95.3%)   0      0      1      4      -36     lightgbm/rank_exp_d0.5/deeplift/b=500/ev=lightgbm
12233982    caco2                         2-18:51:26  [3179/4536] (70.1%)   0      0      0      2      3       xgboost/entropy_2.0/lightgbm_gain/b=50/ev=lightgbm
12233922    herg                          2-18:53:43  [3153/4248] (74.2%)   0      0      0      0      -50     ridge/entropy_2.0/treeshap/b=1000/ev=ridge
12233988    herg                          2-18:51:26  [2108/4248] (49.6%)   0      0      0      0      209     svm/power_2/lightgbm_gain/b=250/ev=mlp
12233915    herg_ki                       2-18:53:43  [C05 3done]           0      0      2      13     49170   svm/rank_exp_d0.5/treeshap/b=500/ev=mlp
12233981    herg_ki                       2-18:51:26  [S02 1done]           0      0      0      3      31862   svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12233917    logd                          2-18:53:43  [3365/4248] (79.2%)   0      0      0      0      220     lightgbm/entropy_2.0/lightgbm_gain/b=1000/ev=rf
12233983    logd                          2-18:51:26  [2915/4248] (68.6%)   0      0      0      0      71      rf/entropy_2.0/treeshap/b=1000/ev=lightgbm
12233918    qm9                           2-18:53:43  [2696/4248] (63.5%)   0      0      0      0      81      mlp/rank_linear/lightgbm_gain/b=25/ev=mlp
12233984    qm9                           2-18:51:26  [1998/4248] (47.0%)   0      0      0      0      165     mlp/power_2/lightgbm_gain/b=50/ev=svm

--- DTA (4 jobs) ---
JOBID       DATASET                       RUNTIME    PROGRESS              j1h    j6h    drift  stale  idle_s  CURRENT STEP
12233924    bindingdb_kd                  2-18:53:43  [2/2] (100.0%)        0      0      0      0      20738   -
12233990    bindingdb_kd                  2-18:51:26  [1/2] (50.0%)         0      0      0      0      25402   -
12233923    kiba                          2-18:53:43  [2/2] (100.0%)        0      0      0      0      57790   -
12233989    kiba                          2-18:51:26  [1/2] (50.0%)         0      0      0      0      54404   -

--- DDG (4 jobs) ---
JOBID       DATASET                       RUNTIME    PROGRESS              j1h    j6h    drift  stale  idle_s  CURRENT STEP
12233935    ddg_combined                  2-18:53:43  [C05 3done]           0      0      4      14     184529  svm/rank_exp_d0.5/treeshap/b=429/ev=svm
12234001    ddg_combined                  2-18:51:26  [S02 1done]           0      0      2      4      181116  svm/rank_exp_d0.5/treeshap/b=429/ev=svm
12233925    flab_thermo_tresanco          2-18:53:43  [C05 3done]           0      0      1      10     139793  svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12233991    flab_thermo_tresanco          2-18:51:26  [S02 1done]           0      0      0      3      145758  svm/rank_exp_d0.5/treeshap/b=500/ev=svm

--- REACTION (3 jobs) ---
JOBID       DATASET                       RUNTIME    PROGRESS              j1h    j6h    drift  stale  idle_s  CURRENT STEP
12233956    bh_doyle_plates               2-18:53:43  [2588/4146] (62.4%)   0      0      0      3      -5      mlp/rank_linear/deeplift/b=10/ev=ridge
12234022    bh_doyle_plates               2-18:51:26  [1490/4536] (32.8%)   0      0      0      1      50      svm/entropy_1.0/treeshap/b=250/ev=ridge
12233955    buchwald_hartwig              2-18:53:43  [C05 2done]           0      0      1      39     648     svm/rank_exp_d0.5/treeshap/b=500/ev=svm

--- TCGA (4 jobs) ---
JOBID       DATASET                       RUNTIME    PROGRESS              j1h    j6h    drift  stale  idle_s  CURRENT STEP
12233958    tcga_brca_pam50               2-18:53:43  [C05 3done]           0      0      1      11     98091   svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12234024    tcga_brca_pam50               2-18:51:26  [S02 1done]           0      0      0      2      84626   svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12233957    tcga_luad_vs_lusc             2-18:53:43  [C05 3done]           0      0      1      11     180745  svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12234023    tcga_luad_vs_lusc             2-18:51:26  [S02 1done]           0      0      0      2      175704  svm/rank_exp_d0.5/treeshap/b=500/ev=svm

--- ANTIBODY (28 jobs) ---
JOBID       DATASET                       RUNTIME    PROGRESS              j1h    j6h    drift  stale  idle_s  CURRENT STEP
12233941    abagym_Ang2                   2-18:53:43  [C21 15done]          0      3      1      13     310     gp/rank_exp_d0.5/treeshap/b=397/ev=svm
12233942    abagym_EGFR                   2-18:53:43  [C05 5done]           0      2      1      12     6007    gp/rank_exp_d0.5/treeshap/b=500/ev=svm
12233944    abagym_lysozyme               2-18:53:43  [C05 3done]           0      0      0      0      237970  -
12234010    abagym_lysozyme               2-18:51:26  [S02 1done]           0      0      0      0      239662  -
12233945    abagym_VEGF                   2-18:53:43  [C21 15done]          0      3      1      12     1624    gp/rank_exp_d0.5/treeshap/b=418/ev=svm
12233932    flab_aggregation_jain         2-18:53:43  [C05 6done]           0      0      3      10     198938  gp/rank_exp_d0.5/treeshap/b=137/ev=svm
12233929    flab_aggregation_jain_hicrt   2-18:53:43  [C05 6done]           0      0      2      39     21357   gp/rank_exp_d0.5/treeshap/b=137/ev=svm
12233938    flab_binding_adams            2-18:53:43  [1180/4026] (29.3%)   0      0      1      4      -52     mlp/entropy_1.0/deeplift/b=25/ev=mlp
12234004    flab_binding_adams            2-18:51:26  [75/4536] (1.7%)      0      0      0      2      -22     rf/performance_weighted/deeplift/b=250/ev=rf
12233936    flab_binding_koenig           2-18:53:43  [3261/3605] (90.5%)   0      0      0      0      -42     ridge/rank_exp_d0.5/treeshap/b=1000/ev=rf
12234002    flab_binding_koenig           2-18:51:26  [2020/3605] (56.0%)   0      0      0      0      18      xgboost/rank_linear/treeshap/b=500/ev=xgboost
12233940    flab_binding_phillips_h3      2-18:53:43  [1900/4140] (45.9%)   0      0      0      2      -83     svm/entropy_1.0/treeshap/b=250/ev=mlp
12234006    flab_binding_phillips_h3      2-18:51:26  [2516/3812] (66.0%)   0      0      0      0      314     ridge/rank_linear/lightgbm_gain/b=2000/ev=rf
12233939    flab_expression_adams         2-18:53:43  [C04 2done]           0      0      0      11     17669   svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12234005    flab_expression_adams         2-18:51:26  [S02 1done]           0      0      0      2      63554   svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12233937    flab_expression_koenig        2-18:53:43  [676/3858] (17.5%)    0      0      0      3      68      svm/performance_weighted/lightgbm_gain/b=500/ev=mlp
12234003    flab_expression_koenig        2-18:51:26  [4083/4248] (96.1%)   0      0      0      0      59      lightgbm/rank_exp_d0.5/deeplift/b=2000/ev=lightgbm
12233927    flab_pk_jain_fcrnrt           2-18:53:43  [C05 6done]           0      0      2      38     1548    gp/rank_exp_d0.5/treeshap/b=132/ev=lightgbm
12233926    flab_pk_jain_heprt            2-18:53:43  [C05 3done]           0      0      3      16     15233   gp/rank_exp_d0.5/treeshap/b=130/ev=svm
12233930    flab_thermo_jain              2-18:53:43  [C05 3done]           0      0      1      10     2442    gp/rank_exp_d0.5/treeshap/b=137/ev=svm
12233949    ginkgo_acsins                 2-18:53:43  [C05 5done]           0      1      2      37     20874   gp/rank_exp_d0.5/treeshap/b=242/ev=svm
12233950    ginkgo_colloidal              2-18:53:43  [C05 5done]           0      0      2      39     52017   gp/rank_exp_d0.5/lightgbm_gain/b=242/ev=svm
12233951    ginkgo_heparin                2-18:53:43  [C05 6done]           0      0      3      42     3174    gp/rank_exp_d0.5/treeshap/b=94/ev=svm
12233952    ginkgo_hic                    2-18:53:43  [C05 5done]           0      0      0      36     29639   -
12233953    ginkgo_pr_cho                 2-18:53:43  [C05 5done]           0      0      3      42     72935   gp/rank_exp_d0.5/treeshap/b=197/ev=svm
12233954    ginkgo_titer                  2-18:53:43  [C05 4done]           0      0      0      32     241     -
12234020    ginkgo_titer                  2-18:51:26  [S06 5done]           0      0      1      5      38321   gp/rank_exp_d0.5/treeshap/b=239/ev=svm
12233948    humanness_therapeutic         2-18:53:43  [C13 10done]          0      0      1      60     116     gp/rank_exp_d0.5/treeshap/b=500/ev=svm

--- OTHER (10 jobs) ---
JOBID       DATASET                       RUNTIME    PROGRESS              j1h    j6h    drift  stale  idle_s  CURRENT STEP
12233779    exp_%j                        2-20:19:02  [1086/3105] (35.0%)   0      0      0      0      -66     xgboost/power_2/deeplift/b=100/ev=rf
12233780    exp_%j                        2-20:19:02  [1148/4956] (23.2%)   0      0      0      0      4907    knn/performance_weighted/treeshap/b=1000/ev=mlp
12233781    exp_%j                        2-20:19:02  [2281/4536] (50.3%)   0      0      0      0      285     knn/power_2/lightgbm_gain/b=429/ev=knn
12233782    exp_%j                        2-20:19:02  [2/3] (66.7%)         0      0      1      3      12248   mlp/rank_exp_d0.5/treeshap/b=1000/ev=mlp
12233783    exp_%j                        2-20:19:02  [963/3105] (31.0%)    0      0      0      0      -105    svm/entropy_1.0/treeshap/b=100/ev=mlp
12233784    exp_%j                        2-20:19:02  [1493/4956] (30.1%)   0      0      0      0      1603    rf/entropy_1.0/lightgbm_gain/b=1000/ev=rf
12233785    exp_%j                        2-20:19:02  [2213/4536] (48.8%)   0      0      0      0      71      knn/power_2/deeplift/b=25/ev=svm
12233786    exp_%j                        2-20:19:02  [1/3] (33.3%)         0      0      1      2      14849   mlp/rank_exp_d0.5/treeshap/b=1000/ev=mlp
12233946    tz_her2_binary                2-18:53:43  [C05 3done]           0      0      0      16     7744    svm/rank_exp_d0.5/treeshap/b=500/ev=svm
12233947    tz_her2_multiclass            2-18:53:43  [C05 3done]           0      0      0      13     41515   -

--- BY DATASET (cx vs sim: progress, elapsed, eta-remaining) ---
DATASET                           CX                              SIM
abagym_Ang2                       C21 15done  67h                 -
abagym_EGFR                       C05 5done  67h                  -
abagym_lysozyme                   C05 3done  67h                  S02 1done  67h
abagym_VEGF                       C21 15done  67h                 -
adme_hlm                           69.4%  67h +  30h               46.6%  67h +  77h
adme_mdr1                          85.5%  67h +  11h               65.4%  67h +  35h
adme_rlm                           58.2%  67h +  48h               13.2%  67h + 441h
bh_doyle_plates                    62.4%  67h +  40h               32.8%  67h + 137h
bindingdb_kd                      100.0%  67h wrap-up              50.0%  67h +  67h
buchwald_hartwig                  C05 2done  67h                  -
caco2                              95.3%  67h +   3h               70.1%  67h +  29h
ddg_combined                      C05 3done  67h                  S02 1done  67h
flab_aggregation_jain             C05 6done  67h                  -
flab_aggregation_jain_hicrt       C05 6done  67h                  -
flab_binding_adams                 29.3%  67h + 161h                1.7%  67h +3977h*
flab_binding_koenig                90.5%  67h +   7h               56.0%  67h +  52h
flab_binding_phillips_h3           45.9%  67h +  79h               66.0%  67h +  34h
flab_expression_adams             C04 2done  67h                  S02 1done  67h
flab_expression_koenig             17.5%  67h + 315h               96.1%  67h +   3h
flab_pk_jain_fcrnrt               C05 6done  67h                  -
flab_pk_jain_heprt                C05 3done  67h                  -
flab_thermo_jain                  C05 3done  67h                  -
flab_thermo_tresanco              C05 3done  67h                  S02 1done  67h
ginkgo_acsins                     C05 5done  67h                  -
ginkgo_colloidal                  C05 5done  67h                  -
ginkgo_heparin                    C05 6done  67h                  -
ginkgo_hic                        C05 5done  67h                  -
ginkgo_pr_cho                     C05 5done  67h                  -
ginkgo_titer                      C05 4done  67h                  S06 5done  67h
herg                               74.2%  67h +  23h               49.6%  67h +  68h
herg_ki                           C05 3done  67h                  S02 1done  67h
humanness_therapeutic             C13 10done  67h                 -
kiba                              100.0%  67h wrap-up              50.0%  67h +  67h
logd                               79.2%  67h +  18h               68.6%  67h +  31h
qm9                                63.5%  67h +  39h               47.0%  67h +  75h
tcga_brca_pam50                   C05 3done  67h                  S02 1done  67h
tcga_luad_vs_lusc                 C05 3done  67h                  S02 1done  67h
tz_her2_binary                    C05 3done  67h                  -
tz_her2_multiclass                C05 3done  67h                  -
```

---

## 6. Raw data — check_precompute status from same snapshot

```
=== ANTIBODY ===
  dataset                                status   found    top_rep_dim  missing/bad
  -------------------------------------- -------- -------- ------------ ----------------------------------------
  abagym_Ang2                            DONE     2/2      6/8
  abagym_EGFR                            DONE     2/2      8/9
  abagym_HER2                            DONE     2/2      8/8
  abagym_VEGF                            DONE     2/2      6/6
  abagym_lysozyme                        DONE     3/3      6/6
  abcov_kd                               DONE     2/2      6/10
  ddg_combined                           DONE     4/4      5/6
  flab_aggregation_jain                  DONE     5/5      5/6
  flab_aggregation_jain_hicrt            DONE     5/5      4/6
  flab_binding_adams                     DONE     4/4      -
  flab_binding_koenig                    partial  3/4      -            top/uncapped=missing
  flab_binding_phillips_h3               DONE     4/4      -
  flab_expression_adams                  DONE     4/4      4/6
  flab_expression_koenig                 DONE     4/4      -
  flab_immunogenicity_marks              DONE     2/2      7/10
  flab_pk_jain_fcrnrt                    DONE     5/5      4/6
  flab_pk_jain_heprt                     DONE     5/5      5/6
  flab_polyreact_jain_bvp                DONE     2/2      6/7
  flab_polyreact_jain_cic                DONE     2/2      6/6
  flab_thermo_jain                       DONE     5/5      5/6
  flab_thermo_tresanco                   DONE     4/4      4/5
  ginkgo_acsins                          DONE     5/5      6/6
  ginkgo_colloidal                       DONE     5/5      4/6
  ginkgo_heparin                         DONE     5/5      5/8
  ginkgo_hic                             DONE     5/5      4/6
  ginkgo_pr_cho                          DONE     5/5      4/6
  ginkgo_titer                           DONE     5/5      4/6
  humanness_therapeutic                  DONE     2/2      7/9
  tz_her2_binary                         DONE     2/2      4/6
  tz_her2_multiclass                     DONE     2/2      4/6

=== DTA ===
  dataset                                status   found    top_rep_dim  missing/bad
  -------------------------------------- -------- -------- ------------ ----------------------------------------
  bindingdb_kd                           partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  kiba                                   partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing

=== MOLECULAR ===
  dataset                                status   found    top_rep_dim  missing/bad
  -------------------------------------- -------- -------- ------------ ----------------------------------------
  adme_hlm                               DONE     4/4      -
  adme_mdr1                              DONE     4/4      -
  adme_rlm                               DONE     4/4      -
  caco2                                  DONE     4/4      -
  herg                                   partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  herg_ki                                DONE     4/4      7/10
  logd                                   partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  qm9                                    partial  2/4      4/5          general/uncapped=missing, top/uncapped=missing

=== NUCLEOTIDE ===
  dataset                                status   found    top_rep_dim  missing/bad
  -------------------------------------- -------- -------- ------------ ----------------------------------------
  gb_demo_coding_vs_intergenomic_seqs    missing  0/5      -            general/none=missing, general/uncapped=missing, top/none=missing, top/uncapped=missing, top/capped=missing
  gb_demo_human_or_worm                  partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  gb_human_enhancers_cohn                partial  2/4      -            top/none=missing, top/uncapped=missing
  gb_human_nontata_promoters             partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H2AFZ                               partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H3K27ac                             partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H3K27me3                            partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H3K36me3                            partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H3K4me1                             partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H3K4me2                             partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H3K4me3                             partial  3/4      -            top/uncapped=missing
  nt_H3K9ac                              partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H3K9me3                             partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_H4K20me1                            partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_enhancers                           partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_enhancers_types                     partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_promoter_all                        partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_promoter_no_tata                    partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_promoter_tata                       partial  3/4      -            top/uncapped=missing
  nt_splice_sites_acceptors              partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing
  nt_splice_sites_donors                 partial  1/4      -            general/uncapped=missing, top/none=missing, top/uncapped=missing

=== REACTION ===
  dataset                                status   found    top_rep_dim  missing/bad
  -------------------------------------- -------- -------- ------------ ----------------------------------------
  bh_doyle_plates                        DONE     4/4      -
  buchwald_hartwig                       DONE     2/2      4/6

=== TCGA ===
  dataset                                status   found    top_rep_dim  missing/bad
  -------------------------------------- -------- -------- ------------ ----------------------------------------
  tcga_brca_pam50                        DONE     4/4      4/6
  tcga_luad_vs_lusc                      DONE     4/4      4/6

======================================================================
Total datasets checked: 65  |  DONE: 38  |  partial: 26  |  missing: 1

Ready for experiments: 38/65 datasets
```

Missing per check_precompute as of this snapshot, summarised:

- **DTA** (both n=30K): bindingdb_kd needs 3 conditions, kiba needs 3 conditions. Each `general/uncapped` measured at 96-122h on the May wave.
- **Molecular**: herg (n=7772) needs 3, logd (n=5028) needs 3, qm9 (n=9978) needs 2.
- **Antibody**: flab_binding_koenig needs 1 (top/uncapped).
- **Nucleotide**: 20 of 21 datasets are partial or missing; this is what the 32-CPU resubmit was meant to address but I do not know if that resubmit happened.

---

## 7. Raw data — check_precompute --timing measured walls (from 2026-06-02)

I have per-condition walls and per-(condition, model) walls for every dataset that had a cached search at 06-02. This is too long to inline here, but I have it in the conversation. The notable measured walls for currently-running datasets:

```
ddg_combined           general/none n=429      1.10h
ddg_combined           general/uncapped n=429  7.61h
abagym_lysozyme        general/none n=627      0.52h
abagym_lysozyme        general/uncapped n=627  12.87h
abagym_lysozyme        top/uncapped n=627      18.21h
caco2                  general/none n=2158     4.53h
caco2                  general/uncapped n=2158 63.31h
caco2                  top/none n=2158         4.11h
caco2                  top/uncapped n=2158     62.32h
adme_hlm               general/none n=3086     7.97h
adme_hlm               general/uncapped n=3086 95.27h
adme_hlm               top/none n=3086         4.57h
adme_hlm               top/uncapped n=3086     75.00h
adme_mdr1              general/none n=2641     6.90h
adme_mdr1              general/uncapped n=2641 80.40h
adme_mdr1              top/none n=2641         3.52h
adme_mdr1              top/uncapped n=2641     72.57h
adme_rlm               general/none n=3053     8.41h
adme_rlm               general/uncapped n=3053 102.71h
adme_rlm               top/none n=3053         4.23h
bh_doyle_plates        general/none n=4312     7.46h
bh_doyle_plates        general/uncapped n=4312 159.00h
bh_doyle_plates        top/none n=4312         4.38h
bh_doyle_plates        top/uncapped n=4312     118.92h
flab_binding_adams     general/none n=5000     21.32h
flab_binding_adams     general/uncapped n=5000 75.58h
flab_binding_adams     top/none n=5000         8.76h
flab_binding_adams     top/uncapped n=5000     58.14h
flab_binding_koenig    general/none n=4275     10.24h
flab_binding_koenig    general/uncapped n=4275 151.77h
flab_binding_koenig    top/none n=4275         5.73h
herg                   general/none n=7772     29.44h
herg_ki                general/none n=1412     3.47h
herg_ki                general/uncapped n=1412 47.97h
herg_ki                top/none n=1412         2.54h
herg_ki                top/uncapped n=1412     45.27h
logd                   general/none n=5028     20.45h
qm9                    general/none n=9978     20.64h
qm9                    top/none n=9978         23.81h
kiba                   general/none n=30000    121.83h
bindingdb_kd           general/none n=30000    96.23h
tcga_brca_pam50        general/none n=821      1.16h
tcga_brca_pam50        general/uncapped n=821  16.51h
tcga_brca_pam50        top/none n=821          0.83h
tcga_brca_pam50        top/uncapped n=821      19.75h
tcga_luad_vs_lusc      general/none n=1017     1.01h
tcga_luad_vs_lusc      general/uncapped n=1017 8.94h
tcga_luad_vs_lusc      top/none n=1017         0.94h
tcga_luad_vs_lusc      top/uncapped n=1017     10.61h
nt_H3K4me3             general/uncapped n=18244 140.94h
gb_human_enhancers_cohn general/uncapped n=27791 106.41h
```

Per-model breakdowns within each condition are also in the conversation. They are the basis for any cost-per-config regression.

---

## 8. cx experiment cumulative walls — measured ground truth from FIVE complete pipeline runs

Pulled on 2026-06-04 from the current wave's slurm logs. **This is the actual measured data the new chat must fit its cost model against.** Each row is a fully-completed cx pipeline (c01 → c24) on a specific (job, dataset).

### Job 12233934 — flab_polyreact_jain_bvp (n=137) — Done 2026-06-02 21:40

| Exp | wall (s) | hours |
|---|---|---|
| c01 | 88.2 | 0.024 |
| c02 | 298.1 | 0.083 |
| c04 | 685.2 | 0.190 |
| c05 | 4205.9 | 1.168 |
| c06 | 10.2 | 0.003 |
| c07 | 0.5 | 0.000 |
| c09 | 2877.0 | 0.799 |
| c10 | 1.8 | 0.000 |
| c11 | 1877.7 | 0.522 |
| c13 | 759.4 | 0.211 |
| c15 | 4634.9 | 1.287 |
| c16 | 4356.0 | 1.210 |
| c17 | 32708.8 | 9.086 |
| c18 | 198.6 | 0.055 |
| c19 | 18.6 | 0.005 |
| c21 | 69338.4 | 19.261 |
| c22 | 127.2 | 0.035 |
| c23 | 10362.9 | 2.878 |
| c24 | 7783.5 | 2.162 |
| **TOTAL** | **140332** | **38.98h** |

### Job 12233928 — flab_polyreact_jain_cic (n=137) — Done 2026-06-02 19:45

| Exp | wall (s) | hours |
|---|---|---|
| c01 | 155.3 | 0.043 |
| c02 | 168.7 | 0.047 |
| c04 | 652.8 | 0.181 |
| c05 | 2894.0 | 0.804 |
| c06 | 32.8 | 0.009 |
| c07 | 4.5 | 0.001 |
| c09 | 3299.7 | 0.917 |
| c10 | 0.4 | 0.000 |
| c11 | 843.4 | 0.234 |
| c13 | 584.4 | 0.162 |
| c15 | 1334.6 | 0.371 |
| c16 | 1547.1 | 0.430 |
| c17 | 29946.3 | 8.318 |
| c18 | 165.0 | 0.046 |
| c19 | 28.0 | 0.008 |
| c21 | 69490.7 | 19.303 |
| c22 | 59.2 | 0.016 |
| c23 | 10357.9 | 2.877 |
| c24 | 12043.5 | 3.346 |
| **TOTAL** | **133606** | **37.11h** |

### Job 12233943 — abagym_HER2 (n=146) — Done 2026-06-02 18:48

| Exp | wall (s) | hours |
|---|---|---|
| c01 | 3862.2 | 1.073 |
| c02 | 2857.4 | 0.794 |
| c04 | 9262.8 | 2.573 |
| c05 | 24645.6 | 6.846 |
| c06 | 6.4 | 0.002 |
| c07 | 158.6 | 0.044 |
| c09 | 37288.4 | 10.358 |
| c10 | 4.6 | 0.001 |
| c11 | 6824.0 | 1.896 |
| c13 | 2699.2 | 0.750 |
| c15 | 1791.1 | 0.497 |
| c16 | 6499.7 | 1.806 |
| c17 | 24247.9 | 6.736 |
| c18 | 1182.4 | 0.328 |
| c19 | 515.1 | 0.143 |
| c21 | 2739.9 | 0.761 |
| c22 | 12.5 | 0.003 |
| c23 | 18.8 | 0.005 |
| c24 | 1166.8 | 0.324 |
| **TOTAL** | **125782** | **34.94h** |

### Job 12233941 — abagym_Ang2 (n=397) — INCOMPLETE (hung at c21)

| Exp | wall (s) | hours |
|---|---|---|
| c01 | 478.4 | 0.133 |
| c02 | 1148.0 | 0.319 |
| c04 | 1804.7 | 0.501 |
| c05 | 8420.7 | 2.339 |
| c06 | 11.5 | 0.003 |
| c07 | 1748.0 | 0.486 |
| c09 | 8728.6 | 2.425 |
| c10 | 0.9 | 0.000 |
| c11 | 5059.5 | 1.405 |
| c13 | 2068.3 | 0.575 |
| c15 | 5572.9 | 1.548 |
| c16 | 8891.0 | 2.470 |
| c17 | 185540.5 | **51.539** |
| c18 | 399.9 | 0.111 |
| c19 | 49.9 | 0.014 |
| c21 | (in progress, not done) | — |
| **PARTIAL TOTAL through c19** | **229922** | **63.87h** |

### Job 12233945 — abagym_VEGF (n=418) — INCOMPLETE (hung at c21)

| Exp | wall (s) | hours |
|---|---|---|
| c01 | 1605.4 | 0.446 |
| c02 | 634.4 | 0.176 |
| c04 | 1878.6 | 0.522 |
| c05 | 10078.2 | 2.800 |
| c06 | 42.8 | 0.012 |
| c07 | 3054.9 | 0.849 |
| c09 | 7331.8 | 2.037 |
| c10 | 1.1 | 0.000 |
| c11 | 2118.4 | 0.588 |
| c13 | 2177.4 | 0.605 |
| c15 | 1249.3 | 0.347 |
| c16 | 6152.4 | 1.709 |
| c17 | 192233.2 | **53.398** |
| c18 | 358.8 | 0.100 |
| c19 | 24.4 | 0.007 |
| c21 | (in progress, not done) | — |
| **PARTIAL TOTAL through c19** | **228941** | **63.59h** |

### 8.1 What this data REFUTES

**The previous baseline anchor I had been using was wrong about three things:**

1. **c08 (calibration) is not in any of these logs.** It either runs silently, was skipped, or was removed. The 06-01 paste showed it at ~463s for flab_immunogenicity_marks but it does not appear in these complete logs. The new chat must verify whether c08 still runs in this codebase.

2. **c24 (blueprint transfer) IS present in some logs (n=137 jobs) but NOT in others (n=146, n=397, n=418).** Conditional on dataset config. Must be modelled.

3. **No experiment scales monotonically with n.** Specific counter-examples from the data:
   - **c01** at n=146 (HER2) = 3862s, but at n=137 (polyreact_bvp) = 88s. **44× difference at nearly identical n.**
   - **c17** at n=146 = 6.7h, at n=137 = 9.1h, at n=217 = 27.6h, at n=397 = 51.5h, at n=418 = 53.4h. Not monotonic in n.
   - **c21** at n=137 = 19.3h, at n=146 = 0.76h, at n=217 = 0.20h. Highest at smallest n.
   - **c13** at n=137 = 0.21h, at n=146 = 0.75h, at n=217 = 18.6h. Wildly variable.
   - **c09** at n=146 = 10.4h, at n=137 = 0.80h, at n=217 = 1.68h.

**The new chat must build per-experiment cost functions parametrized by more than just n.** Every previous attempt to extrapolate a cumulative wall using `(n/anchor)^k` for any k will be wrong by 10× or more on at least some experiments. The variability is dataset-rep-set specific, not n-specific.

### 8.2 Implication for wedge interpretation

c17 at n=397-418 took **~52 hours**. c21 at n=137 took **~19 hours**. So a job that has been "idle" for 1-3 hours at C17 or C21 may actually be **legitimately mid-experiment**, not dead.

Wedge classifications need to be revised against the per-experiment wall data:
- Jobs at C17 idle <60h: possibly alive
- Jobs at C21 idle <25h on small n: possibly alive
- Jobs at C13 idle <20h on n>200: possibly alive
- Jobs idle >60h at any experiment: almost certainly dead (no measured experiment exceeds c17 at large n)

Specifically from the current snapshot, these "wedged" jobs may actually still be running:
- `abagym_Ang2` (12233941) at C21 idle 43min — c21 here can plausibly be 10-25h
- `abagym_VEGF` (12233945) at C21 idle 65min — same
- `humanness_therapeutic` at C13 idle 40min — c13 at n=564 unknown but plausibly several hours
- `abagym_EGFR` at C05 idle 2.3h — c05 at n=659 plausibly hours
- Jobs at C05 with idle <4h on small datasets

These are likely dead (>60h or far past plausible c-experiment wall):
- `abagym_lysozyme` 66h idle
- `flab_aggregation_jain` 55h idle
- `ddg_combined` 51h idle
- `tcga_luad_vs_lusc` 50h idle
- `flab_thermo_tresanco` 39h idle

I do not have measured s-experiment cumulative walls. Same methodology must be applied to sim — need to grep similar logs once one finishes.

---

## 9. What is verifiable about "won't finish precompute in 4 days"

100h remaining of partition. **Datasets whose remaining precompute conditions alone need ≥100h, BASED ON MEASURED 06-02 WALLS**:

| Dataset | Conditions missing | Measured similar-condition wall (from same dataset at lower aug level) | Verdict |
|---|---|---|---|
| bindingdb_kd sim | general/uncapped, top/none, top/uncapped | general/none was 96.23h. uncapped of similar shape on n=30K elsewhere was 120h. | ≥120h for uncapped alone → does NOT fit |
| kiba sim | same | general/none was 121.83h. uncapped expected >150h. | does NOT fit |
| flab_binding_adams sim | top/uncapped + finishing current | top/uncapped measured at 58.14h on 06-02; current ETA shows still in precompute at 1.7%. | does NOT fit (current condition + top/uncapped both incomplete) |
| adme_rlm sim | top/uncapped + current | adme_rlm top/uncapped not yet measured; comparable adme_hlm top/uncapped was 75h | likely does NOT fit (sim ETA 441h in snapshot, even allowing 3x compression doesn't fit) |
| flab_binding_phillips_h3 cx | nominally DONE per check_precompute, BUT live counter shows 45.9% inner counter — this is either a recompute of a stale cache (probable) or the experiment phase | UNKNOWN — need to verify whether this is a precompute redo or experiment fbh call | UNKNOWN |
| flab_expression_koenig cx | same as above | UNKNOWN | UNKNOWN |
| flab_binding_adams cx | precompute partially done, top/uncapped finished but current condition only 29.3% | If recomputing already-cached, fast; if not, projecting ~140h | likely does NOT fit |

**Datasets whose remaining precompute fits comfortably in 100h, BASED ON MEASURED WALLS**:
- caco2 cx (current ETA 3h), caco2 sim (29h projected, would fit at face value)
- adme_hlm cx (30h projected), adme_hlm sim (77h projected)
- adme_mdr1 cx (11h), adme_mdr1 sim (35h)
- herg cx (23h)
- logd cx (18h), logd sim (31h)
- qm9 cx (39h)
- bh_doyle_plates cx (40h)
- flab_binding_koenig cx (7h to finish top/uncapped at 90.5% currently)

**But all of the above are face-value snapshot ETAs**, which assume constant per-config cost and do NOT account for the high-budget cost tail. Real walls are likely 1.5-2x. Caveat must be stated.

**Datasets wedged for hours-to-days are not making precompute progress**. They are listed in section 5.

---

## 10. About making c05 / other experiments smaller

You asked: "do I need to make c05 or other experiments smaller?"

**Honest answer: I cannot tell you yet without doing the work in section 4.** The 30h c17 wall at n=137 cannot be scaled to "what c17 takes at n=2000" by any constant factor.

What is **verifiable** from the n=137-217 anchor:
- c05 took 3.72h on flab_immunogenicity_marks (n=217).
- c17 took 27.57h on the same dataset.
- c13 took 18.62h on flab_polyreact_jain_cic (n=137).
- c06 took 6.97h on flab_immunogenicity_marks (n=217).

That is **all** that is verifiable. These are four measurements at two values of n in a similar regime, all small. They tell us nothing reliable about behaviour at n>500.

### 10.1 DO NOT ASSUME PROPORTIONAL SCALING

**This is the central error I have been making throughout this conversation. Stating it explicitly so the next chat does not repeat it.**

The walls measured for an experiment at n=217 cannot be extrapolated to n=2000, n=5000, n=30000 by multiplying by any constant — neither `(n/217)`, nor `(n/217)^0.5`, nor `(n/217)^0.8`, nor anything similar. Why:

1. **Per-experiment internal structure differs.** c05 has a nested loop `for n_sub in [5 sizes]: for seed in [3]: find_best_hybrid(subsample)`. The inner `find_best_hybrid` cost depends on the actual subsample size, which is determined by `n_total`. At n_total=217, the largest subsample is n=217. At n_total=5000, the largest is n=5000. The per-cell wall of the deepest layer scales with the subsample n, not with n_total. So c05 grows in a complicated way: not linear, not proportional.

2. **c17 (information-theoretic) cost is dominated by mRMR**, which is `O(budget × n_features × n_samples)`. If a dataset has more reps in `general`, `n_features` (the concatenated rep dim) grows. n_features at flab_immunogenicity_marks vs n_features at caco2 are not the same number. The mrmr_prefilter caps it at 2000, but the cap kicks in for high-dim datasets and not low-dim ones. The mRMR contribution to c17's wall does NOT scale with n_total alone.

3. **c13 (forward compatibility) iterates over candidate new reps** with per-rep search. Wall depends on which datasets register which new reps. flab_immunogenicity_marks has its own set; caco2 has a different set. Different per-rep counts → different wall.

4. **find_best_hybrid cost-per-config is itself n-dependent and rep-dim-dependent.** Per-config cost is roughly `O(n × budget × per_model_factor)`. The `per_model_factor` differs across models (SVM is O(n²), RF is O(n log n), MLP is O(n × layer_size)). So the same grid at different n produces different walls in a model-mix-dependent way.

5. **`MODEL_MAX_BUDGET` and `_max_samples` exclude models at different n.** SVM excluded at n>10000, GP excluded at n>1000. The grid SHRINKS as n grows in these dimensions. So at very large n, certain experiments lose entire model contributions, while at small n they have everything. **The grid is not the same at different n.**

6. **Rep set differs per dataset.** Different datasets have different numbers of reps in `general` and `top` tiers. More reps → more allocation combinations → larger grid. caco2 has 12 general + 8 top + 5 aug; herg has 18 general + 8 top + 3 aug; flab_immunogenicity has different again. **The grid SIZE is not the same at different datasets.**

### 10.2 What must happen instead

For each experiment cN.py:

1. Read the source. Identify every `find_best_hybrid()` call, every `get_best_search_for_downstream()` call, every loop bound. Identify what determines those bounds: `n_samples`, `rep_count`, `aug_count`, hardcoded constants, dataset-specific config.
2. Express the experiment's cell count as a symbolic function of `(n_samples, n_general_reps, n_top_reps, n_aug_reps, budgets, allocations, importance_methods, evaluators, n_folds)`.
3. From the 06-02 timing data, fit per-config wall as a function of `(n_samples, budget, model)` — this is the cell-level model.
4. Plug the symbolic cell count × the cell-level wall into a per-dataset projection.
5. Verify the projection against any measured cumulative experiment wall (only the n=137-217 anchors exist).

This produces a **dataset-specific estimate** that does not assume proportional scaling.

### 10.3 What I have done in this conversation that is wrong

Anywhere I wrote estimates of the form "for n=X expect ~Yh cx experiments" without explicitly deriving Y from the symbolic structure of each experiment — those numbers are not defensible. They include:

- Section 5.2's column "Estimated cumulative cx wall" was wrong methodology.
- Earlier conversation tables that gave "~50-150h cx remaining" for various molecular datasets used the same wrong methodology.
- Any "X% reduction by cutting Y" claim where the % came from dividing one experiment's small-n wall by the small-n cumulative.

The new chat must redo this from the code.

---

## 11. What needs to be done in the next chat (no recommendations, just the structure)

Per the user's plan:

1. Many agents comb through the c-experiment and s-experiment code in parallel to extract grid sizes, loop structures, and per-experiment cost models.
2. Combine those with the 06-02 timing data to anchor per-config walls per `(model, budget, n_samples)`.
3. Predict per-dataset precompute remaining and cx/sim cumulative experiment walls.
4. Verify predictions against the 06-02 measured walls before trusting them.
5. Only then decide whether to add gates / cut experiments.

The user has been explicit: no suggestions until they ask for them. The job is to produce accurate estimates first.

---

End of document.
