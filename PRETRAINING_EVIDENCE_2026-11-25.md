# What my stored results say about pretraining for small molecules

Written 2026-11-25, from files opened that day. Unorganized on purpose — this is the raw
record of the search, not a document to hand anyone.

**Provenance of every number below.** Three levels, marked per section:
- `[read here]` — I opened the file in the session that wrote this page.
- `[search]` — a read-only search I launched in that session opened the file and reported the
  number. I did not open it myself. Re-check before quoting one of these anywhere.
- `[code]` — read out of source code, not a result file.

**Gap.** `~/repos/qsar_qm_models` (the noise paper's own tables) was NOT checked. The search
covering it was stopped by a model safeguard before it ran, twice, with `Details: [bio]`
(request IDs req_011ChQUftM3R8GDW6Wf6p6Ha, req_011ChQV1wTcyMGyacv5sphwD). Anything about
ChemBERTa or MHG-GNN in that paper is still unchecked.

---

## 1. There is already a random-weight network in the results, unlabelled

This is the condition the colleague's experiment is about.

`[code]` `src/kirby/representations/molecular.py`, `create_schnet`: the argument
`pretrained_target` defaults to `None`, and its own docstring says `None` means random
initialization. `tests/find_best_hybrid_molecular.py:737` registers `'schnet': create_schnet`
with no arguments passed. So every `schnet` row in the March surveys is a SchNet whose
weights were never trained.

`[code]` `create_dimenetpp` in the same file never loads weights either, and is registered
the same way at `tests/find_best_hybrid_molecular.py:738`.

`[search]` Scores, R² (higher better), 5 folds, from
`tests/results_server/universal_survey/<dataset>.json`. One row is one dataset and
downstream model.

| dataset | model | untrained SchNet | MolCLR (pretrained) | ECFP4 | MHG-GNN | GraphMVP |
|---|---|---|---|---|---|---|
| logd | ridge | 0.6405 | — | 0.7655 | — | — |
| logd | lightgbm | 0.6290 | 0.5627 | — | 0.8043 | 0.6454 |
| logd | rf | 0.5692 | — | — | — | — |
| caco2 | lightgbm | 0.3390 | 0.2462 | — | — | — |
| herg | lightgbm | 0.4009 | 0.3850 | — | — | — |
| qm9 | ridge | 0.6231 | — | — | — | — |
| qm9 | lightgbm | 0.5696 | 0.5678 | — | — | — |

The untrained network is above the pretrained MolCLR on all four datasets, and within 0.02
R² of GraphMVP on logd. `dimenetpp` sits at about -0.01 R² everywhere, i.e. a near-constant
output, so it says nothing either way.

Caveats, mine: no seed was found for these two builders (not exhaustively checked). The
pretrained-vs-untrained comparison here is an accident of a default argument, not a designed
condition, and the two networks differ in architecture as well as in training, so SchNet
against MolCLR is not a clean swap.

---

## 2. Pretraining the downstream network, then fine-tuning it, loses to from scratch

### 2a. ChEMBL-pretrained network, three assay datasets

`[read here]` `tests/results_server/comparisons/imrie_transfer_learning.json`, 16 March 2026.
Pretraining data is bulk ChEMBL bioactivity. 5 folds grouped by scaffold. RMSE, lower better.
Conditions as named in the file: `dnn_scratch`, `dnn_tl_full` (all layers fine-tuned),
`dnn_tl_frozen` (pretrained body, linear layer only), `ridge`.

One row is one dataset and representation.

| dataset | representation | scratch | pretrained + fine-tuned | pretrained + linear only | ridge |
|---|---|---|---|---|---|
| logd | ecfp4 | 0.5251 | 0.6813 | 1.1767 | 0.5377 |
| logd | maccs | 0.6631 | 0.7774 | 1.1659 | 0.7882 |
| logd | avalon | 0.5337 | 0.7879 | 1.1561 | 0.5515 |
| logd | pdv | 0.4709 | empty | empty | 0.5989 |
| logd | mordred | 0.4504 | empty | empty | 0.5086 |
| logd | mol2vec | 0.4710 | — | — | 0.5734 |
| caco2 | ecfp4 | 0.3001 | 0.3496 | 0.4311 | 0.3045 |
| caco2 | maccs | 0.3558 | 0.3617 | 0.4264 | 0.3641 |
| caco2 | avalon | 0.3079 | 0.3754 | 0.4303 | 0.3187 |
| caco2 | pdv | 0.2992 | empty | empty | 0.3297 |
| caco2 | mordred | 0.3040 | empty | empty | 0.3257 |
| herg | ecfp4 | 0.6419 | 0.7181 | 0.9148 | 0.6073 |
| herg | maccs | 0.7236 | 0.7816 | 0.9071 | 0.7911 |
| herg | avalon | 0.6427 | 0.7839 | 0.9158 | 0.6416 |
| herg | pdv | 0.6917 | empty | empty | 0.7610 |
| herg | mordred | 0.6679 | empty | empty | 0.7091 |

Spearman for the linear-probe condition on logd ecfp4 is 0.057, i.e. no signal.

Four representations have empty fine-tuning cells (pdv and mordred on all three datasets), so
the run is incomplete — the checkpoints for those were missing or failed. The Gaussian process
condition is in the file for maccs/pdv/mol2vec only and is catastrophic (herg maccs R² -33.9);
`MASTER_ISSUES_2026_06_04.md:262` records leaving that asymmetry as-is by decision of
2026-06-05.

Leakage note: `MASTER_ISSUES_2026_06_04.md:224` logged OPIG-1, the pretraining corpus
excluding `herg` but not `herg_ki`, as a P0 leak; line 254 records it FIXED in
`chembl_bioactivity.py:328-331`. Unchecked whether this file predates or postdates the fix.

### 2b. Same target, four small veterinary ChEMBL sets

`[read here]` `tests/results/cross_species/same_target_transfer.json`, 16 May 2026. Human
ChEMBL at the same protein target as the pretraining data. RMSE, lower better. `zero_shot` =
pretrained weights applied with no fine-tuning at all.

One row is one dataset and representation. Scaffold GroupKFold, 5 folds, 3 folds below 150
molecules.

| dataset (molecules, folds) | representation | scratch | fine-tuned | linear only | no fine-tune | rf |
|---|---|---|---|---|---|---|
| cow_CA4 (1250, 5) | ecfp4 | 0.866 | 1.444 | 1.760 | 2.440 | 0.988 |
| cow_CA4 | maccs | 1.168 | 1.383 | 1.734 | 2.336 | 1.193 |
| cow_CA4 | pdv | 0.961 | 1.394 | 1.783 | 1.989 | 1.306 |
| cow_CA4 | mol2vec | 1.014 | 1.239 | 1.755 | 10.216 | 1.341 |
| cow_CA4 | mordred | 1.097 | 1.497 | 1.721 | 3.262 | 1.509 |
| cow_CA4 | avalon | 1.237 | 1.392 | 1.757 | 2.504 | 1.149 |
| cow_DRD2 (250, 5) | ecfp4 | 0.944 | 1.018 | 1.079 | 1.500 | 0.843 |
| cow_DRD2 | mordred | 0.852 | 0.931 | 1.082 | 1.831 | 0.879 |
| cow_DRD2 | pdv | 0.928 | 1.061 | 1.069 | 1.345 | 0.885 |
| dog_PTGS2 (107, 3) | ecfp4 | 0.586 | 0.608 | 0.609 | 0.971 | 0.575 |
| dog_PTGS2 | mol2vec | 0.603 | 0.567 | 0.606 | 9.671 | 0.519 |
| dog_PTGS2 | pdv | 0.624 | 0.581 | 0.620 | 1.106 | 0.584 |
| cow_PTGS2 (27, 3) | — | all conditions degenerate, mordred RMSE 6.4e10 | | | | |

Smaller datasets did not make pretraining help. The hybrid rows in the same file (condition
`kirby_hybrid`, non-transfer conditions only): cow_CA4 rf 0.742 at 50 features against 0.988
for ECFP4 — best number on that dataset; dog_PTGS2 rf 0.477 against 0.519 for the best
single; cow_DRD2 rf 0.835 against 0.843.

---

## 3. Fine-tuning the encoder loses to leaving it frozen

`[read here]` `tests/results_server/comparisons/molecular_finetune_comparison.json`,
15 March 2026. RidgeCV on top, 5 folds grouped by scaffold, RMSE lower better. logd 5,028
molecules / 2,273 scaffolds, caco2 2,158 / 1,056, herg 1,412 / 751.

One row is one method; a method is an encoder plus how it was trained.

| method | logd | caco2 | herg |
|---|---|---|---|
| grover_frozen (4800 features) | **0.4661** | **0.3150** | **0.6385** |
| molformer_lora | 0.5492 | 0.3223 | 0.6758 |
| molformer_contrastive | 0.5570 | 0.3327 | 0.6457 |
| graphmvp_ft | 0.5598 | 0.3392 | 0.6756 |
| molformer_frozen (768) | 0.5730 | 0.3373 | 0.6430 |
| chembert_contrastive | 0.5935 | 0.3567 | 0.7145 |
| chembert_lora | 0.6148 | 0.3517 | 0.7413 |
| molformer_freeze_head | 0.6209 | 0.3408 | 0.6672 |
| chemberta_frozen (384) | 0.6219 | 0.3349 | 0.6936 |
| chemberta_lora | 0.6383 | 0.3516 | 0.7423 |
| chemberta_contrastive | 0.6492 | 0.3712 | 0.7437 |
| selformer_frozen (768) | 0.6554 | 0.3541 | 0.6662 |
| graphmvp_frozen (300) | 0.6670 | 0.3448 | 0.7018 |
| chembert_frozen (256) | 0.6785 | 0.3503 | 0.7035 |
| molclr_ft | 0.6902 | 0.3592 | 0.8133 |
| selformer_contrastive | 0.7572 | 0.3648 | 0.7200 |
| selformer_lora | 0.7613 | 0.3590 | 0.7329 |
| chembert_freeze_head | 0.8583 | 0.3748 | 0.7462 |
| chemberta_freeze_head | 0.8765 | 0.3806 | 0.7499 |
| selformer_freeze_head | 0.8789 | 0.3763 | 0.7510 |
| molclr_frozen (512) | 0.8865 | 0.3688 | 0.7999 |
| grover_ft, gp_molformer_* (4 methods) | zero folds completed | | |

Frozen GROVER is first on all three datasets.

### Decisions already recorded, 2026-03-25

`[read here]` `EXPERIMENT_DESIGN_DISCUSSION.md` §9, lines 157-235, repeated at
`PIPELINE_READINESS_TODO.md:195-206`. Verdicts there are the ones written at the time:

- Replaces frozen where it wins: `lora_molformer` and `contrastive_molformer` on logd and
  caco2; `graphmvp_ft` on all three. herg_ki keeps frozen MoLFormer.
- Excluded everywhere as worse on all datasets: all ChemBERTa fine-tuning, all SELFormer
  fine-tuning, all freeze-head.
- Broken: `grover_ft`, `gp_molformer` (NaN), `antiberty_ft` (R² -166 million).
- `~/.cache/kirby/finetuned_reps/` recorded as empty; all fine-tuned representations are
  per-fold factories, no precomputed arrays.

### Fine-tuned representations offered to the search, and ignored

`[read here]` `tests/results_server/finetune_survey/ft_mol_*.json` (25 files, 25-26 February
2026, 2 folds, R²) and `tests/results/molecular_ft_fill/*.json` (31 March, 3 folds, RMSE).
Both are small-fold runs, debug grade by the local-runs rule.

In all 25 files the hybrid's features come from ecfp4, maccs and pdv only. Zero features from
`lora_molformer`, `contrastive_molformer`, `gnn_graphmvp`, `gnn_molclr`, `lora_chemberta`,
`freeze_*`, although those were in the searched set. Example, `adme_rlm_ft.json`, RMSE:
hybrid rf 0.5637 at 150 features (ecfp4 17, maccs 47, pdv 86); singles lora_molformer 0.6024,
contrastive_molformer 0.6057, gnn_graphmvp 0.6298, pdv 0.5889, ecfp4 0.6523.

`qm9_ft.json`, RMSE: hybrid xgboost 0.4139 at 300 features (ecfp4 15, maccs 100, pdv 185);
singles pdv 0.4264, lora_molformer 0.5342, gnn_graphmvp 0.6177.

---

## 4. Does the hybrid help, and does it need the pretrained features

`[search]` `/Volumes/seagate/kirby_results_2026-10-10b/complexity/c01_rep_characterization/`
(singles) and `.../c02_complexity_matching_grid/` (hybrid). mtimes 2026-10-06 to 10-09.
5 outer scaffold folds, `selection_scope: fold_train_rows`, status ok. Ten representations,
two downstream models per dataset. The only runs here with the search inside each fold, so
the only hybrid scores not inflated by the earlier whole-dataset search. Reported as the
source of thesis Table `tab:hybrid_vs_best_single`.

One row is one dataset and downstream model. RMSE lower better, herg is AUROC higher better.
"picked single" = the representation ranked first on each fold's training molecules, scored
held-out; the other two single columns are best-in-hindsight.

| dataset | model | metric | best pretrained | best classical | graph_kernel | picked single | hybrid |
|---|---|---|---|---|---|---|---|
| adme_hlm | svm | RMSE | mhg_gnn 0.4654 | pdv 0.4574 | 0.5089 | 0.4574 | 0.4267 |
| adme_hlm | dnn | RMSE | mhg_gnn 0.4689 | pdv 0.4591 | 0.5173 | 0.4660 | 0.4273 |
| adme_mdr1 | svm | RMSE | smited 0.4701 | pdv 0.4678 | 0.5205 | 0.4678 | 0.4370 |
| adme_mdr1 | dnn | RMSE | smited 0.4794 | pdv 0.4732 | 0.5268 | 0.4732 | 0.4438 |
| adme_rlm | svm | RMSE | mhg_gnn 0.5331 | pdv 0.5209 | 0.5765 | 0.5209 | 0.4818 |
| adme_rlm | dnn | RMSE | mhg_gnn 0.5416 | pdv 0.5332 | 0.5865 | 0.5318 | 0.4974 |
| caco2 | dnn | RMSE | mhg_gnn 0.3024 | pdv 0.2980 | 0.3018 | 0.3042 | 0.2775 |
| caco2 | svm | RMSE | mhg_gnn 0.3077 | pdv 0.3001 | 0.3149 | 0.3036 | 0.2862 |
| herg | lightgbm | AUROC | mhg_gnn 0.8603 | mordred 0.8566 | 0.8442 | 0.8555 | 0.8666 |
| herg | xgboost | AUROC | mhg_gnn 0.8546 | mordred 0.8550 | 0.8441 | 0.8532 | 0.8644 |
| herg_ki | svm | RMSE | mhg_gnn 0.5881 | ecfp4 0.5980 | 0.6109 | 0.5881 | 0.5590 |
| herg_ki | dnn | RMSE | mhg_gnn 0.5926 | pdv 0.6056 | 0.6225 | 0.6031 | 0.5691 |
| logd | dnn | RMSE | mhg_gnn 0.4466 | mordred 0.4478 | not in pool | 0.4497 | 0.4089 |
| logd | svm | RMSE | mhg_gnn 0.4930 | mordred 0.5056 | not in pool | 0.4930 | 0.4302 |
| qm9 | dnn | RMSE | mol2vec 0.4097 | mordred 0.3619 | not in pool | 0.3619 | 0.3155 |
| qm9 | lightgbm | RMSE | smited 0.5620 | mordred 0.3720 | not in pool | 0.3720 | 0.3463 |

The hybrid is ahead of both best singles on all 16 rows.

Pretrained ahead of classical on herg_ki (both models), logd (both), herg lightgbm. Classical
ahead on the three adme sets, caco2, qm9 (both models each), herg xgboost. Outside qm9 the
largest gap either way is 0.013 RMSE. On qm9 it is large: dnn 0.4097 against 0.3619,
lightgbm 0.5620 against 0.3720.

### Which pretrained representation

Best pretrained is MHG-GNN on 12 of the 16 rows, SMI-TED on 3, Mol2vec on 1. The SMILES
language models alone are mid-table or last: logd dnn MoLFormer 0.5054, SMI-TED 0.5123,
ChemBERT 0.5662, GraphMVP 0.6297, against ECFP4 0.5281. The qm9 pool has no MHG-GNN and no
GROVER.

Best classical is pdv or mordred on 15 of the 16 rows (ecfp4 once, herg_ki svm). Hashed
fingerprints alone are usually below the best pretrained representation.

### Features taken from pretrained representations

`[search]` `c02` `hybrids[0].features_by_rep`, summed over 5 folds. "offered" = share of
features in the representations handed to the search, from `c01 _rep_dims`.

One row is one dataset and downstream model. Counts are features.

| dataset | model | pretrained | classical | graph_kernel | total | share taken | share offered |
|---|---|---|---|---|---|---|---|
| adme_hlm | svm | 1879 | 1667 | 375 | 3921 | 0.48 | 0.56 |
| adme_hlm | dnn | 4929 | 4219 | 677 | 9825 | 0.50 | 0.56 |
| adme_mdr1 | svm | 2062 | 3727 | 442 | 6231 | 0.33 | 0.44 |
| adme_mdr1 | dnn | 2858 | 5161 | 649 | 8668 | 0.33 | 0.44 |
| adme_rlm | svm | 3832 | 4323 | 630 | 8785 | 0.44 | 0.48 |
| adme_rlm | dnn | 4305 | 3811 | 558 | 8674 | 0.50 | 0.48 |
| caco2 | dnn | 3542 | 3347 | 984 | 7873 | 0.45 | 0.46 |
| caco2 | svm | 1236 | 1453 | 311 | 3000 | 0.41 | 0.46 |
| herg | lightgbm | 3482 | 3416 | 400 | 7298 | 0.48 | 0.41 |
| herg | xgboost | 2384 | 3201 | 516 | 6101 | 0.39 | 0.41 |
| herg_ki | svm | 1001 | 1251 | 248 | 2500 | 0.40 | 0.45 |
| herg_ki | dnn | 2640 | 2889 | 386 | 5915 | 0.45 | 0.45 |
| logd | dnn | 5834 | 4166 | 0 | 10000 | 0.58 | 0.51 |
| logd | svm | 3637 | 2363 | 0 | 6000 | 0.61 | 0.51 |
| qm9 | dnn | 2110 | 3635 | 0 | 5745 | 0.37 | 0.38 |
| qm9 | lightgbm | 0 | 4299 | 0 | 4299 | 0.00 | 0.38 |

Share taken tracks share offered on every row but logd (above) and qm9 lightgbm, which takes
no pretrained feature on any of the 5 folds. qm9 dnn takes none on fold 2.

Per-fold example, logd dnn fold 0, of 2000 features: mhg_gnn 217, mordred 223, grover 214,
ecfp4 196, smited 208, molformer 212, avalon 194, top_torsion 186, chembert 186,
graphmvp 164.

### A hybrid built without any pretrained representation

`[search]` `.../complexity/c13_forward_compatibility/`. A hybrid over the non-pretrained
representations only, then the same plus one pretrained representation, then the full hybrid.
RMSE lower better, herg AUROC higher better.

One row is one dataset and downstream model.

| dataset | model | classical-only hybrid | full hybrid |
|---|---|---|---|
| logd | dnn | 0.5122 | 0.4902 |
| logd | svm | 0.5364 | 0.5175 |
| caco2 | dnn | 0.2717 | 0.2788 |
| caco2 | svm | 0.2864 | 0.2861 |
| herg_ki | svm | 0.5615 | 0.5596 |
| adme_mdr1 | svm | 0.4371 | 0.4325 |
| adme_mdr1 | dnn | 0.4418 | 0.4407 |
| herg | lightgbm (AUROC) | 0.8305 | 0.8290 |
| herg | xgboost (AUROC) | 0.8092 | 0.8215 |

On caco2, adme_mdr1 and herg_ki svm the classical-only hybrid is already ahead of every single
representation in the §4 table. On logd the pretrained features are worth 0.022 RMSE with the
dnn. herg and logd here ran on 2,000 subsampled molecules; adme_mdr1 and herg have 3 fold
entries for one condition; herg_ki dnn classical-only is 0.6547, worse than its singles, cause
not found.

Random-feature controls in the same file: a Gaussian matrix of matched width alone, and a
pretrained representation with rows shuffled alone, both at chance (logd dnn 1.194 and 1.191
RMSE; herg AUROC 0.46-0.53). Also the full hybrid with one representation's features replaced
by Gaussian noise: logd dnn 0.5164-0.5931 against 0.4868-0.5033 with the real features.

### Attribution within the finished hybrid

`[search]` `.../complexity/c22_explainability/`, 5 folds, 5 repetitions. Drop in held-out
score when one representation's features are shuffled. logd dnn: mordred 0.248, mhg_gnn 0.172,
grover 0.149. herg lightgbm: mordred 0.163, mhg_gnn 0.100. herg_ki svm: mhg_gnn 0.065,
mordred 0.057, graph_kernel 0.057. qm9 dnn: mordred 0.544, avalon 0.182, chembert 0.144.
qm9 lightgbm: mordred 1.276.

SHAP share, pretrained / classical / other: logd dnn 0.66/0.34/0, logd svm 0.74/0.26/0,
caco2 dnn 0.59/0.29/0.12, herg_ki svm 0.52/0.38/0.11, adme_mdr1 dnn 0.45/0.47/0.09,
qm9 dnn 0.40/0.60/0, qm9 lightgbm 0/1.00/0. ECFP4, Avalon, atom pair and topological torsion
each account for 0.010 or less outside logd.

---

## 5. Older single-representation surveys, same question

All `[search]`. Hybrid scores from before October 2026 come from searches over whole
datasets and are optimistic; single-representation scores are unaffected.

### phase5 baselines, 2026-03-21/22, 5 folds, RMSE, 22-24 representations, up to 10 models
`tests/results_server/phase5_baselines/molecular/<dataset>.json`. Best classical / best
pretrained per model. "herg" here is the regression version, 1,412 molecules.

- logd: catboost mordred 0.4701 / mhg_gnn 0.4852; dnn mordred 0.4462 / mhg_gnn 0.4396;
  gp mordred 0.4594 / mhg_gnn 0.4438; knn pdv 0.6698 / grover 0.6905; lightgbm mordred 0.5081
  / mhg_gnn 0.5121; mlp pdv 0.5261 / mhg_gnn 0.4898; rf pdv 0.5876 / mhg_gnn 0.6166;
  ridge mordred 0.4858 / grover 0.4506; xgboost mordred 0.5342 / mhg_gnn 0.5521.
- caco2: catboost pdv 0.2926 / mhg_gnn 0.2958; dnn ecfp4 0.3023 / mhg_gnn 0.2967; gp ecfp4
  0.2926 / mol2vec 0.2906; knn pdv 0.3418 / grover 0.3449; lightgbm ecfp4 0.3005 / mhg_gnn
  0.2959 (graph_kernel 0.2932); mlp ecfp4 0.3055 / mol2vec 0.3597; rf pdv 0.3096 / mhg_gnn
  0.3176; ridge ecfp4 0.3023 / mhg_gnn 0.3128; xgboost atom_pair 0.3034 / mhg_gnn 0.3200
  (graph_kernel 0.2983).
- herg: catboost mordred 0.5776 / mhg_gnn 0.5747; dnn ecfp4 0.5996 / mhg_gnn 0.5858;
  gp avalon 0.5913 / molformer 0.6081; knn mordred 0.6293 / mol2vec 0.6341; lightgbm pdv
  0.6026 / mhg_gnn 0.5830; mlp atom_pair 0.7029 / chembert 0.8397; rf ecfp4 0.5955 / mhg_gnn
  0.5983; ridge ecfp4 0.5995 / molformer 0.6397; xgboost ecfp4 0.6154 / mhg_gnn 0.6094.
- adme_hlm: catboost mordred 0.4561 / mhg_gnn 0.4708; dnn atom_pair 0.5172 / mhg_gnn 0.4617;
  gp pdv 0.4596 / mhg_gnn 0.4601; lightgbm mordred 0.4592 / mhg_gnn 0.4761; rf pdv 0.4958 /
  mhg_gnn 0.5170; ridge atom_pair 0.5079 / grover 0.4892; xgboost mordred 0.4853 / mhg_gnn
  0.5113; knn pdv 0.5378 / smited 0.5559; mlp atom_pair 0.5217 / mol2vec 0.5898.
- adme_rlm: catboost mordred 0.5403 / mhg_gnn 0.5519; dnn atom_pair 0.6020 / mhg_gnn 0.5491;
  gp pdv 0.5340 / mhg_gnn 0.5346; lightgbm mordred 0.5464 / mhg_gnn 0.5549; rf pdv 0.5799 /
  mhg_gnn 0.6070; ridge ecfp4 0.5981 / grover 0.5495; xgboost mordred 0.5701 / mhg_gnn 0.5937;
  knn pdv 0.6390 / smited 0.6757; mlp atom_pair 0.6149 / chembert 0.6822.
- adme_mdr1 (5 models): lightgbm mordred 0.4787 / mhg_gnn 0.4856; rf pdv 0.5093 / 0.5335;
  ridge mordred 0.4831 / 0.4899; xgboost mordred 0.5075 / 0.5142; mlp avalon 0.5277 /
  chembert 0.5816.
- adme_solubility (5 models): lightgbm mordred 0.5681 / mhg_gnn 0.5648; rf mordred 0.5788 /
  0.5876; ridge mordred 0.5622 / 0.5699; xgboost mordred 0.6037 / 0.6006; mlp atom_pair
  0.6006 / molclr 0.6559.
- qm9 (5 models): lightgbm mordred 0.3762 / grover 0.4130; rf mordred 0.4609 / grover 0.4848;
  ridge pdv 0.4963 / grover 0.4034; xgboost mordred 0.3919 / mhg_gnn 0.4442; mlp mordred
  0.5881 / grover 0.5182.
- The cnn column is poor for every representation (logd 0.87-1.15) — a model fault, not a
  representation result.

### universal survey, 2026-03-10, 5 folds, R² higher better, 25 representations
`tests/results_server/universal_survey/<dataset>.json`. Best classical / best pretrained.
- logd: rf pdv 0.7427 / mhg_gnn 0.7266; xgboost pdv 0.7812 / 0.7787; ridge ecfp4 0.7904 /
  grover 0.8503; knn pdv 0.6676 / mhg_gnn 0.6452; lightgbm pdv 0.7910 / mhg_gnn 0.8043.
- caco2: rf ecfp4 0.4741 / mhg_gnn 0.4462; xgboost 0.4761 / 0.4512; ridge 0.5097 / grover
  0.4747; knn 0.3450 / 0.3434; lightgbm pdv 0.5092 / mhg_gnn 0.5117 (graph_kernel 0.5196).
- herg: rf ecfp4 0.5649 / mhg_gnn 0.5586; xgboost 0.5387 / 0.5283; ridge 0.5484 / 0.5074;
  knn 0.5162 / mol2vec 0.5184; lightgbm pdv 0.5709 / mhg_gnn 0.5794.
- The 50-feature PCA version of each pretrained representation is always below the full one.
- qm9 server file holds pretrained representations only (mhg_gnn lightgbm 0.8862). The local
  copy has R² ≤ 0.14 for everything — unusable.

### Other surveys
- `tests/results/framework/f29_{logd,caco2}_full__*.json`, 2026-09-03, 5 folds, RMSE,
  8 models. File states fold scores were lost and it was rebuilt from the run log. logd:
  catboost mordred 0.4699 / mhg_gnn 0.4778; dnn 0.4475 / 0.4523; mlp avalon 0.5191 / mhg_gnn
  0.4895; ridge 0.4802 / 0.5269. caco2: catboost pdv 0.2851 / mhg_gnn 0.2993; dnn ecfp4
  0.3016 / 0.2936; mlp ecfp4 0.3018 / 0.3737; ridge top_torsion 0.3287 / 0.3619.
- `tests/results/molecular_aug_fill/`, 03-25, 3 folds: logd ridge grover 0.4682 vs mordred
  0.524; logd rf mhg_gnn 0.6437 vs pdv 0.6145; caco2 ridge mhg_gnn 0.3199 vs ecfp4 0.3037;
  qm9 ridge grover 0.4066 vs pdv 0.499; qm9 xgboost mhg_gnn 0.4579 vs mordred 0.403.
- `tests/results/schnet_e2e/`, 04-12 to 04-20, 3 folds (herg_ki 5): herg_ki lightgbm mhg_gnn
  0.5875 vs pdv 0.5992; herg_ki ridge 0.6317 vs ecfp4 0.594; logd ridge grover 0.4808 vs
  mordred 0.4911. Fine-tuned encoders in the same files are worse than both: logd ridge LoRA
  ChemBERTa 0.5654, LoRA MoLFormer 0.6094, SchNet trained end to end 0.7568.
- `tests/results_server/molecular_graph_survey/`, 03-18, 5 folds: herg rf mhg_gnn 0.5942 vs
  ecfp4 0.6070; logd ridge 0.5375 vs mordred 0.4836; qm9 rf grover 0.4572 vs mordred 0.4048.
- `tests/results_server/baseline_survey/molecular_baselines.json`, 02-27, R², fold count not
  stored: logd lgbm mhg_gnn 0.822 vs pdv 0.8044; caco2 0.5255 vs pdv 0.5354; herg 0.5638 vs
  ecfp4 0.5582.
- `tests/results_server/complexity_*_a/complexity_full_20260216_*.json`, 02-16, 5 folds, R²,
  only ChemBERTa/MoLFormer/SELFormer as pretrained: classical ahead nearly everywhere, e.g.
  logd lightgbm pdv 0.7913 vs molformer 0.6905; caco2 rf pdv 0.5059 vs molformer 0.3966. Its
  qm9 file (1,981 molecules) is near zero for everything — unusable.
- `tests/results/framework/f00_rep_screening/herg.json`, 03-17, RF, 5 folds, RMSE: ecfp4
  0.6178, graph_kernel 0.6226, mhg_gnn 0.6276; grover/molformer/chemberta/smited 0.68-0.69.

### Noise-paper representations as copied into this repo
`tests/results_server/{cox2,esol,herg,openadmet_caco2,openadmet_logd}/all_results.csv` and
`tests/results_server/alternative_full/*/all_results.csv`, mtime 2026-03-17. One pretrained
representation (MHG-GNN) against ECFP4, PDV, SNS, at no added noise. RMSE unless stated.
- alternative_full/logd, 5 folds: DNN ecfp4 0.5923 / mhg_gnn 0.5322 / sns 0.5380 / pdv 1.1981;
  LightGBM 0.6228 / 0.5552 / 0.5807 / 0.5870; SVM 0.5890 / 0.5639 / 0.5642 / 1.1797;
  XGBoost 0.6300 / 0.6083 / 0.5694 / 0.6012; RF 0.6902 / 0.6395 / 0.6274 / 0.6370.
- alternative_full/caco2: DNN 0.3115 / 0.3158 / 0.3036 / 0.4399; LightGBM 0.3214 / 0.3176 /
  0.3088 / 0.3138; XGBoost 0.3246 / 0.3438 / 0.3101 / 0.3400.
- cox2 RF 1.0387 / 1.0144 / 1.0298 / 1.0454. esol RF ecfp4 0.6974 / mhg_gnn 0.4649 / sns
  0.5374. herg_fluid AUROC RF 0.8968 / 0.9098 / 0.9152 / pdv 0.9236.
- PDV with DNN, SVM and GP gives 1.17-1.27 on logd, 3949 on cox2, AUROC 0.5000 on herg_fluid.
  Inferred, mine: an unscaled-input fault, not a representation result.
- `tests/results/validation/*`, 05-17, 13 models: MHG-GNN far worse with neural models there
  (logd DNN 1.0015 vs pdv 0.5339). drd2: QRF ecfp4 0.8161 / mhg_gnn 0.8212 / sns 0.8110.

### Older hybrid numbers (whole-dataset searches — optimistic)
- `tests/results/precompute_registry.json`, 04-26: herg_ki best pretrained molformer 0.5763,
  best classical avalon 0.5919, hybrid svm 0.5401; caco2 grover 0.2894, pdv 0.2901, hybrid gp
  0.2686; qm9 grover 0.387, mordred 0.372, hybrid svm 0.3306. Single and hybrid scores there
  are not from the same model, and no feature counts are stored.
- `tests/results/complexity_c02/full/c02_complexity_matching_grid/herg_ki.json`, 04-26:
  svm molformer 0.5763 / hybrid 0.5401; lightgbm mhg_gnn 0.5790 / 0.5660; gp avalon 0.5919 /
  0.5586; mlp atom_pair 0.6886 / hybrid 0.6903.
- `tests/results_server/noise_robustness_v3/qm9/hybrid_info.json` `[read here]`: R² pdv
  0.8577, mhggnn 0.8251, hybrid of the two 0.8758; allocation pdv 35 features, mhggnn 10, at a
  budget of 100, by greedy — greedy is cut.
- Framework build experiments: f32, 09-09, lightgbm, 5 folds, held-out hybrid at 300
  features: caco2 0.2892, herg_ki 0.5924, logd 0.4996. Pretrained features of 300 per fold:
  caco2 126/140/155/121/98, herg_ki 150/103/133/90/97, logd 189/199/202/199/191. f31 caco2 at
  budget 100: 36-50 pretrained of 100. f00 herg at budget 100: hybrid 0.6078-0.6240 vs ecfp4
  0.6178, 10-30 pretrained features of 60-100. f01/f03/f04/f12/f13 are single-fold runs,
  debug grade, pretrained share anywhere from 0 of 150 to 299 of 300.
- f34, 10-02/04, local, nearest-neighbour prediction, best setting chosen on held-out scores:
  adme_mdr1 pdv 0.5165 / smited 0.5266 / hybrid 0.4862; caco2 pdv 0.3213 / grover 0.3298 /
  0.3137; herg AUROC atom_pair 0.8273 / mhg_gnn 0.8334 / 0.8490; herg_ki atom_pair 0.5948 /
  mhg_gnn 0.6148 / 0.5786.
- February similarity files `tests/results_server/similarity_*_{a,b}`: only ChemBERTa,
  MoLFormer, SELFormer as pretrained; feature counts caco2 0 pretrained of 80, herg 10 of 50,
  qm9 46 of 100.

### Adjacent files worth knowing about
- `fed03_synthetic_private/{logd,caco2}.json` `[search]`, 3 seeds × 5 folds, RMSE, source of
  thesis Table `tab:method:family_sites`. All features of one family joined, no selection:
  logd dnn learned 7,916 features 0.4422, mordred 1,613 0.4510, fingerprints 6,144 0.4570;
  logd svm 0.4598 / 0.5060 / 0.5140; caco2 dnn learned 7,360 0.2896, pdv 200 0.2998,
  fingerprints plus graph_kernel 8,509 0.2773. One selection over all families: logd dnn
  0.4191, svm 0.4382; caco2 dnn 0.2839, svm 0.2847.
- c18 `[search]`: budget filled from the top representation alone / the search's hybrid / an
  even split over all representations — logd dnn 0.4610 / 0.4093 / 0.4057; qm9 lightgbm
  0.3721 / 0.3455 / 0.3591.
- c11 `[search]`: matched-width PCA of all representations joined beats the hybrid on logd dnn
  (0.3895 vs 0.4086) and caco2 dnn (0.2725 vs 0.2776).
- `tests/results/comparisons/opig_*.json` and `.../opig/qsar.json` `[search]`, MAE, 5 folds,
  GROVER and SMI-TED requested but not built: D2 lightgbm mhg_gnn 0.5188 / avalon 0.5282 /
  hybrid 0.5139; D2 dnn 0.5229 / 0.5284 / 0.4989; Factor Xa lightgbm mhg_gnn 0.7363 / ecfp4
  0.6835 / hybrid 0.7144 (the one row where the hybrid loses to a single); Factor Xa dnn
  0.7330 / 0.7165 / 0.6612; Mpro lightgbm 0.4654 / 0.4830 / 0.4516; Mpro dnn 0.4965 / 0.5025 /
  0.4904.
- s01 `[search]`, Spearman between Manhattan distance and label difference, higher better:
  adme_mdr1 mhg_gnn 0.194 vs atom_pair 0.281; caco2 mhg_gnn 0.168 vs top_torsion 0.229;
  herg 0.025 vs 0.024; herg_ki smited 0.117 vs pdv 0.107; logd molformer 0.147 vs ecfp4 0.100;
  qm9 chemberta 0.279 vs mordred 0.186.
- noise c10 `[search]`, pretrained features in the hybrid at noise level 0: caco2 dnn 689 of
  1568, herg_ki svm 201 of 499, logd dnn 1202 of 1978 (3 folds), qm9 lightgbm 0 of 959
  (partial).

---

## 6. What does not exist

- No experiment is labelled as an untrained, re-initialised or shuffled-weight encoder
  control. A search of `src/` and `tests/` for `random_init`, `untrained`, `random_weights`,
  `reinit`, shuffled weights, `gin_random` and similar found none, and no result file name or
  key matches `[search]`. §1 above is an accident of a default argument, not a control.
- `c04_importance_as_metric.py` shuffles importance weights on features, not encoder weights.
- `PRETRAIN_REP_ROADMAP.md` is the work that was dropped for compute. Its harness
  `tests/experiments/pretrain_rep/pretrain_rep_experiment.py` exists and is offline-tested;
  line 347 of the roadmap records it as not yet run on real data. `tests/results/
  pretrain_rep_experiment/` is an empty directory, and `EMPTY_RESULT_FILES_2026_08_27.md:52`
  logs its only output as an error, `insufficient_shared_reps`, 2026-05-15. It was written for
  antibodies first; the molecular phase in §5 of that roadmap was never built.
- `tests/experiments/pretrain_rep/pretrain_rep_model_transfer.py` has a from-scratch arm, but
  that is the downstream model, and it has no results.
- `issues_molecular.md:208` records that there is no Chemprop/D-MPNN representation —
  a network trained on the target property is absent from every comparison above.

---

## 7. File-copy discrepancies noticed in passing

`[search]`, not chased:
- `kirby_results_2026-10-08`, `-08b`, `-09`, `-10`, `-10b` are byte-identical for c01/c02
  except: adme_rlm c02 changed between 10-09 and 10-10; qm9 absent from 10-08 and 10-08b;
  adme_hlm c02 present only in 10-08, 10-08b and 10-09.
- c22 and c18 differ between the 10-08 and 10-10b copies. Cause not checked.
