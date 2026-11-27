# Pretraining and fine-tuning with adaptive hybrid representations — implementation plan

Written 2026-11-27. Not Paper 2 and not Paper 3: this is the collaborator's paper, planned as a
step of the same experiments ledger so it reuses that run's representation files and searches.

Every word for a thing comes from `VOCABULARY.md`. Every setting in §3 is an input with no
default. Nothing in this plan picks one.

Every claim about the repository was read on 2026-11-27 and carries its file and line. Claims
marked NOT CHECKED were not opened.

---

## 1. What the experiment answers

Four scenarios, in the author's words:

1. **Pretrained encoder** — a published encoder's weights fixed, its activations used as a
   representation, the downstream model fitted from scratch.
2. **Fine-tuned encoder** — the encoder itself trained further on this dataset's molecules, so
   its activations change.
3. **Pretrained model** — a downstream network's weights carried over from a larger dataset,
   with no further training.
4. **Fine-tuned model** — those carried-over weights trained further on this dataset.

Two questions, one per pair.

**Scenarios 1 and 2.** Do pretrained and fine-tuned embeddings perform well, and are their
features useful? Four things are compared per dataset and downstream model: a classical
representation on its own, a hybrid over the classical representations, a hybrid over the
pretrained and fine-tuned representations, and those pretrained and fine-tuned representations on
their own. Usefulness is read off how many features each contributes to a winning hybrid.

**Scenarios 3 and 4.** Does pretraining on the hybrid beat pretraining on a standard
representation? A downstream model is fitted on a larger dataset using the features the winning
blueprint selects, then training continues on the smaller dataset's hybrid. The comparator is the
same network pretrained on ECFP4 instead, and a published pretrained model where one exists.

---

## 2. Six things to fix before any arm runs

Each is two things that disagree. None is a setting.

### 2.1 No pretraining checkpoint exists on this machine

`imrie_transfer_learning.py:81` defaults `--pretrain-dir` to `/Volumes/seagate/kirby/chembl_pretrain`.
That directory does not exist, with the drive mounted (checked 2026-11-27). No `.pt` file exists
anywhere under `/Volumes/seagate/kirby`.

`list_conditions` (`imrie_transfer_learning.py:103-109`) therefore drops both transfer conditions,
prints `WARNING: No pre-trained weights at ...`, and the run completes reporting only
`dnn_scratch` and `ridge`. `same_target_transfer.py:117-132` and `rodent_to_vet.py:98-104` do the
same.

So scenarios 3 and 4 have zero inputs today. `pretrain_chembl_dnn` has to run first, and no
measurement of how long that takes exists anywhere in the repository.

### 2.2 `pretrain_chembl_dnn` accepts six representations and the hybrid is not one of them

`src/kirby/models/pretrain_chembl_dnn.py:75-82` supports ecfp4, maccs, avalon, pdv, mordred,
mol2vec. Any other name prints `ERROR`, returns `None`, and writes no checkpoint — which is what
makes a transfer condition silently disappear downstream.

The one path that pretrains on a hybrid is
`tests/experiments/pretrain_rep/pretrain_rep_model_transfer.py:67`, which writes its own
checkpoint in `PretrainedDNNRegressor`'s format. It has never been run.

### 2.3 Two scripts fit with a random split

`pretrain_rep_model_transfer.py:283` calls `m.fit(X_tr, y_tr)` with no groups.
`same_target_transfer.py` `evaluate_condition` does the same. `PretrainedDNNRegressor` then carves
its early-stopping holdout with `np.random.RandomState(42)` instead of the fold's groups, and warns
once.

No random splitting is a standing rule. Both calls take `groups=`.

### 2.4 Four arms can report a score with weights that never loaded

| where | what happens |
|---|---|
| `src/kirby/representations/molecular.py:2983` | `create_chemformer` prints `Loaded model weights` after `load_state_dict(strict=False)` however many tensors matched |
| `src/clamp/clamp/models/pretrained.py:32` | vendored CLAMP logs `Loaded pretrained model from ...` unconditionally after `strict=False` |
| `src/kirby/representations/reaction.py:642-646` | `_find_cgr_smiles_checkpoint` falls back to base ChemBERTa with a printed WARNING |
| `src/kirby/models/dnn_pretrained.py:47` | `pretrained_path=None` trains a random net with no warning at all |

`forward_compat_antibody.py:625-637` is the one loader in the repository that counts the tensors
that landed and raises on zero. Every arm of this experiment asserts what loaded, the same way.

### 2.5 The hybrid search cannot select a pretrained network

`DEFAULT_MODELS` (`src/kirby/find_best_hybrid.py:2790`) holds catboost, dnn, dnn_small, dnn_wide,
gp, knn, lightgbm, lightgbm_huber, linear, mlp, rf, rf_l1, ridge, svm, xgboost. No pretrained or
transfer entry.

So scenarios 3 and 4 cannot be a downstream model inside a search. The hybrid is built in one
`apply_blueprint` call and the network is fitted outside it. Scenarios 1 and 2 are representations
and do go through the search.

### 2.6 No experiment builds a fine-tuned representation

`load_data`'s `finetune` defaults to False (`tests/experiments/_shared.py:4647,4680`) and all three
call sites read `_build_rep_factories(...) if finetune else {}`. A repository-wide grep for
`finetune=True` returns nothing.

So scenario 2 inside a search exists only behind `tests/find_best_hybrid_molecular.py --finetune`,
which the papers pipeline never calls. The new step passes `finetune=True`.

---

## 3. The settings file

`tests/experiments/pretraining/SETTINGS.json`, edited only by the author. Every key below has no
default in code. A key left `null` holds every unit that needs it; a key given a list runs every
value in the list. No axis is collapsed to make anything fit.

```json
{
  "datasets":            null,
  "pretraining_pairs":   null,
  "pretraining_corpus":  null,
  "budgets":             null,
  "downstream_models":   null,
  "classical_reps":      null,
  "pretrained_encoders": null,
  "finetuned_encoders":  null,
  "encoder_layer":       null,
  "encoder_pooling":     null,
  "standard_comparator": null,
  "untrained_controls":  null,
  "blueprint_source":    null,
  "n_outer_folds":       null,
  "evidence_grade":      null,
  "collaborator_list":   null
}
```

What each one decides, and the values measured for it. The candidates are what exists; the choice
is not made here.

### datasets

Molecular regression datasets with a registered metric and task, molecule counts measured from the
cached tables on 2026-11-27:

| dataset | measures | molecules | metric |
|---|---|---|---|
| logd | octanol/water distribution | 5,028 | rmse |
| caco2 | Caco-2 P-gp efflux ratio, log10 | 2,158 | rmse |
| herg_ki | hERG pKi, ChEMBL CHEMBL240 | 1,412 | rmse |
| herg_ic50 | hERG pIC50, one row per measurement | 12,008 rows of 9,802 molecules | rmse |
| adme_hlm | human liver microsome CLint, log | 3,087 | rmse |
| adme_rlm | rat liver microsome CLint, log | 3,054 | rmse |
| adme_mdr1 | MDR1-MDCK efflux ratio, log | 2,642 | rmse |
| chembl_qsar_d2 | dopamine D2 pKi | 6,333 | rmse |
| chembl_qsar_fxa | factor Xa pKi | 3,605 | rmse |
| chembl_qsar_mpro | SARS-CoV-2 Mpro pIC50 | 1,924 | rmse |
| qm9 | HOMO-LUMO gap, DFT | 5,000 of 133,885 | rmse |

Not usable as written: adme_solubility (2,173), cox2, drd2 and esol have no entry in
`tests/experiments/dataset_policy.py`, so no metric and no task. esol is a MoleculeNet set. davis,
kiba, bindingdb_kd and metz load through TDC.

One dataset blocks itself: qm9's stored representation files hold the earlier 10,000-molecule
sample and are refused against the current 5,000 (`src/kirby/registry.py:57-63` still says
10,000). A qm9 arm needs those rebuilt and nothing measures that rebuild.

### pretraining_pairs

One larger dataset and one smaller one, for scenarios 3 and 4. Both counts measured 2026-11-27:

| larger | smaller | molecules | molecules | relation |
|---|---|---|---|---|
| logd | herg_ki | 5,028 | 1,412 | lipophilicity drives hERG block; the thesis precedent at `thesis.tex:2080` |
| logd | caco2 | 5,028 | 2,158 | same OpenADMET table; lipophilicity sets passive permeability |
| logd | adme_mdr1 | 5,028 | 2,642 | same relation, different assay and source table |
| herg_ic50 | herg_ki | 9,802 | 1,412 | same ChEMBL target, pIC50 against pKi |
| chembl_qsar_d2 | chembl_qsar_mpro | 6,333 | 1,924 | same curation, different protein family |
| chembl_qsar_d2 | herg_ki | 6,333 | 1,412 | aminergic GPCR ligands enriched for hERG block |
| adme_mdr1 | caco2 | 2,642 | 2,158 | both log efflux ratios, different monolayer |
| chembl_qsar_fxa | chembl_qsar_mpro | 3,605 | 1,924 | same curation, both proteases |

A pairing from the same source table shares molecules: of 5,326 OpenADMET rows, 5,039 carry a
LogD and 2,161 a Caco-2 efflux value. Overlapping molecules are removed from the larger side
before pretraining, by InChIKey, the way `pretrain_chembl_dnn` already filters downstream keys.

Two pairings are near-equal in size and are not large-to-small: adme_hlm against adme_rlm (3,087
and 3,054), and any two qm9 properties (same 5,000 molecules, different label).

### pretraining_corpus

For a corpus rather than a dataset pairing:

- **Bulk ChEMBL pActivity** through `pretrain_chembl_dnn`, `--n-compounds` default 500,000,
  `--max-targets` 200. Needs a download. Not measured.
- **Full QM9**, 133,885 molecules with 12 DFT labels, of which the repository loads 5,000
  (`src/kirby/datasets/qm9.py:216`). No download and no registry gap. The labels are computed, so
  they carry no experimental error.
- **The larger side of a pairing** above, its own labels.
- `src/kirby/datasets/pretrain_corpora.py` resolves named corpora; its contents were NOT CHECKED.

### budgets

This decides whether the identity-keyed first layer is needed at all. Measured 2026-11-27 from
`/Volumes/seagate/kirby_results_2026-10-10b/complexity/c02_complexity_matching_grid/*.json`, one
row per dataset and downstream model, features in the hybrid on each of five folds:

| dataset, downstream model | budget | features per fold |
|---|---|---|
| herg_ki dnn | 250 | 250, 250, 250, 250, 250 |
| herg_ki dnn | 500 | 500, 500, 500, 500, 500 |
| herg_ki dnn | 1000 | 1000, 973, 944, 997, 977 |
| herg_ki dnn | 2000 | 1778, 1757, 1817, 1845, 1759 |
| herg_ki svm | 1000 | 989, 973, 1000, 1000, 1000 |
| logd dnn | 1000 | 1000, 1000, 1000, 1000, 1000 |
| logd dnn | 2000 | 2000, 1983, 2000, 1962, 1952 |
| logd svm | 2000 | 2000, 2000, 2000, 2000, 2000 |
| caco2 dnn | 500 | 500, 500, 500, 500, 499 |
| caco2 dnn | 2000 | 1962, 1953, 1961, 1969, 1760 |
| adme_mdr1 svm | 500 | 500, 500, 500, 444, 500 |
| adme_mdr1 dnn | 2000 | 1962, 1761, 1955, 1952, 1959 |
| qm9 lightgbm | 1000 | 858, 927, 913, 916, 919 |
| qm9 lightgbm | 2000 | 1049, 1057, 1032, 1036, 1055 |

At 250 and 500 the width equals the budget on 23 of the 25 cells measured. At 1000 and 2000 the
representation pool runs out and the shortfall differs fold to fold. `budget_fill` defaults to
False (`src/kirby/hybrid.py:6519`), so nothing pads the width back up to the budget.

All five default allocation methods split a budget (`find_best_hybrid.py:3773`:
performance_weighted, power_2, softmax_T1, rank_exp_d0.3, wglobal_power2), so the width equals the
budget whenever no representation caps out.

### downstream_models

Scenarios 1 and 2 take any model in `DEFAULT_MODELS`.

Scenarios 3 and 4 need a model whose fitted parameters can be carried and trained further. Read
2026-11-27:

| model | carrying weights |
|---|---|
| dnn, dnn_small, dnn_wide | defined. `PretrainedDNNRegressor` already does it for one representation |
| lightgbm | defined by the library (`init_model`), used nowhere in the repository. Its trees split on positions in the matrix they were grown on |
| catboost | same shape as lightgbm; its tier is `off` (`find_best_hybrid.py:2945`), so no search selects it |
| gp | not defined. The fitted object is the training molecules plus a Cholesky factorisation |
| svm | not defined. The fitted object is support vectors drawn from the training molecules |
| knn | not defined. It has no fitted parameters |

So scenarios 3 and 4 are dnn-only unless lightgbm's `init_model` is wired, which is a separate
build.

### classical_reps

Computed in this repository and needing no GPU and no checkpoint. Widths measured from the
representation cache where a cached file exists, else the default in the builder:

ecfp4 2048, ecfp4_count 1024, maccs 167, avalon 2048, atom_pair 2048, atompair_count 1024,
top_torsion 2048, rdkfp 2048, pdv 200, desc208 208, mordred 1613, coulomb_matrix 50, usr 12,
usrcat 60, pharmacophore3d 49, tda 28, e3fp 1024, map4 1024, graph_kernel dataset-dependent (141,
276 and 317 features in three cached files).

Two need more than RDKit: padel needs a Java runtime and its docstring says it has not been
smoke-tested locally; graph_kernel and tda are in the GPU-stack list and do not build on the
laptop.

`graph_kernel`, `create_pca_rep` and `create_sns` without a reference featurizer have a feature
space discovered from the molecules they are given, so feature index *j* does not name the same
feature on two datasets. They are out of any carried-over hybrid, or built against a fixed
vocabulary.

The author named ECFP4, PDV and SNS as the standard set. SNS is `create_sns`
(`molecular.py:1226`) and takes a reference featurizer; whether one is passed is not decided here.

### pretrained_encoders

Wired and loading published weights in this repository, widths measured from the cache where one
exists:

| name in the code | input | features | checkpoint |
|---|---|---|---|
| chemberta | SMILES | 384 | DeepChem/ChemBERTa-77M-MTR |
| chembert | SMILES | 256 | jonghyunlee/ChemBERT_ChEMBL_pretrained |
| molformer | SMILES | 768 | ibm/MoLFormer-XL-both-10pct, revision pinned |
| gp_molformer | SMILES | 768 (docstring) | ibm-research/GP-MoLFormer-Uniq |
| selformer | SELFIES | 768 | HUBioDataLab/SELFormer |
| smited | SMILES | 768 | light variant default |
| mol2vec | ECFP substructures | 300 | model_300dim.pkl on disk |
| grover | 2D graph | 4800 | cloned repo plus grover_large.pt |
| molclr | 2D graph | 512 | ckpt/pretrained_gin |
| graphmvp | 2D graph | 300 | GraphMVP.pth |
| mhg_gnn | 2D graph | NOT CHECKED | |
| unimol, unimol_v2 | 3D conformer | docstring only | |
| clamp, rmat, minimol, chemformer | mixed | docstring only | |

Not measured: molclr, minimol, mole, unimol, unimol_v2, clamp, rmat, map4, padel, e3fp,
gp_molformer are docstring claims; chemfm reads its width from the checkpoint at load. A docstring
is not a measurement.

Standard in the field and absent here, with the field's reason for noticing:

| encoder | year | input | why it would be queried |
|---|---|---|---|
| CheMeleon | 2025 | 2D graph | the only one of 28 pretrained models rated better than an ECFP count fingerprint; mean rank 5.24 of 28 against ECFP 8.80 |
| MolBERT | 2020 | SMILES | mean rank 8.16 of 28; CPU-feasible |
| Uni-Mol / Uni-Mol2 | 2023 / 2024 | 3D conformer | the standard 3D baselines; Uni-Mol2 mean rank 16.08 of 28 |
| ContextPred (Hu et al. GIN) | 2020 | 2D graph | the field's default checkpoint and its worst performer, 24.24 of 28 |

Costs for adding them: GEM ships in PaddlePaddle, CDDD in TensorFlow 1.x, and Uni-Mol, Uni-Mol2,
GEM and R-MAT need a generated conformer per molecule, which is CPU time per molecule rather than
once. `create_ouroboros` hardcodes `.cuda()` and raises without a GPU, and the
`forward_compat_molecular` module header says none of its seven encoders has been run here.

### finetuned_encoders

Factories that train the encoder on the fold's molecules:

| name | what trains | location |
|---|---|---|
| lora_chemberta, lora_molformer, lora_selformer, lora_gp_molformer, lora_chembert | LoRA adapters rank 8 alpha 16 plus a head, MSE | `src/kirby/finetuning/molecular_lora.py:208` |
| contrastive_<model>, same five | the same adapters under a supervised contrastive loss, temperature 0.1 | `molecular_contrastive.py:267` |
| gnn_molclr, gnn_graphmvp | the whole GIN plus a head, `requires_grad=True` | `molecular_gnn.py:453` |
| grover_ft | the whole GROVER plus an FFN head | `molecular_grover.py:319`, never run |

Three names in the repository say fine-tuning and do something else:

- `freeze_<model>` sets `requires_grad=False` on the entire backbone (`molecular_freeze.py:71`),
  so its embedding is identical to the frozen encoder's. It is scenario 1 under a scenario 2 name,
  and keeping it spends GPU hours for no new condition.
- `ft_schnet` and `ft_dimenetpp` load no checkpoint anywhere in `molecular_3d.py`; the only
  `load_state_dict` calls restore the best epoch of the same run.
- `ft_chemprop` loads none either (`molecular_chemprop.py:177`).

Those three are train-from-scratch. They belong in `untrained_controls` under a name that says so,
or out.

One name means two things and a results table keyed by name would double-count: the same GraphMVP
fine-tune is `gnn_graphmvp` in `DATASET_REPS` and `graphmvp_ft` in
`molecular_finetune_comparison.py:310`.

**Order, when a fine-tuned encoder feeds a hybrid.** Fine-tune the encoder, recompute its
embedding, then run the blueprint. The blueprint selects features from the embedding, and the
embedding has changed; running them the other way round selects features that no longer exist.

### encoder_layer and encoder_pooling

Not one condition until both are fixed. `create_chemfm` means over non-padding last-layer states is
this repository's choice, not the model card's (`forward_compat_molecular.py:621`).

Published, 22 TDC ADMET tasks, five encoders: an intermediate layer beat the conventional final
layer in 81% of model-task pairs, average +5.4%, rising to +8.5% when fine-tuning stops at that
layer (arXiv:2506.06443). Pos-EGNN layer 3 on lipophilicity-astrazeneca reached MAE 0.459 against
a prior best of 0.467; MolFormer layer 8 on ppbr-az reached 7.221 against 7.526.

Every encoder here pools the last layer. Taking it understates these encoders by about 5%.

### standard_comparator

What scenario 3 and 4 pretraining is compared against. Candidates: ECFP4 2048, the author's named
non-SMILES comparator; pdv 200; mordred 1613; mol2vec 300. All four are among the six
`pretrain_chembl_dnn` already supports.

A published pretrained model as a comparator is harder than it reads — see §6.

### untrained_controls

Three are already in the repository by accident and one is deliberate:

- `create_schnet` defaults `pretrained_target=None` (`molecular.py:1786`) and **no caller anywhere
  passes it**: `tests/find_best_hybrid_molecular.py:737`, `scripts/_generate_single_rep.py:82`,
  `scripts/verify_models.py:200`, `scripts/verify_server_models.py:83`,
  `tests/experiments/qm9_3d_comparison.py:282`. Two further disablers: any download exception falls
  through to a random SchNet after printing `Using randomly initialized model instead`
  (lines 1842-1851), and the module global `_SCHNET_MODEL` is built only when None (line 1828), so
  the first call in a process fixes the weights.
- `ft_schnet`, `ft_dimenetpp`, `ft_chemprop` as above.
- `mock_lora` (`tests/find_best_hybrid_molecular.py:1611`) returns random embeddings.

A random-weight encoder of matched width is the control the published literature almost never
reports, and it is the control the collaborator's result rests on. There is no seed for either 3D
builder: `grep -n manual_seed src/kirby/representations/molecular.py` returns nothing.

`c13_forward_compatibility` already carries a matched-width Gaussian control,
`D_random_established`, and the real features beat it in 50 of 54 cells.

### blueprint_source

Where the winning blueprint for scenarios 3 and 4 comes from.

**A blueprint names the whole pool, not the representations that got features.**
`find_best_hybrid.py:5485-5494` sets `reps = list(base_reps.keys())` plus the factory names, and
the comment at 1537-1544 says so. One blueprint unpickled from `herg_ki.db` on 2026-11-27 names 6
representations while its `allocation_per_fold` shows 2 getting features on each fold. So a
blueprint from a 10-representation pool demands all 10 on the pretraining dataset.

`apply_blueprint` raises `ValueError: base_reps is missing representations required by the
blueprint` (`find_best_hybrid.py:10038-10044`). No fallback, no partial rebuild.

What exists now:

| source | blueprints | state |
|---|---|---|
| `~/.cache/kirby/result_db/*.db` | 6,798, one per scored configuration | 2,498 go straight through `apply_blueprint`. 4,300 carry `rep_count` smaller than the pool with `rep_scores` empty, and `restrict_to_rep_count` raises (`find_best_hybrid.py:9948-9952`); `apply_blueprint` has no parameter for passing fold allocation scores. 17 of 18 files open; `flab_expression_koenig__ocv1f0s42.db` raises `StaleResultDB`. Local laptop runs, Sep-Oct 2026 |
| `tests/results/precompute_registry.json` | 5,810 over 45 datasets | 26 April, every one `rep_count=None`, model set still holds mlp and ridge, predates the representation-count setting. mlp is cut and greedy allocation is cut |
| `~/.cache/kirby/hybrid_registry.json` | 0 | holds one key, `__fake__`. A `get_best_hybrid` call with the default registry path raises KeyError for every real dataset |
| Papers 2 and 3 | one result_db row per scored configuration | per dataset, per outer fold, per search kind, per search space, each carrying its own pickled blueprint, under `result_db/<dataset>__ocv5f<fold>s42.db`. 37 of 39 datasets have a row in `decided_grid.json`; cdr_flexibility_h3 and cdr_flexibility_l3 do not |

The 4,300 can be rebuilt through `_shared.build_fold_hybrid` when the caller supplies that fold's
allocation scores (`tests/experiments/_shared.py:5996-6024`).

The 4,300 is the number that decides whether this experiment needs searches of its own. Which
(dataset, downstream model, budget) cells survive after removing them was NOT CHECKED.

### n_outer_folds

The papers run uses 5 (`RUN_SETTINGS.json` of the 2026-11-03 smoke run).

### evidence_grade

Whether the local October c01/c02/c13/c22 numbers count as evidence or only as design input.
They are local laptop runs. Memory records local runs as debug, not results. This plan uses them
only to size the design, and every number quoted from them says where it came from.

### collaborator_list

The collaborator's own datasets, encoders and pretrained models. Two arms stay blank until he
answers: `standard_comparator` where it means a published pretrained model, and whichever of his
datasets are not in the table above.

---

## 4. The arms

One unit is one (dataset, downstream model, arm) on all `n_outer_folds` outer folds, scored on
each fold's held-out molecules. Every search and every rebuild sees only that fold's training
molecules.

### Scenarios 1 and 2 — four arms

| arm | what is fitted | reuses |
|---|---|---|
| A1 | each classical representation on its own | the papers run's `c01` single-representation scores where the dataset and model match |
| A2 | a hybrid over the classical representations only | a new search, or a papers-run blueprint whose pool is classical only |
| A3 | each pretrained and each fine-tuned representation on its own | nothing; fine-tuned ones need a factory per fold |
| A4 | a hybrid over the pretrained and fine-tuned representations only | a new search |

A fifth arm answers the usefulness question without a new fit: for each winning hybrid, how many
features each representation contributed, read from
`feature_info[rep]['selected_indices']`. `c02` already writes this as `features_by_rep`.

The search reaches a fine-tuned representation only through `rep_factories`
(`find_best_hybrid.py:8479`), which calls each factory once per fold before the grid and caches the
embedding under `(dataset, factory_id, rep, fold, n_folds, seed)`. `cache=True` with
`rep_factories` requires `rep_factories_id` or it raises (line 9247). A factory that raises on any
fold aborts the run (lines 4305-4316).

### Scenarios 3 and 4 — six arms

All on one pretraining pair or corpus, one smaller dataset, dnn only.

| arm | pretraining input | then |
|---|---|---|
| B1 | none | fit from random init on the smaller dataset's hybrid. The floor |
| B2 | the hybrid, built on the larger dataset by the winning blueprint | carry the weights, no further training (scenario 3) |
| B3 | the same | continue training on the smaller dataset's hybrid (scenario 4) |
| B4 | `standard_comparator`, e.g. ECFP4 | carry the weights, no further training |
| B5 | the same | continue training on the smaller dataset's own `standard_comparator` features |
| B6 | random weights of the same width | continue training. The control the literature does not report |

B2 against B4 and B3 against B5 is the question. B1 and B6 are what make the answer readable.

**Two ways to carry the features, and they answer different questions.**

1. **Carry the exact selected features.** `kirby.hybrid.apply_feature_selection(new_reps,
   feature_info)` takes the stored (representation, index) pairs straight onto the new molecules.
   Width is identical by construction and the first layer needs no change. This is what
   `pretrain_rep_model_transfer.py` already does. It raises if a representation is absent
   (`hybrid.py:5729`) or an index is out of bounds (`hybrid.py:5736`).
2. **Re-run the blueprint on each side.** `apply_blueprint` scores the representations on the rows
   it is given, so the pretraining rebuild and each fold's rebuild pick their own features. This
   is the author's intended use, and it is the one that needs the first layer keyed by feature
   identity.

Both run. They are not the same experiment: (1) asks whether the pretrained weights transfer, (2)
asks whether the blueprint transfers.

### The identity-keyed first layer, for route 2

Needed only where the width moves fold to fold, which §3 measures as budget 1000 and above.

A feature's identity is the pair (representation name, integer index into that representation's own
full feature matrix), written at `hybrid.py:2669-2673`. There are no feature name strings:
`grep -rn feature_names src/kirby/*.py src/kirby/models/*.py` returns nothing.

Four facts about how stable that pair is:

- The index is in the representation's **original** feature space. Every allocation path maps back
  through `filter_index_maps` before returning (`hybrid.py:6299, 6360, 6412, 6438, 6486`). That is
  what makes the pair comparable between two rebuilds.
- Position in the matrix is **not** the identity. Within one representation the stored order is
  importance-descending (`hybrid.py:2665`), and `apply_feature_selection` builds the matrix in that
  stored order (`hybrid.py:5724, 5744, 5786`). Block order follows the representation dict's
  insertion order.
- The within-representation order is not one convention: the budget top-up re-sorts a
  representation's indices ascending (`hybrid.py:6591`).
- Index *j* means the same feature on two datasets for hashed fingerprints of fixed width and for
  fixed descriptor lists, and **not** for graph_kernel, pca_rep or sns without a reference
  featurizer.

Six places change. No code is written here.

1. `DNNRegressor._build_net` (`dnn.py:112`) and `fit` (`dnn.py:124`, net built at `dnn.py:185`),
   and the classifier twin (`dnn.py:307, 319, 356`). Both take a width and nothing else, and `fit`
   records only `input_dim_` (`dnn.py:265, 422`). They need the identity list for the matrix being
   fitted, a stored table mapping each identity to its row of the first layer's weight, and
   assembly of that weight by lookup, with a fresh initialisation for an identity absent from the
   table.
2. The checkpoint contract. `pretrain_chembl_dnn` writes `state_dict` and `input_dim`
   (`pretrain_chembl_dnn.py:219-220, 241`) and nothing that names a feature.
   `PretrainedDNNRegressor._load_pretrained` accepts or refuses on width equality alone
   (`dnn_pretrained.py:87`) and then loads by position (`dnn_pretrained.py:97`). The identity list
   goes beside the state_dict and is read here instead of the width check.
3. A route from the hybrid build to the estimator. `feature_info` reaches the caller from
   `apply_blueprint` (`find_best_hybrid.py:10244`), but `_outer_fit_score`
   (`find_best_hybrid.py:10652`) is given two matrices and a model name and builds the estimator
   through a no-argument factory. That signature and the dnn `fit` signature both take the identity
   list.
4. One ordering convention, read off `feature_info` rather than assumed, because of the two orders
   above.
5. Per-feature scaling. `_outer_fit_score` fits a `FeatureScaler` on the outer-training molecules
   of the dataset at hand (`find_best_hybrid.py:10700`), so a carried weight would meet a
   differently scaled input. The pretraining dataset's per-feature mean and spread travel with the
   weights — which the existing zero-shot path already does for one representation
   (`dnn_pretrained.py:109-124`, written at `pretrain_chembl_dnn.py:228`).
6. Representation admission: graph_kernel, pca_rep and sns without a reference featurizer are out,
   or built against a fixed vocabulary.

Pretrained weights are shared across folds and each fold trains its own copy. No leakage, because
every rebuild sees only that fold's training molecules.

---

## 5. Where this goes and what each unit writes

The experiments ledger is `tests/experiments/papers/plan.py`. Its steps are
`('reps',) + ('search_pruned','search_fast','search_full') + ('paper2','paper3')`.

This experiment is a new step after `paper3`, gated the same way: held until
`RUN_SETTINGS.json`'s `open` block names it, and held per dataset until that dataset has a row in
`decided_grid.json`. It reuses the run's representation files and its blueprints, so it never
rebuilds either.

Units:

- One per (dataset, downstream model, arm) for A1-A4, each over all outer folds.
- One per (pretraining pair, smaller dataset, arm) for B1-B6.
- One checkpoint-writing unit per (pretraining input, representation or blueprint), which every B
  arm on that pairing waits for.

Each unit writes one result file carrying: the arm name, the dataset, the downstream model, the
budget, per-fold held-out scores in the dataset's own metric, the features per fold, the features
contributed per representation, what loaded (§2.4), and the seconds it took.

A result file records `selection_scope`. The October Seagate copies are the only files in the
repository that carry that field, which is why nothing older can be cleared by inspection
(`SPLIT_AUDIT_2026_09_03.md`).

---

## 6. What the literature supports and what has to be built

The comparison in scenarios 3 and 4 is close to unpublished. One study pretrains a network whose
input is a fixed-width fingerprint and continues on a smaller related assay: Liu R, Laxminarayan S,
Reifman J, Wallqvist A, *Enabling data-limited chemical bioactivity predictions through deep neural
network transfer learning*, J Comput Aided Mol Des 36(12):867-878, 2022,
doi 10.1007/s10822-022-00486-x. A feedforward net on atom- and bond-based structural fingerprints,
over 400 assay pairs, dense layers transferred. It reports only relative change — mean squared
prediction error down about 10-20% per 0.1 increase in r-squared between source and target assay,
and the training set cut up to tenfold with no loss — and the full text is paywalled, so no
absolute score pair, no assay names and no code.

What **is** well published on fingerprint inputs is multi-task, not pretrain-then-continue, and
none of it carries a random-initialisation control:

| paper | input | numbers |
|---|---|---|
| Dahl, Jaitly, Salakhutdinov 2014, arXiv:1406.1231 | 3,764 Dragon descriptors | test AUC single-task to multi-task: PubChem 488912 0.664 → 0.816; 488915 0.723 → 0.873; 488917 0.835 → 0.894. Beat the best baseline on 14 of 19 assays, lost or tied on 3 |
| Kearnes, Goldman, Pande 2016, arXiv:1606.08793 | 1024-bit radius-2 circular fingerprint | 22 industrial ADMET endpoints; modest benefit over single-task, larger on smaller datasets. Per-task tables not read |
| MELLODDY, J Chem Inf Model 64(7):2331-2344, 2023 | ECFP6 folded to 32k bits | 2.6 billion measurements, 21 million compounds. Most tasks gained over 4% relative AUC-PR; 8 of 10 partners' median above 2.5% |

The cleanest absolute pretrained-against-scratch pairs on a fixed-width numeric input are from
materials, not chemistry. ElemNet, Nat Commun 10:5316, 2019: an 86-element composition vector,
341,000 DFT formation energies pretraining, MAE in eV/atom scratch against transfer — experimental
0.1299 ± 0.0136 to 0.0642 ± 0.0061 on 1,963 measurements; JARVIS 0.0546 to 0.0311; Materials
Project 0.0326 to 0.0248. Code at github.com/dipendra009/ElemNet. XenonPy, ACS Cent Sci
5(10):1717-1730, 2019: fingerprint-input nets, polymer heat capacity 5-fold MAE 0.043 to 0.028
cal/g·°C on 58 samples.

The closest ADME work pretrains a graph model and deliberately leaves the fingerprint one from
scratch. Peteani et al., J Chem Inf Model 66(22):14319-14329, 2026: 273,706 structures labelled by
teacher models, Chemprop v2 fine-tuned, prospective 2025 test MAE — rat liver microsome CLint 0.368
→ 0.300, human liver microsome CLint 0.367 → 0.287, LogD 0.487 → 0.338, LE-MDCK Papp 0.272 →
0.217. Its own Morgan-2048 XGBoost and DNN stay from-scratch baselines, and its comparator is not a
random initialisation. Code at github.com/Novartis/SurrogateADMEModels.

Negative evidence, with the caveat that decides whether it can be cited here:

| study | numbers | caveat |
|---|---|---|
| Praski, Adamczyk, Czech 2025, arXiv:2508.06199 | 28 pretrained models, 25 datasets. ECFP count fingerprint mean rank 8.80 of 28, mean AUROC 79.95%. Only CheMeleon rated better, 5.24 and 81.17%. On 11 external datasets 82.13% against 80.98%, hierarchical t-test inconclusive, probability of practical equivalence 66-68% | **classification only**, so it cannot support a claim about an RMSE endpoint. Its version 5 ranks CLAMP 13.56 of 28, below the ECFP count fingerprint; an earlier version had CLAMP as the only model beating it |
| Sieben, Zimmermann 2026, arXiv:2610.18651 | descriptor reconstruction median R² with a ridge head: descriptor-supervised pretraining 0.82, unsupervised 0.59, ECFP4 0.69, **random untrained encoder 0.34**. Assay-label and shuffled-descriptor pretraining scored below the random encoder. XGBoost on fingerprints plus descriptors won on virtually all 64 datasets. 8M to 100M molecules gave no robust gain | SMILES input, so the wrong shape — but it is the clearest published random-init control |
| Kirschbaum, Bande 2024, arXiv:2404.13393v2 | test MAE scratch against pretrained: HOPV 7.6 meV → 6.3 and 6.1 (helped); FreeSolv 0.56 kcal/mol → 0.61 and 0.66 (hurt) | graph input. Joint pretraining, discriminative fine-tuning and target-like filtering all failed to help |
| Jiang et al. 2021, J Cheminform 13:12 | descriptor-based SVM, XGBoost, RF and DNN beat GCN, GAT, MPNN and AttentiveFP on average; SVM best on the regression tasks | per-task RMSE in Table S4, not read |
| Deng et al. 2023, Nat Commun 14:6395 | 62,820 trained models; representation learning limited on most datasets, dataset size the factor that let it win | |
| Sun, Dai, Yu 2022, arXiv:2207.06010 | self-supervised GNN pretraining no statistically significant advantage in many settings | graph input |

And the counterweight, on ADME regression specifically, which is the endpoint type here: frozen
Uni-Mol2 embeddings had lower MAE than ECFP4 plus LightGBM on 5 of 6 regression endpoints
(arXiv:2607.19977 Table IV — LD50 0.682 against 0.659, Caco-2 0.338 against 0.477, PPBR 8.985
against 9.646, lipophilicity 0.602 against 0.627, ESOL RMSE 0.655 against 1.069, FreeSolv 1.169
against 2.178), while a from-scratch D-MPNN was lower still on 3 of the 4 TDC endpoints.

One more comparator worth knowing exists: a tabular model pretrained only on synthetic tables and
never on molecules (TabPFN) run in context on descriptors beat a fine-tuned molecular foundation
model over 58 tasks, win rate 67.2% on Mordred descriptors against 41.4% for fine-tuned CheMeleon,
with Morgan fingerprints at 22.4% (arXiv:2604.16123).

So the fair comparison the collaborator's result needs is one almost nobody publishes: the same
network, same input width, pretrained weights against a random initialisation. That is arm B6, and
it is the reason B6 is in the list.

---

## 7. Cost

**Measured.** `CHECK_PRECOMPUTE_TIMING_2026_06_02.md`, one number per (dataset, condition), wall
clock captured at the time the cache file was written. General/none: herg 29.44 h over 5,048
configurations at 7,772 molecules; qm9 20.64 h over 4,391 at 9,978; logd 20.45 h over 6,097 at
5,028; adme_rlm 8.41 h over 6,480 at 3,053; adme_hlm 7.97 h; adme_mdr1 6.90 h; caco2 4.53 h;
herg_ki 3.47 h over 6,480 at 1,412. The uncapped condition is far larger: adme_rlm 102.71 h,
adme_mdr1 80.40 h, caco2 63.31 h, herg_ki 47.97 h.

`TIME_ESTIMATION_METHOD.md` reconstructs every one of those walls to ±0.1% structurally, and
states its own caveats: node contention inflates the same code 3 to 48 times, cache state swings
cache-consuming experiments up to 3,700 times, and its forward per-configuration fit is ±35% in
the molecular mid-large regime. Both files predate the current model set and the
representation-count setting.

**Measured, per arm, from result files.** grover frozen 24.2 s and molformer LoRA 177.5 s, both one
fit on logd.

**Not measured.** A per-fold search under the current grid. A fine-tuning factory per fold. A
pretraining run of `pretrain_chembl_dnn` at any corpus size. The qm9 representation rebuild. Every
arm of this experiment.

The one lever that costs nothing: reuse. Papers 2 and 3 will have run the searches, and arms A1 and
A2 read their scores and blueprints rather than searching again. The 4,300 unusable blueprints in
§3 are the reason that lever may not reach every cell.

---

## 8. Open

- The settings file in §3. Every key.
- Whether `freeze_<model>`, `ft_schnet`, `ft_dimenetpp` and `ft_chemprop` stay under corrected
  names, move to `untrained_controls`, or come out.
- Whether `create_schnet` and `create_dimenetpp` are passed `pretrained_target`, dropped, or kept
  as named random-weight controls.
- Whether `lightgbm`'s `init_model` is wired, which is the only way scenarios 3 and 4 reach a
  model that is not the dnn.
- Which name the GraphMVP fine-tune carries: `gnn_graphmvp` or `graphmvp_ft`.
- How many (dataset, downstream model, budget) cells survive after the 4,300 unusable blueprints
  are removed. NOT CHECKED, and it decides whether this experiment needs searches of its own.
- `tests/experiments/pretrain_rep/pretrain_rep_experiment.py` already transfers a blueprint's
  settings or its exact selected features across datasets without touching weights, under
  conditions A, B, F, G, H. Whether those conditions are this experiment's A arms or a separate
  thing was not settled.
- The collaborator's list.
