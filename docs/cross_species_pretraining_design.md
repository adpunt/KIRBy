# Cross-Species Pretraining Design (T-PT.0)

**Status:** Decisions locked 2026-05-13. T-PT.1 shipped 2026-05-13. T-PT.2 shipping 2026-05-13.
**Date:** 2026-05-13. **Renamed from Greek (PT-α…ζ) to plain names 2026-05-13.**
**Scope:** Defines protocols for the 6 cross-species pretraining variants — `same_target_transfer`, `species_token_multispecies`, `rodent_to_vet`, `microsome_transfer`, `foundation_hybrid`, `joint_multitask` — listed in `CROSS_SPECIES_ROADMAP.md` §3 / §5 / §6.

**Variant name map (old → new), for grep continuity:** PT-α=`same_target_transfer`, PT-β=`species_token_multispecies`, PT-γ=`rodent_to_vet`, PT-δ=`microsome_transfer`, PT-ε=`foundation_hybrid`, PT-ζ=`joint_multitask`.

---

## 1. Why pretraining

T-Mol.2b confirmed vet ChEMBL is sparse:

| species | tax_id | total IC50/Ki activities (strict) | concentrated on |
|---|---|---|---|
| dog | 9615 | 1,055 | 1,032 on TRPM8 alone |
| cat | 9685 | 59 | none usable for QSAR |
| cow | 9913 | 18,958 | rich (173 targets) — in-vitro pharmacology heritage |
| horse | 9796 | 6,306 | 5,609 on cholinesterase |

Of the human anchors (KCNH2 / PTGS2 / DRD2): only **3 of 12** non-human combos have ≥20 rows (dog PTGS2 = 110, cow PTGS2 = 28, cow DRD2 = 406). Direct cross-species QSAR/ADMET hits the small-data wall.

Literature consensus (MolPMoFiT, MTL-BERT, CL-MFAP, Imrie 2018): pretraining on bulk chemistry data + downstream fine-tune is the canonical fix when label counts drop below ~200. KIRBy already has the Imrie infra in `src/kirby/models/pretrain_chembl_dnn.py` + `dnn_pretrained.py` and a working sweep in `tests/experiments/imrie_transfer_learning.py` (C4). X-PT is a generalisation of C4 — same pretrain → fine-tune backbone, different corpora.

**Gate.** foundation_hybrid runs first as the cheap baseline anchor (foundation-model frozen reps + small head, no new pretraining). If frozen ChemBERTa/MolFormer/MolE reps already beat from-scratch DNN on small vet data, that is the cleanest possible result and the expensive same_target_transfer–ζ runs can be deprioritised. **same_target_transfer–ζ are conditional on foundation_hybrid failing to dominate.**

---

## 2. Variants — corpus / fine-tune / what it tests

All fine-tunes target the vet tasks built in X-Mol (`organism='dog'|'cow'|'horse'|'cat'` slices of `chembl_bioactivity` / `cox2` / `drd2` / `canine_cyp`). Test rows are held out from every pretraining corpus by InChIKey, using the existing `collect_downstream_inchikeys()` mechanism already used by `pretrain_chembl_dnn.py`.

### same_target_transfer — Same-target transfer

**Question.** Does human chemistry at target T transfer to dog/cow at the same target T?

| | |
|---|---|
| Corpus | `chembl_bioactivity(organism='human', max_targets=200)` — current C4 corpus, ~500K activities. |
| Fine-tune set | Dog COX-2 (110), Cow PTGS2 (28), Cow DRD2 (406), Cow CA4 (1,541) — the four ≥20-row combos with a clear human ortholog. |
| Model | Reuse `PretrainedDNNRegressor` with `freeze_layers=0` (full fine-tune) and `freeze_layers=-1` (linear probe). |
| Reps tested | `ecfp4, maccs, pdv, mol2vec, mordred, avalon` — same set as C4. |
| Splits | ScaffoldGroupKFold, 5-fold. |
| Baselines (mandatory) | `dnn_scratch`, `ridge`, `kirby_hybrid` (from current `find_best_hybrid`), `zero_shot` (pretrained model, no fine-tune). |

### species_token_multispecies — Multi-species + species token

**Question.** Does joint training on all species at all targets with a species token outperform per-species pretrains?

| | |
|---|---|
| Corpus | Union of `chembl_bioactivity(organism=s)` for s ∈ {human, rat, mouse, dog, cow, horse}, with `species_token` (**learned 16-D embedding, locked**) + `target_token` (target_chembl_id one-hot, top-N targets only). |
| Target-token sweep | **N ∈ {10, 50, 200}** — all three run as a deliberate ablation. N=10 = vet eval targets + their human/rodent orthologs only; N=50 = targets with reasonable cross-species coverage; N=200 = matches `pretrain_chembl_dnn` default. |
| Fine-tune | Same vet eval targets as same_target_transfer, evaluating both "no species token" and "vet species token" inference. |
| Model | New class `SpeciesTokenDNNRegressor` extending `PretrainedDNNRegressor`. Concatenates a learned 16-D species embedding + N-D target one-hot to the molecular rep before the first linear layer. |
| Held-out toggle | Run inference twice per test compound: with the test-species token vs forcing human token. Difference quantifies what the species token learned. |
| Baselines | same_target_transfer (no species token), per-species RidgeCV, KIRBy hybrid (standard rep set). |

### rodent_to_vet — Rodent → vet transfer

**Question.** Is "evolutionary closeness" real? Are rat+mouse closer to dog/cat than human is for these endpoints?

| | |
|---|---|
| Corpus | `chembl_bioactivity(organism=s, max_targets=200)` for s ∈ {rat (tax_id=10116), mouse (tax_id=10090)} unioned. Excludes human. |
| Fine-tune | Same vet eval targets as same_target_transfer. |
| Model | Reuse `PretrainedDNNRegressor`. |
| Comparison | RMSE delta vs same_target_transfer at every (rep × target). Positive delta = rodent transfer beats human transfer = "evolutionary closeness" supported. |

### microsome_transfer — Microsomal stability transfer

**Question.** Does microsomal clearance transfer organ-system-wise? HLM+RLM → DLM/FLM.

| | |
|---|---|
| Corpus | Polaris HLM (`adme_hlm`, 3,087 cpds) + RLM (`adme_rlm`, 3,054 cpds). Both already loaded in KIRBy. |
| Fine-tune | DLM (dog liver microsomes, X-Data.5) + FLM (cat/feline liver microsomes, X-Data.6). |
| Model | `PretrainedDNNRegressor` with `freeze_layers ∈ {0, -1}`. |
| Baselines | from-scratch DLM/FLM, RidgeCV on the rep, KIRBy hybrid on DLM/FLM. |
| Gating | **Blocked on X-Data.5 + X-Data.6 loaders shipping.** |

### foundation_hybrid — KIRBy hybrid search on vet targets ← RUNS FIRST

**Question.** Can `find_best_hybrid` on KIRBy's existing molecular rep set beat from-scratch baselines on small vet ChEMBL tasks, without any cross-species pretraining?

| | |
|---|---|
| Corpus | None. No new pretraining step. |
| Method | `find_best_hybrid` on the **General-Tier molecular reps from PIPELINE_READINESS_TODO.md**. These are the reps that the surveys actually validated on caco2, herg_ki, logd, qm9, adme_hlm, adme_rlm, adme_mdr1 — the union of General Tier across the 7 molecular datasets. |
| Rep set (18 reps, audit-grounded 2026-05-13) | `ecfp4`, `maccs`, `avalon`, `atom_pair`, `top_torsion`, `rdkfp` (6 fingerprints); `pdv`, `mordred` (2 descriptors); `mol2vec`, `mhg_gnn`, `grover`, `graphmvp`, `graph_kernel` (5 GNN/learned); `chemberta`, `chembert`, `molformer`, `smited`, `selformer` (5 SMILES LM). Excluded: `mole`, `gp_molformer`, `minimol` — wired but **zero measured numbers** in `tests/results/`, marked "running" in BOM 2A/10A. They can be added later once benchmarked. |
| Eval targets | 4 vet datasets, target IDs locked in `docs/cross_species_vet_eval_targets.json` (resolved 2026-05-13 via `scripts/lookup_vet_target_ids.py`): cow CA4 = CHEMBL281 (n=1,541), cow DRD2 = CHEMBL3998 (n=406), dog PTGS2 = CHEMBL4033 (n=110), cow PTGS2 = CHEMBL3331 (n=28). |
| Splits | 5-fold ScaffoldGroupKFold (3-fold fallback at n < 150 — i.e. dog PTGS2 + cow PTGS2). |
| Hybrid settings | `FULL_BUDGETS`, `metric='rmse'`, `augmentation_strategy='none'`, `adaptive_pruning=0.25` — matches C4 (`imrie_transfer_learning.py`). |
| Mandatory baselines per (rep × target) | 1. `ridge` (RidgeCV) — simple control. 2. `dnn_scratch` — DNNRegressor with no pretraining. |
| Decision rule | Reported per (target). foundation_hybrid passes if it beats both baselines by ≥0.05 RMSE on ≥3 of 4 vet targets. No kill-switch on the other variants regardless of outcome (Q5 lock). |

### joint_multitask — Joint multi-target multi-species end-to-end (molecular-only)

**Question.** Best-case ceiling: one model, all KIRBy *molecular* data, species + target tokens, no transfer step.

| | |
|---|---|
| Corpus | **Molecular-only** (Q7 lock — matches the C04/C05/C16/C19 KIRBy precedent of single-domain experiments): all of `chembl_bioactivity` (human + rat + mouse + dog + cow + horse) + Polaris HLM + RLM + DLM + FLM, with species_token (16-D learned) + target_token + assay_type_token. **Excludes reaction (CGR) and DTA.** |
| Model | Multi-head DNN (one regression head per (species, target)) or shared backbone with token conditioning. |
| Fine-tune | Same vet eval held-out targets. Per-target read-off only. |
| Baselines | Best result from same_target_transfer–ε per (rep × target). |
| Gating | **Blocked on same_target_transfer through foundation_hybrid results.** Runs last. |

---

## 3. Reuse map vs C4 Imrie infra

| Component | Status | What changes for X-PT |
|---|---|---|
| `src/kirby/datasets/chembl_bioactivity.py` | shipped with `organism` kwarg (T-Mol.2c) | unchanged; called per-species by the new corpus assembler |
| `src/kirby/datasets/pretrain_corpora.py` | shipped (T-PT.1, 2026-05-13) | named corpora: `human_chembl`, `rodent_chembl`, `mammalian_chembl`, `human_microsomes`, `microsome_all`, `multispecies_chembl`. Composes per-species `chembl_bioactivity` calls + Polaris microsomes. Honors `exclude_inchikeys` and provides `collect_vet_inchikeys()` cache-only helper. Output schema matches `download_chembl_bioactivity`. Rat (10116) + mouse (10090) added to `SPECIES_REGISTRY` for rodent_to_vet. |
| `src/kirby/models/pretrain_chembl_dnn.py` | shipped | accept `--corpus <name>` flag wired to `pretrain_corpora`. Output path becomes `{output_dir}/{corpus}/{rep}_{hs}.pt`. No change to training loop. |
| `src/kirby/models/dnn_pretrained.py` | shipped | unchanged. Used for same_target_transfer/γ/δ. |
| `src/kirby/models/species_token_dnn.py` | **new (T-PT.5)** | extends `PretrainedDNNRegressor`; appends species + target token vectors to the molecular rep before the first linear layer. Required for species_token_multispecies. |
| `src/kirby/models/multitask_dnn.py` | **new (T-PT.7)** | multi-head DNN, one regression head per (species, target). Required for joint_multitask. |
| `tests/experiments/cross_species/x_pt_*.py` | **new (T-PT.2–T-PT.7)** | one experiment script per variant. Follow `imrie_transfer_learning.py` structure (datasets × reps × conditions matrix, ScaffoldGroupKFold, JSON results). |
| `tests/experiments/cross_species/rd_analysis.py` | **new (T-PT.8)** | given a SMILES, run all fitted PT-* models, return per-(species, target) prediction with conformal interval + agreement flag. |
| `src/kirby/uq/conformal.py` | **not yet built (T-UQ.1)** | MAPIE wrapper. Hard dependency for PT-* per-prediction intervals. |

**No code changes to `find_best_hybrid` for X-PT.** Hybrids are an *input* baseline, called as a non-TL condition exactly the way C4 already calls them.

---

## 4. Evaluation protocol

**Splits.** ScaffoldGroupKFold, 5-fold. `assign_scaffold_groups` from `kirby.utils.scaffold_groups`. No random splits anywhere (per standing user rule).

**Primary metrics.** RMSE (regression), Spearman ρ, MAE. Conformal coverage @ α = 0.1 (target 90%), mean interval width.

**Leakage controls.** Every pretraining corpus filtered by `collect_downstream_inchikeys()` against ALL vet eval test InChIKeys, not just the human ones. Extend `collect_downstream_inchikeys` to include vet target datasets.

**Conformal split policy (Q6 lock).** Tiered by training set size:

| n (training set) | Policy | Reason |
|---|---|---|
| n ≥ 500 | **Fold-out calibration** — use 1 of 5 folds as a permanent calibration set, 4 for train/test. | Calib size ≥ 100; meets MAPIE stability guidance. |
| 150 ≤ n < 500 | **Within-fold calibration** — split each training fold 80/20 train/calib. | Fold-out calib size too small. |
| n < 150 | **No conformal interval** — report point estimate + bootstrap CI only. | Conformal coverage unstable at this size. |

Vet eval target placement under this rule: Cow CA4 (n=1,541, fold-out), Cow DRD2 (n=406, within-fold), Dog COX-2 (n=110, bootstrap-only), Cow PTGS2 (n=28, bootstrap-only).

**Mandatory baselines per PT-*.**
1. `dnn_scratch` — same architecture, no pretrained init.
2. `ridge` — RidgeCV(alphas=logspace(-4,4,50)).
3. `kirby_hybrid` — `find_best_hybrid` on the **standard rep set** (`ecfp4`, `maccs`, `pdv`, `mol2vec`, `mordred`, `avalon`), with `FULL_BUDGETS`, `metric='rmse'`, `augmentation_strategy='none'`, `adaptive_pruning=0.25`. Distinct from foundation_hybrid (which uses the foundation-augmented rep set).
4. `zero_shot` — pretrained model frozen, predict on vet test directly (no fine-tune). Skipped for foundation_hybrid (no separate pretrained model exists for it).

**Reporting.** One JSON per experiment in the run dir, plus a summary table: per (dataset × rep × condition) → mean ± std across folds. Same shape as `imrie_transfer_learning.py` output.

**Decision rules.**
- "Pretraining works for variant X" ⇔ ≥0.05 RMSE improvement over `dnn_scratch` AND non-overlapping 1σ bands AND ≥3 of 4 reps reproduce the lift.
- "Evolutionary closeness real" (rodent_to_vet vs same_target_transfer) ⇔ ≥0.03 RMSE lift of rodent_to_vet over same_target_transfer on ≥3 of 4 vet eval combos.
- "Species token earns its keep" (species_token_multispecies) ⇔ toggling species_token changes prediction RMSE by ≥0.03 on held-out vet rows.
- **No kill-switch on same_target_transfer / rodent_to_vet (Q5 lock).** same_target_transfer and rodent_to_vet run to completion regardless of foundation_hybrid outcome, to produce a publishable "foundation vs custom-pretrain" ablation.

---

## 5. Files written by X-PT (anchor for T-PT.1–T-PT.8 acceptance)

```
src/kirby/datasets/pretrain_corpora.py                   # T-PT.1
src/kirby/models/species_token_dnn.py                    # T-PT.5
src/kirby/models/multitask_dnn.py                        # T-PT.7
src/kirby/uq/conformal.py                                # T-UQ.1 (dep)
tests/experiments/cross_species/__init__.py
tests/experiments/cross_species/foundation_hybrid_vet.py  # T-PT.2 (first)
tests/experiments/cross_species/same_target_transfer.py   # T-PT.3
tests/experiments/cross_species/rodent_to_vet.py        # T-PT.4
tests/experiments/cross_species/species_token_multispecies.py   # T-PT.5
tests/experiments/cross_species/microsome_transfer.py    # T-PT.6
tests/experiments/cross_species/joint_multitask.py          # T-PT.7
tests/experiments/cross_species/rd_analysis.py         # T-PT.8
docs/cross_species_pretraining_design.md                 # THIS FILE (T-PT.0)
```

SLURM scripts (one per variant) deferred to each T-PT.* implementation task — no `--output`/`--error` overrides per repo rule.

---

## 6. Locked decisions (2026-05-13)

| Q | Topic | Lock |
|---|---|---|
| 1 | Pretrain corpus size | **500K per corpus** (matches C4). |
| 2 | foundation_hybrid method | **`find_best_hybrid` with a foundation-augmented rep set** (subset of wired `chemberta`, `molformer`, `gp_molformer`, `mole`, `minimol`). Distinct from the `kirby_hybrid` baseline (standard rep set). (Previous draft named a 4th rep "Polaris-MolE" that does not exist; corrected 2026-05-13.) |
| 3 | Species token | **Learned 16-D embedding.** Lets model discover dog ≈ cat ≈ mammal structure; collapses to one-hot if not. |
| 4 | species_token_multispecies target token N | **Sweep all three: N ∈ {10, 50, 200}.** Deliberate ablation. ~3× compute on species_token_multispecies only. |
| 5 | foundation_hybrid kill switch | **None — same_target_transfer and rodent_to_vet run regardless of foundation_hybrid outcome** for the publishable ablation. |
| 6 | Conformal split | **Tiered:** fold-out at n ≥ 500, within-fold at 150 ≤ n < 500, bootstrap-only at n < 150 (see §4 table). |
| 7 | joint_multitask scope | **Molecular-only** (matches KIRBy single-domain experiment precedent). Excludes reaction (CGR) + DTA. |

---

## 7. Acceptance — what "T-PT.0 done" means

This document is accepted when the user signs off on §6 open questions and either confirms or amends §2 variant specs and §4 decision rules. No code lands until then.

Once accepted, downstream task acceptance becomes:

- **T-PT.1** — `pretrain_corpora.compose('rodent_chembl')` returns a DataFrame with rat + mouse rows, leakage-filtered against vet test InChIKeys. Unit test passes.
- **T-PT.2** (foundation_hybrid) — dry-run on dog COX-2 with frozen ChemBERTa + RidgeCV head completes; full sweep submits to ARC.
- **T-PT.3** (same_target_transfer) — dry-run completes; result either beats `dnn_scratch` per §4 decision rule or fails fast with documented "no transfer signal."
- **T-PT.4** (rodent_to_vet) — RMSE delta vs same_target_transfer computed per (rep × target); evolutionary-closeness verdict explicit.
- **T-PT.5** (species_token_multispecies) — species-token toggle measurable; ≥0.03 RMSE delta one direction or other.
- **T-PT.6** (microsome_transfer) — gated on X-Data.5 + X-Data.6.
- **T-PT.7** (joint_multitask) — gated on same_target_transfer–ε results.
- **T-PT.8** — one-table per-species profile output on the pole-network demo compound set.

---

## 8. References

- Imrie et al. (2018). *JCIM* 58(11):2319-2330 — transfer-learned DNN backbone (C4 prior art).
- MolPMoFiT — inductive TL for QSAR. *JCheminf* 12 (2020).
- MTL-BERT — multitask BERT for molecular property. *Research* (Science Partner Journal) 2022.
- CL-MFAP — contrastive learning molecular foundation for ADMET.
- Two-Stage Pretraining for Molecular Property Prediction in the Wild — arXiv:2411.03537.
- TACTIC — species-conditional multi-task QSAR. (species_token_multispecies precedent.)
