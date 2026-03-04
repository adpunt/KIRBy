# KIRBy Pipeline Overhaul Roadmap

Started: 2026-03-04

This document collects all changes needed across the BOM, survey scripts, and
representation code based on new information. Each section corresponds to a
piece of information provided during the planning phase.

---

## 1. Fact-Check Audit (compass artifact)

External audit of the BOM. Cross-referenced against actual code — several of
the auditor's claims are themselves wrong. Sorted into confirmed vs rejected.

### 1A. Confirmed Issues (fix in BOM)

| Item | BOM claim | Actual (verified from code) | Action |
|------|-----------|----------------------------|--------|
| `chemberta` dim | 384 | **768** (RoBERTa-base, `DeepChem/ChemBERTa-77M-MTR`) | Fix dim in BOM §2A |
| `smited` dim | 768 | **512** (IBM SMI-TED light variant) | Fix dim in BOM §2A |
| `smited` deps | "HuggingFace" | Local IBM clone (`$KIRBY_MODELS_DIR/materials/`) | Fix deps in BOM §2A |
| `grover` dim | 4800 | **~5000** (GROVERlarge `both`, runtime-determined) | Document variability in BOM §2A |

### 1B. Rejected Claims (fact-checker was wrong, BOM is correct)

| Auditor claim | Why it's wrong |
|---------------|----------------|
| "SMITED is fabricated" | It's IBM's SMI-TED. Auditor searched HuggingFace; model is distributed as a local clone from IBM Research, not on HF Hub |
| "All 5 antibody LM dims are wrong" | Auditor compared raw model hidden_size, not KIRBy's output. KIRBy concatenates H+L for single-chain models: ESM2 1280×2=2560, AntiBERTy 512×2=1024, ProtT5 1024×2=2048. All correct. |
| AbLang2 "actual 480" | Auditor confused v1 (768) / v2 (480). Code uses `ablang2-paired` with `mode='seqcoding'` → 768D. Code and BOM agree. |
| IgBert "actual 1024 from ProtBERT-BFD" | IgBert is RoFormer-based, hidden_size=768. Code loads `Exscientia/IgBert`, mean-pools → 768D. BOM correct. |
| PDV "actual 208" | KIRBy uses a curated 200-descriptor list (verified in `DEFAULT_DESCRIPTOR_LIST`). BOM correct. |
| MolCLR "actual ~300" | Auditor looked at GIN backbone (300), not projection head. Code uses `feat_dim=512`. BOM correct. |
| WSI dims "all should be 768-1536" | Those are native tile encoder dims. KIRBy uses GigaSSL **slide-level** projections (Mendeley dataset) which are 512D. Code auto-detects from .npy, fallback is 512. BOM correct for what we use. |

### 1C. Needs Runtime Verification

| Item | BOM claim | Question | How to verify |
|------|-----------|----------|---------------|
| `geneformer` dim | 1152 | Auditor says 768; code reads `config.hidden_size` at runtime | `python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('ctheodoris/Geneformer'); print(c.hidden_size)"` |
| WSI .npy dims | 512 | Code auto-detects from data. Are the actual server .npy files 512D? | Check `first_emb.shape` from server cache |

### 1D. Suggested Additions (from auditor, evaluate priority)

| Domain | Suggestion | Priority | Status |
|--------|-----------|----------|--------|
| Molecular | Uni-Mol v2 (NeurIPS 2024) | LOW | **DONE** — `create_unimol_v2()` in molecular.py (768D) |
| Molecular | MAP4 fingerprints | LOW | **DONE** — `create_map4()` in molecular.py (1024D) |
| Antibody | FLAb2 benchmark (Dec 2025) | MEDIUM | **DONE** — 12 new datasets in `FLAB_DATASETS` (7 properties) |
| Antibody | ProteinGym, FLIP2, AbDev | LOW | Deferred — complementary but scope creep |
| TCGA | scGPT, UCE | LOW | **DONE** — `create_scgpt_embeddings()` (512D), `create_uce_embeddings()` (1280D) |
| DTA | DeepDTA, GraphDTA | **HIGH** | **DONE** — `tests/experiments/dta_e2e_comparison.py` (needs server run) |

---

## 2. Validation Addendum: New Models, Fine-Tuning & HP Reference

Source: `KIRBy_Validation_Addendum_Models_FineTuning.docx` (March 2026)

### 2A. New Representations to Add (priority order)

| # | Rep | Domain | Dim | Why | Effort | Deps |
|---|-----|--------|-----|-----|--------|------|
| 1 | **BulkRNABert** | TCGA | varies | Purpose-built for bulk RNA-seq. Geneformer devs explicitly said their model is NOT designed for bulk data. Critical replacement. | Medium (custom tokenizer, `log10(1+x)` preprocessing) | `transformers` (from source), `InstaDeepAI/BulkRNABert` |
| 2 | **CLAMP** | Molecular | 512 | ONLY model that statistically significantly beat ECFP in Praski 2025 benchmark (25 models × 25 datasets). Learned refinement of classical FPs. | Low (FP concat → MLP forward pass) | `ml-jku/clamp` |
| 3 | **ESM-C 600M** | Antibody/Protein | 1152 | Drop-in ESM-2 replacement. Matches ESM-2 3B quality at 5× fewer params. Dec 2024 release. | Low (same pipeline as ESM-2) | `esm` package, `EvolutionaryScale/esmc-600m-2024-12` |
| 4 | **R-MAT** | Molecular | 768 | Second-best in Praski benchmark (mean rank 7.23). 3D-aware transformer. | Low-Medium | GitHub repo |

### 2B. Geneformer → BulkRNABert (Critical Change)

**Problem:** Geneformer was pretrained on single-cell data (Genecorpus-104M). The
developers explicitly stated it was NOT designed for bulk RNA-seq and recommend
fine-tuning since bulk is out-of-distribution. KIRBy uses it on TCGA bulk data.

**Solution:** Add BulkRNABert (InstaDeep, ML4H 2024) as the primary foundation model
for TCGA. It was pretrained ON TCGA bulk RNA-seq via masked gene expression modeling.

**Implementation:**
- Checkpoint: `InstaDeepAI/BulkRNABert`
- Preprocessing: `np.log10(1 + gene_expression_array)`
- Extraction: `model(input_ids)["embeddings_4"].mean(axis=1)` (layer 4 = last)
- Install: `pip install --upgrade git+https://github.com/huggingface/transformers.git`
- Keep Geneformer for comparison but BulkRNABert should be the default

**Impact on BOM:** Geneformer stays but BulkRNABert gets added as §2C foundation model.
If BulkRNABert outperforms, Geneformer fine-tuning can be retired (saves GPU time).

### 2C. ESM-C 600M Resolves Multiple Issues

ESM-C 600M (Dec 2024) has embedding dim **1152**. This is notable because the BOM
claims Geneformer is 1152D — possible confusion between the two models. Either way,
ESM-C 600M should be added as a new antibody/protein rep:

- Drop-in replacement for ESM-2 650M (same mean-pool extraction)
- H+L concat → **2304D** (vs ESM-2's 2560D)
- Should be strictly better than ESM-2 650M per the Sci Reports 2025 eval
- Non-commercial license (note in BOM)

### 2D. Intermediate Layer Extraction (ESM-2/ESM-C)

Key finding: final layer is often NOT the best for frozen-embedding transfer. Averaging
layers 20–33 for ESM-2 650M improved unsupervised clustering ARI from 0.268 → 0.354.

**Implementation:** Pass `output_hidden_states=True`, average layers 20–33.
This could be a new rep variant: `esm2_midlayers` / `esmc_midlayers`.

**Priority:** MEDIUM — easy to implement, moderate expected improvement.

### 2E. CLAMP Integration

CLAMP = contrastive learning over concatenated ECFP + RDKit + MACCS fingerprints
through a 2-layer MLP. Trained on ChEMBL bioactivity data. NOT a traditional LM —
it's a supervised-contrastive refinement of classical fingerprints.

- 512D output from MLP encoder
- Code: `github.com/ml-jku/clamp`
- Input: SMILES → fingerprint concatenation → MLP forward pass
- CPU-friendly (~2GB GPU or CPU-only)
- The ONLY model statistically significantly outperforming ECFP (Praski 2025)

### 2F. Fine-Tuning Regime Guidance

The addendum confirms KIRBy's existing FT findings and provides size-based guidelines:

| Dataset Size | Recommendation | KIRBy status |
|-------------|----------------|-------------|
| < 200 | Frozen ONLY | Already doing this (antibody FT deleted) |
| 200–1,000 | LoRA rank 4, aggressive regularization | Most antibody datasets fall here — confirms frozen-only decision |
| 1,000–5,000 | LoRA rank 4–8 | FLAb Koenig (4,275) is in this range — could revisit |
| > 5,000 | Full FT or LoRA rank 8–16 | Davis (30K), KIBA (118K) — FT could help here |

**Key insight:** LoRA degradation on antibody data is EXPECTED behavior, not a bug.
Published evidence: frozen MLP (AUPR 0.684) beat LoRA (0.600) and full FT (0.623)
for PPI with ESM-2 650M (Sledzieski PNAS 2024).

**Action:** For the DTA domain (>5K samples), fine-tuning ESM-2/ESM-C on the protein
side could be worthwhile. This was never tested. Add to survey.

### 2G. Downstream Model HP Tuning

Current KIRBy approach vs recommended:

| Parameter | KIRBy current | Recommended | Action |
|-----------|--------------|-------------|--------|
| RidgeCV alphas | `logspace(-6, 6, 13)` | `logspace(-4, 4, 50)` | Widen grid to 50 points, narrow range |
| L2-normalize before Ridge | Not done | Recommended | Add option |
| XGB/LGBM tuning | Fixed defaults | Optuna 50-100 trials | Consider for Phase 2 full runs |
| RF max_features | Default | `"sqrt"` for high-dim embeddings | Fix in survey scripts |
| XGB colsample_bytree | Default (1.0?) | 0.3–0.5 for embeddings | Fix in survey scripts |

**Priority:** MEDIUM — these are survey script / model config changes, not rep code.
The biggest wins are `colsample_bytree` and `max_features` for high-dim embeddings.

### 2H. Dimension Confirmations from Cheat Sheet

The addendum's embedding cheat sheet (Table 10) independently confirms KIRBy's
antibody dims are correct:

| Model | Addendum dim | KIRBy dim (H+L concat) | Match? |
|-------|-------------|----------------------|--------|
| ESM-2 650M | 1280 | 2560 (×2) | YES |
| ESM-C 600M | 1152 | N/A (new) | — |
| ProtT5-XL | 1024 | 2048 (×2) | YES |
| AbLang2 | 768 (seqcoding) | 768 | YES |
| IgBert | 768 | 768 | YES |

This independently validates the §1B rejection of the fact-checker's antibody claims.

### 2I. WSI: ABMIL Aggregation

Current approach: mean pooling over GigaSSL tile embeddings.
Recommended: ABMIL (attention-based MIL) gives +2–7% AUC over mean pooling.

**Priority:** MEDIUM — requires new module but expected meaningful improvement on
WSI-based TCGA tasks. The current WSI results (0.78–0.85 lung, 0.51–0.65 BRCA)
have room to improve.

### 2J. CDR-Specific Embeddings (Antibody)

For binding tasks, extracting embeddings from CDR regions only (rather than full
sequence) gives +4–19% R² improvement per published results.

**Priority:** HIGH for binding-specific datasets (flab_binding_koenig, abcov_kd,
ddg_combined, abagym). Would require identifying CDR positions (ANARCI/IMGT
numbering) and masking the mean-pool to CDR residues only.

---

## 3. Missing Pipelines & Mismatched Representations

The current BOM evaluates everything on ADMET tasks and antibody biophysics — domains
where classical fingerprints and simple embeddings dominate. Several "failing" reps
aren't bad; they're on the wrong task. And several entire task categories are missing.

### 3A. Where "Bad" Reps Actually Work

| Rep | KIRBy result | Why it underperforms here | Where it shines |
|-----|-------------|--------------------------|-----------------|
| `schnet` | R² 0.06–0.65 | Tested on 2K QM9 subsample with SMILES-inferred geometry. Needs: (1) 50K+ samples, (2) DFT-quality 3D coords, (3) end-to-end training, not frozen embeddings | Full QM9 (134K) with 3D coords → chemical accuracy on most properties. Also MD17 forces, OC20 catalysis. |
| `molclr` / `graphmvp` | Mid-pack on ADMET | Graph topology less informative than FPs for single-molecule ADMET | Reaction prediction, scaffold hopping, low-data molecular tasks where graph topology matters |
| `mol2vec` | Never best on ADMET | Lightweight but beaten by FPs | DTA baseline — fast, cheap drug embedding for concatenation with protein embeddings |
| `chemberta` | Loses to ECFP | Small model, ADMET favors FPs | DTA sequence-based baseline (DeepDTA-style SMILES encoder). Fine-tuning on >5K samples. |

### 3B. Missing Pipeline 1: DTA Baselines (READY — infrastructure exists)

**Status:** Dataset loaders, pair concatenation, cold-target GroupKFold, and hybrid
search ALL exist. The baselines just never successfully ran (protein_kmer timeout,
server code was stale). This is the lowest-hanging fruit.

**What exists:**
- `src/kirby/datasets/dta.py` — Davis (30K pairs) and KIBA (118K pairs)
- `src/kirby/datasets/pdbbind.py` — PDBBind refined (~5316) and core (~290) sets
- `tests/dta_baseline_survey.py` — pairs drug+protein reps via `np.hstack`
- `tests/find_best_hybrid_dta.py` — hybrid search over drug+protein reps
- `tests/experiments/dta_e2e_comparison.py` — DeepDTA/GraphDTA vs KIRBy frozen comparison
- 11 drug reps: ecfp4, maccs, pdv, mol2vec, chemberta, molformer, selformer, clamp, rmat, map4, unimol_v2
- 7 protein reps: esm2, esmc, esm2_midlayers, amplify, aac, descriptors, kmer

**What's needed:**
- Actually run the survey on server (all drug×protein combos)
- Run DeepDTA/GraphDTA comparison on server
- Fill BOM §2D with fresh numbers

**Graph+sequence idea:** Already works. The hybrid search freely mixes drug graph/FP
reps with protein sequence reps. No code changes needed — just run it.

### 3C. Missing Pipeline 2: Full QM9 End-to-End (New)

**Purpose:** Demonstrate the boundary of the frozen-embedding paradigm. Full QM9
(134K molecules) with DFT-quality 3D coordinates and end-to-end SchNet/DimeNet++.

**Three-way comparison:**
1. Frozen SchNet embeddings + Ridge (current approach, expected to lose)
2. End-to-end SchNet on full QM9 (expected to win — this is what it was designed for)
3. Frozen ECFP4 + XGBoost (KIRBy's standard approach)

**What exists:**
- `src/kirby/datasets/qm9.py` — loads from PyTorch Geometric, has 3D coords in
  the PyG data objects but **currently discards them** (only extracts SMILES + labels)
- All 12 QM9 properties available (mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv)

**What's needed:**
- Expose 3D coordinates from the QM9 loader (they're in `data.pos`)
- Increase default `n_samples` from 2K to full 134K
- Add end-to-end SchNet training loop (PyG `SchNet` model, supervised)
- Possibly add DimeNet++ for comparison

**Why it matters for thesis:** Shows WHEN and WHY frozen embeddings break down
(answer: tasks requiring 3D geometry + sufficient data for end-to-end training).

### 3D. Missing Pipeline 3: Reaction Yield Prediction (New)

**Purpose:** Task category where graph/reaction-aware reps genuinely outperform ECFP.

**Standard benchmarks:**
- Buchwald-Hartwig amination (~3,955 reactions, 4 components)
- Suzuki-Miyaura coupling (~5,760 reactions)

**Relevant reps:**
- `rxnfp` — reaction fingerprints (Schwaller et al., Nature Machine Intelligence 2021)
- `drfp` — differential reaction fingerprints
- Standard: concatenated reactant/product ECFP4 as baseline
- GraphMVP/MolCLR graph reps of reactants (where graph topology actually helps)

**What exists (implemented session 2):**
- `src/kirby/datasets/reaction_yields.py` — Buchwald-Hartwig (~3955) + Suzuki-Miyaura (~5760)
- `src/kirby/representations/reaction.py` — DRFP (2048D) + concat ECFP4
- `tests/run_reaction_yield_survey.py` — full survey script (DRFP + concat ECFP4 × Ridge/RF/XGB × 10 splits)

**What's needed:** Run on server (`pip install drfp`).

### 3E. Missing Pipeline 4: Structure-Based Binding Affinity (New)

**Purpose:** PDBBind refined set. Unlike Davis/KIBA (sequence-based), PDBBind has
co-crystal structures — the 3D arrangement of drug in the binding pocket.

**KIRBy approach:** Frozen protein pocket embeddings + frozen ligand embeddings →
Ridge/XGBoost. Compare against OnionNet, PLIP features, structure-based methods.

**What exists (implemented session 3):**
- `src/kirby/datasets/pdbbind.py` — Loader for refined (~5316) and core (~290) sets
- Sequence-based approach: extracts protein seq from PDB + ligand SMILES from SDF
- Returns DTA-compatible dict → plugs into existing DTA pipeline
- Registered in `dta_baseline_survey.py`, `find_best_hybrid_dta.py`, `run_baseline_survey.py`

**What's needed:** Manual data download from pdbbind.org.cn (registration required).

### 3F. Multi-Modal Concatenation (Already Works)

The hybrid system already handles multi-modal concatenation transparently:

- **DTA:** `Dict[str, ndarray]` with both `drug_*` and `protein_*` entries → greedy
  allocation freely mixes features across modalities. No special code needed.
- **TCGA:** Expression + CNV + RPPA + miRNA + WSI entries already coexist in the
  same rep dict. The `multimodal` preset tests this.
- **Antibody:** Could add structural features (net charge, CDR length, pI) alongside
  LM embeddings. Framework supports it — just add the rep functions.

No architectural changes needed. The infrastructure is modality-agnostic by design.

### 3G. Pipeline Priority Summary

| Pipeline | Exists? | Effort | Impact | Priority |
|----------|---------|--------|--------|----------|
| DTA baselines | YES (code complete, need server run) | LOW | HIGH | **P0** — run surveys + DeepDTA/GraphDTA comparison |
| Full QM9 (end-to-end) | YES (loader + comparison script) | LOW | HIGH (thesis argument) | **P1** — run 3-way comparison on server |
| Reaction yield | YES (datasets + reps + survey) | LOW | MEDIUM | **P2** — run survey on server (`pip install drfp`) |
| PDBBind structure-based | YES (loader + DTA integration) | LOW | LOW (scope) | **P3** — needs manual data download |

---

## 4. [PLACEHOLDER — next piece of information]

---

## Action Summary

### Immediate (BOM text fixes, no code changes)
- [x] Fix `chemberta` dim: 384 → 768 in BOM §2A
- [x] Fix `smited` dim: 768 → 512, deps: "HuggingFace" → "smi-ted (IBM)" in BOM §2A
- [x] Document `grover` dim variability (~5000 for GROVERlarge 'both') in BOM §2A
- [ ] Verify `geneformer` dim on server (1152 vs 768 vs ESM-C confusion)

### P0: Run Existing DTA Pipeline
- [x] Fix protein_kmer: LightGBM feature importance pre-filter 8000D → 200D
- [ ] Run `dta_baseline_survey.py` on server (all drug×protein combos)
- [ ] Run `find_best_hybrid_dta.py` to validate graph+sequence mixing
- [ ] Fill BOM §2D with fresh numbers

### High Priority (new reps — code + BOM)
- [x] Add **BulkRNABert** as TCGA foundation model (256D, maps HUGO→Ensembl, log10(1+TPM))
- [x] Add **CLAMP** as molecular rep (768D, only model beating ECFP statistically)
- [x] Add **ESM-C 600M** as antibody rep `esmc` (2304D H+L concat) and protein rep `protein_esmc` (1152D)
- [x] Add ESM-C as `protein_esmc` in DTA pipeline (fast+lm and full presets)
- [x] Add CLAMP as `drug_clamp` in DTA pipeline (fast+lm and full presets) + baseline survey

### DeepDTA/GraphDTA Comparison
- [x] Add DeepDTA + GraphDTA-GIN end-to-end comparison experiment (`tests/experiments/dta_e2e_comparison.py`)
- [ ] Run on Davis/KIBA with cold-target GroupKFold — needs server

### P1: Full QM9 Pipeline
- [x] Expose 3D coordinates from QM9 loader (`load_qm9_with_3d()` in qm9.py)
- [x] Increase QM9 default sample size to full 134K (n_samples=None → full dataset)
- [x] Add end-to-end SchNet training loop (`tests/experiments/qm9_3d_comparison.py`)
- [ ] Run three-way comparison (frozen SchNet vs end-to-end vs ECFP+XGB) — needs server

### Medium Priority (improvements to existing pipeline)
- [x] CDR-specific antibody embeddings (esm2_cdr, esmc_cdr, antiberty_cdr, prot_t5_cdr)
- [x] ESM-2 intermediate layer extraction (layers 20-33): `esm2_midlayers` antibody + `protein_esm2_midlayers` DTA
- [x] ABMIL aggregation for WSI: frozen gated attention + trainable ABMIL + comparison experiment
- [x] Fix RF `max_features="sqrt"` and XGB `colsample_bytree=0.5` in all survey scripts
- [x] Widen RidgeCV alpha grid to `logspace(-4, 4, 50)` in all scripts
- [x] Fix DTA survey: `Ridge(alpha=1.0)` → `RidgeCV(alphas=logspace(-4,4,50))`
- [x] Fix `_shared.py`: `Ridge(alpha=1.0)` → `RidgeCV`, add max_features/colsample_bytree
- [x] Update HP tuning search space: RF `max_features`, XGB `colsample_bytree` lower values
- [x] DTA protein-side LoRA fine-tuning (`dta_protein_lora.py`, `--finetune protein_esm2_lora`)

### P2: Reaction Yield Pipeline (if scope allows)
- [x] Add Buchwald-Hartwig and Suzuki coupling dataset loaders (`datasets/reaction_yields.py`)
- [x] Add DRFP + concat ECFP4 reaction fingerprints (`representations/reaction.py`)
- [x] Create reaction baseline survey script (`tests/run_reaction_yield_survey.py`)
- [ ] Test GraphMVP/MolCLR on reactions (where graph topology matters) — needs server
- [ ] Run reaction yield survey — needs server + `pip install drfp`

### Low Priority (evaluate later)
- [x] R-MAT molecular rep (768D, 3D-aware transformer via huggingmolecules)
- [x] Optuna HP tuning for downstream models (`--tune 50` flag + `tune_xgboost_optuna()` in _shared.py)
- [x] L2-normalize embeddings before Ridge (`--l2-norm` flag → `ridge_l2norm` model)
- [x] FLAb2 as antibody benchmark dataset (12 new datasets spanning 7 properties)
- [x] DNA methylation modality for TCGA (HM450 via Xena, PCA50 + HVG reps)
- [x] PDBBind structure-based binding (loader for refined/core sets, sequence-based)
- [x] Uni-Mol v2 (768D, NeurIPS 2024), MAP4 (1024D fingerprints)
- [x] scGPT (512D), UCE (1280D) genomic foundation models
- [x] AMPLIFY 350M (960D protein LM, antibody + DTA)
- [ ] TwinBooster, GP-MoLFormer (deferred)

*(Duplicate section removed — see above for canonical checklist)*
