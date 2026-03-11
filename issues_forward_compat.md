# KIRBy Forward Compatibility & Downstream Model Audit

Research date: 2026-03-10

---

## PART 1: Forward Compatibility — New Models (2024-2026)

The thesis: any new model can be converted to a fixed-size embedding vector and plugged into the KIRBy framework with ZERO code changes. Below are candidate models to prove this.

---

### 1.1 MINT — Multimeric Interaction Transformer

- **URL**: https://github.com/VarunUllanat/mint
- **Paper**: https://www.nature.com/articles/s41467-025-67971-3 (Nature Communications, Jan 2026)
- **Domain**: Protein-Protein Interactions
- **Released**: January 2026
- **Can embeddings be extracted?** YES — fixed-size 1280D vector per protein, or 2560D for a protein pair (concat of two 1280D vectors with `sep_chains=True`).
- **How to extract embeddings**:
  ```python
  from mint.helpers.extract import load_config, CSVDataset, CollateFn, MINTWrapper
  import torch
  cfg = load_config("data/esm2_t33_650M_UR50D.json")
  wrapper = MINTWrapper(cfg, checkpoint_path, device='cuda:0')
  # dataset = CSVDataset(csv, 'Protein_Sequence_1', 'Protein_Sequence_2')
  embeddings = wrapper(chains, chain_ids)  # shape (batch, 1280)
  ```
  Checkpoint: `wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt`
- **Architecture**: Built on ESM-2 650M backbone + cross-attention for multi-chain context. Trained on 96M PPIs from STRING.
- **Benchmarks**: 29% improvement on SKEMPI binding affinity (sequence-only SOTA). Strong on PPI classification, mutational effects.
- **KIRBy relevance**: **HIGH for DTA domain**. MINT embeddings are PPI-context-aware, unlike vanilla ESM-2. Can directly replace/augment `esm2` protein rep in DTA pipeline. For DTA, the target protein interacts with a drug — MINT captures that interaction bias.
- **Forward-compat experiment**: Plug MINT 1280D into DTA pipeline as a new protein rep. Compare: MINT alone vs ESM-2 alone vs MINT+PDV hybrid.

---

### 1.2 ESM3 (ESM3-open 1.4B)

- **URL**: https://github.com/evolutionaryscale/esm
- **Paper**: https://www.evolutionaryscale.ai/blog/esm3-release (Science, Jan 2025)
- **Domain**: Protein (multimodal: sequence + structure + function)
- **Released**: June 2024 (paper), open model Jan 2025
- **Can embeddings be extracted?** YES — per-residue embeddings from hidden layers, mean-pool to fixed-size vector. Open model is 1.4B params.
- **How to extract embeddings**:
  ```python
  from esm.models.esm3 import ESM3
  from esm.sdk.api import ESMProtein, GenerationConfig
  model = ESM3.from_pretrained("esm3_sm_open_v1")
  protein = ESMProtein(sequence="MKTL...")
  # Forward pass returns per-residue hidden states
  # Mean-pool over residue dimension -> fixed-size vector
  ```
  Embedding dimension: depends on model, open 1.4B likely ~1536D or similar.
- **Architecture**: 98B param full model (open 1.4B). Multimodal: jointly reasons over sequence, structure, function as discrete token tracks. Pre-LN transformer.
- **Benchmarks**: Designed a functional green fluorescent protein with only 58% sequence identity to known GFPs (equivalent to 500M years of evolution). State-of-art on contact prediction.
- **KIRBy relevance**: **HIGH for protein & antibody**. ESM3 is the successor to ESM-2 (already in KIRBy). Multimodal training means it captures structure/function info even from sequence alone.
- **Forward-compat experiment**: Extract ESM3-open 1.4B embeddings for DTA protein targets and antibody sequences. Compare: ESM3 vs ESM-2 vs ESM-C. Show KIRBy can absorb the new model with zero framework changes.

---

### 1.3 ESM Cambrian (ESM-C)

- **URL**: https://github.com/evolutionaryscale/esm
- **Blog**: https://www.evolutionaryscale.ai/blog/esm-cambrian
- **Domain**: Protein representation learning
- **Released**: Late 2024 / Early 2025
- **Can embeddings be extracted?** YES — 960D mean-pooled embeddings.
- **How to extract embeddings**:
  ```python
  from esm.models.esmc import ESMC
  from esm.sdk.api import ESMProtein, LogitsConfig
  client = ESMC.from_pretrained("esmc_300m").to("cuda")
  protein = ESMProtein(sequence="MKTL...")
  protein_tensor = client.encode(protein)
  output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
  emb = output.embeddings  # per-residue -> mean pool -> 960D
  ```
- **Architecture**: Transformer with Pre-LN, rotary embeddings, SwiGLU. Sizes: 300M, 600M, 6B. Trained in 2 stages with metagenomic data.
- **Benchmarks**: ESM-C 600M outperforms ESM-2 15B on contact prediction despite being ~25x smaller. SOTA on protein representation benchmarks.
- **KIRBy relevance**: **Already partially integrated** (esmc rep exists in protein.py). But ESM-C 600M and 6B variants could be additional reps. Confirms forward compatibility since ESM-C was added with zero framework changes.

---

### 1.4 Boltz-2

- **URL**: https://github.com/jwohlwend/boltz
- **Paper**: https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1 (bioRxiv, June 2025)
- **Domain**: Structure prediction + binding affinity + molecular representation
- **Released**: June 2025
- **Can embeddings be extracted?** YES — `--write_embeddings` flag outputs `prot_embeddings.npz`. Internal trunk produces single representations (per-token, shape N x D) and pairwise representations (N x N x D). Mean-pool single reps for fixed-size vector.
- **Boltz as molecular representation**: A Feb 2026 paper (arxiv:2602.13249) shows Boltz atom-level representations match or outperform specialized molecular foundation models (MiniMol, MolGPS) on TDC ADMET benchmark (22 tasks). This means Boltz reps are competitive for BOTH proteins AND small molecules.
- **Architecture**: Pairformer trunk (64 layers) + MSA modules + diffusion head. MIT license.
- **KIRBy relevance**: **HIGH for molecular & protein**. Boltz-2 reps are complementary to existing molecular reps (MolE, MolFormer etc). The structure-prediction pretraining captures 3D physics that sequence-only models miss.
- **Forward-compat experiment**: Extract Boltz-2 single representations for molecules in ESOL/hERG/QM9, mean-pool to fixed vector. Plug into molecular survey. Compare: Boltz-2 alone vs ECFP4 vs Boltz-2+PDV hybrid.

---

### 1.5 AlphaFold3

- **URL**: https://github.com/google-deepmind/alphafold3
- **Domain**: Structure prediction (protein, nucleic acid, ligand, ion)
- **Released**: May 2024 (paper), Nov 2024 (code)
- **Can embeddings be extracted?** YES — with `--save_embeddings=true`, outputs `embeddings.npz` containing:
  - `single_embeddings`: shape (num_tokens, 384) — per-residue
  - `pair_embeddings`: shape (num_tokens, num_tokens, 128)
  Mean-pool single_embeddings -> 384D fixed vector per protein.
- **Caveats**: Restrictive academic license. Embeddings are float16. Very large for big proteins (~6 GiB for 5000 tokens). Requires significant GPU.
- **KIRBy relevance**: **MEDIUM**. 384D is relatively low-dimensional. Already have AlphaFold DB features (15D) in protein.py. The full AF3 embeddings would be a richer alternative but license is restrictive.
- **Forward-compat experiment**: Extract AF3 384D embeddings for DTA protein targets. Compare: AF3 vs ESM-2 vs AF3+ESM-2 hybrid.

---

### 1.6 Chai-1

- **URL**: https://github.com/chaidiscovery/chai-lab
- **Paper**: https://www.biorxiv.org/content/10.1101/2024.10.10.615955v1.full
- **Domain**: Biomolecular structure prediction
- **Released**: October 2024
- **Can embeddings be extracted?** PARTIALLY — Chai-1 generates per-residue embeddings from a 3B parameter protein language model internally. The `run_folding_on_context` API allows custom MSAs/embeddings/constraints. However, extracting the internal trunk representations requires code modification (hooking into forward pass).
- **License**: Apache 2.0 (code + weights) — excellent for academic/commercial use.
- **KIRBy relevance**: **MEDIUM**. Chai-1 uses its own internal pLM (3B) that generates per-residue embeddings. If extractable, these would be rich protein representations. Less straightforward than ESM-based models.
- **Forward-compat experiment**: Extract Chai-1 internal pLM embeddings for DTA targets. This would require a forward hook on the trunk module.

---

### 1.7 MolE (Recursion)

- **URL**: https://github.com/recursionpharma/mole_public
- **Paper**: https://www.nature.com/articles/s41467-024-53751-y (Nature Communications, Nov 2024)
- **Domain**: Molecular (small molecule foundation model)
- **Released**: November 2024
- **Can embeddings be extracted?** YES — trivially:
  ```python
  from mole import mole_predict
  smiles = ['CCC', 'CCCCCC', 'CC']
  embeddings = mole_predict.encode(smiles=smiles, pretrained_model=PATH, batch_size=32, num_workers=4)
  ```
- **Architecture**: DeBERTa-style transformer on molecular graphs with disentangled attention. 2-step pretraining: (1) self-supervised on ~842M molecules, (2) massive multi-task biological assay training.
- **Benchmarks**: Top results on 10/22 TDC ADMET tasks.
- **License**: CC-BY-NC 4.0 (non-commercial)
- **KIRBy relevance**: **HIGH for molecular domain**. Direct competitor to MolFormer, ChemBERTa etc already in KIRBy. Graph-based (like MHG-GNN) but much larger scale pretraining.
- **Forward-compat experiment**: Extract MolE embeddings for ESOL/hERG/QM9/DRD2/Cox2. Plug into molecular survey. Compare: MolE vs ChemBERTa vs MolFormer vs MolE+ECFP4 hybrid.

---

### 1.8 GP-MoLFormer

- **URL**: https://huggingface.co/ibm-research/GP-MoLFormer-Uniq
- **Paper**: https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00122f
- **Domain**: Molecular (generative + representation)
- **Released**: 2025
- **Can embeddings be extracted?** YES — 46.8M parameter decoder-only transformer. Uses rotary positional embeddings. Can extract last hidden state and mean-pool.
- **Architecture**: Autoregressive decoder trained on 1.1B SMILES (ZINC + PubChem). Linear attention + RoPE. Variant GP-MoLFormer-Uniq trained on 650M deduplicated subset.
- **KIRBy relevance**: **MEDIUM**. MoLFormer (encoder) is already in KIRBy. GP-MoLFormer is the generative successor — its embeddings might capture different distributional properties.

---

### 1.9 MiniMol

- **URL**: https://pypi.org/project/minimol/
- **Paper**: ICML 2024 Workshop on Accessible and Efficient Foundation Models for Biological Discovery
- **Domain**: Molecular (parameter-efficient foundation model)
- **Released**: 2024
- **Can embeddings be extracted?** YES — "four lines of code" per PyPI description. Uses GINE GNN backbone with graph pooling.
- **Architecture**: GINE layers + pooling. Pre-trained on ~6M molecules, 500M labels across ~3300 tasks (quantum + biological). Only 10% parameters of MolE.
- **Benchmarks**: Outperforms MolE on 17 datasets despite 10x fewer parameters.
- **KIRBy relevance**: **HIGH**. Extremely parameter-efficient. GNN-based (complementary to SMILES-based models). Easy pip install.

---

### 1.10 Antibody-Specific New Models

#### IgBERT / IgT5 (Exscientia, Dec 2024)
- **URL**: https://huggingface.co/Exscientia/IgBert, https://huggingface.co/Exscientia/IgT5
- **Paper**: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012646
- **Can embeddings be extracted?** YES — IgBERT outputs per-residue embeddings (BERT hidden states), mean-pool for fixed vector. IgT5 uses encoder-only mode for embeddings.
  ```python
  from transformers import BertModel, BertTokenizer
  model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)
  # Forward pass -> last_hidden_state -> mask special tokens -> mean pool
  ```
- **Architecture**: IgBERT (BERT-based), IgT5 (T5 encoder). Trained on 2B+ unpaired + 2M paired antibody sequences from OAS.
- **Benchmarks**: Best-performing antibody-specific LMs to date. Handle both paired and unpaired VH/VL.
- **KIRBy relevance**: **HIGH for antibody domain**. Direct competitors to AntiBERTy already in KIRBy. Paired-aware training is unique.
- **Forward-compat experiment**: Extract IgBERT embeddings for FLAb2/SAbDab datasets. Compare: IgBERT vs AntiBERTy vs ESM-2 vs IgBERT+structural hybrid.

#### S2ALM (May 2025)
- **URL**: https://spj.science.org/doi/10.34133/research.0721
- **Domain**: Antibody (sequence + structure multimodal)
- **Architecture**: Built on ESM-2 650M. 650M trainable params. Pre-trained on 75M sequences + 11.7M structures. Two objectives: Sequence-Structure Matching (SSM) + Cross-Level Reconstruction (CLR).
- **Can embeddings be extracted?** YES — ESM-2 architecture means standard HuggingFace extraction. Mean-pool hidden states.
- **Benchmarks**: SOTA on binding affinity prediction, B cell maturation, CDR identification.
- **KIRBy relevance**: **HIGH**. Structure-aware antibody embeddings — richer than pure sequence models.

#### Ab-RoBERTa (2026)
- **Paper**: https://arxiv.org/pdf/2506.13006
- **Architecture**: RoBERTa-based, 125M params. 5x faster than IgT5 with competitive performance.
- **Can embeddings be extracted?** YES — standard RoBERTa hidden states.
- **KIRBy relevance**: **MEDIUM**. Smaller/faster alternative to IgBERT/IgT5.

#### IgBlend (2024)
- **Paper**: https://www.biorxiv.org/content/10.1101/2024.10.01.615796v2.full
- **Architecture**: Multi-modal (sequence + 3D structure). Structure encoder produces 512D embedding. Trained on 4M structures + 200M sequences.
- **Can embeddings be extracted?** YES — 512D structure embedding + sequence embedding.
- **KIRBy relevance**: **HIGH**. Structure-aware antibody model. 512D is manageable.

---

### 1.11 Structure-Aware Protein Models

#### SaProt (ICLR 2024 Spotlight)
- **URL**: https://github.com/westlake-repl/SaProt
- **Architecture**: ESM-2 retrained with structure-aware vocabulary (AA + Foldseek 3Di tokens). 650M params. 1280D embeddings.
- **Benchmarks**: Ranked #1 on ProteinGym (April 2024). SOTA on 10 protein function tasks.
- **Can embeddings be extracted?** YES — standard ESM-2 extraction pipeline. 1280D.
- **KIRBy relevance**: **VERY HIGH**. Same architecture as ESM-2 (already in KIRBy) but with structure info baked in. Drop-in replacement for ESM-2 embeddings.
- **Forward-compat experiment**: SaProt 1280D into DTA and antibody pipelines. Compare: SaProt vs ESM-2 vs SaProt+PDV.

#### ProSST (NeurIPS 2024)
- **URL**: https://github.com/ai4protein/ProSST
- **Architecture**: Structure quantization module + disentangled attention transformer. Pre-trained on 18.8M structures.
- **Benchmarks**: Best on thermostability, metal ion binding, protein localization. Smaller than ESM-GearNet.
- **Can embeddings be extracted?** YES — transformer hidden states, mean-pool.
- **KIRBy relevance**: **HIGH**. Structure-aware, compact.

#### S3F (NeurIPS 2024)
- **URL**: https://github.com/DeepGraphLearning/S3F
- **Architecture**: Combines ESM-2 embeddings + GVP structure encoder + surface topology encoder. Multi-scale.
- **Benchmarks**: SOTA on ProteinGym 217 substitution DMS assays (among non-MSA methods). Outperforms TranceptEVE.
- **Can embeddings be extracted?** YES — the combined representation is a fixed-size vector.
- **KIRBy relevance**: **HIGH for mutation/fitness**. Multi-scale reps capture surface features that sequence-only models miss.

#### Protenix (ByteDance, Jan 2026)
- **URL**: https://github.com/bytedance/Protenix
- **Architecture**: AlphaFold3 reproduction. Uses ESM2-3B embeddings internally. Apache 2.0 license.
- **Can embeddings be extracted?** YES — ESM2 embeddings are part of the input pipeline. Internal Pairformer trunk representations can be hooked.
- **KIRBy relevance**: **MEDIUM**. Open-source AF3 alternative with better license terms.

---

### 1.12 Designed Forward Compatibility Experiments

#### Experiment 1: MINT for DTA (Protein-Interaction-Aware Embeddings)
- **Hypothesis**: MINT embeddings capture PPI context that vanilla ESM-2 misses, improving DTA predictions.
- **Protocol**:
  1. Extract MINT 1280D embeddings for all protein targets in Davis and KIBA datasets
  2. Run KIRBy survey with MINT as a new rep (zero code changes to framework)
  3. Compare: `mint` vs `esm2` vs `mint + pdv_descriptors` hybrid
  4. Expected: MINT alone >= ESM-2 alone; MINT+small_features hybrid > MINT alone
- **What it proves**: Brand-new 2026 model plugs into existing KIRBy pipeline. Framework is forward-compatible.

#### Experiment 2: SaProt/ESM3 for Antibody & DTA (Structure-Aware Protein LMs)
- **Hypothesis**: Structure-aware LMs provide richer embeddings than ESM-2 for fitness/binding tasks.
- **Protocol**:
  1. Extract SaProt 1280D and ESM3-open embeddings for:
     - DTA protein targets (Davis, KIBA)
     - Antibody sequences (FLAb2 datasets)
  2. Run KIRBy survey with each as a new rep
  3. Compare: `saprot` vs `esm3` vs `esm2` vs `esmc` across all tasks
  4. Augmentation: `saprot + aac` hybrid, `esm3 + structural` hybrid
- **What it proves**: Next-generation protein LMs integrate with zero changes. Structure info accessible via embeddings alone.

#### Experiment 3: MolE/MiniMol for Molecular (New Foundation Models)
- **Hypothesis**: New molecular foundation models provide complementary representations to existing KIRBy reps.
- **Protocol**:
  1. Extract MolE embeddings and MiniMol embeddings for ESOL, hERG, Cox2, DRD2, QM9
  2. Run KIRBy molecular survey with each as new reps
  3. Compare: `mole` vs `minimol` vs `molformer` vs `chemberta` vs `ecfp4`
  4. Hybrid: `mole + pdv` vs `minimol + ecfp4` vs `best_existing`
- **What it proves**: Molecular foundation models from completely different research groups plug in identically. Framework-agnostic.

#### Experiment 4: IgBERT/S2ALM for Antibody (Paired & Multimodal AbLMs)
- **Hypothesis**: Paired-trained (IgBERT) and structure-aware (S2ALM) antibody LMs improve on AntiBERTy.
- **Protocol**:
  1. Extract IgBERT and S2ALM embeddings for all FLAb2 + SAbDab datasets
  2. Run KIRBy antibody survey
  3. Compare: `igbert` vs `s2alm` vs `antiberty` vs `esm2` across all 18 FLAb2 tasks
  4. Hybrid: `igbert + structural` vs `s2alm + cdr_embeddings`
- **What it proves**: Specialized antibody LMs from 2024-2025 integrate seamlessly.

#### Experiment 5: Boltz-2 for Cross-Domain (Structure Prediction as Universal Embedder)
- **Hypothesis**: Boltz-2 reps, learned from co-folding, are competitive for both molecular and protein tasks.
- **Protocol**:
  1. Extract Boltz-2 atom-level representations for small molecules (ESOL, hERG)
  2. Extract Boltz-2 protein representations for DTA targets
  3. Run KIRBy surveys in both molecular and DTA domains
  4. Compare: `boltz2_mol` vs `ecfp4` vs `molformer`; `boltz2_prot` vs `esm2`
  5. Cross-domain hybrid: `boltz2 + domain_specific_rep`
- **What it proves**: A single structure-prediction model provides useful embeddings across domains. Ultimate forward-compatibility test.

---

## PART 2: Downstream Model Audit

KIRBy currently uses: **RidgeCV, Random Forest, XGBoost, LightGBM**

---

### 2.1 Comprehensive Algorithm Ranking (Literature)

From the definitive 16-algorithm QSAR study (Briefings in Bioinformatics, 2021):

**Overall ranking (best to worst):**
```
rbf-SVM > XGBoost > rbf-GPR > Cubist > GBM > DNN > RF > pca-ANN > MARS >
linear-GPR ~ KNN > linear-SVM ~ PLSR > CART ~ PCR ~ MLR
```

Source: https://pubmed.ncbi.nlm.nih.gov/33313673/

**Key findings:**
- Nonlinear algorithms MUCH better than linear for molecular descriptor-based QSAR
- rbf-SVM is overall #1, XGBoost is #2
- Ridge regression (≈ PLSR/PCR tier) is in the BOTTOM THIRD
- DNN is #6 overall — strong but not dominant
- GPR (Gaussian Process Regression) is #3 — excellent + provides uncertainty

**Implications for KIRBy:**
- Ridge is a weak baseline (by design, but it's VERY weak)
- SVM and GP should be added as they outperform tree methods on small datasets
- DNN should be added for completeness
- CatBoost is comparable to LightGBM (already have LightGBM)

---

### 2.2 Model-by-Model Analysis

#### Currently in KIRBy:
| Model | Tier | Notes |
|-------|------|-------|
| RidgeCV | Low | Bottom third. Good as linear baseline only. |
| Random Forest | Mid | #7 in ranking. Solid but not top-tier. |
| XGBoost | High | #2 overall. Good for large datasets. ~5% better than LightGBM/CatBoost. |
| LightGBM | High | Similar to CatBoost. Fastest gradient booster. Good for medium datasets. |

#### Should Add:

##### SVR (Support Vector Regression) — PRIORITY: HIGH
- **Ranking**: #1 overall with RBF kernel
- **Why it matters**: Consistently best on small datasets (< 5000 samples) which is most QSAR datasets. Excellent with high-dimensional fingerprints. Well-established in chemoinformatics.
- **Implementation**: `sklearn.svm.SVR(kernel='rbf')`
- **Hyperparameters**: C, gamma, epsilon. Can use `GridSearchCV` or default C=1.0.
- **Evidence**: SVM classification achieved F1 > 0.95 with both MACCS and ECFP4. SVR achieves Pearson > 0.8 with ECFP4.
- **Source**: https://pubs.acs.org/doi/10.1021/acsomega.7b01079

##### Gaussian Process Regression (GPR) — PRIORITY: HIGH
- **Ranking**: #3 overall (with RBF kernel)
- **Why it matters**:
  1. Provides uncertainty quantification (unique among KIRBy models)
  2. Kermut (NeurIPS 2024 Spotlight) uses GP with ESM-2 embeddings = SOTA on ProteinGym supervised benchmark
  3. GAUCHE library provides chemistry-specific kernels (Tanimoto, SMILES string kernel)
  4. Excellent in low-data regimes (many KIRBy datasets are small)
  5. Deep kernel learning combines DNN feature extraction with GP inference
- **Implementation**:
  - Basic: `sklearn.gaussian_process.GaussianProcessRegressor`
  - Advanced: GPyTorch (for Kermut-style composite kernels)
  - Chemistry: GAUCHE library (https://github.com/leojklarner/gauche)
- **Kermut details**: Composite kernel with ESM-2 sequence kernel + inverse-folding structure kernel. Uses GPyTorch + AdamW. NeurIPS 2024 spotlight. Code: https://github.com/petergroth/kermut
- **Caveats**: O(n^3) scaling — problematic for datasets > ~5000 samples. Use sparse GP approximations (inducing points) for larger datasets. GPyTorch handles this.
- **Evidence**: https://proceedings.neurips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf

##### MLP / FNN (Multi-Layer Perceptron) — PRIORITY: HIGH
- **Ranking**: ~#6 (as DNN in the 16-algorithm study)
- **Why it matters**:
  1. VespaG uses single-hidden-layer FNN (256 units) on ESM-2 embeddings = competitive on ProteinGym
  2. Standard downstream head for all protein LM papers
  3. Can learn nonlinear feature interactions that tree models miss
  4. ProteinGym supervised baselines use MLP prediction heads
- **Implementation**: `sklearn.neural_network.MLPRegressor` or PyTorch
- **Architecture**: 1-3 hidden layers, 256-512 units, ReLU, dropout 0.1-0.3
- **VespaG specifics**: 1 hidden layer, 256 units, ESM-2 2560D input, 660K params. MSE loss against GEMME evolutionary scores.
- **Evidence**:
  - VespaG: https://academic.oup.com/bioinformatics/article/40/11/btae621/7907184
  - Medium-sized pLMs study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11601519/

##### CatBoost — PRIORITY: MEDIUM
- **Why it matters**:
  1. Handles categorical features natively (useful if KIRBy ever uses discrete features)
  2. Better than LightGBM on medium-sized datasets (7K-40K compounds)
  3. User's qsar_qm_models repo already uses it
  4. Oblivious tree architecture provides good regularization
- **Performance vs existing**: Similar to LightGBM. Worse on small datasets (< 1K), better on medium (7K-40K).
- **Implementation**: `catboost.CatBoostRegressor`
- **Evidence**: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00743-7

##### DNN (Deep Neural Network) — PRIORITY: MEDIUM
- **Why it matters**:
  1. User's qsar_qm_models uses it
  2. #6 in comprehensive QSAR ranking
  3. Better than RF for large diverse datasets
  4. Architecture: 2-3 hidden layers, 80+ nodes per layer
- **Implementation**: PyTorch or `sklearn.neural_network.MLPRegressor` (with more layers)
- **Difference from MLP**: DNN typically means >= 3 hidden layers with dropout, batch norm, etc. MLP is simpler (1-2 layers).
- **Evidence**: https://pubs.acs.org/doi/10.1021/ci500747n

##### 1D-CNN — PRIORITY: LOW-MEDIUM
- **Why it matters**:
  1. ProteinGym supervised baselines include CNN prediction heads
  2. Captures local patterns in per-residue embeddings (before pooling)
  3. Standard in protein function prediction
- **When useful**: When using per-residue (not mean-pooled) embeddings as input. NOT useful for pre-pooled fixed-size vectors (which is KIRBy's current approach).
- **KIRBy limitation**: KIRBy works with fixed-size vectors, not variable-length sequences. CNN is only useful if KIRBy adds support for per-residue embedding inputs.
- **Implementation**: PyTorch Conv1d layers

##### TabPFN — PRIORITY: LOW-MEDIUM (EXPERIMENTAL)
- **URL**: https://github.com/PriorLabs/TabPFN
- **Paper**: https://www.nature.com/articles/s41586-024-08328-6 (Nature, Dec 2024)
- **Why it matters**:
  1. Foundation model for tabular data — fits KIRBy's paradigm perfectly
  2. Requires NO hyperparameter tuning (aligns with KIRBy's no-HP-tuning philosophy)
  3. Maps inputs to nearly linearly separable embedding space
  4. Excellent on small datasets (< 10K samples) which is most QSAR/bioactivity data
  5. TabPFN v2.5 handles up to 100K samples, 2000 features
- **Caveats**: Relatively new. May not be standard enough for a thesis.
- **Implementation**: `pip install tabpfn`

---

### 2.3 Model Hierarchy by Domain

#### Molecular Property Prediction (QSAR)
```
Tier 1 (SOTA):     rbf-SVM, XGBoost, GP (with Tanimoto kernel)
Tier 2 (Strong):   LightGBM, CatBoost, DNN (2-3 layers), GBM
Tier 3 (Baseline): RF, MLP (1 layer), KNN
Tier 4 (Weak):     Ridge, Linear SVM, PCR
```

#### Protein Fitness Prediction
```
Tier 1 (SOTA):     GP (Kermut-style composite kernel), CNN on per-residue
Tier 2 (Strong):   MLP/FNN (VespaG-style), Ridge + zero-shot augmentation
Tier 3 (Baseline): Ridge on embeddings, RF on embeddings
Tier 4 (Weak):     Linear regression
```

#### Drug-Target Affinity (DTA)
```
Tier 1 (SOTA):     DNN on concat embeddings, XGBoost
Tier 2 (Strong):   GP, SVR, LightGBM
Tier 3 (Baseline): RF, Ridge
```

#### Antibody Binding/Function
```
Tier 1 (SOTA):     GP (on LM embeddings), MLP/FNN
Tier 2 (Strong):   XGBoost, SVR, LightGBM
Tier 3 (Baseline): RF, Ridge
```

---

### 2.4 What KIRBy Should Add (Priority Order)

| Priority | Model | Effort | Impact | Notes |
|----------|-------|--------|--------|-------|
| 1 | **SVR** | Low | High | sklearn one-liner. #1 ranked algorithm. |
| 2 | **GP (basic)** | Low | High | sklearn GPR. Uncertainty for free. |
| 3 | **MLP** | Low | High | sklearn MLPRegressor or simple PyTorch. Standard in protein LM literature. |
| 4 | **CatBoost** | Low | Medium | pip install catboost. User already has it in qsar_qm_models. |
| 5 | **DNN** | Medium | Medium | PyTorch 3-layer net. User already has it in qsar_qm_models. |
| 6 | **GP (advanced)** | High | High | GPyTorch with Tanimoto/composite kernels (Kermut-style). |
| 7 | **TabPFN** | Low | Unknown | Experimental but fascinating. Zero-HP model. |
| 8 | **1D-CNN** | High | Medium | Requires per-residue embedding support (architecture change). |

---

### 2.5 Key Literature on "Model Matters"

1. **16-algorithm QSAR study**: rbf-SVM > XGBoost > rbf-GPR > ... > Ridge
   - https://academic.oup.com/bib/article/22/4/bbaa321/6032614

2. **Gradient boosting for molecular property prediction**: XGBoost ~5% better than LightGBM/CatBoost
   - https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00743-7

3. **Kermut (NeurIPS 2024 Spotlight)**: GP with ESM-2 embeddings = SOTA supervised protein fitness
   - https://github.com/petergroth/kermut

4. **VespaG**: 1-layer FNN on ESM-2 = competitive with much more complex methods
   - https://academic.oup.com/bioinformatics/article/40/11/btae621/7907184

5. **Medium-sized pLMs study**: downstream architecture matters less than embedding quality, but MLP slightly outperforms ridge
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC11601519/

6. **S3F (NeurIPS 2024)**: Multi-scale reps + ridge = SOTA, showing representation quality dominates model choice
   - https://github.com/DeepGraphLearning/S3F

7. **Deep kernel learning for reactions**: GP + DNN features outperform standalone GP or DNN
   - https://www.nature.com/articles/s42004-024-01219-x

8. **GAUCHE**: GP library for chemistry with Tanimoto kernel — excellent for molecular fingerprints
   - https://github.com/leojklarner/gauche

---

### 2.6 Summary Table: Model vs Domain Recommendation

| Model | Molecular | DTA | Antibody | Protein Fitness | Cost |
|-------|-----------|-----|----------|-----------------|------|
| Ridge (current) | Weak | Weak | Weak | Weak (but fast) | Negligible |
| RF (current) | Mid | Mid | Mid | Mid | Low |
| XGBoost (current) | Strong | Strong | Strong | Strong | Low |
| LightGBM (current) | Strong | Strong | Strong | Strong | Low |
| **SVR (add)** | **SOTA** | **Strong** | **Strong** | **Strong** | Low |
| **GPR (add)** | **SOTA** | **Strong** | **Strong** | **SOTA** | Medium |
| **MLP (add)** | **Strong** | **Strong** | **Strong** | **Strong** | Low |
| **CatBoost (add)** | Strong | Strong | Strong | Strong | Low |
| **DNN (add)** | Strong | Strong | Mid | Mid | Medium |
| **TabPFN (add)** | Unknown | Unknown | Unknown | Unknown | Low |

---

### 2.7 Implementation Notes

**For SVR**: Add to existing model loop. Use `sklearn.svm.SVR(kernel='rbf', C=1.0)`. For hyperparameter search: C in [0.1, 1, 10, 100], gamma in ['scale', 'auto']. SVR scales O(n^2) in memory — may need subsampling for datasets > 10K.

**For GPR**: Add `sklearn.gaussian_process.GaussianProcessRegressor(kernel=RBF() + WhiteKernel())`. For molecular fingerprints, consider Tanimoto kernel from GAUCHE. For protein embeddings, RBF works well. Scales O(n^3) — use sparse approximation for n > 3000.

**For MLP**: Add `sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True)`. Or PyTorch for more control. Match VespaG architecture: 1 hidden layer, 256 units, no output activation.

**For CatBoost**: `catboost.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=0)`. Already in user's qsar_qm_models.

**For DNN**: PyTorch. 3 hidden layers (256-128-64), ReLU, dropout 0.2, Adam optimizer, MSE loss. Already in user's qsar_qm_models.

---

## APPENDIX: All Repository URLs

### New Models
| Model | Repository | License |
|-------|-----------|---------|
| MINT | https://github.com/VarunUllanat/mint | MIT |
| ESM3 | https://github.com/evolutionaryscale/esm | EvolutionaryScale License |
| ESM-C | https://github.com/evolutionaryscale/esm | EvolutionaryScale License |
| Boltz-2 | https://github.com/jwohlwend/boltz | MIT |
| AlphaFold3 | https://github.com/google-deepmind/alphafold3 | Academic only |
| Chai-1 | https://github.com/chaidiscovery/chai-lab | Apache 2.0 |
| MolE | https://github.com/recursionpharma/mole_public | CC-BY-NC 4.0 |
| GP-MoLFormer | https://huggingface.co/ibm-research/GP-MoLFormer-Uniq | Apache 2.0 |
| MiniMol | https://pypi.org/project/minimol/ | — |
| IgBERT/IgT5 | https://huggingface.co/Exscientia/IgBert | — |
| S2ALM | https://spj.science.org/doi/10.34133/research.0721 | — |
| Ab-RoBERTa | https://arxiv.org/pdf/2506.13006 | — |
| IgBlend | https://www.biorxiv.org/content/10.1101/2024.10.01.615796v2 | — |
| SaProt | https://github.com/westlake-repl/SaProt | MIT |
| ProSST | https://github.com/ai4protein/ProSST | — |
| S3F | https://github.com/DeepGraphLearning/S3F | MIT |
| Protenix | https://github.com/bytedance/Protenix | Apache 2.0 |

### Downstream Model Libraries
| Tool | Repository | Use |
|------|-----------|-----|
| Kermut | https://github.com/petergroth/kermut | GP for protein fitness |
| GAUCHE | https://github.com/leojklarner/gauche | GP for chemistry |
| VespaG | https://github.com/JSchlensok/VespaG | MLP for protein fitness |
| ProteinGym | https://github.com/OATML-Markslab/ProteinGym | Benchmarks |
| TabPFN | https://github.com/PriorLabs/TabPFN | Tabular foundation model |
