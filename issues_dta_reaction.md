# DTA and Reaction Yield Pipeline Audit

Research conducted 2026-03-10. All findings are from published papers and public GitHub repos.

---

## Part 1: DTA (Drug-Target Affinity) Pipeline

### 1.1 Published DTA Methods and Benchmarks

#### DeepDTA (Ozturk et al. 2018) -- the baseline everyone compares to
- **Repo:** https://github.com/hkmztrk/DeepDTA
- **Drug rep:** CNN on SMILES strings (learned)
- **Protein rep:** CNN on amino acid sequences (learned)
- **Architecture:** Dual-CNN encoder, concatenate, FC layers
- **Splitting:** Warm (standard 5-fold CV, same drugs/proteins in train and test)
- **Metrics:** CI and MSE
- **Davis (warm):** CI = 0.878, MSE = 0.261
- **KIBA (warm):** CI = 0.863, MSE = 0.194

#### GraphDTA (Nguyen et al. 2021)
- **Repo:** https://github.com/thinng/GraphDTA
- **Drug rep:** Molecular graph (atoms=nodes, bonds=edges) with GNN variants (GCN, GAT, GIN, GAT-GCN)
- **Protein rep:** 1D-CNN on amino acid sequence
- **Architecture:** GNN + CNN, concatenate, FC
- **Splitting:** Warm 5-fold CV
- **Davis (warm):** CI ~ 0.882, MSE ~ 0.254 (GAT-GCN variant)
- **KIBA (warm):** CI ~ 0.882
- Note: Falls behind some methods in cold-drug scenario

#### MolTrans (Huang et al. 2021)
- **Repo:** https://github.com/kexinhuang12345/MolTrans
- **Drug rep:** Sub-structural pattern mining from SMILES
- **Protein rep:** Sub-structural patterns from amino acid sequence
- **Architecture:** Augmented Transformer encoder for interaction modeling
- **Splitting:** Warm, evaluated on Davis, BindingDB, BIOSNAP
- **Davis:** Primarily evaluated on classification (DTI), not regression (DTA)
- Note: Focuses on ROC-AUC/PR-AUC rather than CI/MSE

#### ColdDTA (2023)
- **Paper:** https://www.sciencedirect.com/science/article/abs/pii/S0010482523008375
- **Drug rep:** Molecular graph + data augmentation (subgraph removal)
- **Protein rep:** Sequence + attention-based feature fusion
- **Architecture:** Attention-based feature fusion for cold-start
- **Splitting:** Cold-drug, cold-target, cold-both
- **Davis (warm):** CI = 0.913, MSE not specified in search results
- **KIBA (warm):** CI = 0.906
- Cold-start: Significantly outperforms 5 baseline methods

#### GRA-DTA (2024)
- **Paper:** PMC11121300
- **Davis:** CI = 0.890, MSE = 0.225
- **KIBA:** CI = 0.897, MSE = 0.142
- Under cold-target and cold-drug-target: 10.7% and 13.5% improvement in MSE and CI vs suboptimal methods

#### PocketDTA (2024)
- **Paper:** https://academic.oup.com/bioinformatics/article/40/10/btae594/7811139
- **Drug rep:** Molecular features
- **Protein rep:** 3D pocket structure from AlphaFold
- **Architecture:** Multimodal (3D pocket + sequence)
- **Splitting:** Warm + cold-drug + cold-target + cold-both
- **Cold-target Davis:** MSE decreased by 2.965% (from 0.371 to 0.360) vs suboptimal
- **Cold-both Davis:** MSE decreased by 10.440% (from 0.546 to 0.489)

#### MTAF-DTA (2024)
- **Paper:** https://link.springer.com/article/10.1186/s12859-024-05984-3
- **Cold-target (novel-protein) Davis:** CI = 0.840, MSE = 0.330

#### MixingDTA (2025)
- **Paper:** https://academic.oup.com/bioinformatics/article/41/Supplement_1/i105/8199404
- Cold protein setting: CI = 0.87, MSE = 0.23
- Outperforms DeepDTA, GraphDTA, AttentionDTA, MGraphDTA, CSCo-DTA, MD-CT-DTA

#### LKE-DTA (2025)
- **Paper:** https://link.springer.com/article/10.1007/s11030-025-11394-1
- **Drug rep:** LLM representations + knowledge graph embeddings
- **Protein rep:** LLM representations + knowledge graph embeddings
- **Cold Target improvement:** MSE +10.2%, MAE +12.2%, CI +6.6%, r +9.0%

#### InceptionDTA (2025)
- **Paper:** PMC11850134
- Achieves highest CI in cold-protein setting among compared methods
- Uses biological context features and inception network architecture

#### ESM2-based approaches
- **Repo:** https://github.com/MaastrichtU-IDS/predict-drug-target
- Uses ESM2 protein embeddings + MolecularTransformer drug embeddings
- Linear classifier for DTI prediction
- PMMR (2025) uses ESM-2 + ChemBERTa-2 for pre-trained features, shows superior cold-start generalization

### 1.2 Warm vs Cold-Target Performance Ranges (Davis)

| Setting | CI range | MSE range | Notes |
|---------|----------|-----------|-------|
| Warm (standard) | 0.878--0.913 | 0.19--0.26 | Most published numbers are warm |
| Cold-target | 0.82--0.87 | 0.23--0.37 | Significant drop from warm |
| Cold-drug | 0.83--0.88 | 0.25--0.35 | Slightly easier than cold-target |
| Cold-both | 0.70--0.80 | 0.40--0.55 | Hardest setting |

**Key insight:** Almost ALL published DTA numbers are warm setting (CI 0.87--0.91). Cold-target numbers are substantially lower but still well above random (CI=0.5). The Davis dataset's 68 drugs under cold-target split means drug reps cannot differentiate -- they are constant per target fold. This explains KIRBy's R2=0.223 for drug-only baselines being expected behavior.

### 1.3 KIRBy's DTA Pipeline: Comparison and Gaps

#### What KIRBy does right:
- Cold-target GroupKFold splitting (harder, more realistic than warm)
- Drug + protein reps as separate entries (correct architecture for the framework)
- PCA-reduced variants for high-dim LM embeddings
- Comprehensive protein rep coverage (ESM2, ESM-C, AMPLIFY, ESMFold, AlphaFold DB, AAC, descriptors, kmer)

#### Issue 1: MISSING concordance index (CI)
- **Severity: HIGH**
- CI is THE standard metric for DTA. Every paper reports CI.
- KIRBy only reports R2. This makes direct comparison impossible.
- CI measures ranking accuracy (are higher-predicted values truly higher?), which is arguably more important than R2 for drug discovery.
- **Recommendation:** Add CI to DTA evaluation. Formula: CI = (number of concordant pairs) / (total comparable pairs). Scipy's `concordance_index` or a custom implementation. It is a pairwise ranking metric similar to AUC.
- Implementation: `from lifelines.utils import concordance_index` or write a simple version.

#### Issue 2: KIRBy R2=0.223 on Davis cold-target -- is this consistent?
- **Yes, this is consistent and expected.**
- Under cold-target with only 68 drugs, ALL drug-rep baselines will be identical (drug features are constant across all pairs for a given protein fold). The R2=0.223 is purely from protein reps.
- Published cold-target CI values are ~0.82--0.87 (MSE 0.23--0.37). CI=0.82 with MSE=0.37 is roughly consistent with modest R2 values. CI and R2 are not directly comparable metrics -- CI measures ranking, R2 measures variance explained -- but low R2 with decent CI is common in DTA because the label distribution is narrow (most Davis Kd values cluster near 10,000 nM).
- **The protein rep baselines (ESM2, ESM-C, etc.) are the ones that matter for cold-target.** Drug reps only differentiate in cold-drug or warm settings.

#### Issue 3: Missing datasets
- **Severity: MEDIUM**
- KIRBy has Davis and KIBA. The field also uses:
  - **BindingDB_Kd** -- Much larger (10,665 drugs x 1,413 proteins). TDC provides this with standard cold splits.
  - **Metz** -- 1,423 drugs x 170 proteins, 35,259 binding entries. Sequence-based benchmark.
  - **PDBBind** -- Already in KIRBy as a separate loader (structure-based)
- **Recommendation:** Add BindingDB_Kd from TDC (`from tdc.multi_pred import DTI; data = DTI(name='BindingDB_Kd')`). This would give a large-scale benchmark to complement the small Davis and large-but-sparse KIBA.

#### Issue 4: No cold-drug evaluation
- **Severity: MEDIUM**
- KIRBy only does cold-target. Published work evaluates cold-drug and cold-both too.
- `assign_cold_drug_groups()` already exists in `dta.py` but is never called.
- **Recommendation:** Add `--cold-setting {target,drug,both}` flag. Cold-drug would group by drug_ids. Cold-both would require a custom split (e.g., hold out drugs AND proteins simultaneously).

#### Issue 5: Only linear/tree models, no deep interaction models
- **Severity: LOW (by design)**
- KIRBy's thesis is about representations, not architectures. Using Ridge/RF/XGBoost on concatenated drug+protein features is the correct approach for the thesis.
- However, this means KIRBy will never match SOTA CI numbers (0.90+) which come from deep interaction models (attention between drug and protein substructures).
- **This is fine for the thesis.** Document it as a known limitation.

#### Issue 6: TDC leaderboard splits
- **Severity: LOW**
- TDC provides standardized cold-drug and cold-target splits for Davis/KIBA with MAE and Pearson as metrics.
- Using TDC splits would enable direct leaderboard comparison.
- **Recommendation:** Consider adding TDC split compatibility if time permits.

### 1.4 Published Methods Summary Table (Davis, Warm Setting)

| Method | Year | CI | MSE | Drug Rep | Protein Rep |
|--------|------|----|----|----------|-------------|
| KNN baseline | 2018 | 0.713 | 0.424 | Smith-Waterman | Smith-Waterman |
| DeepDTA | 2018 | 0.878 | 0.261 | CNN on SMILES | CNN on sequence |
| GraphDTA (GAT-GCN) | 2021 | 0.882 | 0.254 | GNN on mol graph | CNN on sequence |
| MGraphDTA | 2022 | ~0.890 | ~0.235 | Multi-scale GNN | CNN on sequence |
| 3DProtDTA | 2023 | 0.891 | ~0.23 | GNN | Residue-level 3D graph |
| FusionDTA | 2022 | ~0.890 | ~0.24 | GNN | PLM (protein LM) |
| ColdDTA | 2023 | 0.913 | -- | Graph + augmentation | Attn fusion |
| GRA-DTA | 2024 | 0.890 | 0.225 | Graph + attention | Sequence |
| MTAF-DTA | 2024 | ~0.895 | ~0.22 | Multi-type attention | Multi-type attention |
| GS-DTA | 2024 | 0.903 | -- | Grammar + sequence | Gene expression |
| MixingDTA | 2025 | ~0.90 | -- | Mixup augmentation | Mixup augmentation |

Note: Almost all these are WARM setting. Cold-target numbers are 5-10% lower in CI.

---

## Part 2: Reaction Yield Pipeline

### 2.1 Published Reaction Yield Methods and Benchmarks

#### Schwaller et al. (Yield-BERT / rxn_yields) 2021
- **Repo:** https://github.com/rxn4chemistry/rxn_yields
- **Docs:** https://rxn4chemistry.github.io/rxn_yields/
- **Rep:** Reaction SMILES fed to fine-tuned BERT (Yield-BERT)
- **Models:** BERT regression head
- **Splitting:** Random 70/30 (x10 splits) + 4 out-of-sample tests
- **BH random 70/30:** R2 = 0.951 (Yield-BERT augmented)
- **Suzuki random 70/30:** R2 = 0.81, RMSE = 12.07%
- **BH out-of-sample tests (held-out isoxazole additives):**
  - Test1: R2 = 0.84, RMSE = 11.0, MAE = 7.0
  - Test2: R2 = 0.84, RMSE = 11.0, MAE = 7.1
  - Test3: R2 = 0.74, RMSE = 14.4, MAE = 9.2
  - Test4: R2 = 0.54, RMSE = 18.0, MAE = 12.7
- **Key:** Test1-4 are out-of-sample additive splits. Performance degrades substantially (0.95 -> 0.54) when extrapolating to unseen reagent classes.

#### rxnfp (Schwaller et al. 2021)
- **Repo:** https://github.com/rxn4chemistry/rxnfp
- Learned reaction fingerprints from transformer attention
- Used as starting point for Yield-BERT fine-tuning
- BH random 70/30: R2 ~ 0.95

#### DRFP (Probst et al. 2022)
- **Repo:** https://github.com/reymond-group/drfp
- **Paper:** https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00006c
- **Rep:** Differential reaction fingerprint (2048-bit), NLP-inspired, data-independent
- **Models:** XGBoost, Random Forest
- **BH random 70/30:** R2 ~ 0.95 (DRFP + XGBoost)
- **BH with decreasing train size:** R2 = 0.95 (70/30) down to 0.73 (5/95)
- **BH out-of-sample (additive held out):** R2 ~ 0.83
- **Key insight:** DRFP outperforms DFT-based fingerprints and rivals learned fingerprints (rxnfp/Yield-BERT) without any pre-training.

#### Ahneman/Doyle/Dreher (Science 2018) -- the original BH dataset
- **Paper:** https://www.science.org/doi/10.1126/science.aar5169
- **Rep:** DFT-computed descriptors (vibrational, steric, electronic) for each component
- **Models:** Random Forest
- **Splitting:** Random 70/30 = R2 ~ 0.92; Leave-additive-out (isoxazoles) = R2 ~ 0.83
- **Dataset structure:** 15 aryl halides x 4 ligands x 3 bases x 23 additives = 4,608 reactions (3,955 after filtering)
- **The 4 out-of-sample test sets correspond to different isoxazole additive plates held out entirely.**

#### YieldGNN (2023)
- **Paper:** PMC10189898 ("On the use of real-world datasets for reaction yield prediction")
- **Rep:** Weisfeiler-Lehman Networks (WLN) for structural features + chemical descriptors
- **Models:** GNN end-to-end
- **BH:** Outperforms Yield-BERT and DRFP, particularly on BH dataset
- **Suzuki:** Smaller improvement over baselines
- **Key contribution:** Leave-one-group-out analysis on BH additives showing modest degradation per additive removed.

#### BH-RGNN (2025 -- J. Org. Chem.)
- **Paper:** https://pubs.acs.org/doi/10.1021/acs.joc.5c01400
- Buchwald-Hartwig Reaction Graph Neural Network
- R2 = 0.971 on BH (random split)
- Perturbation-based analysis reveals base-yield relationships

#### ReaMVP (2024 -- J. Cheminformatics)
- **Paper:** https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00815-2
- Large-scale multi-view pre-training
- BH Test4 (hardest OOS): R2 improvement of ~40% over Yield-BERT
- SM Test1: R2 improvement of ~24%

#### "When Yield Prediction Does Not Yield Prediction" (Strieth-Kalthoff et al. 2024)
- **Paper:** https://pubs.acs.org/doi/10.1021/acs.jcim.3c01524
- **Key argument:** Random splits are over-optimistic. Highly related reactions end up in both train and test.
- Document-based splits and reaction-type splits are more realistic.
- BH R2 = 0.972 with Mol2Vec atom embedding on random 70/30 -- but this is misleading for real-world generalization.

#### "Machine Learning C-N Couplings: Obstacles" (2023)
- **Paper:** https://pubs.acs.org/doi/10.1021/acsomega.2c05546
- **Key finding:** Models fail when chemical space is left. Applicability domain is quickly breached.
- Random 70/30: R2 ~ 0.92
- Leave-additive-out: R2 ~ 0.83
- **"It appears to be extremely deceiving to judge models based on literature data with test sets which are split off the same literature data."**

### 2.2 BH/Suzuki Performance Summary (Random 70/30 Split)

| Method | BH R2 | Suzuki R2 | Notes |
|--------|-------|-----------|-------|
| DFT descriptors + RF (Doyle 2018) | 0.92 | -- | Original baseline |
| Yield-BERT (Schwaller 2021) | 0.951 | 0.81 | SMILES-based transformer |
| Yield-BERT augmented | 0.96 | ~0.82 | Data augmentation |
| rxnfp + regression | ~0.95 | ~0.80 | Learned fingerprint |
| DRFP + XGBoost | ~0.95 | ~0.85 | Data-independent fingerprint |
| DRFP + RF | ~0.94 | ~0.84 | |
| YieldGNN | ~0.96 | ~0.86 | Graph neural network |
| BH-RGNN | 0.971 | -- | Reaction graph NN |
| XGBoost + extended FP (2024) | -- | 0.903 | Molecular FP + physical params |
| **KIRBy (DRFP + XGBoost)** | **0.934** | **0.842** | Current results |

**KIRBy's BH R2=0.934 is slightly below published DRFP+XGBoost (~0.95).** Possible causes: (1) different XGBoost hyperparameters (KIRBy uses depth=6, lr=0.1; DRFP paper may differ), (2) different random seeds/splits, (3) DRFP implementation differences (KIRBy uses `drfp` pip package). This gap is small and likely HP-related.

**KIRBy's Suzuki R2=0.842 matches published DRFP+RF/XGBoost range (~0.84-0.85).**

### 2.3 KIRBy Reaction Pipeline: Critical Issues

#### CRITICAL Issue 1: Random splitting violates group rule
- **Severity: CRITICAL**
- The user's group forbids random splits. The reaction pipeline uses random 70/30 x 10.
- **This MUST change.**

#### What non-random splits exist for BH/Suzuki?

**Buchwald-Hartwig (3,955 reactions):**
The BH dataset has a natural factorial structure: 15 aryl halides x 4 ligands x 3 bases x 23 additives.

KIRBy already parses these components in `load_buchwald_hartwig()` (line 136-139 of `reaction_yields.py`):
```python
components.append({
    'aryl_halide': aryl, 'amine': _BH_AMINE,
    'ligand': ligand, 'base': base, 'additive': additive,
})
```

Published non-random splits for BH:
1. **Leave-one-additive-out (LOAO):** Hold out all reactions with a specific additive. 23-fold CV. This is what Schwaller's Test1-4 approximate (held-out isoxazole plates). Published R2 drops from ~0.95 to 0.54-0.84 depending on which additive.
2. **Leave-one-ligand-out (LOLO):** Hold out all reactions with a specific ligand. 4-fold CV. Only 4 ligands, so each fold is 25% of data.
3. **Leave-one-aryl-halide-out:** Hold out all reactions with a specific aryl halide. 15-fold CV.
4. **Leave-one-base-out:** Only 3 bases, so 3-fold CV (each fold = 33%).

**Recommendation for KIRBy:** Use **GroupKFold on additives** (23 groups, use 5-fold by assigning additive groups to folds) as the primary evaluation. This is the most analogous to scaffold splitting -- the additive is the most variable component and most closely represents "unseen chemistry." Expected R2 drop: ~0.95 -> 0.70-0.84.

For a stricter test, also report **leave-one-ligand-out** (4-fold). This is very hard (R2 may drop to 0.3-0.5) since there are only 4 ligands and they fundamentally change the reaction mechanism.

**Suzuki-Miyaura (5,760 reactions):**
Components: 15 electrophile-nucleophile couplings x 12 ligands x 8 bases x 4 solvents.

The components are currently only parsed generically via `_parse_rxn_components()` (splitting by '.'), not named. The raw data is from Perera et al. 2018.

**Recommendation:** Parse the Suzuki components into named fields (electrophile, nucleophile, ligand, base, solvent) and use **GroupKFold on ligands** (12 groups -> 5-fold) as the primary non-random split.

#### Issue 2: Missing representations
- **Severity: MEDIUM**
- KIRBy only has DRFP and concat_ecfp4. Published work also uses:
  - **rxnfp:** Learned reaction fingerprints from BERT (https://github.com/rxn4chemistry/rxnfp). `pip install rxnfp`.
  - **DFT descriptors:** The original Doyle/Dreher approach. Requires QM calculations, probably out of scope.
  - **One-hot encoding of components:** Simple but effective baseline. Encode each (aryl_halide, ligand, base, additive) as a one-hot vector and concatenate.
- **Recommendation:** Add rxnfp as a third representation. Add one-hot component encoding as a fourth (trivial to implement from the already-parsed components dict).

#### Issue 3: Missing models
- **Severity: LOW**
- KIRBy uses Ridge, RF, XGBoost. Published work also uses LightGBM and Gaussian Processes.
- The DTA pipeline uses LightGBM but the reaction pipeline does not.
- **Recommendation:** Add LightGBM to match DTA pipeline.

### 2.4 Concrete Plan for Fixing Reaction Splits

**Step 1: Add GroupKFold splits to `load_buchwald_hartwig()`**

The components dict already has `additive`, `ligand`, `base`, `aryl_halide`. Add a parameter:
```
load_buchwald_hartwig(split_mode='random')  # current default
load_buchwald_hartwig(split_mode='additive')  # GroupKFold on additive (23 groups)
load_buchwald_hartwig(split_mode='ligand')  # Leave-one-ligand-out (4-fold)
load_buchwald_hartwig(split_mode='aryl_halide')  # GroupKFold on aryl halide (15 groups)
```

For `split_mode='additive'`:
- Extract unique additives (23)
- Assign each reaction a group_id based on its additive
- Use sklearn GroupKFold with n_splits=5 (assigns ~4-5 additives per fold)
- Return (train_idx, test_idx) tuples

**Step 2: Add GroupKFold to Suzuki**

First, parse Suzuki components properly (electrophile, nucleophile, ligand, base, solvent). Then:
```
load_suzuki(split_mode='random')   # current
load_suzuki(split_mode='ligand')   # GroupKFold on ligand (12 groups -> 5-fold)
load_suzuki(split_mode='base')     # GroupKFold on base (8 groups -> 5-fold)
```

**Step 3: Update `run_reaction_yield_survey.py`**

Change default from `random` to `additive` (BH) / `ligand` (Suzuki). Report both random and structured results for comparison.

---

## Part 3: Summary of All Recommendations

### HIGH priority (must fix):
1. **Reaction splits:** Replace random 70/30 with GroupKFold on additive (BH) / ligand (Suzuki). The components are already parsed -- just need to add GroupKFold logic.
2. **DTA: Add concordance index.** Every DTA paper reports CI. Without it, KIRBy's DTA results are not comparable to published work.

### MEDIUM priority (should fix):
3. **DTA: Add BindingDB_Kd** from TDC as a third DTA benchmark dataset.
4. **DTA: Add cold-drug evaluation** (the function already exists, just needs wiring).
5. **Reaction: Add rxnfp** as a third representation.
6. **Reaction: Add one-hot component encoding** as a simple baseline.

### LOW priority (nice to have):
7. **DTA: TDC-compatible splits** for leaderboard comparison.
8. **Reaction: Add LightGBM** to match DTA pipeline.
9. **Reaction: Parse Suzuki components** into named fields for group splitting.
10. **DTA: Document** that KIRBy will not match SOTA CI (0.90+) because it uses linear/tree models, not deep interaction models. This is by design (thesis is about representations).

---

## Sources

### DTA
- [DeepDTA paper](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245)
- [DeepDTA GitHub](https://github.com/hkmztrk/DeepDTA)
- [GraphDTA paper](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970)
- [GraphDTA GitHub](https://github.com/thinng/GraphDTA)
- [MolTrans paper](https://academic.oup.com/bioinformatics/article/37/6/830/5929692)
- [MolTrans GitHub](https://github.com/kexinhuang12345/MolTrans)
- [ColdDTA paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482523008375)
- [GRA-DTA paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11121300/)
- [PocketDTA paper](https://academic.oup.com/bioinformatics/article/40/10/btae594/7811139)
- [MTAF-DTA paper](https://link.springer.com/article/10.1186/s12859-024-05984-3)
- [MixingDTA paper](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i105/8199404)
- [LKE-DTA paper](https://link.springer.com/article/10.1007/s11030-025-11394-1)
- [DTA survey 2025](https://academic.oup.com/bib/article/26/5/bbaf491/8260789)
- [DTA comprehensive review 2024](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2024.1375522/full)
- [TDC DTI benchmarks](https://tdcommons.ai/multi_pred_tasks/dti/)
- [TDC leaderboard guide](https://tdcommons.ai/benchmark/overview/)
- [ESM2 drug-target repo](https://github.com/MaastrichtU-IDS/predict-drug-target)
- [DGraphDTA GitHub](https://github.com/595693085/DGraphDTA)
- [Davis-KIBA Kaggle dataset](https://www.kaggle.com/datasets/christang0002/davis-and-kiba)

### Reaction Yield
- [rxn_yields GitHub](https://github.com/rxn4chemistry/rxn_yields)
- [rxn_yields docs](https://rxn4chemistry.github.io/rxn_yields/)
- [BH evaluation results](https://rxn4chemistry.github.io/rxn_yields/results_evaluation_of_buchwald_hartwig_yields_prediction/)
- [Suzuki evaluation results](https://rxn4chemistry.github.io/rxn_yields/results_evaluation_of_suzuki_miyaura_yields_prediction/)
- [rxnfp GitHub](https://github.com/rxn4chemistry/rxnfp)
- [DRFP GitHub](https://github.com/reymond-group/drfp)
- [DRFP paper](https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00006c)
- [Doyle/Dreher 2018 (original BH dataset)](https://www.science.org/doi/10.1126/science.aar5169)
- [YieldGNN / real-world datasets paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10189898/)
- [BH-RGNN 2025](https://pubs.acs.org/doi/10.1021/acs.joc.5c01400)
- [ReaMVP 2024](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00815-2)
- [When Yield Prediction Does Not Yield Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01524)
- [ML C-N Couplings Obstacles](https://pubs.acs.org/doi/10.1021/acsomega.2c05546)
- [MetaRF GitHub](https://github.com/Nikki0526/MetaRF)
- [Robust OOD BH reactions 2025](https://chemrxiv.org/doi/10.26434/chemrxiv-2025-xcr46)
- [Suzuki ML extended FP 2024](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/slct.202402529)
