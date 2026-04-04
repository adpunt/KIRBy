# KIRBy Preprocessing & Data Handling Audit

**Date:** 2026-03-10
**Auditor:** Claude Opus 4.6 (automated line-by-line review)
**Scope:** All dataset loaders, representation modules, and core evaluation engine

---

## CRITICAL Issues (breaks results)

### C1. `create_maccs()` silently drops invalid molecules — row count mismatch

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, lines 206-228
**What:** When an invalid SMILES is encountered, `create_maccs()` skips it with `continue` (line 224), producing fewer rows than input. Compare with `create_ecfp4()` (line 194-201) which correctly appends a zero-vector for invalid molecules.
**Impact:** The returned array has fewer rows than `len(smiles_list)`. When this is later indexed alongside labels/groups that still have the full row count, the arrays are misaligned. Every downstream result using MACCS is silently corrupted.
**Should do:** Append `np.zeros(167)` for invalid molecules (same pattern as `create_ecfp4`).
**Severity:** CRITICAL

### C2. `create_pdv()` silently drops invalid molecules — row count mismatch

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, lines 231-256
**What:** Same issue as C1. Line 249: `if mol is None: continue` skips the sample entirely instead of appending a zero vector.
**Impact:** Same as C1 — misaligned arrays corrupt all PDV-based results, including `create_pdv_binary_mean()` and `create_pdv_rank()` which call `create_pdv()` internally.
**Severity:** CRITICAL

### C3. `create_pdv_binary_mean()` computes column means on FULL dataset — data leakage

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, lines 259-275
**What:** Binarizes each descriptor by comparing to the column mean computed over ALL samples (line 274: `col_means = pdv.mean(axis=0)`). This is called once on the full dataset BEFORE any train/test split. The test set's values influence the binarization threshold used for training data.
**Impact:** Test-set information leaks into the feature representation. The binary features will be slightly biased toward the test distribution.
**Should do:** Either (a) compute means on training set only inside the CV loop, or (b) document that this is an intentional approximation and quantify the bias.
**Severity:** CRITICAL

### C4. `create_pdv_rank()` rank-transforms on FULL dataset — data leakage

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, lines 278-298
**What:** `rankdata()` is applied to the entire dataset (line 297: `ranked[:, j] = rankdata(pdv[:, j], ...)`) before splitting. Ranks are relative to ALL samples, so the rank of a test molecule depends on the training molecules and vice versa.
**Impact:** Same leakage as C3. The rank of each molecule is influenced by molecules in the other fold.
**Should do:** Rank-transform should be fitted on train only, then applied to test (e.g., using percentile mapping).
**Severity:** CRITICAL

### C5. `create_pca_rep()` fits PCA on FULL dataset — data leakage

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, lines 301-324
**What:** Both `StandardScaler().fit_transform(X)` (line 322) and `PCA().fit_transform(X)` (line 323) are applied to the entire dataset before any train/test split. The PCA basis vectors are learned from all data including the test set.
**Impact:** Test-set information leaks into the PCA transformation. This inflates performance of PCA-reduced representations.
**Should do:** PCA must be fitted on training data only and then `transform()` applied to test data. This requires restructuring as a per-fold operation.
**Severity:** CRITICAL

### C6. TCGA genomic reps (`create_hvg2000`, `create_gene_pca100`, pathway scores) fitted on full data before CV

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/genomic.py`, lines 181-265 and 309-395
**What:** `create_hvg2000()` applies `StandardScaler().fit_transform()` on full data (line 227). `create_gene_pca100()` applies `StandardScaler().fit_transform()` then `PCA().fit_transform()` on full data (lines 257-260). All pathway scoring functions (`_score_gene_sets`, `_score_weighted_pathways`) apply `StandardScaler().fit_transform()` on full data (lines 327, 369). These are called once before the CV loop in the TCGA survey script.
**Impact:** All TCGA genomic results have data leakage from fitting scalers and dimensionality reduction on the full dataset.
**Should do:** All scaling/fitting must happen inside the CV loop on training data only.
**Severity:** CRITICAL

### C7. `_make_cv_splitter()` falls back to random `KFold` when groups are insufficient

**File:** `/Users/apunt/repos/KIRBy/src/kirby/find_best_hybrid.py`, lines 533-545
**What:** When `n_unique_groups < n_folds`, the code falls back to `KFold(n_splits=n_folds, shuffle=True, random_state=seed)` (line 545), silently discarding the group information entirely. This is also triggered when `groups is None`.
**Impact:** For any dataset where the number of unique groups is less than the requested folds, the system silently switches to random splitting. This directly violates the "no random splits" requirement. The antibody survey has the same fallback at `tests/antibody_baseline_survey.py` line 293.
**Should do:** (a) Reduce n_folds to match n_groups rather than discarding groups, or (b) raise an error/warning when groups are discarded, or (c) never fall back to random KFold.
**Severity:** CRITICAL

---

## HIGH Issues (inflates or deflates results)

### H1. Antibody `_identity_groups()` hashes sequences when too few unique sequences — pseudo-random splitting

**File:** `/Users/apunt/repos/KIRBy/tests/antibody_baseline_survey.py`, lines 196-213
**What:** When there are fewer than 3 unique sequences (common in DMS datasets where most variants differ by 1-2 AA from wildtype), the code falls back to deterministic SHA-256 hash grouping (lines 208-213). This assigns groups based on a hash of the full sequence, which is effectively random partitioning — identical sequences can land in different groups, and similar sequences (differing by a single mutation) will be in different groups with high probability.
**Impact:** For DMS datasets, this hash-based grouping provides no protection against leakage. Mutations at the same position can appear in both train and test, which is the main concern for DMS evaluation.
**Should do:** Group by mutation position for DMS data, or by wildtype-hash for multi-mutant data.
**Severity:** HIGH

### H2. DMS `identity` grouping by full sequence does not prevent same-position leakage

**File:** `/Users/apunt/repos/KIRBy/tests/antibody_baseline_survey.py`, lines 50-76 and 196-214
**What:** DMS datasets (FLAb, AbAgym) use `'identity'` grouping, which groups by unique heavy chain sequence. In SSM/DMS experiments, each sequence IS unique (each has a different single-point mutation), so every sample gets its own group. GroupKFold with N groups = N samples degenerates to leave-one-out CV. This means mutations at adjacent positions (e.g., Y27A and Y27G) can appear in both train and test.
**Impact:** No structural split for DMS data. Models can memorize position-specific patterns. Results may be optimistic relative to evaluating on truly unseen mutation sites.
**Should do:** Group by mutation position (e.g., all mutations at position 27 in same group) for SSM data.
**Severity:** HIGH

### H3. Reaction yield datasets use random 70/30 splits — not scaffold-based

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/reaction_yields.py`, lines 145-151 (Buchwald-Hartwig) and lines 198-212 (Suzuki)
**What:** Both reaction yield loaders create random train/test splits. Buchwald-Hartwig: `rng.permutation(n)` with 70/30 split (line 149). Suzuki: uses pre-shuffled TSV order from the original paper (line 212), which is also random.
**Impact:** This is actually correct for these datasets — the original papers (Ahneman 2018, Perera 2018, Schwaller 2021) all use random splits, and scaffold splitting doesn't apply to reaction prediction in the same way. However, this should be documented as an intentional exception to the scaffold-split policy. No actual inflation.
**Severity:** LOW (documented exception needed, not an actual error)

### H4. `tune_models()` fits StandardScaler outside CV loop

**File:** `/Users/apunt/repos/KIRBy/src/kirby/find_best_hybrid.py`, lines 406-408
**What:** `scaler = StandardScaler(); X = scaler.fit_transform(X)` is applied to the full concatenated dataset before `RandomizedSearchCV`. The CV inside RandomizedSearchCV then splits this pre-scaled data.
**Impact:** Mild leakage: the scaling statistics (mean, std per feature) are computed on all data, so the test fold in each CV iteration has been scaled using its own statistics. For StandardScaler this is usually a negligible effect, but it's technically incorrect.
**Should do:** Use a `Pipeline(StandardScaler(), model)` inside RandomizedSearchCV.
**Severity:** HIGH (mild practical impact, but methodologically wrong for a benchmarking paper)

### H5. `embed_esm2_entropy_top20` computes positional entropy on full dataset — information leakage

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/antibody.py`, lines 820-850
**What:** `_compute_entropy_from_maps(sequences, pos_maps)` computes Shannon entropy across ALL sequences (line 838), then uses the result to select which 20 positions to pool from. Since this is computed on the full dataset before splitting, the position selection is influenced by test-set sequence variation.
**Impact:** The representation itself is data-dependent (which positions are "variable" depends on all samples), creating feature leakage.
**Should do:** Entropy should be computed on training sequences only per fold.
**Severity:** HIGH

### H6. Sort & Slice (`create_sns`) substructure prevalence computed on full data

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, lines 327-415
**What:** The SNS featurizer counts substructure prevalence across ALL molecules to determine the top-k substructures. The docstring (lines 351-354) correctly notes this must be fitted on training data, but in practice it depends on how the caller uses it. If called once on the full dataset (as appears typical from the survey scripts), this is leakage.
**Impact:** Top substructures are selected using test-set molecules. Substructures unique to the test set may be included, inflating apparent feature informativeness.
**Should do:** The SNS API is correct (supports `reference_featurizer`), but needs enforcement that callers always fit on train only.
**Severity:** HIGH (if called on full data, which depends on the survey script)

### H7. KIBA label negation may cause confusion

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/dta.py`, lines 161-163
**What:** KIBA scores are negated: `labels = -labels` (line 163). The comment says "negate so higher = stronger binding (like pKd)". However, KIBA scores in the DeepDTA dataset are already transformed by the original authors (they applied `12.1 - Kd` to make higher = stronger). Negating them again reverses the polarity.
**Impact:** Depends on whether DeepDTA's Y matrix contains raw KIBA scores or pre-transformed ones. The raw KIBA score has higher = weaker binding, so negation would be correct. But if DeepDTA already transformed them, negation makes higher = weaker, which is backward. This needs empirical verification. If wrong, all KIBA results have inverted direction (R^2 would still be the same, but the interpretation of predictions is wrong).
**Should do:** Verify by checking the actual range of Y values loaded. Original KIBA scores range from ~0 to ~17. DeepDTA-transformed range from ~-5 to ~17.
**Severity:** HIGH (if polarity is wrong, predictions are inverted)

---

## MEDIUM Issues (suboptimal but not data leakage)

### M1. `create_mol2vec()` silently drops invalid molecules — row count mismatch

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, lines 502-522
**What:** Same pattern as C1/C2: `if mol is None: continue` (line 505) skips invalid molecules without appending a placeholder vector.
**Impact:** Same row-count mismatch issue. However, mol2vec failure rate is typically very low.
**Severity:** CRITICAL (same class as C1/C2, listed here but severity upgraded)

### M2. AbAgym label choice: MinMax normalized score preferred over raw DMS score

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/abagym.py`, lines 282-286
**What:** Uses `MinMax_normalized_DMS_score` as the label, falling back to raw `DMS_score` only if MinMax is NaN. MinMax normalization is pre-computed by AbAgym authors (per-experiment). For `abagym_all`, these are then z-scored per experiment (line 313-321).
**Impact:** Double normalization: MinMax then z-score. This is defensible (MinMax for within-experiment, z-score for cross-experiment) but should be documented. The z-scoring step is correct for pooling assays.
**Severity:** MEDIUM (defensible, but needs documentation)

### M3. DDG combined dataset outlier filter uses full-dataset statistics

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/ddg_combined.py`, lines 629-637
**What:** Outlier filtering uses mean and std computed on the entire combined dataset (line 631: `mu, sigma = ddg_values.mean(), ddg_values.std()`). The hard cutoff of |ddG| > 8 kcal/mol is fine, but the 3-sigma filter depends on all data.
**Impact:** Minor: the outlier filter is applied during data loading (before any model fitting), so it's the same across all folds. This is standard practice and not leakage per se, but the 3-sigma threshold is data-dependent.
**Severity:** LOW

### M4. ESM-2 protein embeddings truncate sequences to 1022 characters

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/protein.py`, lines 79, 148
**What:** `batch = [s.upper().strip()[:1022] for s in sequences[...]]` hard-truncates protein sequences to 1022 amino acids. ESM-2 max is 1024 tokens (1022 AA + BOS + EOS).
**Impact:** For proteins longer than 1022 AA (e.g., some kinases are 1000+ AA), the C-terminal region is silently discarded. This affects the embedding quality for long proteins.
**Should do:** Document the truncation. Consider processing long sequences in chunks and averaging.
**Severity:** MEDIUM

### M5. `_is_abag_entry()` in DDG combined uses overly permissive OR logic

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/ddg_combined.py`, lines 84-99
**What:** An entry is classified as Ab-Ag if it has `has_ab OR has_ag` (line 99). This means a complex with an antigen keyword but no antibody (e.g., "lysozyme + barnase") would be incorrectly classified as Ab-Ag.
**Impact:** Some non-antibody protein-protein complexes from SKEMPI may be included. This dilutes the dataset with non-Ab-Ag entries.
**Should do:** Use `has_ab AND has_ag` for keyword-only classification, or `has_ab` alone since Hold_out_type already handles the primary case.
**Severity:** MEDIUM

### M6. Antibody `_mean_pool()` includes BOS/EOS tokens (ESM-2 special tokens)

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/antibody.py`, lines 246-251
**What:** The `_mean_pool()` function in antibody.py pools over ALL non-padding tokens, including BOS and EOS special tokens. Compare with `protein.py` lines 87-95 which explicitly zeros out BOS and the last-token positions.
**Impact:** BOS/EOS embeddings are included in the mean pool, diluting the per-residue signal. For ESM-2, BOS is at position 0 and EOS at position seq_len+1. The effect is larger for short sequences (e.g., VL ~107 AA: 2 special tokens out of 109 = 1.8% dilution).
**Should do:** Exclude BOS/EOS from mean pooling, as done in protein.py.
**Severity:** MEDIUM (consistent bias, not leakage, but degrades representation quality)

### M7. AMPLIFY protein embeddings include BOS/EOS in mean pool

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/protein.py`, lines 274-276
**What:** `mask = inputs['attention_mask'].unsqueeze(-1).float()` — uses full attention mask without excluding special tokens. Consistent with the antibody.py `_mean_pool()` issue (M6) but different from the ESM-2 protein embedding code that correctly excludes them.
**Impact:** Same dilution effect as M6.
**Severity:** MEDIUM

### M8. `_identity_groups` for antibody survey: hash fallback uses modular arithmetic

**File:** `/Users/apunt/repos/KIRBy/tests/antibody_baseline_survey.py`, lines 206-213
**What:** The hash fallback `int(hashlib.sha256(s.encode()).hexdigest(), 16) % n_buckets` uses modular arithmetic on SHA-256 hashes with `n_buckets = max(min_groups, len(seqs) // 5)`. This creates pseudo-random group assignments that may be unbalanced.
**Impact:** Group sizes will be approximately equal (by the uniformity of SHA-256), but this is still fundamentally random splitting disguised as "identity" grouping.
**Severity:** MEDIUM (same issue as H1, restated)

### M9. ESMFold truncates proteins to 512 residues

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/protein.py`, line 438
**What:** `seq = seq.upper().strip()[:512]` — hard truncation to 512 for ESMFold structural prediction.
**Impact:** Many kinase proteins in DAVIS are longer than 512 residues. The structural features will only reflect the N-terminal 512 residues.
**Severity:** MEDIUM

### M10. DTA cold-target splitting may create very small test folds

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/dta.py`, lines 214-229
**What:** `assign_cold_target_groups()` groups by protein ID. DAVIS has 442 proteins, but the pairs per protein are very unbalanced (68 drugs each, but some proteins may be more/less represented). GroupKFold with 5 folds will put ~88 proteins per fold, but the number of PAIRS varies.
**Impact:** Some folds may have very different sizes, increasing variance of CV estimates.
**Severity:** LOW

---

## LOW Issues (cosmetic or minor)

### L1. `create_ecfp4()` returns float32 for binary data

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py`, line 203
**What:** Returns `np.float32` array for binary {0,1} fingerprints. The `_is_binary_rep()` check in the evaluation code (line 667-669) checks for `dtype == np.uint8` first, then checks unique values. Since ECFP4 is float32, it falls through to the unique-values check, which iterates the entire array.
**Impact:** Wasteful memory usage (4x vs uint8) and slower binary detection. No correctness impact.
**Severity:** LOW

### L2. FLAb wildtype identification assumes most-common sequence is wildtype

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/flab.py`, lines 166-173
**What:** `_identify_wildtype()` uses `Counter(heavy_seqs).most_common(1)[0][0]`. This assumes the mode sequence is wildtype. For some DMS experiments, the wildtype may not appear at all (it may be excluded) or may not be the most common variant.
**Impact:** If wildtype identification is wrong, all mutation annotations are wrong, affecting mutation-aware representations.
**Severity:** MEDIUM (depends on dataset, but could be HIGH for datasets where WT is not the mode)

### L3. AbAgym: `all_labels` appended after `all_heavy_seqs` and before NaN check

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/abagym.py`, lines 277-303
**What:** Heavy/light seqs are appended (lines 278-279) BEFORE the NaN check on labels (lines 283-286). If `label` is NaN and `DMS_score` is also NaN, `continue` is executed at line 286, but only AFTER the sequences have already been appended at lines 278-279.
**Impact:** The `all_heavy_seqs` list would have one more entry than `all_labels`, causing misalignment.
**Severity:** CRITICAL (reclassifying from LOW — this is a bug)

### L4. `_is_binary_rep()` check is fragile

**File:** `/Users/apunt/repos/KIRBy/src/kirby/find_best_hybrid.py`, lines 667-669
**What:** `is_binary = (X.dtype == np.uint8 or (len(np.unique(X)) <= 2 and set(np.unique(X).tolist()).issubset({0, 0.0, 1, 1.0})))`. The `np.unique()` call on a large matrix is O(n*d*log(n*d)). Also, binary fingerprints that have a small number of unique values per column but >2 globally will not be detected (though this is unlikely for ECFP4 which has many 0s and some 1s).
**Impact:** Performance issue only — the check itself is correct for ECFP4/MACCS.
**Severity:** LOW

### L5. QM9 loader uses `random.sample()` instead of numpy RNG

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/qm9.py`, lines 117-119
**What:** Uses Python's `random.seed(random_seed)` and `random.sample()` instead of `np.random.RandomState`. Python's random module state is global and could be affected by other code.
**Impact:** Subsampling is still deterministic with the same seed, but could be affected by other random calls between sessions.
**Severity:** LOW

### L6. `get_qm9_splits()` includes random and stratified splitters

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/qm9.py`, lines 272-381
**What:** The `get_qm9_splits()` function offers `'random'` (line 328-338) and `'stratified'` (line 340-359) splitting options using `np.random.permutation` and `train_test_split`. These are available to callers despite the project's no-random-split policy.
**Impact:** If a caller uses `splitter='random'` or `splitter='stratified'`, they get random splits. The default is `'scaffold'`, which is correct.
**Severity:** LOW (API design issue — dangerous option exists but is not the default)

### L7. hERG `get_herg_splits()` offers random splitting

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/herg.py`, lines 325-336
**What:** Same as L6 — `splitter='random'` option available in the hERG loader.
**Severity:** LOW

### L8. `_parse_site()` discards insertion codes

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/abagym.py`, lines 108-118
**What:** AbAgym sites like `100A` (insertion code A) are parsed to integer 100, losing the insertion code. Two different sites (100 and 100A) would map to the same integer.
**Impact:** Could cause incorrect mutation application if a PDB has both residue 100 and 100A. In practice this is rare for antibody variable domains.
**Severity:** LOW

### L9. ESOL `get_esol_combined_train()` normalizes labels

**File:** `/Users/apunt/repos/KIRBy/src/kirby/datasets/esol.py`, lines 81-120
**What:** Labels are z-score normalized using train+val statistics (lines 103-107). This is a label transformation applied at loading time.
**Impact:** If the evaluation pipeline also applies label normalization, this would double-normalize. Not a leakage issue since the function is designed for single train/test use, not CV.
**Severity:** LOW

### L10. ESM-C embeddings include all tokens in mean pool (no BOS/EOS exclusion)

**File:** `/Users/apunt/repos/KIRBy/src/kirby/representations/protein.py`, lines 222-223
**What:** `emb = output.embeddings.mean(dim=1)` — pools over all token positions without excluding special tokens.
**Impact:** Same class of issue as M6/M7, but for ESM-C. Consistent behavior at least.
**Severity:** LOW (consistent, but suboptimal)

---

## Summary of reclassified issues

After full review, L3 (AbAgym sequence-label misalignment) has been reclassified as CRITICAL.

| Severity | Count | Examples |
|----------|-------|---------|
| CRITICAL | 8 | C1-C7, L3 (reclassified) |
| HIGH | 7 | H1-H2, H4-H7, M1 (reclassified) |
| MEDIUM | 8 | M2-M10 |
| LOW | 10 | L1-L2, L4-L10 |

---

## Top priority fixes (blocking correctness)

1. **C1/C2/M1**: Fix `create_maccs()`, `create_pdv()`, and `create_mol2vec()` to return zero vectors for invalid molecules instead of skipping them. One-line fix each.

2. **L3 (CRITICAL)**: Fix AbAgym label append ordering — move `all_heavy_seqs.append()` and `all_light_seqs.append()` to AFTER the NaN check on labels.

3. **C3/C4/C5/C6/H5/H6**: All data-leakage issues from fitting scalers, PCA, rank transforms, entropy, and SNS on the full dataset. This is the most architecturally significant fix — representations that depend on fitted statistics must either: (a) be restructured as per-fold factories, or (b) accept that the leakage is negligible and document it.

4. **C7/H1/H2**: Splitting fallback issues. The `KFold(shuffle=True)` fallback should at minimum log a warning. DMS grouping needs a fundamentally different strategy (group by mutation position).

5. **H4**: `tune_models()` scaling should use a Pipeline.

6. **H7**: Verify KIBA label polarity empirically.
