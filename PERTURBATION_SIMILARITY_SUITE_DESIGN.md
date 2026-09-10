# Perturbation-Similarity Experiment Suite — Design Doc
**Branch:** `similarity-metrics-study` · **Home:** standalone OPIG experiment · **Date:** 2026-07-19

---

## 0. Framing lock (read first)

**Deliverable:** a *methodology* — representations and distances built for *small perturbations* (one mutation, one matched-molecular-pair transformation) — tested across varied data. Domain (antibody / protein ΔΔG / molecular) is **which data it is tried on**, never the unit of the claim.

**Scope constraints, non-negotiable:**
1. **STANDALONE.** Nothing registers into `src/kirby/representations/` or `DATASET_REPS`. The `dphyschem` registration is already reverted (commit `da8ec98`; `git diff HEAD` on `mutation.py`, `_shared.py`, `complexity_helpers.py` is empty). Two stale doc blocks still claim registration and must be corrected: `SIMILARITY_METRICS_ROADMAP.md:886-905` and `scaffold_feature_isolation.py:23-26`.
2. **No edits** to `find_best_hybrid.py` / `hybrid.py` (rebuild-roadmap lock). Experiment C1 *calls* the augmentation machinery; it does not modify it.
3. Scaffold/group-aware CV only. Baselines reported individually, never averaged. Anti-circularity grading mandatory.

---

## 1. What the previous session actually delivered — and what it overclaimed

**Built and reusable (keep):**
- `tests/experiments/opig_scaffold_similarity.py` (467 lines): 3 modes, 6 datasets, 3 graders, 3 named honesty controls.
- `similarity_metrics/scaffold_reps.py` (236), `scaffold_experiment.py` (281), `scaffold_ddg_experiment.py` (211), `scaffold_feature_isolation.py` (176), `feature_type_study.py` (237).
- `SCAFFOLD_SIMILARITY_RESULTS.md` (246 lines) + `tests/experiments/results/opig_scaffold_similarity.json` (49 KB).

**Genuinely good work — do not discard:**
- **Control C3 caught a fatal circularity.** `dist_position_gated` is bit-for-bit the Grantham grader on a single-mutation library (Spearman 1.000000, max abs diff exactly 0.0). The prior headline 0.519 was grader-vs-grader agreement. This is retracted *in the doc*, correctly.
- **Control C2 (count control) is real.** `BaseDelta` scores 0.796 / 0.924 raw on adams/phillips and 0.000 controlled — the raw number was pure mutation-count leakage.
- The feature study's redundancy evidence is concrete and mechanistic (molecular weight: solo 0.535 on phillips, r=0.98 with a chosen feature, adding it costs −0.057).

**Overclaims that must be corrected in the results doc before anything is built on top:**

| # | Claim as written | What the JSON actually shows |
|---|---|---|
| O1 | "dPhysChem_pos wins 6/6 datasets, both domains" (`SCAFFOLD_SIMILARITY_RESULTS.md:239`) | At most 5, arguably 4. `flab_expression_koenig` and `flab_binding_koenig` are the **same variant library** — identical in every structural column (both 0.387, both 0.516, both 0.619, identical greedy set). The doc concedes this at line 108-112 then counts it twice anyway. |
| O2 | graphinity 0.627 listed in the same column as koenig 0.387 (line 127) | **Different representation.** graphinity routes through `feature_sqdiff()` (14 scales); antibody sets route through `SR.rep_dphys_positional` (5 scales, `scaffold_reps.py:37`). Proof: graphinity `dPhysChem_pos euclidean` = 0.627 = `all_features_score` (all 14), bit-identical. Not the same yardstick. |
| O3 | "euclidean >> cosine, supported, all datasets" (line 241) | On graphinity **cosine was never computed**. `n_rep_rows` = 2, euclidean only. |
| O4 | "5 representations compared" | 3 of 5 produce nothing. MutID = 0.000/0.009/0.017/0.030 (noise). BaseDelta = 0.000 euclidean everywhere. SubScore = `nan` on all 6 rows. It is a 2-way comparison. |
| O5 | Headline dPhysChem_pos = 0.387 on koenig | **Superseded by the suite's own prior finding.** `scaffold_feature_isolation.py` established 0.513 after dropping charge + isoelectric_point. That 3-scale improvement lived only in the reverted `mutation.py` block and now exists in **no runnable code path**. |
| O6 | Suite answers "scaffold-diff beats whole-object" | `aac` and `blosum_mean` are declared in `REP_BASIS` (`opig_scaffold_similarity.py:106-107`) and **never built** — those names appear at no other line. The motivating question is unmeasured. The doc's own claim table (line 246) correctly says "not yet established". |

**Additional defects to fix while touching the file:**
- Dead compute at `opig_scaffold_similarity.py:229`: `SD, truth, ham = feature_sqdiff(ctx)` — all three discarded, recomputed at :231. On graphinity that is a full redundant pass over 58,564 mutations.
- `labels` and `task_type` are loaded into ctx (:132-133) and referenced nowhere. No activity axis exists.
- Single seed (`seed=0`, :152). No bootstrap, no CI, no significance anywhere.
- Zero tests. No `test_*.py` imports any scaffold module.
- **Dataset-policy contradiction:** `flab_binding_phillips_h3` is roadmap-cut ("stays cut, user, 2026-07-18") yet is a headline dataset at :91 and supplies the largest reported number (0.645). It is also the most count-contaminated set (0.924) and a censored classification assay (`dataset_policy.py:244-250`). **Decision needed from user before it appears in any new result.**

---

## 2. Shared infrastructure (build once, all experiments use)

**NEW — `similarity_metrics/perturb_core.py`:**
- `PairSet`: a canonical (i, j, Δy, perturbation-descriptor) container, one per dataset family. Antibody/ΔΔG pairs come from mutation records; molecular pairs from real MMP rows.
- `bootstrap_spearman(x, y, n_boot=1000, seed=0)` → point estimate + 95 % percentile CI. **Every number in every experiment below gets a CI.** Non-negotiable given O1–O5.
- `held_out_grader(name)`: a third grader used *identically* for every representation, so cross-representation comparison becomes possible (the current design makes it structurally impossible — `SCAFFOLD_SIMILARITY_RESULTS.md:217-222`).
- Pair-level group CV: molecules/variants, not pairs, are assigned to folds; a pair is used only if **both** members are in the same fold. Nothing in the repo does this today.

**REUSED as-is:** `scaffold_reps.grantham_matrix / blosum_matrix / identity_matrix`; `simeval.py` (6 coefficients, 9 evals); `structural_yardstick.mcs_distance`; `mcs_cache` npz caches (8000 pairs × 3 datasets); `_data.py` loaders + `CAP_N=1400`; `noise_mitigation_methods.compute_distance_matrix` + `KNNConsistencyFilter`; `noise_estimation.smooth_by_scores`; `noise_spec.py`.

---

## 3. The experiments

### GROUP A — Representation design (incl. counts)

---

#### **A1 — Count-based perturbation features vs binary**
*Status: NEW. Local. ~2 h build, minutes to run.*

**Question.** The similarity results say count fingerprints beat binary on activity read-across. Does that carry into a *perturbation* representation, where the object is a substitution rather than a molecule?

**Inputs.** For each variant pair, three encodings of the same information:
- **binary**: one-hot indicator of which substitution types occurred (20×20 upper-triangle, folded to 210 bits).
- **count**: integer number of times each substitution type occurs in the variant relative to wild type.
- **log-count**: `log1p(count)`.
Plus, per encoding, the same under `dPhysChem` deltas (raw Δ vs |Δ| vs signed-Δ summed vs positionally tagged).

**Procedure.** z-score each column over the dataset (the feature study measured z-scoring is worth **+0.16 to +0.38** structural fidelity — graphinity 0.387→0.627). Compute pairwise euclidean, cosine, and — because these are now genuine count vectors — **Soergel** (`simeval.py:84-99`; provably identical to Tanimoto on binary, `simeval.py:126-128`, so it is a *counts-only* coefficient). Grade against a held-out grader not on the representation's own basis.

**Metric.** `struct_fidelity` = Spearman ρ between pairwise representation distance and pairwise grader distance, over 8000 sampled pairs, **with bootstrap 95 % CI**. ρ = 1.0 means the representation orders every pair exactly as the independent grader does; ρ = 0.0 means the ordering is unrelated to the grader; ρ < 0 means inverted.

**Baselines.** (i) binary encoding of the identical feature set — the direct counts-vs-binary contrast; (ii) `blosum_mean` whole-object rep (structural 0.408, activity 0.48 — "the only all-rounder, the bar to beat"); (iii) `aac`. **Both baselines must actually be built this time** (fixes O6).

**Datasets.** `flab_expression_koenig`, `flab_binding_adams`, `graphinity_foldx`. Koenig binding **dropped** as a duplicate library (fixes O1). Phillips held pending the policy decision.

**Confound to control explicitly.** In the source result, the two top-ranked entries were `ecfp4_count:soergel` (14.4) and `atompair_count:soergel` (14.5) — **counts and Soergel are confounded**, since Soergel exists only for counts. A1 breaks this by running count vectors under tanimoto / dice / cosine / soergel / euclidean separately.

**Paper.** O'Boyle & Sayle, *J Cheminform* 8:36 (2016), DOI 10.1186/s13321-016-0148-0 — the same count fingerprints **improved** performance on the single-assay (close-analogue) benchmark and were "uniformly worse" on the multi-assay (diverse-structure) benchmark. Their mechanism, quoted: *"penalising the absence of a second (or third, etc.) feature to the same extent as penalising the complete absence of a feature yields a worse similarity measure."* **Small perturbations are the close-analogue regime, so this literature predicts counts should win here** — that is the hypothesis, and it is a real, mechanistically-explained prediction rather than a guess. Secondary: Huber & Pollmann, *J Cheminform* (2026), DOI 10.1186/s13321-026-01245-y (counts and log-counts improve specificity; caveat — confounded with a folding/bit-collision fix).

---

#### **A2 — Small-perturbation resolution (does the metric see the change at all?)**
*Status: NEW. Local. ~3 h. **Highest value-per-effort in the suite.***

**Question.** Given a pair that differs by exactly one known perturbation, does the distance return a non-zero, well-spread value — or does it collapse?

**Inputs.** Perturbation pairs with a *known, discrete* difference: single-mutation antibody pairs; single-transformation MMP pairs from the cached `MMP_data_clean.csv` (35,484 rows, `/Volumes/seagate/kirby/chembl_qsar/chembl_dopamine_d2/`); stereoisomer pairs where available.

**Procedure.** For each (representation × coefficient), collect the distance distribution over perturbation pairs only.

**Metrics — all four from the source paper, exactly:**
- **p0** = fraction of pairs with distance exactly 0. **p0 = 0** means the metric resolves every perturbation; **p0 = 1** means total blindness (the metric cannot distinguish the pair at all).
- **CV** = σ/μ of a Beta-fitted distance distribution. **Higher = more resolving power**; the paper reports ChemBERTa collapsing at CV ≈ 0.04–0.05, MolFormer-cosine at 1.178–1.215, MACCS/Dice at 0.73–0.99.
- **RC** (relative contrast) = (d95 − d5)/d5. **RC → 0** means all distances are the same number, i.e. the metric is dead in this space.
- **SN_k** (hubness skewness of the k-occurrence distribution at k=5). **SN_k ≥ 1.5 flags an artifact** — a few points are the nearest neighbour of everything.

**Baselines.** Every base rep the pipeline offers on these datasets (`aac`, `blosum`, `onehot`, `esm2`, `ablang2`, `antiberty`, `ecfp4`, `ecfp4_count`) — the whole-object arm. **Expected published result to reproduce as a sanity check: all fingerprints are 100 % stereo-degenerate (p0,stereo = 100 %).** If our harness does not reproduce that, the harness is wrong.

**Datasets.** All antibody DMS + `graphinity_foldx` + `chembl_dopamine_d2` MMPs.

**Paper.** Dabrowski-Tumanski, Topolski, Plewczynski & Jetka, **"The Geometry of Activity Cliffs: Representation Dependence and Multi-Scale Characterization of Activity Landscapes", arXiv:2605.30831 (29 May 2026)** — Steps 1 and 6. *This is the paper the hand-off said to read and never was.* Its thesis is exactly ours: an activity cliff is a property of the (embedding, distance) pair, not of the molecules — at a fixed 90 % similarity threshold the cliff count varies **55-fold** across representations (142 under Morgan/Tanimoto vs 7,903 under RDKit/Dice) on the same SARS-CoV-2 dataset. **Differentiation to write:** they characterise *existing* representations geometrically; we *design* perturbation-native representations and test them on the same geometry. Nothing in that paper builds a difference/transformation encoding.

**Why this is the flagship.** It needs no labels, no model, no grader, and therefore has no circularity surface at all — the chemistry defines what "one perturbation" is and the metric either resolves it or does not. It is immune to every failure mode that killed `position_gated`.

---

#### **A3 — Position-selection variant of dPhysChem_pos**
*Status: NEW but small. Local. ~2 h.*

**Question.** `feature_type_study` showed *which properties* to include matters hugely (greedy-5 0.6187 > peak-10 0.6078 > all-14 0.5160). Does *which positions* to tag matter equally?

**Procedure.** Greedy forward selection over sequence positions (or position-bins for ΔΔG multi-PDB), reusing `feature_type_study.greedy_forward`. Restore the 3-scale winner (hydro + MW + aromaticity) that was lost in the revert — currently in no runnable code path (fixes O5).

**Metric.** `struct_fidelity` (as A1) + CI. Bar to beat: **0.513** (`scaffold_feature_isolation`, koenig), not the stale 0.387.

**Baselines.** all-positions dPhysChem_pos; random position subsets of matched size (n=20 draws) — without this, greedy selection is guaranteed to "win" by overfitting the grader.

---

### GROUP B — Algorithm × dimensionality

---

#### **B1 — Model class × representation width, same task, same folds**
*Status: NEW experiment; models all exist. **SERVER** (this is the compute-heavy one).*

**Question.** Standard reps are 1280–3072 D; perturbation reps are 5–50 features. Does the winning *algorithm* change across that ~400× width gap?

**Inputs — width ladder, measured not assumed** (`X.shape[1]` at runtime; the registry comments are **wrong** — `ablang2` documented 768 D, measured 480; `igbert` documented 768 D, measured 1024):
| Band | Reps |
|---|---|
| 3–50 D | `esm2_llr` (3), `vhvl_angles` (6), `cdr_canonical` (12), `paratope` (8), `structural` (28), `aac` (40), `mutation_features` (~48), **our perturbation reps (5–50)** |
| 60–210 D | `all_geometry` (162), `maccs` (167), `desc` (208) |
| 480–1280 D | `ablang2` (480), `antiberty` (1024), `igbert` (1024), `esm2_mutsite` (1280) |
| 2048–5600 D | `prot_t5` (2048), `ecfp4` (2048), `esm2` (2560), `esm2_wt_delta` (2560), `onehot`/`blosum` (5600) |

**Models.** `ridge`, `knn`, `svm`, `gp`, `mlp`, `rf`, `xgboost`, `lightgbm`, `catboost`. **Ridge and GP must be passed explicitly** — `_resolve_models` (`find_best_hybrid.py:1674-1677`) hard-gates ridge to `nt_*`/`gb_*` datasets, and `DOWNSTREAM_MODELS` (`_shared.py:324`) omits ridge, knn, gp, catboost entirely. With the defaults, this experiment *cannot* detect the effect.

**Procedure.** Same scaffold/group folds for every cell. Re-tune hyperparameters **per representation** — Boldini 2023 found hyperparameter importance itself varies significantly across feature sets of different width, so a single tuning pass confounds the comparison. Disable `MODEL_MAX_BUDGET` truncation or report it as a covariate (`find_best_hybrid.py:2020`: gp 1000, svm 500, knn 2000, mlp 1000; no cap for trees) — otherwise the caps *are* the result.

**Metric.** Spearman ρ and RMSE per (rep × model × fold), reported **per cell, never averaged**. Primary derived quantity: **rank-of-model-within-rep**, and the Kendall τ between model-rank orderings at the narrow end vs the wide end. **τ = 1** means algorithm choice is width-independent (no interaction); **τ = 0** means the ranking is completely reshuffled by width; **τ = −1** means it inverts.

**Baselines.** Existing latent evidence to reproduce — `tests/results/universal_survey/logd.json` already shows a clean crossover: ridge wins every wide dense embedding (grover 4800 D: ridge 0.841 vs lightgbm 0.752; molformer 768 D: 0.753 vs 0.685) while lightgbm wins every 50 D PCA rep (chemberta_pca50 0.595 vs 0.467) and 2048 D sparse ecfp4 (0.756 vs 0.723). knn collapses on ecfp4 (0.409 vs lightgbm 0.552) and is best-or-near-best on 50 D PCA reps on herg. **This data is on disk and has never been analysed.** A cheap local pre-study (B1a, 1 h) reads those three JSONs and writes the crossover up — do that before spending server time.

**Datasets.** Molecular: `herg_ki`, `logd`, `caco2` (multi-model data already exists). Antibody: 3 DMS sets — **no multi-model antibody data exists anywhere locally**; every file in `tests/results/antibody_survey/` is `models=['ridge']`, 3–4 reps, NaN scores.

**Papers.** Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 D&B, arXiv:2207.08815 — trees beat deep nets on medium tabular data, and the named mechanism is **rotational invariance + non-robustness to uninformative features**, which is exactly why an isotropic distance should benefit from a narrow rep while a boosted tree does not. Boldini, Grisoni, Kuhn, Friedrich & Sieber, *J Cheminform* 15:73 (2023), DOI 10.1186/s13321-023-00743-7 — hyperparameter importance varies with feature-set width. Griffiths et al., GAUCHE, arXiv:2212.04450, NeurIPS 2023 — **counter-evidence**: Tanimoto GP kernels work fine on 1024–2048-bit fingerprints, so "kernels die in high dimension" must be *measured*, not asserted. Ansuini, Laio, Macke & Zoccolan, NeurIPS 2019, arXiv:1905.12784 — intrinsic dimension is orders of magnitude below ambient width, so a 1280 D ESM embedding is not a 1280-dimensional cloud.

**Honest gap this fills.** No published paper crosses algorithm class × representation width as the experimental axis in molecular or protein ML. Boldini is closest and only varies the feature set *within* gradient boosting. This is a genuine contribution.

---

#### **B2 — Distance-geometry diagnostics as the explanation for B1**
*Status: NEW. Local. ~2 h. Reuses A2's machinery.*

**Question.** *Why* does width hurt the distance-based arm and not the tree arm?

**Procedure.** For every rep in B1's ladder compute, on the same data: relative contrast RC, hubness skewness SN_k (k = 5, 10, 20), and TwoNN intrinsic dimension. Then correlate each against the B1 per-model score deltas.

**Metrics.** **RC** — (d95−d5)/d5; → 0 means distances are indistinguishable. **SN_k** — skew of the k-occurrence distribution; 0 = every point is a neighbour of equally many others, large positive = hubs dominate. **Spearman ρ between RC and (knn score − lightgbm score)** across reps: **ρ ≈ +1** would mean distance contrast fully explains when distance-based models win; **ρ ≈ 0** means the curse-of-dimensionality story does not apply here and we should stop invoking it.

**Also test the cheap lever.** Aggarwal, Hinneburg & Keim (ICDT 2001, DOI 10.1007/3-540-44503-X_27): L1 beats L2 in high dimension, and fractional L_p (p ≈ 0.3–0.4) preserves contrast further. Run the wide reps under L2 / L1 / L_0.35 and report whether the metric choice recovers the narrow-rep advantage. If it does, "just use L1" is a cheaper finding than any representation work.

**Papers.** Beyer, Goldstein, Ramakrishnan & Shaft, ICDT 1999, pp. 217–235 (nearest/farthest distance convergence, bites from 10–15 dimensions). Radovanović, Nanopoulos & Ivanović, *JMLR* 11:2487–2531 (2010) (hubness). Aggarwal et al. ICDT 2001.

---

### GROUP C — Augmentation vs base representation

---

#### **C1 — Are perturbation features an augmentation, not a base rep?**
*Status: NEW experiment, ZERO new machinery. **SERVER** (calls `find_best_hybrid`). ~3 h build.*

**Question.** Perturbation features are 5–50 columns against a 1280 D base. As a base representation they get drowned; as an augmentation source they are tested one column at a time and kept only if they help. Which routing wins?

**Key structural fact (from the code):** augmentations are **not a distinct feature type**. `create_hybrid(augmentations={name: ndarray})` (`hybrid.py:3280`) takes plain arrays loaded by the *same* loader as base reps (`complexity_helpers.py:971-977`). A rep is an "augmentation" purely by list membership. **So this experiment costs one dict entry, not a code change** — and it stays standalone, since we pass the arrays directly rather than registering them.

**Arms.**
| Arm | Config |
|---|---|
| A0 | base rep alone (e.g. `esm2` 2560 D) |
| A1 | base rep ⊕ perturbation features, `augmentation_strategy='uncapped'`, `candidate_ranking='rf_importance'` |
| A2 | same, `candidate_ranking='residual_correlation'` |
| A3 | `augmentation_strategy='capped'` (fill-then-replace, holds total width fixed) |
| A4 | `augmentation_strategy='all'` (no testing — top-`augmentation_budget` by RF importance, hstacked) |
| A5 | plain `np.hstack([base, perturb])` with no selection — the naive concat control |

**Why A2 matters specifically.** `residual_correlation` (`hybrid.py:1987-2009`) fits the model on base features, takes `residuals = y_train − m.predict(X_base_s)`, and scores each augmentation column by `abs(corrcoef(col, residuals))`. **It is the only place in the entire codebase where complementarity is REWARDED rather than merely penalised** — every allocator upper-bounds a rep's budget by its own standalone score, so complementarity can only subtract. Perturbation features are exactly the case where the signal is complementary-but-individually-weak. Its empirical value vs `rf_importance` has never been measured anywhere.

**Metric.** Spearman ρ and RMSE on held-out folds, per model, per arm. Primary: **Δ = score(arm) − score(A0)**, signed so positive = augmentation helped, reported per fold with a paired Wilcoxon across folds. Secondary: **n_kept** = how many perturbation columns survived the A/B gate (0 means the augmentation path rejected them entirely — a clean negative).

**Gotchas that will silently corrupt this if missed.**
- Pass `augmentation_strategy='none'` explicitly when `augmentations=None`, or you hit the documented **cache-key collision** (`AUDIT_OPIG.md` P0-3).
- `aug_names`, `aug_hashes`, `augmentation_strategy`, `augmentation_threshold`, `augmentation_budget` all enter the cache key (`find_best_hybrid.py:433-445`) — adding a source forces a rebuild.
- `_select_augmentation_features` returns **all** columns untouched when `n_select >= n_features` (`hybrid.py:1456-1458`). With `augmentation_budget=20` and a 14-column perturbation rep, no selection happens at all. Set the budget below the width or the arm is meaningless.
- `'iterative'` is **not** in the dispatch chain (only `greedy_ablation` / `capped` / `uncapped` / `all` at :3845/:3871/:3907/:3936). `f06_augmentation_test.py:124` passes it and it falls through silently applying no augmentation. Do not use that string. (F06 is a framework experiment — do not modify it.)
- No weighting or scaling exists at concatenation: every path is `np.hstack` (:1498, :3934, :3957) with only clip + `nan_to_num`. A 5-column z-scored perturbation block next to a 2560 D unscaled embedding is a scale mismatch nobody has addressed. Report base-vs-aug column-norm ratio as a diagnostic.

**Datasets.** 3 antibody DMS + `graphinity_foldx` + `herg_ki`.

**Paper anchor.** Wittmann, Yue & Arnold, *Cell Systems* 12(11):1026–1045 (2021), DOI 10.1016/j.cels.2021.07.008 — training-set/feature *composition* dominates model choice in MLDE. Grinsztajn (above) for why uninformative columns hurt some learners and not others.

---

### GROUP D — Noise-mitigation similarity tasks (from the repo's own noise code)

These are **real downstream tasks**, each already implemented in `src/kirby/utils/`. Each swaps our distance in where the existing code takes `precomputed_distances`. `DISTANCE_AWARE_METHODS` (`noise_mitigation_methods.py:858`) = 9 of the 16 registered detectors accept one.

---

#### **D1 — Noise detection by neighbour label disagreement (ENN / Wilson editing)**
*Status: mostly BUILT. Local. ~1 h wiring.*

**Question.** Whose neighbourhoods expose an injected label error fastest?

**Procedure.** `inject_label_noise` (`simeval.py:292`; Gaussian, σ default 0.6, frac 0.15, shared realisation per (dataset, seed) so arms are paired). For each candidate distance, flag each variant by `|y_noisy − median(y_noisy of its k nearest neighbours in THIS metric's space)|`. Sweep σ ∈ {0.3, 0.6, 0.9} × rate ∈ {0.10, 0.20, 0.30} × seeds {0,1,2} (`noise_sweep_scores`, :355) — justified because noise-sensitivity rankings can **invert** between low and high noise (Cortes-Ciriano & Bender, JCIM 2015).

**Metric.** **AUROC of the disagreement score against the known injected-noise mask.** **1.0** = the metric's neighbourhoods rank every corrupted point above every clean one; **0.5** = no better than chance; **< 0.5** = the metric actively hides noise.

**Baselines.** `ecfp4:tanimoto`, `blosum_mean`, `aac`, best whole-object embedding. Report the **stability caveat measured previously**: ECFP4:Tanimoto's noise-detection rank is 20 ± 16.3 on herg (swings 6th → 41st) because metrics are near-tied — so noise-detection alone is not a rankable headline, and D3 (recovery, rank stability ρ ≈ 0.9) is the trustworthy one.

**Built:** `simeval.eval_noise_detect` (:320), `KNNConsistencyFilter` (`noise_mitigation_methods.py:182`), `compute_distance_matrix` (:67). **New:** perturbation distances plugged in; pair-level CV honesty.

**Papers.** Wilson, *IEEE Trans SMC* SMC-2(3):408–421 (1972) — the ENN origin. **Explicit gap: no chemistry or biology paper applies ENN/RENN as a label-cleaning step on bioactivity data with a demonstrated gain.** ENN appears in chemistry only inside SMOTE-ENN class rebalancing. Northcutt, Jiang & Chuang, *JAIR* 70:1373–1411 (2021), arXiv:1911.00068 — Confident Learning; **also never applied to ChEMBL or any bioactivity set**. Both are real white space.

---

#### **D2 — Cliff-vs-noise discrimination**
*Status: BUILT (two independent implementations). Local. ~1 h.*

**Question.** A point disagreeing with its neighbours is either a mislabel or a real activity cliff. Does a perturbation-native distance separate the two better than a whole-object one?

**Procedure.** `NeighborhoodConsensusFilter` (`noise_mitigation_methods.py:357`) implements exactly the disambiguation logic: *noise* = sample disagrees with neighbours **and** neighbours agree with each other (`high_disagreement & neighborhood_consistent`); *cliff* = sample disagrees **and** neighbours also disagree. Continuous twin: `ActivityCliffAwareEstimator` (`noise_estimation.py:289`), `deviation × consistency`, cliff flags at `(deviation > 1.0) & (consistency < 0.5)`.

**Metric.** **`cliff_contamination_rate` = |ground-truth cliffs flagged as noise| / |total flagged as noise|** (`s07_noise_bridge.py:43`). **0.0** = every flag is a genuine mislabel, no real cliff destroyed; **1.0** = the detector is doing nothing but deleting real biology. Must return **NaN, not 0.0**, when nothing is flagged — returning 0.0 hands a perfect score to a detector that never fires (this bug is already fixed in `s07`; preserve the behaviour). CV-honest variant: scaler stats **and** the flag threshold fit on train folds only, held-out neighbours drawn from train only.

**Ground truth (must be representation-independent).** Antibody: edit distance. Molecular: MMP-cliff from the cached transformation rows, **not** a fingerprint. Threshold: potency gap must exceed the experimental error floor — Kramer, Kalliokoski, Gedeck & Vulpetti, *J Med Chem* 55(11):5165–5173 (2012), DOI 10.1021/jm300131x: ChEMBL Ki mean absolute error 0.44 pKi, SD 0.54, median 0.34, **max achievable R² ≈ 0.81**. Differences under ~0.5 log units are indistinguishable from measurement noise.

**Honest gap:** I could not find any published paper that operationalises cliff-vs-error disambiguation as an algorithm with a validation metric. The ingredients (Kramer's error distribution + MMP-Cliffs) exist; the protocol does not. That is a paper.

---

#### **D3 — Label correction / recovery**
*Status: BUILT. Local. ~1 h.*

**Procedure.** Distance-weighted kNN leave-one-out estimate `est = Σ(w·y_noisy)/Σw`, `w = 1/(d + 1e-9)` (`simeval.py:332`), and the four `smooth_by_scores` schedules (`noise_estimation.py:1285`): proportional, threshold, soft_threshold (sigmoid), confidence.

**Metrics.** (i) **`corr(applied correction, −ε)` on corrupted points only** — Pearson; **1.0** = the correction exactly undoes the injected error, **0.0** = the correction is unrelated to the error, **negative** = it makes labels worse. (ii) **recovery % = (denoised − noisy)/(clean − noisy) × 100**, gated by a minimum-gap check (`n11:532`) — 100 % = full restoration of clean-label performance, 0 % = no improvement, negative = harm.

**Note:** this is the *stable* ranking (Spearman ρ ≈ 0.9 across seeds), unlike D1. Headline from here, not from D1.

---

#### **D4 — Real (not injected) noise: replicate disagreement**
*Status: BUILT (`n06`). Local, but data-gated.*

**Procedure.** BindingDB per-(drug_id, target_id) replicate standard deviation in pKd is the ground truth — 3,487 pairs with disagreeing replicates over ~9.7k rows, mean SD 0.99 pKd. Score each metric's per-sample noise estimate against it.

**Metric.** **Spearman ρ between the metric's noise score and the replicate SD, over replicate samples only.** 1.0 = perfect ordering of genuinely noisy measurements; 0.0 = unrelated.

**Why it matters.** Every other noise result in the suite uses injected Gaussian noise — a rigged ground truth. This is the only real-noise arm in the entire repo. **Historical warning to not repeat:** an earlier version reused `data['groups']` (one group per protein) as the replicate ID, which measured inter-*drug* variance and made every number meaningless (2026-05-25 audit).

**Caveat.** No antibody/DMS equivalent exists — this arm is molecular-only.

---

#### **D5 — Roughness indices as a metric-quality scoreboard**
*Status: NEW. Local. ~3 h.*

**Question.** Which distance makes a dataset look *least* rough — and does low roughness actually predict low out-of-sample model error?

**Procedure.** Compute **ROGI**, **MODI** and **SARI** under each candidate distance. All three are functions of a pairwise distance matrix, so the distance is the free parameter. Reference implementation: the `rogi` package (github.com/coleygroup/rogi), which ships SARI, MODI and RMODI alongside.

**Metrics.** **MODI** = activity-class-weighted fraction of variants whose single nearest neighbour shares their activity class. **1.0** = the metric perfectly co-locates same-class variants; **0.0** = nearest neighbours are anti-correlated with class; published threshold **0.65** separates modelable from non-modelable datasets. **ROGI** → validated by **Spearman ρ between ROGI and the actual cross-validated RMSE achieved by models on that dataset**; ρ = 1.0 means the roughness index fully predicts modelability, 0.0 means it is uninformative.

**Why this is the strongest non-noise framing.** MODI *is* a 1-NN agreement rate — it is literally the ENN intuition, already published in chemistry under a different name. It is the purest available benchmark of "does this metric put same-labelled objects together", it needs no model on the metric side, and it converts our similarity study into an established, citable task.

**Papers.** Aldeghi, Graff, Frey, Morrone, Pyzer-Knapp, Jordan & Coley, *JCIM* 62(19) (2022), arXiv:2207.09250, DOI 10.1021/acs.jcim.2c00903. Golbraikh, Muratov, Fourches & Tropsha, *JCIM* 54(1):1–4 (2014), DOI 10.1021/ci400572x, PMID 24251851. Peltason & Bajorath, *J Med Chem* 50(23):5571–5578 (2007), DOI 10.1021/jm0705713.

---

### GROUP E — Non-noise downstream tasks

---

#### **E1 — kNN read-across (the activity axis the current suite dropped)**
*Status: BUILT in `scaffold_experiment.py:173`, DROPPED from the formalised suite. Local. ~1 h to restore.*

**Procedure.** Scaffold/group GroupKFold 5-fold, distance-weighted k=5 kNN on the precomputed distance matrix (`simeval.eval_knn_readacross`, :143). Classification path for censored assays (`readacross_classification`, :141-165).

**Metric.** **Spearman ρ between predicted and true held-out activity** (and negative RMSE as the second entry). 1.0 = perfect rank recovery from neighbours alone; 0.0 = neighbours carry no activity information.

**Baselines.** `blosum_mean` at **0.48** and `dPhysChem` at **0.47** — the recorded bar. Whole-object embeddings.

**Why restore it.** `opig_scaffold_similarity.py` loads `labels` and `task_type` and never uses them; `SCAFFOLD_SIMILARITY_RESULTS.md:35` states "No model is trained anywhere in this experiment." The brief's own carried finding — *structural similarity and activity are decoupled* — is therefore unmeasurable by the formalised suite. That is the single biggest hole.

---

#### **E2 — Diversity selection scored by retrospective coverage**
*Status: NEW. Local. Fast (no model). ~3 h.*

**Question.** Pick n variants using only the distance. Does a perturbation-native distance find more of the measured functional classes than a whole-object one?

**Procedure.** MaxMin and sphere-exclusion/leader picking (RDKit `MaxMinPicker`, `LeaderPicker`) plus k-medoids, at fixed budgets n ∈ {96, 384} (real wet-lab plate granularity), under each candidate distance. Labels hidden from the picker.

**Metrics.**
- **Class coverage** = number of distinct measured fitness strata / functional clusters represented in the selected subset. Normalised: **1.0** = every stratum in the full library is represented; **0.0** = all n picks fall in one stratum. Compared against random selection at the same n over 100 draws.
- **Recall of the true top-k** = |selected ∩ true top-k| / k. 1.0 = the diversity pick alone recovered every top performer; 0.0 = none.
- **#Circles** = maximum number of disjoint radius-t balls centred on set members (the packing number). Higher = wider genuine coverage. Computable directly from any distance matrix, so it ports to DMS sequence distances unmodified. Report normalised (#Circles / n).

**Papers.** Snarey, Terrett, Willett & Wilton, *J Mol Graph Model* 15(6):372–385 (1997), DOI 10.1016/s1093-3263(98)00008-4 — the canonical protocol: pick a subset from a fully-annotated database using only the distance, then count how many known activity classes you captured. Model-free, so it isolates the distance perfectly. Xie, Xu, Ma & Mei, ICLR 2023, arXiv:2112.12542 — #Circles.

**Honest gap:** no paper treats the *choice of distance metric* as the independent variable in a protein/DMS diversity-selection benchmark. GeneDisco varies the acquisition function with representation fixed; Snarey is molecular-only and from 1997. **No application of #Circles to protein sequence space exists.**

---

#### **E3 — Retrospective batch active learning on a fully-measured DMS library**
*Status: NEW. Local (the oracle is a lookup — seconds per campaign). ~4 h.*

**Question.** In a simulated wet-lab campaign, does a distance-driven batch selector beat random and beat uncertainty-only?

**Procedure.** Hide all labels on a fully-measured DMS library. Seed with a small random batch. Run R = 5–20 rounds of (train surrogate → acquire batch of B ∈ {32, 96} → reveal true labels). Acquirers: **random**, **uncertainty-greedy**, **Coreset / k-means-on-embedding (distance-driven)**, **BADGE**. The distance-driven acquirers are where our metric enters. 3–5 seeds.

**Metrics.**
- **Hit ratio** = fraction of the true top-5 % of variants discovered, as a function of acquisition budget. **1.0** = every top performer found; **0.0** = none.
- **Top-k recovery curve** (MolPAL form): fraction of the true top-k found after acquiring x % of the library.
- **Budget-to-target** = number of measurements needed before the trained model's RMSE is within 10 % of the full-data RMSE. Lower is better; report as a fraction of library size.
- **Unique-cluster count among recovered hits** — separates "found many actives" from "found many *different* actives".

**Papers.** Mehrjou, Soleymani, Jesson, Notin, Gal, Bauer & Schwab, **GeneDisco**, arXiv:2110.11875 (ICLR 2022) — up to 40 cycles, batch 16–512, hit ratio = top-5 % discovered; **four of its nine acquisition functions are purely distance/embedding-driven (Coreset, k-means-on-data, k-means-on-embedding, BADGE)**, so swapping the underlying distance converts it directly into a similarity-metric benchmark. Yang, Lal, Bowden, Astudillo, Hameedi, Kaur, Hill, Yue & Arnold, **ALDE**, *Nat Commun* 16 (2025), DOI 10.1038/s41467-025-55987-8 — in-silico simulations on GB1 and TrpB 4-site landscapes (~160,000 variants nearly all measured) used to set batch size / rounds / acquisition before spending wet-lab money. Bailey et al., *eLife* 89679 (2023) — budget-to-within-10 %-of-min-RMSE metric, batch 30, 13 datasets. Fralish & Reker, *Beilstein J Org Chem* 20:2152–2162 (2024) — unique-scaffold-among-hits metric.

**Honest caveat to state in the write-up.** ALDE is a **negative result for a distance-centric framing**: its selected configuration is a DNN ensemble with one-hot encodings and Thompson sampling; GP and deep-kernel-learning variants were tested and underperformed. Distance is structurally central to batch construction but is not empirically the winning implementation there. Do not oversell.

---

#### **E4 — Landscape ruggedness features on antibody DMS**
*Status: NEW. Local. ~3 h.*

**Procedure.** Build the Hamming-distance-1 variant graph and compute the five named ruggedness features — **fraction of local optima (φ_lo)**, **roughness-to-slope ratio (r/s)**, **autocorrelation (ρ_a)**, **gamma statistic**, **neighbour-fitness correlation (NFC)** — plus the epistasis panel. Then relate each to model accuracy on the same library by Spearman ρ.

**Metric.** **NFC** = correlation of fitness between distance-1 neighbours. **1.0** = perfectly smooth landscape, neighbours have identical fitness; **0.0** = fitness is uncorrelated across a single mutation (a landscape no similarity method can help on). This is the direct quantitative answer to "is similarity even the right tool for this dataset" — a go/no-go gate, and it is cheap.

**Paper.** **GraphFLA**, arXiv:2510.24826, NeurIPS 2025 — applied to 5,300+ landscapes from ProteinGym, RNAGym, CIS-BP.

**Honest gap:** I found **no paper measuring landscape smoothness specifically on antibody DMS data**. The ruggedness literature is enzyme- and viral-protein-heavy. Our antibody roster is the contribution.

---

#### **E5 — Cliff-restricted regression error**
*Status: NEW for our reps; the benchmark exists. **SERVER** for the full battery.*

**Procedure.** Train the downstream model, then report error separately on cliff-participating variants.

**Metric.** **RMSE_cliff** = RMSE computed only over test members belonging to ≥1 cliff pair, reported **alongside** overall RMSE. Published range: 0.62–1.60 log units across 30 targets. Lower is better; the informative quantity is the **gap RMSE_cliff − RMSE_all** — 0 means the model handles cliffs as well as anything else, large positive means cliffs are where it fails.

**The experiment.** Cliff membership is defined by the similarity rule (≥90 % on any of ECFP-Tanimoto / scaffold-ECFP-Tanimoto / SMILES-Levenshtein, **and** ≥10-fold potency difference). **Swap the similarity, re-measure RMSE_cliff.** That directly tests whether a perturbation-native metric selects a more meaningful cliff set.

**Paper.** van Tilborg, Alenicheva & Grisoni, *JCIM* 62(23):5938–5951 (2022), DOI 10.1021/acs.jcim.2c01073, PMID 36456532 — 30 targets, 48,707 molecules, 7–52 % cliff compounds; **descriptor-based classical ML outperformed deep learning on cliffs**. Note two published Corrections (PMC10091401, PMC11683855) — cite carefully.

**Threshold honesty.** There is **no consensus cliff threshold**: 10-fold (MoleculeACE), 100-fold (MMP-Cliffs, Hu/Bajorath *JCIM* 2012, PMID 22489665), 1 log unit (arXiv:2605.30831) are all in current use. Report sensitivity across all three; never write "the standard threshold".

---

### GROUP F — The molecular pillar

---

#### **F1 — Molecular perturbation representation from real MMP transformations**
*Status: NEW, and the single biggest missing piece. Local. ~5 h.*

**The asset the hand-off did not know about.** `src/kirby/datasets/chembl_qsar.py` already downloads real MMP data with columns `smiles_1, smiles_2, transformation, constant_part` (:241). Verified real content: transformation `[*:1]c1cccc(F)c1>>[*:1]c1ccccc1C`, constant part `[*:1]/C(CCN1CCN(c2ccccn2)CC1)=N/OC`. **35,484 such pairs for the d2 target are already cached** at `/Volumes/seagate/kirby/chembl_qsar/chembl_dopamine_d2/MMP_data_clean.csv` (6.9 MB), alongside 6,333 compounds.

**The gap.** **No code anywhere consumes `transformation` or `constant_part`.** A repo-wide grep returns only that one comment line. `load_chembl_qsar` reads `smiles_1`/`smiles_2` (:252-253) and discards the transformation entirely, returning bare `(i, j, delta)` tuples (:270). That discarded column *is* the molecular scaffold-diff representation.

**Representations to build (the molecular analogue of dPhysChem):**
1. **dPhysChem-mol** — Δ of RDKit physicochemical descriptors between the two molecules, z-scored (mirrors the antibody winner exactly).
2. **TransformFP-XOR/AND** — split each molecule's fingerprint into core (constant part) and substituent; substituent features by **XOR** (unique to one side) and **AND** (shared). This is the Asawa MMP-kernel construction and is the closest published thing to "encode the transformation, not the objects".
3. **TransformFP-count** — the same, with count vectors and Soergel (the A1 hypothesis, molecular side).
4. **ΔECFP** — plain fingerprint subtraction (the Siamese-difference-vector baseline).

**Metrics.** A2's p0 / CV / RC / SN_k on MMP pairs; A1's `struct_fidelity` against **MCS-Tanimoto** as the independent grader (`structural_yardstick.mcs_distance`, already built and cached).

**Critical data caveat.** The three existing MCS caches sample pairs **uniformly at random** (`mcs_cache.py:45-48`), giving mean MCS *distance* 0.816–0.859 — i.e. mean similarity ~0.15. They are dominated by dissimilar pairs and are **badly matched to a matched-pair study**. A similarity-stratified re-draw is required. Budget for it (~0.09 s/pair MCS).

**Also required.** Pair-level group CV: no molecule may appear in both a train pair and a test pair. Nothing in the repo does this.

**Papers.** Asawa et al., *Molecules* 26(16):4916 (2021), DOI 10.3390/molecules26164916 — MMP kernel = product of Tanimoto kernels over core and substituent; XOR/AND substituent construction. Heikamp, Hu, Yan & Bajorath, *JCIM* 52(9):2354–2365 (2012), DOI 10.1021/ci300306a — kernels on molecule *pairs*. Zhang, Menke, He, Nittinger, Tyrchan, Koch & Zhao, *J Cheminform* 15 (2023), DOI 10.1186/s13321-023-00744-6 — ECFP4 difference vectors, and the key finding that **pairing on high similarity beats exhaustive pairing on all three datasets while dropping complexity from O(n²) to O(n)**. Fralish, Chen, Skaluba & Reker, **DeepDelta**, *J Cheminform* 15:101 (2023), DOI 10.1186/s13321-023-00769-x — pair-based property-difference prediction, beats ChemProp and RF-on-fingerprints on 70 % of 10 ADMET benchmarks.

**Boundary to state clearly.** Ramakrishnan, Dral, Rupp & von Lilienfeld, *JCTC* 11(5):2087–2096 (2015) — Δ-machine-learning — learns a difference between two **levels of theory for the same molecule**, not between two objects. It is frequently mis-cited as pair-based learning. Cite it to define the boundary, not as support.

**The white space, stated plainly.** Every delta representation in the literature — DeepDelta, Siamese-difference, MMP kernel, pLM-SAV, Δ-ML — **terminates in a supervised regressor or classifier**. Not one is evaluated as a standalone *similarity*. The MMP kernel is the partial exception (a kernel is formally a similarity) but it is consumed only by an SVM. **Nobody has benchmarked a distance between two perturbations.** That is the contribution.

---

#### **F2 — MoleculeACE 30-target battery**
*Status: staged code, NEVER RUN. **SERVER.***

**Current state, verified:** `MoleculeACE` is not installed (`ModuleNotFoundError`). The output directory `similarity_metrics/sim_data/` is **empty** — no `sim_CHEMBL*.npz`, no `moleculeace_rows.json`, no `moleculeace_summary.json`. `build_moleculeace.py` (5,397 B, dated 11 Jul, predates the scaffold thread) and `run_battery_moleculeace.py` are unexecuted. **The "30 targets" figure is a docstring claim (`build_moleculeace.py:3-4`), never an observed count**, and `:110-111` drops any target with <100 molecules, so the realised count is ≤30.

**Procedure (already written).** GroupKFold(5) on Bemis-Murcko scaffolds, all metrics on the held-out fold only, triplet encoder trained on train and applied to held-out. Per-target rank → mean rank → **paired Wilcoxon vs `ecfp4:tanimoto`**, requiring ≥6 paired targets.

**Metric.** Mean rank across the 9-eval battery (`knn_rho`, `knn_neg_rmse`, `modi`, `patterson`, `cliff_auroc`, `violation_neg`, `bedroc`, `noise_detect_auroc`, `noise_recovery`) over 47 metrics. Lower rank = better. Plus the Wilcoxon p vs the ECFP4:Tanimoto reference.

**Also note:** `cliff_mol`, MoleculeACE's own per-molecule cliff label, is named in the docstring as a verified column but **`build_moleculeace.py` never reads it** (only `smiles` and the y column at :100), so it is not carried into the npz. Add it — it is a free, package-provided ground truth.

**Why this matters for credibility.** The headline count claim (mean ranks 14.4 / 14.5 / 14.8 / 15.0 vs ecfp4:tanimoto 20.2, `SIMILARITY_METRICS_RESULTS_2026_07_11.md:99-107`) rests on `run_grid_robust.py`, `run_noise_sweep.py` and `verify_self.py` — **none of which exist in the repo** (`git ls-files` lists 23 files, none of those three). No raw per-metric battery output is on disk. That number is currently unauditable and unreproducible. F2 regenerates it or retracts it.

---

## 4. Compute placement

| Local (laptop, no ML) | Server (ARC/htc) |
|---|---|
| A1, A2, A3 — pairwise distances + Spearman only | **B1** — full model × width sweep, 9 models × ~18 reps × 5 datasets |
| B1a (read existing `universal_survey` JSONs), B2 | **C1** — calls `find_best_hybrid`, 6 arms × 5 datasets |
| D1, D2, D3, D5 | **E5** — trains downstream models across the cliff battery |
| E1, E2, E4 | **F2** — 30 targets × 47 metrics × 9 evals × 5 folds |
| F1 (rep construction + geometry; MCS re-draw is the only slow part) | D4 if BindingDB reps need recomputing |

Per project rules: no heavy local compute, no `find_best_hybrid` locally — import/syntax checks only. Server ops are the user's; Claude never ssh/scp.

---

## 5. Ranking by value-per-effort

| Rank | Exp | Effort | Value | Why |
|---|---|---|---|---|
| 1 | **A2** small-perturbation resolution | ~3 h, local | ★★★ | No labels, no model, no grader → **no circularity surface**. Immune to the failure that killed `position_gated`. Reproducible against published numbers (100 % fingerprint stereo-degeneracy) as a harness sanity check. Directly discharges the "read arXiv:2605.30831 and differentiate" debt. |
| 2 | **F1** molecular perturbation rep | ~5 h, local | ★★★ | 35,484 real MMP transformations already on disk, unused. Removes the "no molecular domain" objection outright. The white space is verified: nobody benchmarks a distance between transformations. |
| 3 | **D5 + E1** MODI/ROGI + read-across | ~4 h, local | ★★★ | Restores the activity axis the formalised suite silently dropped, and reframes the whole study as an established citable task (MODI is a published 1-NN agreement rate with a validated 0.65 threshold). Cheap, and it makes every other result interpretable. |
| 4 | B1a existing-data crossover write-up | ~1 h, local | ★★★ | The ridge-vs-lightgbm × width crossover is already sitting in three JSONs and has never been analysed. Free result. Do it before spending server time on B1. |
| 5 | A1 counts vs binary | ~2 h, local | ★★☆ | Directly answers the user's point 5, with a real mechanism from O'Boyle & Sayle. Held back only because the counts/Soergel confound makes it fiddly. |
| 6 | C1 augmentation routing | ~3 h, server | ★★☆ | Zero new machinery; `residual_correlation` is the one complementarity-rewarding path in the codebase and has never been measured. Server-gated. |
| 7 | E2 diversity selection | ~3 h, local | ★★☆ | Model-free, clean isolation of the distance, and a verified literature gap. |
| 8 | D1/D2/D3 noise trio | ~3 h, local | ★★☆ | Mostly built. D3 is the trustworthy ranking; D1 alone is too unstable to headline. |
| 9 | E4 ruggedness / NFC | ~3 h, local | ★★☆ | Cheap go/no-go gate per dataset; genuine antibody gap. |
| 10 | A3 position selection | ~2 h, local | ★☆☆ | Incremental; needs random-subset controls or it self-inflates. |
| 11 | B1 full sweep | server, heavy | ★★☆ | High value but expensive; gate on B1a. |
| 12 | E3 active learning | ~4 h, local | ★★☆ | Great framing, but ALDE is a partial negative for distance-centric methods — manage expectations. |
| 13 | E5 RMSE_cliff | server | ★★☆ | Strong, but downstream of F1/F2. |
| 14 | D4 real noise | data-gated | ★★☆ | Only real-noise arm; molecular-only. |
| 15 | F2 MoleculeACE | server | ★★★ value, ★ tractability | Needs a package install and a full battery run; but it is the only way to make the headline count claim auditable. |

### Build these THREE first: **A2, F1, D5+E1.**

- **A2** because it is the one experiment in the suite that *cannot* be circular, and the last session's single most valuable output was a circularity retraction. Start where the ground is solid.
- **F1** because "no molecular domain" is the loudest hole (`SCAFFOLD_SIMILARITY_RESULTS.md:230-231` admits it), the data is already cached, and the discarded `transformation` column makes it far cheaper than anyone assumed.
- **D5+E1** because without an activity axis the suite cannot speak to the project's thesis at all, and MODI/ROGI convert the work from a bespoke correlation study into a task the field already scores.

---

## 6. Non-negotiables for every experiment

1. **Bootstrap 95 % CI on every reported number.** The current suite reports 0.757 vs 0.757 vs 0.762 as findings with no uncertainty. That cannot happen again.
2. **Multi-seed.** Minimum 3 seeds; noise arms sweep σ × rate × seed because rankings invert between noise levels.
3. **One held-out grader used identically across all representations**, in addition to the per-representation anti-circularity grader. Without it, cross-representation comparison is structurally impossible.
4. **Both baselines built, not declared.** `aac` and `blosum_mean` must produce numbers.
5. **Never count koenig twice.** One library, one row.
6. **Never place the 14-scale graphinity number in the same column as a 5-scale antibody number.**
7. **Tests.** At minimum a fixture asserting the C3 control still returns Spearman 1.000000 for `position_gated` vs Grantham, and a regression fixture on the results JSON.
8. **Every result lands in a roadmap doc the same day.** No scratchpad-only results.
9. **Stop and discuss scope with the user before any large build** — the process rule the last session broke.