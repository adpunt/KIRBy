# ROGII Wellbore Geology Prediction — Roadmap

**Competition:** Kaggle "ROGII — Wellbore Geology Prediction"
**Target:** TVT (True Vertical Thickness) regression, RMSE metric, ~200 test wells, hidden test set.
**Constraints:** ≤9h Kaggle GPU notebook, no internet at submission time, pretrained weights allowed if shipped as a Kaggle dataset.
**Timeline:** Start 2026-05-08; entry/team-merger 2026-07-29; final submission 2026-08-05. ~3 months.
**Approach:** Mirror the ANNITIA pattern — enumerate every reasonable feature view × every reasonable model, generate honest OOFs, proliferate fusion variants, rank-average + weight, route per-row on anomaly/disagreement signals. Anchor to the published NORCE prior-competition winner. Layer in the alignment-aware features and CV discipline that the public Kaggle baselines miss.

---

## 0. Why ANNITIA's playbook fits this competition

| ANNITIA property                         | ROGII analogue                                         |
|------------------------------------------|--------------------------------------------------------|
| Small N (1253 patients, 47 hep events)   | Small N (~200 wells, 1 TVT trace each)                 |
| Multiple plausible feature views         | Many plausible well-log views (windowed GR, alignment, trajectory, wavelet…) |
| Hand-interpreted labels = noisy          | TVT manually interpreted by geologists = noisy         |
| Score = rank C-index (rank-stable)       | Score = RMSE (NOT rank-stable — calibration matters)  |
| 30% qualitative methodology              | None — pure RMSE leaderboard                           |
| Survival semantics                       | Sequence regression with masked eval zone              |
| Foundation tabular models partly applicable | Foundation time-series models partly applicable     |
| Fusion-variant proliferation paid off (0.91 → 0.93) | Same expected pattern; calibrate to RMSE not rank |

**Three differences that change tactics:**

1. **RMSE punishes calibration**, not just rank. Rank-mean fusion stays useful but must be paired with a calibration step (isotonic on OOF, or anchor-and-smooth) before submission. Pure-rank tricks that won ANNITIA points on C-index will *not* transfer 1:1.
2. **Sequence structure is real** — adjacent depth samples within a well are highly correlated. CV must explicitly model the gap mask shape, or you fall into the FORCE 2020 trap.
3. **Public-test inputs are visible**, only TVT is hidden in the eval zone. That means a *lot* of free EDA before any model runs (gap structure, GR distribution shift, spatial layout, typewell formation prevalence). This was not available in ANNITIA.

**The ANNITIA failure mode that we must NOT repeat:** the 0.7181 catastrophe came from search-on-search overfit (Coxnet `stack_meta` fit on full-row OOFs and evaluated in-sample). Our fix is the same: every fusion variant is evaluated on held-out OOFs, never on its own training-fold predictions. Nested CV for any learned-weight aggregator.

---

## 1. Compute & deployment plan

| Stage                   | Where                | What runs there                                    |
|-------------------------|----------------------|----------------------------------------------------|
| Until PC built          | MacBook              | EDA, CV scaffold, gap-mask sampler, KIRBy fast-tier hybrid on engineered features, NORCE reproduction on small subset (CPU, ~hours), submission packaging |
| PC online (deadline-blocking) | Local PC      | NN training (S/A-tier), foundation-model PEFT, full ensemble training, hyperparameter ablations |
| Final submission        | Kaggle GPU notebook  | Inference only — load pretrained weights from Kaggle Dataset, run forward pass per test well, write `submission.csv` |
| **Not used**            | Oxford ARC           | User direction. No ssh/scp, no SLURM. Treat as a hard constraint. |

**Implication for the roadmap:** the "training" phase happens locally; the "submission" phase happens on Kaggle. Every artefact must be packagable into a Kaggle Dataset (model weights, normalisation params, fold blueprints, fusion weights). Any code that depends on internet at inference time is disqualified.

**Kaggle Dataset bundle (final form, ~few GB):**
- `weights/{m1_norce, m2_unet_xattn, a1_patchtst, a4_moment_peft, b3_kirby_blueprints}.pt`
- `weights/fusion_meta.json` (rank weights for F13–F17, per-bucket variant assignments)
- `weights/normalisation_anchors.json` (pooled-typewell P5/P95, EGFDL/BUDA two-point reference)
- `weights/cv_fold_index.json` (well-id → fold for reproducibility)
- `weights/seed_list.json`

---

## 2. Repo & directory layout (mirror ANNITIA)

**Recommendation: separate repo `kirby-rogii`**, sibling to `kirby-finance` and `kirby-av`. KIRBy itself is imported as a library for the hybrid-search tier (just like `kirby-av` consumes it for AV reps). This isolates the Kaggle-specific code from the research pipeline and matches the precedent the user already set with the other domain spinoffs.

If you'd rather start under `KIRBy/tests/experiments/rogii/` as a sandbox until it stabilises, that's fine — it's how ANNITIA started. Migration to a standalone repo is a `git filter-repo` away.

```
kirby-rogii/
├── README.md                          # problem statement + 6-tier model menu + variant catalogue
├── SUBMISSION_PLAN.md                 # per-submission hypothesis + predicted_lb (mirror SUBMISSION_PLAN_2026_05_08.md)
├── GENERALIZATION_IDEAS.md            # brainstorm sink (mirror annitia)
├── USER_SUGGESTIONS.md                # running TODO from session conversations
├── EXPERIMENTS.csv                    # append-only log: run_id, commit, cv_mean, lb, hypothesis, outcome, next_step
├── PROBES.md                          # constant-probe results, adversarial-validation snapshots, dated test-distribution inferences
│
├── src/kirby_rogii/
│   ├── datasets/rogii.py              # loader: per-well dict (horizontal_well, typewell), CV folds, gap-mask sampler
│   └── eda/                           # one-shot EDA notebooks/scripts (run on day zero, never re-run)
│
├── tests/experiments/rogii/
│   ├── _cv.py                         # WellGroupKFold, StratifiedGroupKFold-by-formation, SpatialKMeansKFold, GapMaskSampler
│   ├── _oof.py                        # init_oof_buffer / fill_oof_fold / save_oof — copy from annitia/_oof.py verbatim, swap shape (n_wells × max_MD) NaN-padded
│   ├── _schema.py                     # ConditionRecord dataclass (config_id = SHA over views+model+postproc), copy from annitia/_schema.py
│   ├── _calibration.py                # NEW vs annitia: RMSE needs isotonic + anchor-and-smooth, not rank
│   │
│   ├── views.py                       # 16 deterministic feature-view builders (§4)
│   ├── alignment.py                   # DTW / matrix-profile / NCC features against typewell (§4 v7-v9)
│   ├── normalisation.py               # P5–P95 percentile match + two-point shale/carbonate baseline + histogram match
│   ├── postprocess.py                 # anchor-and-smooth (Savitzky–Golay around bracketing TVT_input)
│   ├── pseudolabel.py                 # Babakhin-style pseudo-labelling, retrain-from-scratch
│   │
│   ├── models/                        # one file per backbone (§5)
│   │   ├── m1_norce_convattn.py       # NORCE 1st-place reproduction (Conv1D k=7 → MHA → GAP → MLP, weighted-MSE matching the metric, 7-seed ensemble)
│   │   ├── m2_unet_xattn.py           # 1D U-Net w/ cross-attention to typewell, masked Huber loss
│   │   ├── m3_softdtw_aux.py          # soft-DTW divergence as aux loss (wraps M1 or M2)
│   │   ├── a1_patchtst.py             # PatchTST + cross-attn head, masked-patch SSL pretrain on pooled GR
│   │   ├── a2_tcn.py                  # TCN baseline ensemble member
│   │   ├── a3_encdec_transformer.py   # typewell-as-source, lateral-as-target
│   │   ├── a4_moment.py               # MOMENT foundation model w/ imputation head, PEFT-tune
│   │   ├── a5_siamese.py              # Siamese typewell-vs-lateral window similarity, used as feature extractor
│   │   ├── b1_saits.py                # SAITS partial-label imputation (sanity baseline)
│   │   ├── b2_tabpfn.py               # TabPFN v2.5 over windowed engineered features
│   │   └── b3_kirby_hybrid.py         # KIRBy find_best_hybrid wrapper — see §7
│   │
│   ├── search.py                      # per-(view-subset × model × postproc) enumeration, 5-fold CV w/ gap mask, OOF accumulation
│   ├── run.py                         # orchestrator (CLI): --views, --models, --speed fast|full, writes per-condition OOF + final model
│   │
│   ├── refuse_reselection.py               # nested-CV honest re-fusion (mirror annitia)
│   ├── refuse_variants.py             # 12 simple fusion variants (§6 F1-F12)
│   ├── refuse_meta_aware.py           # 5 per-row routing meta-variants (§6 F13-F17)
│   ├── refuse_diverse.py              # greedy diverse-by-OOF-improvement
│   │
│   ├── sanity_audit.py                # OOF/CV consistency, dup detection, gap-mask leakage check
│   ├── analyse_residuals.py           # per-MD residual heatmap by formation, anomaly-vs-residual correlation, disagreement-vs-residual correlation, fault-spike detection
│   ├── adversarial_validation.py      # train-vs-test classifier, AUC + feature importance, run after every major view-set change
│   ├── oof_diagnostics.py             # OOF calibration plots, RMSE drift across folds
│   ├── leaderboard_probe.py           # generates the 3-constant moment-matching submissions + tracks LB
│   │
│   ├── submit.py                      # writes submission.csv from saved fusion + per-well postprocess
│   ├── build_journal.py               # paired .ipynb + .md per submission (good hygiene even though Kaggle doesn't require it; helps the experiments log)
│   ├── kaggle_inference_notebook.py   # the actual notebook content; deterministic, no internet, loads from Kaggle Dataset
│   └── package_kaggle_dataset.py      # bundles weights/ + fusion_meta.json + normalisation_anchors.json into the format Kaggle expects
│
└── data/                              # local only, .gitignore'd
    ├── train/                         # raw competition data
    ├── test/                          # raw competition data (public stub of ~few wells)
    ├── cache/                         # featurised views per well, per-fold OOF buffers
    └── results/                       # per-condition outputs, fusion CSVs, journals
```

This is verbatim the ANNITIA layout with three additions:
- `_cv.py` is much heavier here — gap-mask sampling is the single most important piece of CV and deserves its own module with unit tests.
- `_calibration.py` and `postprocess.py` are new — RMSE punishes miscalibration in a way C-index does not.
- `leaderboard_probe.py` and `PROBES.md` formalise the moment-matching probing protocol from the Claude research §8.

---

## 3. CV scheme — design this BEFORE any model

This is the highest-leverage decision in the whole roadmap. Get this wrong and every subsequent ablation reads as noise. The Claude research §4 + the FORCE 2020 historical record + your standing "no random splits anywhere" rule converge on the same scheme.

**The composite splitter:**

1. **Outer fold:** `StratifiedGroupKFold(n_splits=5, groups=well_id, stratify=dominant_formation_in_lateral)`. Wells are the leak boundary; dominant formation prevents folds that miss a stratum entirely.
2. **Inner gap-mask sampler:** for each train well, sample 5 gap masks (different seeds) whose length distribution and position-relative-to-MD-range match the empirical distribution of NaN positions in `TVT_input` across **test wells** (visible at day zero — the values are hidden but the positions are not). OOF metrics are computed only over masked indices.
3. **Spatial cluster guardrail (sanity, not primary):** also keep a `KMeans(n_clusters=5)` on `(X, Y)` wellhead → cluster-fold split, used as a held-out check that the primary CV's well-level groups aren't accidentally leaking via field membership. Run this once after the primary CV stabilises; if its score disagrees by >1 RMSE point, the well-level CV is being optimistic about cross-field generalisation.
4. **Adversarial validation:** train LightGBM `well ∈ train` vs `well ∈ test` on every engineered view. Re-run after every major view-set change. AUC > 0.6 → distribution shift; identify and normalise/drop the offending features.

**Acceptance criteria before training any non-baseline model:**

- The gap-mask sampler regenerates masks deterministically given a seed.
- Sampled gap-mask length distribution is within 5% KS-statistic of the test-well NaN-position distribution.
- The composite splitter passes a leakage unit test: no `(well_id, MD)` index appears in both a fold's train and val sets.

**Tracker (mirror the FORCE 2020 lesson):** maintain `EXPERIMENTS.csv`. After 10–15 entries, compute `pearsonr(cv_mean, lb_score)`. Target r > 0.85. Below 0.7 → **stop submitting and fix CV**. This is non-negotiable. The FORCE 2020 leaderboard had #1 → #11 turnover; the ROGII prior comp was won by the team with the cleanest CV + the metric-matching loss.

---

## 4. Feature view catalogue (`views.py`)

Sixteen views, each a deterministic function `build(well_dict, **opts) -> (X_matrix, feature_names)`. No fitting, no leakage, no global state. Every view must work on a single test well in isolation at inference time.

| ID | Name | What it returns | Approx dim | Notes |
|----|------|-----------------|-----------|-------|
| v1 | `raw_logs` | GR + (X,Y,Z) + MD per row | 5 | Floor baseline |
| v2 | `gr_window` | rolling stats over ±{5,10,25,50,100,250} samples × {mean, std, slope, range, MAD, skew, kurt, savgol-d1, savgol-d2} | ~54 | Bestagini SEG-2016 augmentation, used by every FORCE 2020 finisher |
| v3 | `catch22_window` | catch22 22-d fingerprint per ±{50,100,250} window | ~66 | Default minimal time-series feature set; safer than tsfresh |
| v4 | `wavelet` | pywt.cwt at 5 dyadic Morlet scales; per-MD coefficient + envelope per scale | ~10 | |
| v5 | `hilbert` | analytic-signal envelope + instantaneous phase + frequency, on GR | 3 | |
| v6 | `fft_band` | log-band power in 5 frequency bands over rolling ±100 window | 5 | |
| v7 | `align_dtw` | subsequence-DTW best-match offset + distance vs typewell, m∈{50, 150, 300} windows, stretch grid {0.7…1.3} | ~12 | The single highest-EV alignment feature group |
| v8 | `align_xcorr` | sliding-NCC argmax + peak + entropy + corr-at-prev-argmax, on GR derivatives | 4 | |
| v9 | `align_matrixprofile` | stumpy AB-join nearest-neighbour typewell index at m∈{100, 300} | 2 | "Index ≈ TVT estimate" — cleanest single feature |
| v10 | `trajectory` | inclination, az(sin/cos), DLS, build/turn rates, TVD by minimum curvature, lateral-traveled, ratio lateral/TVD, distance from KOP/heel/toe, Frenet curvature/torsion | ~14 | All from Savitzky-Golay-smoothed (X,Y,Z); minimum curvature per Sawaryn–Thorogood SPE-84246-PA |
| v11 | `tvt_input_bracket` | distance to nearest observed `TVT_input`, bracketing observed values + their MDs, linear and cubic-spline interpolation across the gap | 6 | The bracketing values *bound* the answer |
| v12 | `typewell_geology` | ordinal stratigraphic encoding (ANCC<ASTNU<ASTNL<EGFDU<EGFDL<BUDA), per-formation TVT priors (mean, range from train) — used both as feature and as postprocess clip | ~12 | |
| v13 | `gr_normalised` | P5–P95 percentile match to pooled typewell + two-point baseline (EGFDL/BUDA anchors) + histogram match — three normalisation flavours, model can pick the best | 3 | Per-well, fit on full trace at inference |
| v14 | `tsfresh_efficient` | tsfresh `EfficientFCParameters` + `select_features` (Benjamini–Yekutieli FDR) | ~50 (post-select) | Costly; add in week 4+ if v2/v3 are saturating |
| v15 | `anomaly_score` | IsolationForest(n=300) on (v1+v10) joint train+test fit, returned as a single scalar per row | 1 | Used for routing in fusion (§6) |
| v16 | `disagreement` | rank-std across base-model OOFs at each (well, MD) | 1 | Computed AFTER base models are trained; used for routing |

**Forbidden as input features (training-only, leakage trap):** `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` surface depths. These get used **only** as auxiliary supervision (§5).

**Augmentations (small optional feature sets, ANNITIA-style):** none initially. If a future trick adds <10 features and looks helpful in CV, plumb it as an `--augmentations` flag and treat exactly like ANNITIA's `developability` aug pattern.

---

## 5. Model bestiary

Five tiers. Build top-to-bottom. Each model file exports `fit(X, y, mask, groups) -> Model` and `Model.predict(X, mask) -> y_hat`.

### Tier S — build first (week 1–2)

**M1 — NORCE Conv-Attention reproduction** (`m1_norce_convattn.py`).
Source: https://github.com/geosteering-no/Geology-Forecast-Challenge-Gold (1st place on the prior ROGII comp).
Architecture: `Conv1D(64, k=7) → MultiHeadAttention(4 heads, dim=64) → LayerNorm → GAP → Dense(512→64→out)`. Additive position encoding (`col_indices + inputs`, NOT sinusoidal — their ablation confirmed this). Standard-scale both X and y (same space). Drop categorical `geology_id` (it hurt in their ablation).
Loss: weighted MSE matching the competition metric (RMSE is just √MSE — keep MSE in training, no surprise terms). 7-fold seed ensemble.
**This is the floor, not the ceiling.** Reproduce their CV score on training data first, then iterate.
Compute: ~20 min on a P100, ~hours on CPU.

**M2 — 1D U-Net with cross-attention to typewell** (`m2_unet_xattn.py`).
Encoder downsamples (horizontal GR, partially-observed TVT_input, observation mask) → bottleneck cross-attends to a separately-encoded typewell GR (with `Geology` ordinal as an extra channel) → decoder upsamples back to per-MD TVT prediction. Masked Huber loss over eval-zone indices, plus auxiliary masked-Huber over observed-`TVT_input` indices (weight 0.1–0.3). ~1–10M params.
Reference: Ronneberger 2015 (arXiv:1505.04597), Perslev "U-Time" 2019 (arXiv:1910.11162).
**This is the architecture I expect to beat M1**, because cross-attention encodes the typewell-as-reference prior explicitly.

**M3 — Soft-DTW divergence as auxiliary loss** (`m3_softdtw_aux.py`).
Wraps M1 or M2; adds `soft_dtw_divergence(predicted_TVT, typewell_TVT_via_alignment)` weighted 0.1–0.3 alongside masked Huber. Cuturi & Blondel arXiv:1703.01541; divergence variant arXiv:2010.08354. CUDA impl: `Maghoumi/pytorch-softdtw-cuda`. **Mandatory ablation, not optional.**

### Tier A — strong contenders (week 3–5)

**A1 — PatchTST + cross-attention head** (`a1_patchtst.py`). Patch length 16, stride 8. Pretrain self-supervised with masked-patch reconstruction on pooled (typewell + horizontal) GR; fine-tune supervised with siamese cross-attention bridge to typewell. arXiv:2211.14730.

**A2 — TCN** (`a2_tcn.py`). Dilated causal convs, 8 layers, ~1k receptive field. Trains in minutes; **excellent ensemble member**, won't itself top the leaderboard.

**A3 — Encoder-decoder Transformer** (`a3_encdec_transformer.py`). Typewell as source language, horizontal as target. Conceptually cleanest formulation; one fold of testing to validate.

**A4 — MOMENT foundation model** (`a4_moment.py`). HF `AutonLab/MOMENT-1-base`, native imputation head. Linear-probe first, then PEFT tune. Concatenate typewell as extra series via the multivariate hack.

**A5 — Siamese typewell↔lateral window encoder** (`a5_siamese.py`). Pretrain self-supervised contrastive on log windows (Marusov & Zaytsev arXiv:2209.14750). Embeddings used as additional features in the KIRBy hybrid (B3) and as inputs to M1/M2.

### Tier B — niche / ensemble members (week 5–7)

**B1 — SAITS** (`b1_saits.py`). Diagonally-masked self-attention for partial-label imputation. Sanity-baseline; tiny (<5M params).

**B2 — TabPFN v2.5** (`b2_tabpfn.py`). Treat each (well, MD) row as exchangeable, with v2/v3/v7/v8/v9/v10/v11/v13 as columns. Strong ensemble member; loses sequence structure unless you featurise it.

**B3 — KIRBy `find_best_hybrid` over engineered views** (`b3_kirby_hybrid.py`). See §7.

### Auxiliary heads (multi-task supervision, training-only)

Every NN backbone gets:
1. **TVT_input reconstruction head** (weight 0.1–0.3), on observed indices.
2. **Surface-distance heads** — predict signed MD distance to nearest ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA crossing. Training-only; ignored at inference. The cleanest free regulariser available.
3. **Formation classification head** — per-MD softmax over the 6 formations. Training-only. Useful for the per-formation TVT-prior postprocess clip.
4. **Heteroscedastic NLL head** (Kendall & Gal arXiv:1703.04977) — predict `(μ, σ)` and train with NLL alongside the main Huber. Auto-downweights noisy samples. Use σ for the disagreement-style routing in §6.

### Tier C — DO NOT BUILD

Per Claude research §2 + your priors:
- Chronos / TimesFM / Lag-Llama / TimeGPT (univariate causal forecasters, wrong inductive bias).
- Image CNNs on the cross-section PNGs (deterministic StarSteer renderings, no information beyond inputs).
- DISTINGUISH GAN-geomodel / EnKF / RL geosteering (wrong problem — sequential decision-making, not offline regression).
- Informer / Autoformer / FEDformer / iTransformer (forecasting-flavoured).
- BRITS (outclassed by SAITS).
- Mamba / S4 (install pain on Kaggle no-internet, no clear win at L≤5k).

---

## 6. Fusion variant catalogue (`refuse_variants.py` + `refuse_meta_aware.py`)

The ANNITIA empirical lesson: **more variants → better generalisation**, monotonically (0.91 → 0.93). Each variant is a different way of aggregating the same OOF risk matrix; they make different mistakes; rank-averaging across them is immune to scale.

For RMSE (vs C-index) we use **rank then de-rank** for variants F1–F4 (rank-blend on OOF rank percentiles, then map back to value via OOF empirical CDF) — this gets us ANNITIA's rank-stability while preserving RMSE calibration.

### Simple variants (F1–F12)

| ID | Name | Aggregation rule |
|----|------|------------------|
| F1 | `rank_mean` | per-(well, MD) rank across base preds, average, de-rank |
| F2 | `rank_median` | rank median, de-rank |
| F3 | `rank_trimmed` | trim top/bottom 20% of ranks, average, de-rank |
| F4 | `cv_weighted_mean` | per-base weight = `1 / OOF_RMSE`; weighted mean (no rank) |
| F5 | `residual_greedy` | iteratively pick the model with lowest mean residual on the current top-quartile-residual rows |
| F6 | `anchor_smooth_mean` | apply anchor-and-smooth (§postprocess) to each base, then average |
| F7 | `anomaly_route` | bucket rows by IsolationForest score quartile; per-bucket pick best variant |
| F8 | `disagreement_route` | bucket rows by base-pred rank-std; per-bucket pick best variant |
| F9 | `cluster_route` | KMeans(k=5) on (v2+v10) features; per-cluster pick best variant |
| F10 | `specialist_override` | per-formation allow different best model (formation predicted from typewell side) |
| F11 | `soft_blend_disagreement` | continuous sigmoid mix of two specialists, weight = sigmoid(disagreement − median) |
| F12 | `anomaly_blend` | continuous mix using anomaly score |

### Per-row routing meta-variants (F13–F17)

| ID | Name | Aggregation rule |
|----|------|------------------|
| F13 | `weighted_anomaly_per_row` | softmax-weighted blend across all 12 simple variants, weights = softmax(−per-bucket OOF RMSE) |
| F14 | `cluster_router_meta` | KMeans on (anomaly + disagreement + formation prior); per-cluster best simple variant |
| F15 | `disagreement_router_meta` | bucket by ensemble disagreement; per-bucket best simple variant |
| F16 | `anomaly_router_meta` | bucket by anomaly; per-bucket best simple variant |
| F17 | `nested_cv_stacking` | Ridge-style learned stacker on (12 simple variants × OOF rank) + (anomaly, disagreement) features. **Nested 5-fold CV mandatory** — this is the variant that overfits if you blink. Skip if `EXPERIMENTS.csv` Pearson r drops below 0.85. |

### Final submission candidate

**`meta_avg_aware_weighted`** = rank-blend of (12 simple × weight 1) + (5 meta-aware × weight 2), de-ranked, then anchor-and-smooth postprocess.

This is the literal ANNITIA winning recipe transplanted. Every variant's contribution is gated on its OOF RMSE — the weighted blend only includes variants whose nested-CV RMSE beats the rank-mean baseline.

---

## 7. KIRBy `find_best_hybrid` plug-in (`b3_kirby_hybrid.py`)

KIRBy is the hybrid-search engine for the **engineered-feature tabular tier**. It lifts almost cleanly: rows are (well, MD) pairs, reps are the v2/v3/v4/v6/v7/v8/v9/v10/v11/v13 view families.

**Adapter pattern (mirrors `find_best_av_hybrid` from kirby-av):**

```python
from kirby.find_best_hybrid import find_best_hybrid

def find_best_rogii_hybrid(
    base_reps: dict,          # {view_name: array(n_rows, n_features)}
    tvt_target: np.ndarray,   # (n_rows,) with NaN in eval zone
    well_ids: np.ndarray,     # (n_rows,) for grouping
    gap_mask: np.ndarray,     # (n_rows,) bool, True = held-out toe-end gap mimicking test
    speed: str = 'fast',
):
    cv_splitter = WellGroupGapMaskSplitter(
        well_ids=well_ids,
        gap_mask=gap_mask,
        n_splits=5,
        n_mask_seeds=5,
        formation=dominant_formation_per_well,
    )
    return find_best_hybrid(
        base_reps=base_reps,
        labels=tvt_target,
        cv_splitter=cv_splitter,
        cv_splitter_id='rogii_well_gapmask_v1',
        budgets=[50, 200, 500, 1000],
        models=['rf', 'lightgbm', 'mlp'],   # drop 'svm' (slow on big windowed-GR), 'gp' (too expensive)
        metric='rmse',
        task_type='regression',
        speed=speed,
        internal_evaluators=['match', 'rf', 'lightgbm', 'mlp'],
        importance_methods=['lightgbm_gain', 'permutation'],
        cache=True,
        dataset_name='rogii_tvt',
        tier='v1_engineered',
        augmentation_strategy='uncapped',  # explicit per cache_collision_fix memory
    )
```

**Why this works without modification to KIRBy core:** the search is row-iid, but every row already has its sequence context baked in via the windowed/wavelet/alignment views. The temporal dependence is *encoded as features*, not as model state. The custom CV splitter handles the well-grouping + gap-mask correctness.

**What we get back:** a `SearchResult` with `best_by_model['lightgbm']`, `['rf']`, `['mlp']` — each a Blueprint (rep allocation, feature indices, metric). The blueprint is serialised, refitted on full-train + KAGGLE-train concatenated, and predictions on test are one of the base preds going into the fusion stage. KIRBy is *one model* in the bestiary, not the whole pipeline.

**Where KIRBy might NOT help:** if the alignment features (v7-v9) are so dominant that selecting "200 features from the alignment views and 0 from everywhere else" is the optimal blueprint, KIRBy's allocation step adds little. That's fine — it'll converge to that blueprint and the score will reflect what the features can do. The risk is purely that we pay 5-fold-CV time for a thing a simple RF could discover.

---

## 8. Postprocessing — RMSE's free wins

`postprocess.py`. Apply to every base prediction *and* every fusion variant before submission.

### 8.1 Anchor-and-smooth

If `TVT_input` is observed at the last MD before the gap (`T_a`) and the first after (`T_b`):
1. Force endpoints to match exactly: `prediction[gap_start] = T_a`, `prediction[gap_end] = T_b`.
2. Linear-interpolate the anchor: `linear_anchor[i] = T_a + (T_b - T_a) * (i - gap_start) / (gap_end - gap_start)`.
3. Lowpass the residual: `final[i] = linear_anchor[i] + savgol(prediction[i] - linear_anchor[i], window=21, polyorder=2)`.

Free 2–5% RMSE per Claude research §5 P4.

### 8.2 Per-formation TVT clip

For each predicted MD, predicted formation `f` (from auxiliary head or derived from typewell-correlation): clip `prediction[i]` to the train-set TVT range observed for formation `f`. Cheap; catches gross out-of-range predictions.

### 8.3 Fault-aware smoothing

Detect fault candidates from `|d²(GR)/dMD²|` peaks above the 99.5th percentile. Within 5 samples of a candidate, replace Savitzky-Golay with median filter (preserves discontinuities). This is the only postprocess specifically targeting the worst-decile error per Stephen Clark / GWC retrospective.

### 8.4 Isotonic recalibration on OOF

Fit `IsotonicRegression` on (predicted_OOF, true_OOF). Apply to test predictions before fusion. Five-minute experiment that occasionally lifts 0.5–1% on RMSE.

---

## 9. Failure-mode analysis (`analyse_residuals.py`)

Run after every major model addition. Outputs `residuals_<run_id>.parquet` and a HTML summary with:

1. **Per-MD residual heatmap**, rows = wells, columns = relative-MD-position-in-lateral. Reveals whether errors cluster at toe-end (alignment failure) or fault zones (discontinuity failure).
2. **Per-formation residual distribution**. Boxplot RMSE by formation. Identifies which strata the model under-fits.
3. **Anomaly score → residual correlation** (Spearman ρ). Validates IsolationForest as a routing signal. Target ρ > 0.4 (ANNITIA had 0.47 on hep miss-rate).
4. **Disagreement → residual correlation**. Validates rank-std as a routing signal. Target ρ > 0.5 (ANNITIA had 0.57 on death miss-rate).
5. **Fault-detection vs residual spike alignment**. Are the worst residuals collocated with `|d²GR/dMD²|` peaks? If yes, the fault-aware smoother (§8.3) is justified. If no, the residual tail is something else (alignment break, mislabeled training data) and we need a different fix.
6. **Per-well RMSE rank correlation across folds**. If the same wells are consistently the worst-predicted, they're either OOD or mislabeled. Drop or down-weight.

This is the "analyse how I got things wrong, adjust my metrics" cadence from your ANNITIA approach. Every base model addition, every fusion variant addition, every postprocess change → re-run this script → write findings to `EXPERIMENTS.csv:next_step` before submitting.

---

## 10. Leaderboard probing & EXPERIMENTS.csv discipline

### 10.1 Day-zero EDA (no submissions burned)

`src/kirby_rogii/eda/day_zero.ipynb`. Per Claude research §8.5:

- **Eval-zone gap structure**: `TVT_input` NaN positions are visible per test well. Plot length distribution, gap start position relative to MD range, gap-length-to-lateral-length ratio. Confirm the "contiguous toe-end gap, ~30–60% of lateral" hypothesis (or refute it). The output of this notebook directly parameterises the gap-mask sampler in `_cv.py`.
- **GR distribution per well**: P5/P25/P50/P75/P95 percentile bars, train vs test, per well. KS-statistic test vs train pool. Wells with KS > 0.2 are normalisation failures — flagged for review.
- **Spatial scatter** of `(X, Y)` wellheads, train + test combined.
- **Trajectory feature distributions** (DLS, lateral length, mean inclination): test vs train. Outlier test wells = high-uncertainty wells, up-weight uncertainty-aware variants on them.
- **Adversarial validation**: train vs test classifier, AUC + feature importance.
- **Test → train matrix-profile NN**: which test well is closest to which train well? Diversifies fold assignment.
- **Typewell formation prevalence** in test vs train. The `Geology` column is visible in test typewells — use it for the per-formation TVT-prior clip.

None of this costs a submission slot.

### 10.2 First-week probing schedule

Day 1 (3 submissions burned, 2–3 baselines):
- Three constants `c ∈ {0, μ_train, μ_train + 5}` for the moment-matching solve. Recover `E[y_test], Var(y_test)`.
- Linear-interpolation-of-`TVT_input` baseline (no model).
- DTW best-match-against-typewell baseline (no ML).

The delta between trivial-interp and DTW tells you how much "alignment" is worth pre-learning. The constant probes tell you whether test moments differ from train.

Day 2: M1 (NORCE) trained two ways — random K-fold vs gap-mask CV. Same hyperparameters. The LB delta is the **CV-quality signal**. If gap-mask LB > random LB, gap-mask is validated. If not, gap masks are mis-shaped — fix before submitting again.

Day 3–5: single-variable ablations. One feature group toggled per submission (alignment on/off, GR-norm on/off, postprocess on/off, surface-aux-heads on/off).

Day 6: noise-floor seed delta. Submit identical config, two seeds. Difference = LB noise floor; subsequent ablations < that delta are meaningless.

### 10.3 EXPERIMENTS.csv schema

```
run_id, date, commit_hash, model, key_features, cv_method, cv_mean, cv_std,
lb_score, lb_minus_cv, hypothesis, predicted_lb, outcome, next_step
```

Append-only. `hypothesis` and `predicted_lb` written *before* submitting. After 50 entries, your hit rate on `predicted_lb` is your epistemic state — if <50% you don't understand your own model. Pause and investigate.

After 10–15 entries: compute `pearsonr(cv_mean, lb_score)`. Target r > 0.85. Below 0.7: stop submitting, fix CV. **This is the FORCE 2020 lesson, operationalised.**

---

## 11. Phases & timeline (3 months → 2026-08-05)

| Phase | Calendar | Deliverable | Compute target |
|-------|----------|-------------|----------------|
| **0. Bootstrap** | Week 1 (May 8–14) | Repo scaffolded; loader; day-zero EDA done; gap-mask sampler unit-tested; 3 day-1 probes submitted; NORCE-style M1 reproduced on small subset; EXPERIMENTS.csv pattern in place | Laptop |
| **1. Engineered baseline** | Week 2–3 (May 15–28) | Views v1–v13 implemented; KIRBy hybrid (B3) running; first NN baseline (M1) at full scale; CV-LB Pearson r tracked | Laptop / early PC |
| **2. Cross-attention + alignment** | Week 4–5 (May 29 – June 11) | M2 U-Net cross-attention done; alignment features (v7–v9) wired into all base models; auxiliary multi-task heads on M1/M2; M3 soft-DTW aux loss ablated | PC |
| **3. Tier A models** | Week 6–7 (June 12–25) | A1 PatchTST + SSL pretrain; A4 MOMENT PEFT; A2 TCN ensemble member; A3 enc-dec Transformer baseline; A5 Siamese as feature extractor | PC |
| **4. Fusion + routing** | Week 8–9 (June 26 – July 9) | F1–F12 simple variants; F13–F17 meta-aware; per-row routing on anomaly+disagreement+formation; first ensemble submissions; residual analysis cadence | PC |
| **5. Sharpening** | Week 10–11 (July 10–23) | Heteroscedastic NLL on best NN; pseudo-labelling (Babakhin) on high-confidence rows; fault-aware postprocess; isotonic on OOF; per-formation TVT clip | PC |
| **6. Endgame** | Week 12 (July 24 – Aug 5) | Freeze new architectures; submit only candidate ensembles (varying weights/subsets); two-final-submission selection (safe + aggressive); Kaggle Dataset packaging; deterministic inference notebook | PC + Kaggle |

The phase boundaries are adjustable based on PC build ETA. The dependency graph is strict in two places:
- §3 CV scheme **must precede** any model in Phase 1+.
- Day-zero EDA **must precede** the gap-mask sampler.

---

## 12. Compute milestones tied to PC build

- **If PC online by week 2:** Phase 1 NN baseline runs locally end-to-end.
- **If PC online by week 4:** Phase 1 hybrid runs are CPU; first NN baseline waits till week 5. Acceptable — most leverage is in features/CV in weeks 1–4 anyway.
- **If PC delayed past week 6:** **decision point.** Either (a) rent cloud GPU (Lambda / Vast.ai / RunPod, ~$0.50/hr T4) for Phase 2–3, or (b) compress the Tier-A roster to A1+A4 only and accept a smaller ensemble. **Do not** silently use ARC.
- **Always:** Kaggle notebook is inference-only, ≤9h, no internet. Train ONNX-export the final ensemble if PyTorch import time becomes a problem.

---

## 13. What NOT to do (your standing rules + competition-specific)

- **No random splits** — well-group + gap-mask CV is the floor (per `feedback_no_random_splits`).
- **No mean-of-baselines reporting** — every base individually; rank-mean is a fusion variant, not a baseline (per `feedback_no_mean_baselines`).
- **No Ridge as "sensitivity check"** — F4 cv_weighted_mean and F17 nested_cv_stacking are explicit fusion variants, not sensitivity probes (per `feedback_no_ridge`).
- **No PCA-as-rep** — KIRBy's importance pruning handles high-D (per `feedback_importance_over_pca`).
- **No featurising the surfaces** (ANCC etc.) as inputs — auxiliary supervision only.
- **No GR normalisation as a globally fit transform** — per-well, fit on full trace at inference.
- **No ARC servers** — user direction. PC + cloud only.
- **No skipping the gap-mask sampler** — even if it costs a week. CV is the leverage.
- **No `--no-verify` git pushes**, no destructive bulk operations, etc. — standard hygiene.

---

## 14. Open decisions (need user input before week 1 starts)

1. **Repo location**: `kirby-rogii` standalone (sibling of kirby-finance/kirby-av) or `KIRBy/tests/experiments/rogii/` until it stabilises?
2. **PC build ETA**: rough date drives the Phase 2 / Phase 3 plan and whether cloud GPU is needed.
3. **Cloud-GPU budget if PC slips**: yes/no, monthly cap.
4. **Solo or team**: team-merger deadline 2026-07-29; if you're considering merging, that affects how aggressively to publish public notebooks early.
5. **Kaggle account / Dataset upload pipeline**: confirmed working? Any size limits we'll hit on the weights bundle?
6. **Data state**: have you downloaded the Kaggle data yet? Where is it locally? (Should land at `kirby-rogii/data/{train,test}` per layout above.)
7. **Foundation-model checkpoints** (MOMENT, PatchTST): pre-download to a local cache before week 5; they go into the Kaggle Dataset bundle. ~5–10 GB total.

---

## 15. The single sentence of advice

**Build the CV scheme and the day-zero EDA before any model. Reproduce NORCE. Then proliferate views and fusion variants the way ANNITIA did, but evaluate every fusion variant on held-out OOFs only — the 0.7181 catastrophe is one carelessness away on RMSE just like it was on C-index.**
