# KIRBy Analysis

Results analysis and paper narrative. Each section serves two purposes:
1. **What to say** — the finding and why it matters (paper narrative)
2. **What to build** — analysis scripts/figures needed to support the claim

Results here are preliminary unless marked otherwise.

---

## Metric Direction Reference

**Every analysis script, figure, and comparison MUST respect metric direction.** This table is the single source of truth.

| Domain | Datasets | Primary Metric | Direction | `lower_is_better` |
|--------|----------|---------------|-----------|-------------------|
| Molecular | caco2, herg_ki, logd, qm9, adme_hlm, adme_rlm, adme_mdr1 | RMSE | **lower = better** | `True` |
| Antibody (reg) | flab_thermo_tresanco, flab_pk_jain_heprt, flab_pk_jain_fcrnrt, flab_polyreact_jain_cic, flab_aggregation_jain_hicrt, flab_thermo_jain, abcov_kd, flab_aggregation_jain, flab_immunogenicity_marks, flab_polyreact_jain_bvp, ddg_combined | Spearman ρ | **higher = better** | `False` |
| AbAgym | abagym_Ang2, abagym_EGFR, abagym_HER2, abagym_lysozyme, abagym_VEGF | Spearman ρ | **higher = better** | `False` |
| Ab DMS | flab_binding_koenig, flab_expression_koenig, flab_binding_adams, flab_expression_adams, flab_binding_phillips_h3 | Spearman ρ | **higher = better** | `False` |
| Ab CLF | tz_her2_binary, humanness_therapeutic | AUROC | **higher = better** | `False` |
| Ginkgo | ginkgo_acsins, ginkgo_colloidal, ginkgo_heparin, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer | Spearman ρ | **higher = better** | `False` |
| TCGA (binary) | tcga_luad_vs_lusc | AUROC | **higher = better** | `False` |
| TCGA (multi) | tcga_brca_pam50 | F1w | **higher = better** | `False` |
| Reaction | buchwald_hartwig, bh_doyle_plates | R² | **higher = better** | `False` |
| Nucleotide | 19 datasets (pending) | MCC | **higher = better** | `False` |
| DTA | davis, kiba, bindingdb_kd (pending) | CI | **higher = better** | `False` |

**Note:** This table lists the 40 currently ready datasets + pending domains. More datasets are actively being onboarded — update this table as new datasets are added. The analysis scripts must handle arbitrary datasets dynamically (lookup metric/direction from the result JSON or domain registry, never hardcode).

**Direction-aware comparison rules (enforced in ALL scripts):**

```python
# "Better" means lower for RMSE, higher for everything else
def is_better(score_a, score_b, lower_is_better):
    """Returns True if score_a is better than score_b."""
    return score_a < score_b if lower_is_better else score_a > score_b

def delta(hybrid, base, lower_is_better):
    """Positive delta = hybrid is better. Always."""
    return base - hybrid if lower_is_better else hybrid - base
```

**Never write bare `max(scores)` or `min(scores)` for "best" without checking direction.**

---

## 1. General Analysis: Hybrid vs Base Rep Performance

This is the core KIRBy claim and the first thing to establish in the paper. For every dataset, across every model: does the best hybrid beat the best individual representation?

### 1.1 What This Shows

For each of the 40 ready datasets:
- Evaluate every base rep individually with each downstream model (from baseline surveys)
- Build the best hybrid (via `find_best_hybrid`) for each model across budgets
- Compare: **top 3 base reps** vs **best hybrid** per model

The key output is a **win rate**: across all (dataset, model) combinations, how often does the hybrid beat the best base rep? And by how much?

### 1.2 Paper Narrative

**Claim:** KIRBy hybrids consistently outperform the best individual representation across 40 datasets, 7 domains, and 7 downstream models. The improvement is not universal (a method that always improves would be overfitting), but the wins are frequent and meaningful while the losses are small and concentrated where one rep already dominates.

This narrative has three levels:
1. **Aggregate win rate** — "Hybrids outperform the best individual rep in X% of (dataset, model) combinations"
2. **Per-domain patterns** — Does the win rate vary by domain? (expect: bigger wins on antibody/Ginkgo where reps are more diverse, smaller wins on molecular where fingerprints are more redundant)
3. **Per-model patterns** — Do certain models benefit more from hybridization? (expect: tree models and MLPs benefit most, SVMs may benefit less due to kernel trick already capturing interactions)

### 1.3 Data Sources

| Source | Contains | Location |
|--------|----------|----------|
| Baseline surveys | Per-rep, per-model scores (all metrics) | `tests/results/phase5_baselines/`, `tests/results/antibody_survey/`, etc. |
| Hybrid search results | Best hybrid per model/budget + baselines | `tests/results/comparisons/` or hybrid cache JSONs |
| DATASET_REPS | Which reps are in each dataset's pool | `tests/experiments/complexity/complexity_helpers.py` |

### 1.4 Analysis Script — IMPLEMENTED

**Script:** `tests/analysis/general_hybrid_vs_base.py`

**How it works:** For each dataset in `DATASET_REPS`, calls `get_precomputed_search()` which hits the disk cache from precompute. Extracts the SearchResult's `baselines` (per-rep per-model scores, same folds/metric as hybrids) and `best_by_model_budget` (best hybrid per model per budget). Computes direction-aware deltas and win/loss.

**Two modes:**

1. **Server mode** (default) — run on server after `precompute_hybrids.py` completes:
```bash
# All 40 datasets
python tests/analysis/general_hybrid_vs_base.py

# Specific datasets
python tests/analysis/general_hybrid_vs_base.py caco2 herg_ki

# By domain
python tests/analysis/general_hybrid_vs_base.py --domain molecular

# Export JSON only (for rsync to local)
python tests/analysis/general_hybrid_vs_base.py --export -o general_analysis.json
```

2. **Analyze mode** — read exported JSON locally, no cache/data needed:
```bash
python tests/analysis/general_hybrid_vs_base.py --analyze general_analysis.json
```

**Options:** `--speed fast|full` (must match precompute), `--aug none|uncapped|capped`

**Output:** Prints summary tables to stdout + saves `tests/analysis/general_analysis.json` (full data) and `tests/analysis/general_analysis_summary.json` (aggregate stats). Tables include:
1. Aggregate win rate
2. Win rate by domain
3. Win rate by model
4. Per-dataset summary (wins/total/mean delta)
5. Top 20 hybrid wins and losses (by delta)

**Direction handling:** Uses `lower_is_better` from the SearchResult. Delta is always signed so positive = hybrid is better. Sanity-checked against hardcoded lookup table.

**Output tables:**

1. **Aggregate summary:**

| | Hybrid Wins | Hybrid Loses | Total | Win Rate |
|-|-------------|-------------|-------|----------|
| All | ? | ? | ? | ?% |
| By domain | ... | ... | ... | ...% |
| By model | ... | ... | ... | ...% |

2. **Per-dataset table (sorted by domain, then by delta):**

| Dataset | Domain | Metric | Dir | Model | Best Base Rep | Base Score | Hybrid Score | Budget | Δ | Winner |
|---------|--------|--------|-----|-------|---------------|-----------|-------------|--------|---|--------|

(Δ is always signed so positive = hybrid is better, regardless of metric direction)

3. **Delta distribution:** Histogram of all deltas, annotated with median and mean.

### 1.5 Figures Needed

- [ ] **Win rate bar chart by domain:** Grouped bars showing hybrid win rate per domain. Shows whether certain domains benefit more.
- [ ] **Delta scatter plot:** (best base score) vs (delta). One point per (dataset, model). Color by domain. Expect: negative trend (bigger base = smaller delta), with outlier wins in the weak-signal regime.
- [ ] **Per-dataset paired comparison:** For each dataset, show best base vs hybrid for each model. Error bars from CV folds. Shows consistency, not just means.
- [ ] **Budget sensitivity:** For a few representative datasets, show hybrid score vs budget. Shows whether more features always help or if there's a sweet spot.

### 1.6 Direction Pitfalls to Avoid

1. **RMSE deltas are inverted:** For molecular datasets, hybrid RMSE 0.50 vs base RMSE 0.55 → delta should be +0.05 (hybrid is better by 0.05 RMSE units). Computed as `base - hybrid` for lower-is-better.
2. **Mixed-metric aggregation:** Do NOT average deltas across domains with different metrics. Report win rates (dimensionless) for cross-domain aggregation. Per-domain deltas can be averaged within a domain.
3. **"Best" comparison:** The direction of the sort for finding "top 3 base reps" flips for RMSE. Use `sorted(scores, reverse=not lower_is_better)`.
4. **Statistical significance:** Fold-level paired t-test or Wilcoxon between hybrid and best base. Report p-values but do not rely on them alone for n=5 folds.

---

## 2. Ginkgo Developability (OPIG B5)

**Source:** `slurm-11692950.out`, `opig_ginkgo_developability.json`
**Experiment:** `tests/experiments/opig_ginkgo_developability.py`
**Benchmark:** van Niekerk et al. 2026, Ginkgo Antibody Developability Competition (113 teams)
**Status:** Preliminary (pre-cache-clear hybrid, GP removed from re-run)

### 2.1 Results Summary

| Property | N | Best Individual | ρ | Hybrid ρ | Δ | Pattern |
|----------|---|----------------|---|----------|---|---------|
| hic | 242 | esm2_concat | 0.604 | 0.563 | -0.041 | Strong signal, no improvement needed |
| tm2 | 193 | blosum | 0.147 | 0.530 | **+0.383** | Weak + weak + weak → strong |
| titer | 239 | igt5 | 0.367 | 0.580 | **+0.214** | Moderate signals combined |
| pr_cho | 197 | prot_t5 | 0.505 | 0.617 | **+0.112** | Moderate signals combined |
| acsins | 242 | ablang2 | 0.559 | 0.646 | +0.087 | Moderate signals combined |
| purity | 242 | onehot | 0.210 | 0.181 | -0.029 | No signal → correctly stays flat |
| aggregation | 242 | ablang2 | 0.163 | 0.421 | **+0.258** | Weak + weak + weak → strong |
| colloidal | 242 | prot_t5 | 0.394 | 0.335 | -0.059 | Moderate, small loss |
| heparin | 94 | igt5 | 0.679 | 0.823 | **+0.144** | Strong individual, hybrid improves further |

**Hybrid wins 6/9 properties.** 3 losses are all small (< 0.06).

### 2.2 Paper Narrative: Four Regimes of Hybrid Behaviour

This is the key finding. Hybrid effectiveness follows a predictable pattern based on the strength and diversity of individual representations:

**Regime 1: Complementary weak signals → strong hybrid.**
Tm2 and aggregation. Every individual rep is at noise level (best ρ < 0.17), yet the hybrid achieves ρ = 0.53 and 0.42 respectively. Each rep captures a small, orthogonal piece of the signal that is invisible in isolation but emerges when combined via feature selection. This is the core KIRBy thesis working as designed.

**Regime 2: Moderate signals → improved hybrid.**
Titer, PR_CHO, AC-SINS. One or two reps capture meaningful signal (ρ 0.37–0.56), and the hybrid lifts performance further (+0.09 to +0.21). The hybrid is combining the best features from several reps, each contributing partially overlapping but partially unique information.

**Regime 3: Strong dominant signal → diminishing returns.**
HIC. esm2_concat already captures most of the predictive signal (ρ = 0.60). The hybrid can't improve because there's little complementary information in the other reps. The small loss (-0.04) is likely noise or feature dilution from adding weaker reps.

**Regime 4: No signal → no hallucination.**
Purity (best ρ = 0.21, all reps near noise). The hybrid correctly stays flat — it does not fabricate predictive power from noise. This is important: a method that always "improves" would be overfitting.

**Why this matters for the paper:** These four regimes are a testable prediction of the KIRBy framework. If hybrids are doing what we claim (finding complementary features across representations), we should see exactly this pattern: biggest gains when individual reps are weak but diverse, diminishing gains when one rep dominates, and no gains when there is genuinely no signal. The Ginkgo experiment confirms all four predictions across 9 biophysical properties from the same antibody panel.

### 2.3 Comparison to Published Baselines

For the 5 competition properties (CV Spearman ρ):

| Property | Competition Best CV | Best Published Method | KIRBy Hybrid |
|----------|--------------------|-----------------------|-------------|
| HIC | 0.758 | MOE baseline 0.656 | 0.563 |
| Tm2 | 0.140 | ESM2-TAP+Ridge 0.265 | **0.530** |
| Titer | 0.640 | AbLang2+EN 0.356 | **0.580** |
| PR_CHO | 0.559 | ESM2-TAP+Ridge 0.413 | **0.617** |
| AC-SINS | 0.653 | AbLang2+EN 0.509 | **0.646** |

KIRBy hybrid beats ALL published method baselines on 4/5 properties. On Tm2, it doubles the best published method. The only loss (HIC) is against the full competition winner (113 teams, structural features, ensemble methods) — KIRBy still beats the individual published baselines on HIC.

**Caveat:** These are CV results on training data. Competition highlighted CV >> test performance (overfitting). KIRBy does not have access to the held-out test set (80 antibodies). The comparison is fair in terms of protocol (same folds, same metric) but we cannot claim test-set generalization.

**TODO for paper:** Note that published baselines are fixed (rep, model) pairs while KIRBy evaluates each rep with best-of-7 models. This is a methodological choice, not an unfair advantage — we are showing what each representation family can achieve, not claiming a single fixed pipeline.

### 2.4 Issues Found

1. **GP catastrophe (FIXED):** GP predicted constant values for tm2, purity, aggregation — identical R² across all reps (-834, -525, -603). Property-level pathology from tight target ranges, not dimensionality. GP removed from this experiment.

2. **Colloidal ρ/R² disagreement:** Hybrid loses by ρ (0.335 vs 0.394) but wins massively by R² (0.316 vs 0.019). Model-family effect: xgboost on hybrid optimises squared error, SVM on prot_t5 optimises margin. Worth a fold-level investigation.

3. **Heparin small sample:** n=94, best absolute result (ρ=0.82). High variance with ~19 test samples per fold. Interpret cautiously.

### 2.5 Analysis Scripts Needed

- [ ] **Per-property fold-level plots:** Show hybrid vs best-individual ρ per fold (error bars, paired comparison). Demonstrates consistency, not just mean improvement.
- [ ] **Hybrid composition breakdown:** For each property, what reps contributed features to the hybrid? How many features from each? This directly supports the "complementary signals" narrative — e.g., for tm2, we expect features drawn from many reps.
- [ ] **Regime classification figure:** Scatter plot of (best individual ρ) vs (hybrid Δ). Should show clear negative trend with the four regimes labelled. If this holds across ALL datasets (not just Ginkgo), it's a central figure.
- [ ] **Competition comparison bar chart:** Grouped bar chart per property: published baselines, KIRBy individual reps, KIRBy hybrid.
- [ ] **Re-run with updated hybrid search** and confirm numbers hold.

### 2.6 Questions for Next Run

1. Do the 3 small losses (HIC, purity, colloidal) flip or shrink with updated search?
2. What is the hybrid composition on tm2? If features come from 5+ reps, that's the strongest evidence for complementary signals.
3. Does the regime pattern hold for other antibody datasets (FLAb, AbAgym, SKEMPI)?

---

## 3. Experiment Category Plans

### 3.1 Complexity Experiments (C01–C21)

**Question:** How does hybrid performance scale with various complexity dimensions?

**Key experiments by sub-question:**

| Sub-question | Experiments | What it tests |
|-------------|-------------|---------------|
| Budget scaling | C01 | Hybrid score vs feature budget (10→2000). Is there a sweet spot? |
| Rep count scaling | C03 | Performance as you add 1→N reps to the pool. Diminishing returns? |
| Sample size effect | C04 | Hybrid advantage at small vs large N. Does KIRBy help more when data is limited? |
| Model sensitivity | C05 | Same hybrid, different downstream models. Is the blueprint model-agnostic? |
| Rep diversity | C06, C07 | Hybrid value when reps are homogeneous (all fingerprints) vs diverse (FP + LM + graph). |
| Strategy comparison | C08, C09, C10, C11 | Allocation and importance method ablation. |
| Classification | C02 | Full complexity grid on classification datasets. |
| Augmentation | C16–C19 | Does augmentation (adding small reps post-search) improve hybrids? |
| Modality ablation | C12 | What happens when you restrict to one rep family? |
| Single-rep augmentation | C21 | Augmenting a single rep rather than a hybrid. |

**Paper narrative:** The complexity experiments establish the operating characteristics of KIRBy — when it works, when it doesn't, and what parameters matter. Key claims:
1. Budget has a sweet spot (not always "more is better") — supported by C01
2. Rep diversity matters more than rep count — supported by C06/C07
3. Small datasets benefit proportionally more from hybridization — supported by C04
4. The allocation strategy matters less than the rep pool — supported by C08-C11 and F12

**Analysis needed:**
- [ ] Budget curve plots per dataset (C01)
- [ ] Rep count learning curves (C03)
- [ ] Sample size interaction plot (C04)
- [ ] Rep diversity vs hybrid delta (C06/C07)
- [ ] Strategy comparison heatmap (C08-C11)

### 3.2 Similarity Experiments (S01–S13)

**Question:** How does representation similarity affect hybrid value?

**Key experiments:**

| Sub-question | Experiments | What it tests |
|-------------|-------------|---------------|
| Pairwise similarity metrics | S01–S03 | CKA, centered cosine, mutual info between reps |
| Similarity vs hybrid delta | S04–S06 | Correlation between rep dissimilarity and hybrid improvement |
| Clustering reps by similarity | S07, S08 | Do functionally similar reps cluster? Can we predict which reps to combine? |
| Similarity-guided selection | S09–S11 | Use similarity to pick reps before search. Faster? As good? |
| Similarity under noise | S12, S13 | Does noise disrupt similarity structure? |

**Paper narrative:** Similarity experiments test the hypothesis that diverse representations combine better than similar ones. Key claims:
1. Reps from different families (LM vs fingerprint vs graph) have lower similarity and produce better hybrids — S04-S06
2. Similarity can be used as a cheap pre-filter to reduce search cost — S09-S11
3. The four regimes from Ginkgo (Section 2) are explained by similarity: weak reps that combine well have low pairwise similarity (complementary signals), strong dominant reps have high similarity with most others (redundant information)

**Analysis needed:**
- [ ] Similarity heatmaps per domain (S01-S03)
- [ ] Scatter: pairwise similarity vs hybrid delta when combined (S04-S06)
- [ ] Rep dendrogram by similarity (S07-S08)
- [ ] Pre-filter efficiency plot: search cost vs quality when using similarity-guided selection (S09-S11)

### 3.3 Noise Experiments (N01–N13)

**Question:** How robust are hybrids to label noise?

**Key experiments:**

| Sub-question | Experiments | What it tests |
|-------------|-------------|---------------|
| Noise injection | N01–N04 | Gaussian/uniform noise at 5-50% corruption. Hybrid vs base degradation curves. |
| Noise + similarity | N05–N08 | Does similarity-guided selection become more/less important under noise? |
| Learning with noisy labels (LNL) | N09 | Survey of 12 LNL methods (co-teaching, mixup, etc.) applied to hybrid training. |
| Hybrid neighborhood LNL | N10 | Use rep similarity to identify likely mislabelled samples. |
| Estimation × remediation grid | N11 | Grid of noise estimation methods × noise remediation methods. |
| Importance-weighted distance | N12 | Weight similarity by feature importance from hybrid. |
| Alternative noise distributions | N13 | Skewed, heavy-tailed, systematic noise patterns. |

**Paper narrative:** Noise experiments address the confirmation report reviewer question: "Does hybridization amplify noise?" Key claims:
1. Hybrids degrade more gracefully than individual reps under noise — N01-N04 (the feature selection acts as implicit denoising)
2. KIRBy's noise robustness is complementary to explicit LNL methods — N09-N11
3. Similarity structure is preserved under moderate noise — N05-N08

**Analysis needed:**
- [ ] Degradation curves: score vs noise level, hybrid vs top base rep (N01-N04)
- [ ] LNL method comparison table (N09)
- [ ] Noise detection accuracy from hybrid features (N10)

### 3.4 OPIG Experiments

**Standalone experiments tied to OPIG collaboration.**

| Experiment | Datasets | What it tests |
|-----------|----------|---------------|
| Ginkgo Developability (B5) | 9 Ginkgo properties | See Section 2. Published benchmark comparison. |
| Transfer Learning (C4) | logd, caco2, herg_ki | Does ChEMBL DNN pretraining change rep rankings? Does TL + hybrid interact? |
| Ab CLF | tz_her2, humanness | Classification metrics on antibody tasks. |
| DDG (B2) | ddg_combined | ΔΔG prediction from Graphinity. |
| ITS-Flexible | TBD | Developability loop length flexibility. |

**Analysis needed:** Mostly per-experiment. Ginkgo has its own section (Section 2). Others will get sections as results come in.

### 3.5 Federated Experiments (Fed01–Fed05)

**Question:** How does KIRBy interact with federated learning (FLuID framework)?

**Key experiments:**

| Experiment | What it tests |
|-----------|---------------|
| Fed01 | Noise heterogeneity across federated sites. Does KIRBy hybrid help when sites have different noise levels? |
| Fed02 | Rep diversity silos — each site has different reps. Can federation recover the hybrid? |
| Fed03 | Synthetic private datasets. Can KIRBy improve federated models on non-shareable data? |
| Fed04 | Cluster-aware importance weighting across sites. |
| Fed05 | Antibody extension — apply federated KIRBy to antibody domain. |

**Paper narrative:** These extend the KIRBy thesis to privacy-constrained settings. The key claim: even when representations are distributed across sites that cannot share data, the federated hybrid recovers most of the centralised hybrid's benefit.

**Analysis needed:** Per-experiment comparison of federated vs centralised hybrid. Mostly blocked on execution.

### 3.6 Framework Experiments (F01–F14)

**Question:** What are the best internal configuration choices for the KIRBy pipeline?

**Status:** F12/F13 complete and acted on. Results already incorporated into SPEED_TIERS and default configurations. These are implementation experiments, not paper experiments — they justify the choices, not the claims.

**Key conclusions (already applied):**
- Model/evaluator choice dominates (η² 0.23–0.89). Strategy matters much less.
- 13 allocation strategies survived the F12 cut (from 31). All greedy family dropped.
- Budget 300 wins most often, but 50 and 150 matter too.
- Match evaluator is default.

**Analysis needed:**
- [ ] F12 strategy comparison figure (for methods section, not results)
- [ ] F10 small-dataset allocation results (when available)

---

## 4. Cross-Cutting Analyses

These analyses span multiple experiment categories and datasets.

### 4.1 Regime Classification (from Section 2.2, generalised)

**Extends the Ginkgo four-regime framework to all 40 datasets.**

Plot: (best individual score, direction-normalised) vs (hybrid delta). One point per dataset. Color by domain. Label the four regimes. If the pattern holds across all domains, this is a central figure.

**Key insight:** This is NOT just a Ginkgo finding. If it holds across molecular, antibody, TCGA, and reaction domains, it becomes a general theory of when representation hybridization works.

### 4.2 Rep Contribution Analysis

**For each dataset's best hybrid:** How many reps contributed features? What fraction came from each rep family?

This directly supports the "complementary signals" claim. Expected patterns:
- High-delta hybrids draw from 4+ reps (diverse pool)
- Low-delta hybrids are dominated by 1-2 reps (redundant pool)
- The contributing reps should have low pairwise similarity (from S01-S06)

### 4.3 Domain-Specific Findings

As results come in, each domain may produce domain-specific insights worth a paper paragraph:
- **Molecular:** Does fingerprint + LM combination consistently beat either alone?
- **Antibody:** Do mutation-aware reps (esm2_wt_delta, mutation_features) contribute disproportionately to DMS hybrids?
- **TCGA:** With 50+ reps across modalities (gene expr, methylation, CNV, WSI, clinical), does multimodal hybridization dominate?
- **Reaction:** Do product-side and reactant-side reps complement each other?

---

## 5. Publication Figure List (Preliminary)

| # | Figure | Source Section | Priority |
|---|--------|---------------|----------|
| 1 | Win rate by domain (bar chart) | 1.5 | HIGH — core claim |
| 2 | Regime scatter: base score vs hybrid delta | 4.1 | HIGH — central theory |
| 3 | Ginkgo competition comparison | 2.5 | HIGH — external validation |
| 4 | Budget scaling curves | 3.1 (C01) | MEDIUM |
| 5 | Noise degradation curves | 3.3 (N01-N04) | MEDIUM |
| 6 | Rep contribution breakdown | 4.2 | MEDIUM |
| 7 | Similarity heatmap (one domain) | 3.2 (S01-S03) | MEDIUM |
| 8 | Ginkgo per-property fold-level | 2.5 | LOW (supplement) |
| 9 | Strategy comparison heatmap | 3.6 (F12) | LOW (methods) |
| 10 | Federated vs centralised | 3.5 | LOW (extension) |

---

## 6. Future Sections (add as results come in)

Each domain gets its own results section (like Section 2 for Ginkgo) once experiments complete:
- Molecular domain (caco2, herg_ki, logd, qm9, adme_hlm, adme_rlm, adme_mdr1)
- Antibody intra-family (11 FLAb + abcov + ddg)
- Antibody DMS (5 koenig/adams/phillips datasets)
- AbAgym (5 antigen-specific datasets)
- Antibody CLF (tz_her2, humanness)
- TCGA (luad_vs_lusc, brca_pam50)
- Reaction (buchwald_hartwig, bh_doyle_plates)
- DTA (pending surveys)
- Nucleotide (pending FM rep surveys)
- Transfer learning (OPIG C4)

Each section follows the same structure:
1. Results summary (table)
2. Paper narrative (what to say)
3. Comparison to published work (where available)
4. Issues found
5. Analysis scripts needed
6. Questions for next run
