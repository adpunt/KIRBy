# Similarity-metrics roadmap (living doc)

**Owner:** Adelaide · **Started:** 2026-07-11 · **Last update:** 2026-07-13
**Point-in-time results:** `SIMILARITY_METRICS_RESULTS_2026_07_11.md` (pilot).
**Code:** `tests/experiments/similarity_metrics/`. **Branch:** `similarity-metrics-study`.

This is the master record. Every experiment and result goes here so nothing is
lost to a wiped scratchpad again. Sections: Goal → Principle → Done (with numbers)
→ In flight → Planned (local vs server) → Open decisions → Changelog.

---

## Goal

Build/choose a molecular representation whose **pairwise comparison** (Tanimoto,
cosine, …) tells you whether two molecules are genuinely similar — for use
*in place of* ECFP4 Tanimoto, and especially for **label-noise mitigation** (a
major use case). NOT predictive performance.

## The principle that keeps it honest

A similarity metric is graded by an **external yardstick**, never by ECFP4
Tanimoto (that would be circular). Two legitimate yardsticks, and the choice
matters:

1. **Activity coherence** — molecules the metric calls similar have similar
   *measured activity*. (Used in the pilot.)
2. **Structural fidelity** — the metric's distance tracks a *structural* ground
   truth that is conformation-free and not a fingerprint:
   - **MCS** (maximum common substructure): size of the largest shared subgraph ÷
     total size. Interpretable, RDKit-native (`rdFMCS.FindMCS`). Small-molecule only.
   - **Graph-edit distance (GED)**: min atom/bond edits between the two graphs.
     More sensitive, slower (NP-hard; approx via SimGNN). Small-molecule only.
   - **Sequence identity / edit distance** — the antibody/protein analog of MCS
     (for datasets like `flab_expression_koenig`, which are sequences, not graphs).

Standardisation (tautomer + protonation canonicalisation at fixed pH) is applied
before any structural comparison, so "which structure" is unambiguous.

---

## DONE — pilot (2 datasets: herg_ki n=1415, logd n=5039), verified

Full detail + tables in `SIMILARITY_METRICS_RESULTS_2026_07_11.md`. Headlines:

- **47 metrics** (10 representations × 6 coefficients) scored on a **9-method
  activity-grounded battery** (kNN read-across, MODI, Patterson, SALI cliff-AUROC,
  BEDROC, violation, noise-detection AUROC, noise-recovery corr).
- **ECFP4 Tanimoto is mid-pack** by activity: read-across rank **27±5 (herg),
  30±4 (logd)** of 47 — robust across 5 subsamples; ~21/47 metrics beat it.
- **Top of the pack (overall mean rank):** `ecfp4_count:soergel` 14.4,
  `atompair_count:soergel` 14.5, `ecfp4_count:dice` 14.8, `ecfp4_count:tanimoto`
  15.0, FCFP4 ~15.5, `desc:euclidean` 15.6. ECFP4:Tanimoto 20.2.
  Cheapest robust win: **binary → count ECFP4.** ⚠ learned/triplet metrics were
  NOT in this run (server), so "counts beat the logical methods" is premature.
- **kNN read-across winners:** herg `atompair:euclidean` ρ=0.732; logd
  `desc:cosine` ρ=0.688.
- **Hybrid beats every single fingerprint** as a similarity space:
  herg 0.542 vs best-single 0.490 vs ecfp4 0.484; logd 0.560 vs 0.523 vs 0.448.
- **Search-objective swap (Experiment 3):** 6 objectives (predictive, sac,
  struct_align, structural-triplet, activity-triplet, knn-activity) all within
  ~0.03; predictive not beaten; **SAC worst.** Interpretation: the objective only
  controls **allocation** (a weak knob — confirmed by the 2026-07-16 knob-
  sensitivity work, ~0.01 swing); the **feature ranker stayed predictive**, so the
  search never actually selected features *for* similarity. → the real lever is
  the feature ranker (below).
- **Noise σ×rate sweep:** noise-DETECTION winner is regime-dependent (stability
  ρ 0.44–0.57; FCFP4→torsion→count as σ rises); noise-RECOVERY ranking stable
  (ρ ~0.9). No single best noise metric on 2 datasets.
- **Verification:** non-circular design confirmed; ranks/BEDROC/leakage re-checked;
  one overclaim ("less-like-ECFP4 ⇒ better") withdrawn.

## IN FLIGHT — server job 12557946

MoleculeACE 30-target battery (classical grid + ChemBERTa/MolFormer + triplet
metric-learner + noise sweep, paired Wilcoxon/SRD). **Stuck `PD (Priority)`** for
days; bump attempted via `scontrol update jobid=12557946 qos=priority`. Answers
the "do counts beat the logical methods at scale / is the noise regime-map real"
questions. See `project_similarity_metrics_job`.

---

## PLANNED — the feature-ranker program (the real lever)

**Thesis:** the search's *feature importance* score (`hybrid.py:1106` consumes it,
`compute_feature_importance` at `hybrid.py:2382` produces it) is always a
*predictive* model-fit. Replace it with a **similarity-aware** ranker and the
search finally selects features *for* similarity. Tested standalone first (no
`hybrid.py` change until results justify it — rebuild-roadmap lock).

### The ranker menu (score each feature f by how its spread relates to a target)

Base recipe: sample pairs (i,j); `df = |f_i − f_j|`; score f by dependence between
`df` and a target-similarity quantity. **Variants to try (not just Spearman):**

| # | ranker | target | notes |
|---|---|---|---|
| R1 | Spearman(df, \|Δactivity\|) | activity | supervised; cousin of predictive — may collapse (SAC trap) |
| R2 | Spearman(df, structural distance) | MCS or GED | **label-free, the genuinely different one** |
| R3 | distance correlation (Székely) | activity/struct | catches nonlinear dependence Spearman misses |
| R4 | mutual information(df, target) | activity/struct | any dependence |
| R5 | RReliefF (NN-pairs, not random) | activity | local, resists collapse; built in pilot |
| R6 | Laplacian score | — | unsupervised; keep features smooth on kNN graph |
| R7 | kernel-target alignment | activity/struct | whole-feature kernel vs target kernel |
| R8 | HSIC-Lasso | activity/struct | selects a **non-redundant SET** (complementarity) |
| R9 | greedy-forward on space fidelity | activity/struct | captures complementarity; expensive |

### Combinations (user's request — label-free is risky alone for a label problem)

- **R1+R2** (activity + structural), fused — hedge the SAC-collapse against
  pure-structure blindness.
- **R2 gated by R1** (structural rank, break ties by activity), etc.
- Fusion method = same as metric aggregation below (mean-rank / RRF).

### References for the structural target

- Small molecules (herg_ki, logd): **MCS** (primary), **GED** (secondary).
- Antibodies (flab_expression_koenig): **sequence identity / edit distance** —
  MCS/fingerprints DO NOT apply (sequences, not small-molecule graphs).

## PLANNED — metric aggregation ("average or funky rankings")

The user is collecting metrics and wants to combine them. Real methods:

- **Score fusion:** normalise each metric to [0,1], (weighted) average. Classic
  data fusion. ⚠ averaging *redundant* metrics (Tanimoto≈Dice≈Cosine) just
  re-weights one signal.
- **Rank fusion:** combine neighbour *rankings* not scores — mean-rank, Borda,
  **Reciprocal Rank Fusion** (Σ 1/(k+rank), robust, scale-free) = the "funky
  ranking" option.
- **Consensus/vote:** count how many metrics call a pair top-k similar.
- **Key rule (compass DCA):** fuse **complementary** metrics (a topological FP +
  descriptors + MCS + a learned one), NOT redundant ones — measure redundancy by
  rank correlation first, drop near-duplicates.

## PLANNED — standardisation preprocessing

Canonicalise **tautomers** and set **protonation at pH 7.4** (RDKit
`MolStandardize`) before building any representation or structural reference, so
comparisons are on one canonical graph. Cheap; applies to the molecular datasets.

## Datasets (locked by user 2026-07-13)

| dataset | n | type | structural ref | notes |
|---|---|---|---|---|
| herg_ki | 1415 | small mol (pKi) | MCS / GED | primary |
| logd | 5039 | small mol (LogD) | MCS / GED | primary |
| flab_expression_koenig | 4275 | **antibody seq** (BLOSUM) | **seq edit distance** | MCS/fingerprints N/A; check activity signal (antibody reps can be near-constant) |

## Local vs server split

- **LOCAL (no GPU):** rebuild the 3 datasets; ranker menu R1–R9 × targets ×
  references; combinations; metric aggregation; standardisation; MCS yardstick.
  (GED and HSIC-Lasso are the expensive ones — cap n or sample pairs.)
- **SERVER (GPU / scale):** the 30-target MoleculeACE battery (job 12557946);
  learned/triplet embeddings; GED at full n via SimGNN-style approx if pursued.

## Open decisions

1. Structural reference: **MCS vs GED** (start MCS — cheap, RDKit-native).
2. Which ranker variants first: R2 (structural) is the cleanest test; R1+R2
   combo is the user's hedge. (User: "try everything" + combinations.)
3. Does `flab_expression_koenig` have enough variant diversity for a structural
   study, or are variants near-identical (single-Ab DMS)? — verify before trusting.
4. If a similarity-aware ranker wins standalone → propose wiring into
   `compute_feature_importance` (needs signoff; new `method=`, own cache namespace).

---

## Changelog

- **2026-07-13:** roadmap created; antibody dataset scoped (sequences, seq-ref);
  feature-ranker program + aggregation + standardisation planned; datasets locked.
- **2026-07-13:** server job 12557946 submitted (stuck PD/Priority).
- **2026-07-11:** pilot complete + verified (see results doc); server package staged.
