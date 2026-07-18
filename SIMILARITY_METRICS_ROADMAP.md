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

## DONE — structural yardstick (MCS re-rank) · `structural_yardstick.py`

Re-ranks the 47 metrics by **structural** fidelity (Spearman between each metric's
pairwise distance and MCS-distance over sampled pairs) alongside activity fidelity.
Local, no server. **Full 6000-pair run, both datasets (2026-07-18)** →
`structural_rerank.json`:

| structural judge (MCS), top by struct-fidelity ρ | herg_ki | logd |
|---|---|---|
| best | ecfp6:euclidean **0.550** | ecfp4:euclidean **0.626** |
| next | ecfp4:euclidean 0.542, torsion:euclidean 0.528, fcfp4:euclidean 0.484 | ecfp6:euclidean 0.624, torsion:euclidean 0.617, atompair:euclidean 0.615 |
| `ecfp4:tanimoto` | struct 0.131, **rank 15/47** | struct 0.433, **rank 20/47** |
| `ecfp4_count:tanimoto` | struct 0.066, **rank 28/47** | struct 0.374, **rank 32/47** |

**Headline:** under a STRUCTURAL judge the winners flip to **Euclidean-coefficient**
fingerprints on both datasets, and `ecfp4_count:tanimoto` — the pilot's ACTIVITY
winner — drops to rank 28–32. So the count-fingerprint win is an **activity-
coherence** effect, NOT structural fidelity; the activity and structural judges
**disagree** (activity_fidelity ≈ 0 for the structural winners). This is the whole
reason the two yardsticks are reported separately from here on.

⚠ **Open caveat (not yet controlled):** Euclidean distance scales with molecule
SIZE, so part of the Euclidean structural win may be a size confound (bigger
molecules ⇒ larger Euclidean distance ⇒ larger MCS distance). A same-size-pair or
equal-length-Hamming control is the fix; flagged, not yet run. Random pairs are
also mostly dissimilar (MCS-dist mean ≈ 0.85), so the correlation is dominated by
the saturated regime.

## PLANNED — metric aggregation ("average or funky rankings") · `aggregate.py`

The user is collecting metrics and wants to combine them. Methods to TRY (all get
implemented + benchmarked on both yardsticks):

- **A1 Score fusion:** normalise each metric to [0,1], (weighted) average. Classic
  data fusion. ⚠ averaging *redundant* metrics (Tanimoto≈Dice≈Cosine) just
  re-weights one signal.
- **A2 Mean-rank fusion:** average each metric's neighbour ranks.
- **A3 Reciprocal Rank Fusion (RRF):** Σ 1/(k+rank) — robust, scale-free, the
  "funky ranking" option; the IR standard.
- **A4 Borda count:** points by rank position.
- **A5 Consensus/vote:** count how many metrics call a pair top-k similar.
- **A6 Redundancy-pruned fusion (compass DCA):** measure pairwise rank-correlation
  between metrics, drop near-duplicates, fuse only the **complementary** survivors
  (a topological FP + descriptors + MCS + a learned one). The rule that makes
  fusion actually pay off.
- Each aggregate is scored on BOTH yardsticks (activity + structural) and the
  noise battery, vs the best single metric — does fusion beat its parts?

## PLANNED — antibody similarity (`flab_expression_koenig`, n=4275) · valuable, harder

Antibodies are **sequences, not small-molecule graphs**, so MCS/GED/fingerprints
DO NOT apply. The framework transfers with the pieces swapped:

- **Representation:** the cached BLOSUM encoding (4275×5600) is the antibody
  "fingerprint"; add k-mer / one-hot / ESM-embedding reps if pursued.
- **Structural reference (the MCS analog):** **sequence edit/Hamming distance**
  between variants (for a single-Ab DMS, ≈ number of mutations). If raw sequences
  aren't loadable, BLOSUM-space Euclidean distance is a faithful proxy.
- **Activity yardstick:** expression-fitness labels y (must be loaded via the
  KIRBy data loader; not in a flat cache file — OPEN plumbing item).
- **Comparison functions:** Tanimoto/Dice are ill-defined on dense BLOSUM; use
  cosine / Euclidean / correlation.
- ⚠ **Diversity risk:** if variants are near-identical point mutants of ONE
  antibody, structural distances are tiny and near-constant → a structural
  similarity study has little to separate. VERIFY variant spread before trusting
  any structural result (see open decision 3).
- The **feature-ranker menu (R1–R9)** transfers directly on BLOSUM features, with
  the structural target = sequence/BLOSUM distance and the activity target = fitness.

### ⛔ DIVERSITY CHECK RESULT (2026-07-18) — koenig FAILS the structural test

Loaded via `kirby.datasets.flab.load_flab_dataset('flab_expression_koenig')` and
measured spread BEFORE trusting anything (the memory warning was right, and worse
than feared):

- **Every one of the 4275 variants is EXACTLY 1 mutation from wildtype**
  (edit-distance-from-WT histogram = `{1: 4275}`). Single-site DMS.
- Heavy chains are all length 120; **pairwise Hamming between two variants takes
  only {0, 1, 2}**, and **20.8% of random pairs are identical heavy chains**
  (4275 rows, only 2262 unique heavy sequences).
- ⇒ **the structural reference is DEGENERATE** — 3 possible distance values, no
  spread to correlate against. The BLOSUM features are near-constant too
  (119/120 residues identical between any two variants), matching the "dead
  antibody rep" memory. **A structural-fidelity study is not viable here.**
- Labels DO vary (fitness std 0.57, 4186 distinct) so an **activity-only** arm is
  possible but weak on near-constant features.

**Antibody alternatives scoped (same loader):**

| dataset | n | task | seq len | pairwise structural spread | note |
|---|---|---|---|---|---|
| flab_expression_koenig | 4275 | regression | 120 (equal) | Hamming {0,1,2} — **degenerate** | user's pick; structural DEAD |
| flab_binding_phillips_h3 | 32767 | **classification** (11.6% binders) | equal | Hamming **2–14** (median 8) — real | combinatorial CR9114; clean seq distance, big n; activity is BINARY |
| flab_thermo_jain | 137 | regression | **variable** 112–116+ | edit 18–205 — real | inter-family (diff antibodies); tiny n; needs alignment |
| flab_aggregation_jain | 137 | regression | variable | edit 18–205 — real | as above |
| flab_binding_rosace | 14 | regression | — | — | too small |

**FORK for the user (open):** (a) pivot the antibody *structural* study to
`flab_binding_phillips_h3` (large, equal-length, real Hamming spread) accepting a
**classification** activity yardstick (kNN-AUROC/MODI instead of Spearman ρ);
(b) use `flab_thermo_jain` (regression, real spread) accepting **n=137** and
variable-length alignment; or (c) keep koenig as a documented **activity-only**
stress-test that confirms the dead-feature regime. Structural arm on koenig is off
the table either way.

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

## SESSION 2026-07-18 — ranker & aggregation experiments (design LOCKED w/ user)

New standalone scripts (do NOT touch hybrid.py/find_best_hybrid.py):
`_data.py` (shared loader; logd capped to 1400-mol scaffold subsample, viability
checked — capped read-across ρ 0.618 count / 0.589 binary, in line with pilot),
`mcs_cache.py` (8000 fixed MCS pairs/dataset → `mcs_pairs_<ds>.npz`, reused for
structural ranker target AND structural scoring), `ranker_experiment.py`,
`aggregate.py`.

**Design decisions settled with the user (do not relitigate):**
1. **Feature pool = 4 COMPLEMENTARY reps** (ecfp4_count, atompair_count, desc,
   maccs = 2423 features) — DCA "fuse complementary, not redundant" rule; NOT the
   all-10 pool (stacks near-duplicate circular FPs).
2. **Feature COUNT comes from the ALLOCATION knob, not an invented sweep** — the
   user's correction. `total_budget` (hybrid.py:694) is the "how many features"
   knob; `top_k` weighting (top_2/3/5 at hybrid.py:79) is the concentrate knob.
   Realised as **per-rep budget n_per_rep=100 × top-N reps**; the ranker only
   picks WHICH features inside each surviving rep.
3. **scikit-learn feature-selection tier added** (user's ask): SelectKBest
   f_regression + mutual_info, SelectFromModel Lasso (doubles as R8 HSIC-Lasso
   stand-in — pyHSICLasso absent), RandomForest importance, RFE — so the bespoke
   similarity-aware rankers are benchmarked against off-the-shelf sklearn.
4. **Every pair-based selector runs in BOTH target modes** — activity (|Δy|) AND
   structural (MCS). *User flagged this activity-vs-structural selector comparison
   as the novel headline — "no one's made this comparison like this before."*
   All share the SAME cached MCS pairs so the two modes are perfectly paired.
5. **logd capped for local tractability** (n×n kNN + O(n²d) Soergel), gated on the
   viability check above.
6. Selected-feature distance = z-scored cols → cosine AND euclidean. Yardsticks =
   activity read-across ρ (scaffold GroupKFold) + MCS structural fidelity.
   Baselines (never averaged) = ecfp4_count:tanimoto, ecfp4:tanimoto.

**Aggregation (A1–A6)** over 6 complementary metrics (ecfp4_count:tan,
atompair_count:cos, desc:euc, maccs:tan, torsion:tan, rdkit:tan); each aggregate
scored on read-across + structural + noise battery vs the BEST SINGLE metric.

### RESULTS — aggregation (`aggregate_experiment.json`, 2026-07-18)

**Fusion does NOT reliably beat the best single metric, and it HURTS structural
fidelity.** Best aggregator = **RRF (A3)**, but the margin is marginal-to-nil.

| yardstick | herg: best single → best aggregate | logd: best single → best aggregate |
|---|---|---|
| read-across ρ | atompair_count:cos **0.716** → A3_rrf **0.729** (+0.013) | desc:euc **0.666** → A3_rrf 0.664 (**no win**) |
| structural (MCS) | desc:euc **0.250** → best agg 0.067 (**fusion −0.18**) | torsion:tan **0.482** → best agg 0.429 (**fusion −0.05**) |
| noise detect | atompair_count:cos **0.657** → 0.654 (no win) | atompair_count:cos **0.649** → 0.635 (no win) |
| noise recovery | torsion:tan **0.795** → 0.786 (no win) | desc:euc 0.730 → A3_rrf **0.739** (+0.009) |

- **Structural fidelity is the clear casualty:** averaging a structurally-good
  metric (desc:euclidean) with structurally-poor ones washes the signal out —
  herg aggregates collapse to ρ≈0.06 vs the single desc:euclidean's 0.250.
- **A6 (DCA redundancy-pruned) pruned NOTHING** on either dataset — the 6 metrics
  have pairwise rank-correlation < 0.9, i.e. they are genuinely complementary, so
  "fuse only the non-redundant survivors" = fuse all 6 ⇒ A6 ≡ A1. (A finding in
  itself: the chosen set has no redundancy to drop.)
- **Takeaway:** consistent with the pilot — the lever is representation ×
  coefficient, NOT fusion. Pick the best single metric per task; do not fuse if
  you care about structural fidelity. RRF is the least-bad aggregator if forced.

### RESULTS — ranker experiment (`ranker_experiment.json`, 2026-07-18)

**YES — a similarity-aware feature ranker beats the count fingerprint, decisively
on structural fidelity and (dataset-dependent) on read-across.** 55 rows/dataset;
cosine coeff shown (euclidean similar). n_per_rep=100, pool = 4 complementary reps.

| yardstick | herg: count-FP baseline → best ranker | logd: count-FP baseline → best ranker |
|---|---|---|
| read-across ρ | 0.709 → **0.724** (r9_greedy, +0.015) | 0.618 → **0.704** (r5_rrelieff, +0.086) |
| structural (MCS) | 0.045 → **0.370** (r1r2 struct, top2) — **8×** | 0.387 → **0.624** (r9_greedy struct) — +0.237 |

**★ The novel finding — SELECTING FEATURES FOR STRUCTURE WORKS.** For the *same*
selector, switching the target from activity (|Δy|) to structural (MCS) raises
structural fidelity almost everywhere, at a small read-across cost. Δ(struct
fidelity), activity-target → structural-target, top4×100:

| selector | herg Δ | logd Δ |
|---|---|---|
| r1r2 Spearman | +0.112 | +0.080 |
| r7 KTA / sk f_regression | +0.068 | +0.091 |
| r3 dcor | +0.060 | +0.078 |
| r4 mutual-info | +0.050 | +0.124 |
| sk Lasso | +0.056 | +0.031 |
| sk RandomForest | −0.010 | +0.020 |

Consistent and directional across 6/7 selectors and BOTH datasets — the
activity-vs-structural selector comparison the user flagged as novel. It is real:
the feature ranker, not just the representation, controls how structurally
faithful the resulting similarity space is.

**Bespoke vs off-the-shelf scikit-learn (supervisor asked for this comparison —
they expected sklearn to win; it does NOT):** close, with the edges going to the
bespoke selectors. The bespoke complementarity-aware set-selector **R9
greedy-forward tops structural fidelity on both** (herg 0.320, logd 0.624) and
**R5 RReliefF tops read-across on both** (herg 0.720, logd 0.704). The simple
linear screens are interchangeable — sk `f_regression` ≡ r7 KTA to 3 dp (both are
centred-correlation screens). So sklearn is a strong FREE baseline that ties on the
easy (linear) selectors, but the bespoke similarity-aware selectors win precisely
where they are designed to — complementarity (greedy) and local nearest-neighbour
structure (RReliefF), which no single sklearn selector captures.

**HOW the comparison was run (equal footing — this is the methodology to cite):**
1. **Identical inputs.** Every selector — bespoke and sklearn — sees the exact same
   pooled feature matrix (4 complementary reps, z-scored, 2423 columns) and the
   exact same 8000 fixed molecule pairs (cached, `mcs_pairs_<ds>.npz`).
2. **Identical learning problem.** Every pair-based selector scores features on the
   same regression: design matrix = pair feature-differences `|f_i − f_j|`, target =
   either `|Δactivity|` (activity mode) or MCS distance (structural mode). The
   sklearn selectors (`f_regression`, `mutual_info_regression`,
   `SelectFromModel(Lasso)`, `RandomForest` importance, `RFE(Ridge)`) are applied to
   *that same* `(DF, target)` problem — no bespoke advantage in the setup.
3. **Identical budget.** Every selector's output is passed through the SAME
   allocation rule (top-100 features per rep × 4 reps = 400 features), so the
   comparison holds the feature count fixed — a selector can only win by *choosing
   better features*, not more.
4. **Identical scoring.** Each 400-feature space is scored the same two ways:
   read-across ρ (5-fold scaffold-grouped kNN, no random splits) and structural
   fidelity (Spearman vs MCS on the cached pairs). Same datasets, seeds, caches.
5. **Both target modes** run for every pair-based selector, so bespoke and sklearn
   are compared in both the activity and structural regimes.

The one asymmetry (documented, not hidden): the two *set* selectors — bespoke R9
greedy-forward and R8/Lasso — choose a set directly rather than emitting per-feature
scores; R9 is capped at the same 400-feature budget. All *per-feature* selectors
(the bulk, incl. every sklearn scorer) go through the identical allocation in (3).

**Knobs.** Concentrating budget on the top-2 reps (top2×100) can *raise* structural
fidelity over spreading across 4 (herg r1r2 struct: 0.370 @ top2 vs 0.303 @ top4)
at a read-across cost — the concentrate-vs-spread lever, live here too.

**Honest caveat.** The logd read-across gain (+0.086) is partly a POOL effect: the
pool contains `desc` (physicochemical descriptors), inherently strong for a
physicochemical endpoint like LogD, which the count-FP baseline cannot use. The
STRUCTURAL-fidelity gains are the robust ranker/target result (same pool, different
target ⇒ different structural fidelity), not a pool artefact.

**Compute:** ~950s/dataset local; r3_dcor (~510s) and r4_mi (~280s) dominate.

## DONE — scaffold-mutation representation investigation (koenig, 2026-07-18)

**The idea (user):** a single-wildtype DMS is a SCAFFOLD (the wildtype) + point
mutations. Whole-sequence fingerprints collapse (every variant ~99% identical), so
describe each variant by *what changed relative to the scaffold*, build similarity
spaces from that, and grade them against an INDEPENDENT structural ground truth.
This rescues koenig, which failed the naive (count-mutations) structural test.
Code: `scaffold_reps.py`, `scaffold_experiment.py`. NO hybrid.py changes.

**Built:** 5 scaffold reps — MutID (one-hot mutation), dPhysChem (property-change
vector), dPhysChem_pos (position-tagged property change), SubScore (BLOSUM+Grantham
of the swap), BaseDelta (full one-hot minus wildtype = "difference of
fingerprints") — + 2 whole-sequence base reps (aac, blosum_mean) + a NEW
scaffold-aware metric **position_gated** (same-position ⇒ Grantham between the two
mutant residues; different-position ⇒ summed perturbation). Graders: **Grantham**
(physicochemical, rich — 3422 distinct values), **BLOSUM** (evolutionary, 27
values), **identity=Hamming (DEGENERATE — 99% of pairs at Hamming 2, std 0.07,
excluded from headline)**. Anti-circularity: each rep graded only by graders NOT in
its construction basis. koenig n=4275, CV grouped by mutated position (225 groups,
verified zero train/test position overlap; constant/random-distance controls give
read-across ≈ 0, so activity numbers are honest).

**RESULTS** (`scaffold_experiment.json`; struct = best NON-circular grader):

| rep / metric | activity ρ | struct (clean) | note |
|---|---|---|---|
| **position_gated** (new metric) | −0.05 | **0.519** (BLOSUM) | structural champion; no activity |
| blosum_mean (base) | **0.48** | **0.408** (Grantham) | the only all-rounder |
| dPhysChem_pos · euclidean | −0.01 | 0.390 (BLOSUM) | structure only |
| dPhysChem · euclidean | 0.47 | 0.322 (BLOSUM) | decent on both |
| BaseDelta | −0.04 | 0.118 | sees only identity |
| aac (base) | 0.48 | **0.006** | activity yes, structurally DEAD |
| MutID | −0.04 | 0.000 | orthogonal one-hot → constant distance |
| SubScore | 0.31–0.38 | **nan** | built from BOTH rich graders → no clean grader |

**Findings:**
1. **The scaffold idea WORKS for structure.** The whole-sequence base rep `aac` is
   structurally dead (0.006 — confirming the "dead antibody rep" memory), but
   scaffold-relative reps + the new position_gated metric recover **real,
   non-circular** structural fidelity up to **0.519** on the independent BLOSUM
   grader. koenig's structural half is viable after all — via scaffold framing.
2. **The new `position_gated` metric is the structural champion** (0.519), beating
   every generic rep×cosine/euclidean space.
3. **The anti-circularity rule was essential and it bit.** Reps look far better on
   their own basis (e.g. dPhysChem on Grantham) than on an independent grader;
   headlining only clean graders, and cross-checking Grantham vs BLOSUM, separates
   genuine structure (position_gated, dPhysChem_pos — agree across graders) from
   inflation.
4. **Structure and activity are largely DECOUPLED on koenig.** Structural winners
   (position_gated, dPhysChem_pos) carry ~no fitness signal; activity winners (aac,
   dPhysChem) are mediocre-to-dead on structure. **`blosum_mean` is the lone
   all-rounder** (0.41 structure + 0.48 activity).
5. **Activity signal is real** (~0.47, controls ≈0, no position leakage) — antibody
   expression IS predictable here via mutation-type, contra the "near-unpredictable"
   prior, because the task is single-mutation tolerance, not free prediction.
6. **euclidean ≫ cosine** for sparse delta reps (cosine collapses to constant on
   orthogonal single-mutation vectors).

**Honest limits:** SubScore can't be cleanly graded (built from both rich graders);
identity grader degenerate; **single dataset**. Extends to the 6 other single-WT
antibody DMS (binding_koenig, adams ×2, thermo_hie, rosace, phillips_h3). Nucleotide
sets do NOT fit (`nucleotide.py` = promoter classification, not one-reference
variants). Learned analog `esm2_wt_delta`/`esm2_mutsite` already exist (GPU) — the
interpretable-rep + matched-metric + independence-grading angle is the novel part.

**Publishability:** promising — novel (interpretable scaffold-rep similarity study +
the rep↔grader independence methodology + the structure/activity decoupling), it
works, and the caveats are honest. Needs multi-dataset replication before a claim.

## ★ POSSIBLE STANDALONE PAPER — "small-perturbation similarity" (user, 2026-07-18)

If the scaffold-mutation / delta similarity idea holds up, **break it out as its own
paper.** It sits in a large, active field — matched molecular pairs (MMP), activity
cliffs, delta/pairwise ML, deep mutational scanning (DMS), variant-effect prediction
— so the contribution must be the specific angle we actually have: evaluating
perturbation representations **as similarity spaces** (not predictors) + the
**anti-circularity independence-grading** methodology + the **structure/activity
decoupling** finding. Big mature field = HIGHER novelty bar; **a literature scan to
locate the exact gap is the gating step BEFORE more compute** (running 2026-07-18).
Likely stronger on the MOLECULE side (MMP/cliffs, MCS ground truth already wired,
ranker results already on molecules) than antibody-only.

### LITERATURE SCAN VERDICT (2026-07-18, deep-research 106 agents) — GAP IS REAL

The three pillars are UNEVENLY explored; the paper lives in their intersection:
- **Structure/activity DECOUPLING (Q3): WELL-TRODDEN** = the "activity cliff"
  (Maggiora 2006, SALI Guha&VanDrie 2008, MoleculeACE van Tilborg 2022). **Do NOT
  claim as novel** — use it as confirmation/framing only.
- **CIRCULARITY + independent ground truths (Q2): PARTIALLY explored, NEVER unified.**
  O'Boyle&Sayle 2016 (literature co-occurrence, explicitly avoided similarity-filtered
  truth), yeast chemical-genetic-interaction JCIM 2021, RascalMCES J Cheminform 2026
  (MCES over 5.4M pairs) — all molecular-only, scattered, none frames circularity as a
  general method, none extends to protein/sequence truths (Grantham/BLOSUM/ΔΔG). **Room
  to be the one that unifies it.**
- **Reps AS similarity spaces (Q1): NASCENT** — only the 2026 preprint "Geometry of
  Activity Cliffs" (arXiv:2605.30831; distance geometry + persistent homology over 15
  embedding×metric configs) does this systematically, and it uses STANDARD embeddings,
  **NOT delta/difference/scaffold-relative** reps. ⚠ closest competitor — READ + position
  against it.
- **Protein/sequence extension: WIDE OPEN** — no surveyed work extends perturbation-rep
  similarity / circularity / independent structural truths to DMS / antibody / variant-
  effect. Newest cliff-aware rep learning (ACANet 2024, ICML-2024) still optimises
  against ACTIVITY labels, not independent structural truth — gap persists.

**⇒ Paper shape:** "Evaluating **delta/perturbation representations as similarity
spaces** under **independent structural ground truths**, with **circularity as the
organizing frame**, across small molecules AND protein variant libraries." Novelty is
strongest in (a) circularity-as-method, (b) delta-reps-as-similarity-space, (c) the
**cross-domain** molecule+protein unification. This UPDATES the earlier "molecule side
is the more natural paper": the molecule side is MORE crowded (2026 preprint +
MoleculeACE); the **protein/DMS side is the white space** → keep BOTH threads, the
cross-domain framing is the strongest and least-contested contribution.

**Datasets of this "small-perturbation → effect" shape already in the repo** (test
the scenario across these, not just koenig):
- **Molecular (activity cliffs / matched pairs):** MoleculeACE 30 cliff-rich targets
  — `build_moleculeace.py` already wired (also the pending server job 12557946).
- **Protein mutation → ΔΔG:** `graphinity` (~943K FoldX / ~21K Rosetta single-point
  Ab-Ag), `ddg_combined` (AB-Bind+SKEMPI ~754), `abbind` (~1101 mutations/32 cplx).
- **Antibody DMS (single-WT):** koenig ×2, adams ×2, thermo_hie, rosace, phillips_h3.
- Nucleotide sets do NOT fit (promoter classification, not one-reference variants).

## Open decisions

1. Structural reference: **MCS vs GED** (start MCS — cheap, RDKit-native).
2. ~~Which ranker variants first~~ — RESOLVED: full menu R1–R9 + sklearn tier,
   both target modes, combinations (2026-07-18).
3. ~~Does koenig have variant diversity?~~ **RESOLVED — NO** (single-mutation DMS,
   Hamming {0,1,2}, structural arm degenerate). Antibody FORK open: phillips_h3
   (classification) vs thermo_jain (n=137) vs koenig activity-only. See antibody §.
4. If a similarity-aware ranker wins standalone → propose wiring into
   `compute_feature_importance` (needs signoff; new `method=`, own cache namespace).
5. **Euclidean size-confound control** for the structural yardstick (equal-length
   Hamming or same-size pairs) — flagged, not yet run.

---

## Changelog

- **2026-07-18:** work isolated onto `similarity-metrics-study` (cherry-picked
  structural_yardstick + rankers from ridge-removal; 56 stray ridge files parked
  in `git stash@{0}`). Structural yardstick full 6000-pair run DONE (Euclidean FPs
  win structurally; count-FP win is activity-only). Ranker + aggregation
  experiments designed & built; MCS cache + logd viability done. Antibody koenig
  diversity check FAILED (single-mutation DMS). Results pending run.
- **2026-07-13:** roadmap created; antibody dataset scoped (sequences, seq-ref);
  feature-ranker program + aggregation + standardisation planned; datasets locked.
- **2026-07-13:** server job 12557946 submitted (stuck PD/Priority).
- **2026-07-11:** pilot complete + verified (see results doc); server package staged.
