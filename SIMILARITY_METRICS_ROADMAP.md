# Similarity-metrics roadmap (living doc)

**Owner:** Adelaide · **Started:** 2026-07-11 · **Last update:** 2026-07-13
**Point-in-time results:** `SIMILARITY_METRICS_RESULTS_2026_07_11.md` (pilot).
**Code:** `tests/experiments/similarity_metrics/`. **Branch:** `similarity-metrics-study`.

This is the master record. Every experiment and result goes here so nothing is
lost to a wiped scratchpad again. Sections: Goal → Principle → Done (with numbers)
→ In flight → Planned (local vs server) → Open decisions → Changelog.

---

## ★ WHERE WE ARE (re-orientation, 2026-07-18)

Everything below is committed on `similarity-metrics-study`; nothing is lost. Two
threads have emerged from the original goal:

**THREAD A — testing similarity metrics (the ORIGINAL purpose).** Experiments RUN,
but this is only ~STAGE 1 of the brief's staged plan, and the honest read is SOBERING
(NOT "done"):
- ECFP4:Tanimoto is mid-pack; count fingerprints beat it cheaply (pilot).
- Two honest yardsticks DISAGREE: by ACTIVITY, count-FPs win; by STRUCTURE (MCS),
  Euclidean-coefficient FPs win and count-FPs fall to rank ~30.
- **Ranker experiment — the sobering part:** on the ACTIVITY yardstick NOTHING
  meaningfully beat the crude count-FP baseline (herg 0.709→0.724 = +0.015; logd
  0.618→0.704 = +0.086, and that is desc/physchem, itself coarse). The big gains
  (structure-match 0.045→0.37) were only on raw shared-substructure (MCS), which is
  DECOUPLED from activity, so they buy NO activity signal. ⇒ **the activity-predictive
  signal in molecular similarity is SHALLOW; count-FP is a vague-range baseline (the
  user's reaction-dataset analogy) and the deep signal was NOT found.** (Bespoke
  rankers ≥ scikit-learn — supervisor's question answered — but that is a side point.)
- **Aggregation:** fusion does not beat the best single metric; hurts structure.
- **What was SKIPPED — I tunnel-visioned onto feature-ranking + aggregation.** The
  brief has whole untried families: learned/triplet metrics (Stage 3), the n-ary /
  iSIM / iCliff family, 3D/shape similarity, more fingerprints, and the downstream
  NOISE protocol (Deliverable 3). See **WHAT'S NEXT** below.
- (Parallel chat: complementarity selectors do NOT help the PREDICTION task —
  [[project_selector_prediction_negative_2026_07_18]]. That aside is CLOSED, negative.)

**THREAD B — a NEW idea that emerged: "small-perturbation similarity".** When the
antibody dataset (koenig) failed the naive structural test (all variants are 1-letter
tweaks), the fix — describe each variant by WHAT CHANGED from the wildtype scaffold —
turned into its own thing:
- **It works:** whole-sequence antibody reps are structurally dead (0.006) but
  scaffold-relative reps + a new distance metric recover real, non-circular structure
  (up to 0.519). Structure and activity are decoupled.
- **Literature scan (106-agent):** the specific intersection — delta/perturbation reps
  evaluated AS similarity spaces, graded by INDEPENDENT structural truths, with
  CIRCULARITY as the organizing frame, extended to PROTEIN/sequence — is a real, open
  gap. Closest competitor arXiv:2605.30831 uses STANDARD embeddings, molecular-only,
  circularity only implicit. **Thread B is a candidate standalone paper** and a new
  experiment for the OPIG/curiosity suite (same vein as `opig_cdr_flexibility.py`).

**OPEN — how the user wants Thread B tested + organized (not yet decided).** Datasets
ready (MoleculeACE molecular cliffs; graphinity/ddg_combined/abbind ΔΔG; 7 single-WT
antibody DMS). Scan says the molecular version is more CROWDED, the protein version is
WHITE SPACE. Methodology-first: try the method across the variety of data (BOTH), let
results decide. Hand-off written: `SCAFFOLD_DIFF_HANDOFF_PROMPT.md`.

## ★ ULTRACODE PHASE 1 — smoke results + a new avenue (2026-07-19)

Smoke-tested each WHAT'S NEXT avenue (one agent each, standalone script, herg_ki, vs
count-FP baseline `ecfp4_count:tanimoto`). Scripts: `triplet_similarity.py`,
`nary_similarity.py`, `shape_similarity.py`, `extra_fingerprints.py`, `creative_hybrids.py`.

★ **LEADS (worth the full run):**
- **Learned/triplet metric (activity mode) — the standout.** On HELD-OUT scaffolds a
  learned activity-distance beat the baseline on ACTIVITY read-across by **+0.24** (0.706 vs
  0.469). The ONE thing that clearly beat the vague baseline on activity. Costs structural
  fidelity. Small smoke (500 mols/1 fold) → confirm at full scale. MLP-triplet; a real
  graph MPNN is the deeper version (lib broken locally).
- **3D pharmacophore (pharm3d:euclidean) — beats baseline on BOTH yardsticks.** read-across
  +0.033, structural +0.188 (0.193 vs full-data baseline 0.066). Shape captures what 2D
  misses. USR/USRCAT beat on structure only. Conformers cached to `sim_herg_ki_spatial.npz`.

**HONEST NEGATIVES (saves effort):**
- **n-ary / complementary similarity:** complementary similarity is a WEAK outlier/cliff
  detector (below/tied naive baselines); the iCliff-roughness "win" is MECHANICAL; re-
  weighting does NOT help read-across. Not the answer.
- **Extra fingerprints:** Avalon beats on activity only (+0.018); MHFP6/SECFP6 beat on
  structure only (+0.10-0.14); none wins BOTH. MAP4 infeasible locally (no package).
- **Creative hybrids:** no block-stack beats count-alone on ACTIVITY (best −0.010); every
  hybrid CRUSHES structure (0.05→0.60). Control: most activity loss is the COEFFICIENT
  change (Tanimoto→z-euclidean), not the blocks. Stacking is a STRUCTURE lever,
  coefficient-bottlenecked on activity.

### ★ NEW avenue (user, 2026-07-19): FEATURE ABLATION — what HELPS vs HURTS similarity
Inspiration (user's parallel antibody work): the scaffold physchem rep had 5 property
scales; an EXHAUSTIVE 31-subset search found **charge and isoelectric point were SABOTAGING
it** — dropping them lifted structural fidelity on ALL three antibody datasets (koenig
0.387→0.513, adams 0.527→0.637, phillips 0.645→0.723). ⇒ some features actively HURT
similarity, and subset search finds them. Apply to molecular similarity: exhaustive subset
search over the hybrid BLOCKS (count / physchem / shape / fine-2D / keys) + drop-one greedy
WITHIN the physchem/`desc` block, scored on both yardsticks. → `feature_ablation.py`.

## ★ ULTRACODE PHASE 2 — full-scale results (herg_ki + logd, 2026-07-19)

CONFIRMED at full scale on BOTH datasets. Two leads (triplet, shape) had run hiccups
(phase-1 spatial cache was only 250 rows) and are re-running.

★★ **FEATURE ABLATION — the strongest real finding; confirms + extends the antibody
saboteur result.**
- **~40% of the RDKit `desc` descriptors SABOTAGE structural fidelity** — removing them
  RAISES it (88/208 herg, 78/208 logd).
- **Cross-dataset-consistent saboteurs: `fr_Ar_OH`, `fr_HOCCN`, `fr_nitrile`** (sparse
  functional-group counts) hurt structure on BOTH datasets — the molecular analog of
  charge / isoelectric-point sabotaging the antibody physchem rep.
- Block marginals: **physchem (`desc`) is THE read-across block** (beats count-FP on logd
  +0.048; ties on herg −0.006). **count-FP HURTS on logd** (read-across −0.050, structure
  −0.178). **`maccs` keys HURT structure on both.** `feature_ablation.py`.

**CREATIVE HYBRIDS (full):** logd — `all_blocks[coarse_down ×0.5]` beats count-alone on
BOTH yardsticks (activity +0.039, structure 0.387→0.606). herg — activity gains within
noise. The Tanimoto→z-euclidean coefficient swap alone HURTS (control), so the block gains
are real, not an artifact. Structure: every hybrid crushes count-Tanimoto (herg 0.045→0.46,
logd 0.387→0.65).

**NEGATIVES CONFIRMED:** n-ary / complementary similarity weak on both (re-weighting hurts
every metric, both datasets); extra fingerprints — none beat on read-across, SECFP6/MHFP6
beat on STRUCTURE only (+0.11–0.16 both), Avalon loses everywhere; MAP4 infeasible locally.

**TRIPLET (learned metric) — full-scale, 5 scaffold folds, both datasets. The smoke
OVERSOLD it.** activity-mode read-across: **logd +0.132** (0.629→0.761, every fold — real)
but **herg only +0.013** (0.701→0.714, a TIE — the smoke's +0.24 was a 500-mol/1-fold
artifact). And it is ACTIVITY-SUPERVISED (bakes in the labels) and craters structural
fidelity (logd −0.30). The label-FREE structural-mode triplet LOSES on both. So the only
learned lift is on logd (where the baseline was already weaker) and needs the labels.

**3D SHAPE (pharm3d) — full-scale. Also oversold by the smoke.** herg: STRUCTURE only
(pharm3d:euclidean struct +0.106, read-across −0.037 — does NOT beat baseline on activity
at full scale). logd: pharm3d:cosine beats baseline on BOTH (activity +0.034, struct
+0.052). usr/usrcat lose on activity. A structure lever + a modest logd-only activity win.

### ★★ THREAD A VERDICT (full-scale, both datasets) — the deep label-free activity signal is NOT there
Across ALL avenues, the pattern is stark: **every "activity win" is on logd (baseline 0.62,
weaker) and NOT on herg (baseline 0.70, strong).** Nothing — count/binary FP, extra FPs,
n-ary, shape, hybrids, or even a label-FREE learned metric — robustly beats the coarse
count-FP baseline on ACTIVITY across both datasets. The one thing that beats it on logd
(the activity-supervised triplet) requires supervising on the very labels, and still ties
on herg. **Conclusion: the coarse count-FP baseline is at/near the ceiling for label-free
activity similarity; there is no robust deep label-free activity signal to find — the
user's suspicion, confirmed and hardened.** (STRUCTURE is a different story: shape,
SECFP6/MHFP6, hybrids, structural rankers all beat count-FP on MCS fidelity — but structure
is decoupled from activity, so it does not help the activity/noise use-case.)

## ★ WHAT'S NEXT — the FULL menu from the original brief (undo the tunnel-vision)

Source: `~/Downloads/compass_artifact_wf-e835d62c…md` (Deliverables 1-3). Thread A so
far did only classical-FP metrics + feature-selection + fusion. Untried high-value
avenues, in roughly the brief's own Stage order — try any across the variety of data
(methodology-first, not one domain first):

1. **Learned / triplet metrics** (brief §1.4, §3.5, Stage 3) — the untested lever that
   could beat the vague baseline, because it LEARNS the distance instead of using a
   fixed fingerprint. `learned_embeddings.py` exists but was downgraded to an
   MLP-over-fingerprint (graph-NN lib broken) and only wired to the STUCK server job
   12557946 — **never run.** torch works locally → runnable now (MLP version); deeper
   option = hand-roll a real message-passing GNN in pure torch (reads the graph, not a
   fingerprint). ⚠ brief caution: learned embeddings usually only MATCH ECFP
   (arXiv:2508.06199).
2. **Extended n-ary similarity: iSIM / iCliff / complementary similarity** (brief §1.5)
   — completely ignored, cheap (O(N)), and directly relevant to finding deep signal /
   noise. `complementary similarity` = a molecule's impact on the whole set's similarity
   = an OUTLIER/mislabel detector; `iCliff` finds activity-cliff-generator molecules in
   linear time. MACCS-cosine sometimes beats ECFP4 in this analysis. Whole family
   untested.
3. **3D / shape similarity: USR / USRCAT / ElectroShape** (brief §1.3) — alignment-free
   shape; the brief says shape captures binding/cliff detail that 2D fingerprints MISS —
   i.e. exactly the fine signal the coarse baseline lacks. Conformer-dependent but cheap.
4. **Fill out the fingerprint grid** (brief §1.1-1.2): MAP4, MHFP6, Avalon, pharmacophore
   — cheap additions the pilot's fixed 10-rep set omitted.
5. **Proper statistics / unbiased benchmarks** (brief §2.2, §2.7): SRD+ANOVA, paired
   Wilcoxon, Bayesian Bradley-Terry across many targets; LIT-PCBA/MUV. (The stuck
   30-target server job was meant to provide this scale.)
6. **Deliverable 3 — downstream NOISE protocol** (ENN/Wilson editing, confident learning,
   roughness-based cliff-vs-noise). The brief's PRIMARY stance is downstream-noise, and
   it flags "no published paper applies ENN to ChEMBL/MoleculeNet label cleaning" = a
   genuine novelty gap. → user: ANOTHER CHAT.
7. **Structural / spatial AUGMENTATION features — ALREADY in KIRBy, never tried as
   similarity spaces.** `create_usr` (12-D shape moments), `create_usrcat` (60-D shape +
   pharmacophore), `create_pharmacophore3d` (`representations/molecular.py`) ARE the
   brief's §1.3 shape methods — buildable locally (3D conformers, ~mins/1400 mols). Plus
   KIRBy's augmentation-tier reps (small/structural supplementary features:
   `pharmacophore3d`, `all_geometry`, `shape`, `prostt5_3di`, `scaffold`, …). Structure
   is what similarity wants — these were skipped.
8. **Budget sweep for similarity-hybrids — NOT explored.** The ranker experiment FIXED
   per-rep budget at 100 features and only varied top-N (2 vs 4 reps). How the feature
   BUDGET (e.g. 50/100/250/500) changes a hybrid similarity space's quality is untested.
9. **★ Creative hybrid similarity spaces (user, 2026-07-19): mix count-FP + top selected
   features + spatial features.** Expanded below.

## ★ CREATIVE HYBRIDS — expanded plan (user direction, 2026-07-19)

**The intuition (the reaction-dataset analogy made concrete).** In a reaction dataset,
ligand-only features reach R²≈0.7 but only predict a vague general "bindability range" —
they get the coarse magnitude right and the specific protein signal not at all. Thread A
shows molecular similarity is in the same trap: the count fingerprint nails the coarse
activity RANGE (it mostly encodes size/composition) and nothing beats it, because the
FINE, specific signal is not in any single representation we tried. The idea: don't pick
ONE representation — **build a hybrid similarity space that stacks a coarse-range block
with blocks that carry the fine signal counts miss**, and see if the combination finds
what no single block can.

**Ingredient blocks (each z-scored, then concatenated; distance = cosine AND euclidean):**
- **Coarse range:** `ecfp4_count` — the working baseline; encodes gross size/composition.
- **Fine 2D structure:** top-N features selected by the STRUCTURAL-target ranker (from
  atompair / torsion / ecfp — the features that best track shared substructure/MCS).
- **3D shape / pharmacophore (the missing modality):** `create_usr` (12-D shape moments),
  `create_usrcat` (60-D shape + pharmacophore atom types), `create_pharmacophore3d`.
  These capture binding-relevant geometry that 2D fingerprints CANNOT see — the brief's
  §1.3 argument for where cliff/fine signal hides. Buildable locally (ETKDG conformers,
  ~mins for ~1400 mols); build once, cache next to the sim_*.npz.
- **Physicochemical:** `desc` (208-D). **Substructure keys:** `maccs` (167-D).

**Creative combinations to sweep (score each on activity read-across ρ AND structural
fidelity vs MCS, all vs the count-FP baseline):**
1. `count` alone — baseline to beat.
2. `count + desc` — coarse + physicochemical.
3. `count + USRCAT` — coarse + 3D shape/pharmacophore (the key new test).
4. `count + USR + pharmacophore3d` — coarse + pure geometry.
5. `count + top-structural-features` — coarse + fine 2D (no spatial).
6. `count + top-structural-features + USRCAT` — coarse + fine 2D + spatial (the full stack).
7. all blocks stacked.
8. **weighting variants:** equal-weight blocks vs block-weighted (down-weight the coarse
   block so the fine blocks are not drowned out — the count block is high-dimensional).

**Budget dimension (answers the untested budget question, item 8):** for the
top-features block, sweep the number kept — 50 / 100 / 250 / 500 — and for the spatial
blocks, whole-block vs top-k. Report how the hybrid's similarity quality moves with
budget; the ranker work only ever fixed this at 100.

**The question it answers:** does adding fine-2D and 3D-spatial structure to the coarse
count block recover activity signal the count block alone misses (i.e. beat the vague
baseline on the ACTIVITY yardstick, not just the structural one)? If yes → the deep
signal is a MULTI-BLOCK property, not in any single representation. If no → strong
evidence the activity signal really is shallow. Either outcome is worth having.

**Feasibility / cost:** spatial reps = one conformer-embedding pass (cache to
`sim_<ds>_spatial.npz`); the hybrid sweep reuses the existing z-score + cosine/euclidean +
read-across + MCS-fidelity machinery. All local, no server, no hybrid.py changes.

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

## DONE — multi-dataset extension of the scaffold study (2026-07-19)

Ran the remaining antibody DMS sets through `scaffold_experiment.py`
(now takes `--dataset`; koenig expression reproduces its committed numbers
EXACTLY — position_gated 0.519, dPhysChem_pos 0.390, dPhysChem 0.322,
blosum_mean 0.408 — so the parameterisation is clean).

### 1. The dataset roster is much smaller than the plan assumed

| dataset | n | task | muts/variant | single-mut? | usable |
|---|---|---|---|---|---|
| flab_expression_koenig | 4275 | regression | 1.00 | 100% | yes (done) |
| flab_binding_koenig | 4275 | regression | 1.00 | 100% | activity axis only — see below |
| flab_expression_adams | 5000 | regression | 1.75 (max 20) | 32% | structural only, confounded |
| flab_binding_adams | 5000 | classification | 1.76 (max 20) | 32% | structural only, confounded |
| flab_binding_phillips_h3 | 5000 | classification | 7.43 (max 14) | 0% | degenerate, see below |
| **flab_thermo_hie** | **10** | regression | 2.20 | 30% | **DEAD — n=10, not "ready to run"** |
| **flab_binding_rosace** | **14** | regression | 2.43 | 21% | **DEAD — n=14 (already known)** |

`flab_thermo_hie` was listed in the hand-off as ready to run; it is n=10. Correct
that expectation.

### 2. flab_binding_koenig is NOT an independent structural replicate

VERIFIED: `flab_expression_koenig` and `flab_binding_koenig` have **identical,
identically-ordered heavy-chain sequence lists** (n=4275). Same G6 variant library,
two different assays. Therefore every structural-fidelity column is identical BY
CONSTRUCTION (0.519 / 0.408 / 0.390 / 0.322 reproduce exactly) — it is a determinism
check, not a replication.

The only new information is the **activity (read-across)** column, and it is
substantially weaker on binding than on expression:

| rep × metric | expression ρ | binding ρ |
|---|---|---|
| blosum_mean × euclidean | 0.478 | 0.277 |
| dPhysChem × euclidean | 0.467 | **0.284** (now top) |
| aac × cosine | 0.482 | 0.253 |
| position_gated | −0.048 | −0.048 |

So: the structure/activity decoupling **replicates** (position_gated ≈ 0 on both),
but the "blosum_mean is the all-rounder" claim is assay-specific — on binding
everything collapses to ~0.27 and dPhysChem euclidean edges ahead.

**Net: there is currently NO independent structural replicate of the koenig
result from the antibody DMS sets.** Only adams and phillips are new sequence
libraries, and both are compromised — see below.

### 3. ★★★ NEW METHODOLOGICAL FINDING — the mutation-count confound

On the multi-mutation libraries the koenig ranking **inverts**: `BaseDelta` and
`MutID`, the two reps that FAILED on koenig (0.118 / 0.000), become the top
structural performers (adams 0.796 / 0.713; phillips 0.924 / 0.924).

This is not a win — it is a confound. The Grantham and BLOSUM ground truths are
computed as a SUM over positions, so on a variant library where the NUMBER of
mutations varies they are dominated by that count. Measured directly, correlation
of plain Hamming distance with each rich grader:

| dataset | ρ(Hamming, Grantham) | ρ(Hamming, BLOSUM) |
|---|---|---|
| koenig (1 mut, fixed) | 0.096 | 0.118 |
| adams (1.75 muts) | 0.706 | **0.796** |
| phillips_h3 (7.43 muts) | 0.870 | **0.924** |

Compare against the experiment output: `BaseDelta × euclidean` scores clean-best
**0.796** on adams and **0.924** on phillips — *exactly* the Hamming↔BLOSUM
correlation. BaseDelta euclidean IS Hamming distance up to a monotone transform.
Its entire "structural fidelity" is the count confound; it captures **zero**
substitution chemistry. On phillips its identity-grader column is 1.000 exactly,
confirming the space is precisely Hamming.

**The Hamming grader is degenerate at BOTH ends and this is the general rule:**
- single-mutation library → Hamming is CONSTANT (99% of pairs at 2) → carries no
  signal (already known, koenig);
- multi-mutation library → Hamming DOMINATES and **contaminates the other two
  graders**, so any rep that recovers mutation count scores high for free (new).

The anti-circularity framework as it stands checks rep-basis vs grader-basis. It
does **not** catch this, because the confound is in the DATA (variable mutation
count), not in the rep's construction basis. This is a real extension of the
paper's organizing frame, not merely a negative result.

### 4. Consequences

- `position_gated` is **undefined** for multi-mutation variants (it reads only
  `muts[i][0]`). Now explicitly SKIPPED below 99% single-mutation rather than
  silently scoring a metric that ignores all but the first mutation.
- The read-across CV grouping (`groups = first mutated position`) degrades badly:
  21 groups on adams, **13** on phillips where every variant mutates ~7 positions,
  so train/test share positions and the phillips read-across numbers (0.79–0.92)
  are leakage-inflated and should NOT be reported.
- Classification path added (`readacross_classification`): same scaffold-grouped
  GroupKFold + distance-weighted kNN as the regression path, scored AUROC
  re-centred to 0 so the two columns share an origin.

**Headline: koenig's structural result is so far UNREPLICATED, and the two
libraries that could have replicated it are confounded by mutation count.**

### 5. RESOLVED — all three grading options tried (user: "try all of them")

**(a) count-controlled grading (partial correlation) — WORKS. Adopt this.**
Spearman of rep-distance vs grader with the Hamming component removed
(`partial_spearman` in `scaffold_experiment.py`). It cleanly separates real signal
from the confound:

| rep × metric | adams raw | adams count-controlled | phillips raw | phillips count-controlled |
|---|---|---|---|---|
| BaseDelta × euclidean | 0.796 | **0.000** | 0.924 | **0.000** |
| MutID × euclidean | 0.713 | **−0.020** | 0.924 | **0.000** |
| dPhysChem_pos × euclidean | 0.701 | **0.527** | 0.778 | **0.645** |
| blosum_mean × euclidean | 0.467 | 0.267 | 0.406 | 0.228 |

BaseDelta/MutID collapse to exactly zero — they were pure mutation-count. The
physicochemical reps survive at reduced strength, i.e. they carry genuine
substitution chemistry. On koenig the correction changes nothing (0.519→0.513,
0.390→0.387), confirming there was never a confound there.

**(b) equal-mutation-count pair restriction — DOES NOT WORK. Rejected.**
BaseDelta stays at 0.788 (adams) / 0.923 (phillips). Reason, measured directly:
restricting to pairs whose members carry the SAME number of mutations does not
hold Hamming distance fixed, because two variants with equal mutation count can
overlap at zero or many positions. Within equal-count pairs, ρ(Hamming, BLOSUM) is
still **0.788** (adams) and **0.923** (phillips), with Hamming std 0.93 / 1.93.
The confound is the number of DIFFERING POSITIONS BETWEEN the pair, not the
per-variant mutation count.

**(c) ΔΔG replication on graphinity — WORKS. This is the replication.**
New script `scaffold_ddg_experiment.py`. graphinity is verified single-point only
(every row exactly one mutation), 58,564 mutations over 1,388 complexes. Because
it is multi-complex, pairs are drawn WITHIN a complex only (a cross-complex pair
would be graded on wholesale sequence difference, not on the mutation) and then
pooled — 1,310 complexes, **524,000 pooled pairs**. No per-complex averaging.
Confound check passes: ρ(identity-grader, BLOSUM-grader) = 0.303, versus 0.92 on
phillips.

| rep × metric | koenig (antibody, n=4275) | graphinity (protein ΔΔG, 524k pairs) |
|---|---|---|
| dPhysChem_pos × euclidean | 0.390 | **0.440** |
| dPhysChem × euclidean | 0.322 | **0.302** |
| BaseDelta × euclidean | 0.118 | 0.283 |
| MutID × euclidean | 0.000 | 0.023 |
| SubScore | no clean grader | no clean grader |

**The ranking replicates exactly** — dPhysChem_pos > dPhysChem > BaseDelta > MutID —
on an entirely independent dataset, different domain (protein–protein complexes,
not antibody DMS), different label type. euclidean ≫ cosine also replicates. This
is the multi-dataset replication the study needed, and it comes from the protein
pillar rather than from more antibody data.

Structural axis only: graphinity ΔΔG is FoldX/Rosetta-COMPUTED, not measured, so a
read-across number there would correlate against a simulator, not biology.

### 6. ★★★ RETRACTION — `position_gated` is the Grantham grader, not a metric

VERIFIED by direct comparison on koenig: `dist_position_gated` and the Grantham
ground truth are **bit-for-bit identical** (Spearman 1.000000, max absolute
difference exactly 0.0).

This is provable from the definitions. On a single-mutation library the
grader-weighted sequence distance between two variants reduces to: same mutated
position → M[mut_i, mut_j]; different positions → M[wt_i, mut_i] + M[wt_j, mut_j].
That is exactly, line for line, what `dist_position_gated` computes with
M = Grantham.

**Therefore the headline "position_gated achieves 0.519, the best structural
fidelity of anything tested" is NOT a result.** It is the correlation between the
Grantham grader and the BLOSUM grader — a number any method can hit by copying a
ground truth. It measures how much two graders agree, and functions as a ceiling,
not as a discovery. The existing code already flagged its Grantham column as
circular (`G = 1.000*`), but the BLOSUM column was reported as a genuine score.

**Do not cite the 0.519 as a metric result.** `position_gated` is excluded from
`scaffold_ddg_experiment.py` for this reason. The "designing tailor-made
perturbation metrics" thread needs a metric that is NOT a restatement of a grader —
and any future candidate must be checked against every grader for identity before
its score is believed.

### 7. Where this leaves the study

- **Real, surviving result:** `dPhysChem_pos` (physicochemical property change
  tagged by position) with euclidean distance is the best scaffold-diff
  representation, replicated across antibody DMS (0.390) and protein ΔΔG (0.440),
  and it survives count-controlled grading on the confounded sets (0.527 / 0.645).
- **Retracted:** the position_gated 0.519 headline.
- **Confirmed dead:** MutID, and BaseDelta (which is Hamming distance in disguise —
  its apparent multi-mutation wins are entirely the count confound).
- **Standing methodological contribution, now three-part:** grade a rep only with
  graders not built from its own basis (original); control for the
  differing-position count, since it contaminates graders that sum over positions
  (new); and check that a proposed metric is not algebraically identical to a
  grader (new, and the one that caught the biggest error).

Files: `scaffold_experiment_flab_binding_koenig.json`,
`scaffold_experiment_flab_expression_adams.json`,
`scaffold_experiment_flab_binding_adams.json`,
`scaffold_experiment_flab_binding_phillips_h3.json`,
`scaffold_ddg_foldx.json`.

## DONE — reps registered in the main pipeline + feature isolation (2026-07-19)

### A. `dphyschem` / `dphyschem_pos` are now BASE REPS in the main pipeline

Registered in `MUTATION_REPS` (`src/kirby/representations/mutation.py`), signature
`fn(heavy_seqs, light_seqs, mutations) -> (N, D)`. Verified equivalent to the
validated research code: `dphyschem` is bit-identical to `scaffold_reps.rep_dphys`
(max abs diff 0.0) and `dphyschem_pos` gives IDENTICAL pairwise distances (the
research version has extra all-zero columns for never-mutated positions).

Added to the `general` rep list of the 5 datasets with mutation metadata and a
single reference: `flab_expression_koenig`, `flab_binding_koenig`,
`flab_expression_adams`, `flab_binding_adams`, `ddg_combined`
(`tests/experiments/complexity/complexity_helpers.py`). `dphyschem` also gets
`REP_TIER_OVERRIDES['dphyschem'] = 2` so its low dimensionality does not demote it
to augmentation-only — the point is whether 3 chemistry numbers hold their own
against 1280D language-model embeddings.

NOT added, pending a decision: `abagym_*` (DMS, but their surveys exclude
mutation reps entirely) and `graphinity_*` (copies ddg_combined's list at
`complexity_helpers.py:665` — confirm it inherits). `flab_binding_phillips_h3`
stays cut (user, 2026-07-18).

⚠ **Cache:** reps are NOT in the search/result cache key. These 5 datasets' result
DBs must be deleted before re-scoring or stale rows computed over the old pool
will be served. See `REP_PRUNING_LOG.md`.

Relationship to the existing `mutation_features` rep: that one AVERAGES its
property deltas across a variant's mutations and discards position. Averaging is
precisely what fails on multi-mutation variants. These reps are the non-averaging
alternatives (sum, and position-tagged).

### B. ★★ Feature isolation — TWO OF THE FIVE PROPERTIES WERE HARMFUL

`scaffold_feature_isolation.py`. **Metric: Spearman rank correlation over variant
PAIRS, −1..1, higher better, graded by BLOSUM (independent of the physicochemical
basis), with the mutation-count component removed. No model is trained; this is
not R² and not accuracy.**

Drop-one analysis showed removing `charge` or `isoelectric_point` IMPROVED the
score on all three datasets. An exhaustive search over all 31 non-empty subsets
confirmed it:

| kept properties | koenig | adams | phillips |
|---|---|---|---|
| hydrophobicity + MW + aromaticity | **0.513** | **0.637** | 0.723 |
| hydrophobicity + MW | 0.473* | — | **0.765** |
| all five (as originally built) | 0.387 | 0.527 | 0.645 |

`hydro + MW + aromaticity` wins or ties everywhere, so the registered reps now
ship **three** scales, not five. Charge and isoelectric point measure overlapping
things and their noise dilutes the euclidean distance.

**★ This overturns the "loses on single mutations" finding.** With the junk
properties removed, `dphyschem_pos` scores **0.513** on koenig versus
`blosum_mean` at 0.408 — the scaffold-diff rep now WINS on the single-mutation
library too, not just the multi-mutation ones. Caveat, stated plainly: the
anti-circularity rule forces different answer keys on the two reps
(`dphyschem_pos` graded by BLOSUM, `blosum_mean` graded by Grantham), so this is
the study's standard clean-best comparison, not a same-yardstick one.

Individual property strength (koenig, solo): aromaticity 0.390, molecular weight
0.335, hydrophobicity 0.217, isoelectric point 0.081, charge 0.068. Aromaticity
alone matches the whole five-property rep.

### C. Is the signal a few positions or spread out? — DATASET-DEPENDENT

Drop-one-position on `dphyschem_pos`:

| dataset | mutated positions | mean damage per position | share from top 5 |
|---|---|---|---|
| koenig | 225 | 0.0023 | 20.3% |
| adams | 20 | 0.0352 | 90.1% |
| phillips | 15 | 0.0375 | 101.5% |

koenig spreads its signal thinly over 225 sites; adams and phillips concentrate it
in a handful. The >100% on phillips means some positions have NEGATIVE damage —
removing them HELPS, i.e. a few sites are actively harmful. That is a lead for a
position-selection variant of the rep, not yet built.

Files: `scaffold_feature_isolation.json`.

## ★★★ FORMALISED — the scaffold-diff similarity experiment suite (2026-07-19)

The scattered scripts are consolidated into ONE experiment in the experiments
suite, run across every scaffold-compatible dataset, with a publication-shaped
results document. This supersedes the ad-hoc runs recorded above.

**Experiment:** `tests/experiments/opig_scaffold_similarity.py` (OPIG/curiosity
suite conventions, same shape as `opig_cdr_flexibility.py`).
**Results doc:** `SCAFFOLD_SIMILARITY_RESULTS.md` — methods, all tables,
limitations, and a claim-status table. **THIS IS THE DOCUMENT TO WRITE FROM.**
**Machine results:** `tests/experiments/results/opig_scaffold_similarity.json`.

```bash
python tests/experiments/opig_scaffold_similarity.py --mode all          # full run
python tests/experiments/opig_scaffold_similarity.py --mode all --smoke  # ~1 min
python tests/experiments/opig_scaffold_similarity.py --mode features     # one mode
```

Three modes, one per scientific question:
- `representations` — which scaffold-diff rep × distance metric wins.
- `features` — what KIND of feature improves a distance-based similarity space.
- `controls` — C1 basis independence, C2 count control, C3 metric-is-not-a-grader.

Dataset registry `SCAFFOLD_DATASETS` (6: 5 antibody DMS + graphinity ΔΔG, both
single- and multi-scaffold) and `EXCLUDED_DATASETS` with reasons recorded in code
(hie n=10, rosace n=14, ddg_combined needs the multi-mutation generalisation).

### Headline results (all count-controlled, basis-independent)

1. **`dPhysChem_pos` × euclidean wins on 6/6 datasets, both domains** — koenig
   0.387, adams 0.527/0.507, phillips 0.645, graphinity 0.627. Position tagging
   beats position-agnostic summing everywhere; euclidean ≫ cosine everywhere.
2. **★ Feature study (generalisable beyond scaffold-diff):** from a 14-scale pool,
   greedy selection of 2–5 features beats all 14 on 5/6 datasets (up to +0.138) and
   beats top-k-by-solo-score on 5/6. Molecular weight has among the strongest solo
   scores on EVERY dataset (0.33–0.54) and is rejected on EVERY dataset because it
   correlates 0.89–0.98 with an already-chosen axis. **Non-redundancy, not
   individual signal, decides.** Z-scoring is worth +0.16 to +0.38. Mechanism: a
   euclidean distance has no weights, so correlated features double-count their
   shared axis and weak features add spread — a learner can down-weight, a distance
   cannot. Parallels [[project_selector_prediction_negative_2026_07_18]].
3. **Controls:** koenig and graphinity CLEAN (contamination 0.118 / 0.302);
   adams and phillips CONTAMINATED (0.796–0.924) so only controlled columns are
   interpretable there. C3 confirms `position_gated` ≡ Grantham ground truth
   (Spearman 1.000000, max diff 0.0) on both koenig sets.

### Claim status (full table in the results doc)
SUPPORTED: dPhysChem_pos×euclidean best (6/6); position tagging; euclidean≫cosine;
non-redundancy decides (5/6); z-scoring required (6/6).
RETRACTED: `position_gated` 0.519; `BaseDelta` multi-mutation wins (0.000 controlled).
NOT ESTABLISHED: scaffold-diff beats whole-object — still needs a same-basis
baseline on the protein data.

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
   **PARTIALLY RESOLVED 2026-07-18 — NO for the PREDICTION path.** The spinoff study
   `SELECTOR_PREDICTION_RESULTS_2026_07_18.md` applied R9 greedy-forward (re-derived as
   matching pursuit on `(X, y)`), R5 RReliefF, and mRMR to the standard prediction task
   on herg_ki + logd (5-fold scaffold GroupKFold, lightgbm, fixed budget, selection
   inside each training fold). **All 8 pooled comparisons LOSE to marginal
   `lightgbm_gain`** (best −0.0058, none significant). Likely cause: lightgbm already
   absorbs redundancy downstream, so complementarity buys nothing there — whereas in
   the similarity study the features feed a distance directly, where redundancy
   re-weights the metric. Still LIVE for the similarity path (R9/R5 did win there).
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
