# Similarity-metrics study — handoff for a new Claude Code chat

Paste this whole file into a fresh Claude Code session in the KIRBy repo. It carries
(1) what the study is, (2) exactly where the code and results stand, (3) what the
literature says and where the real open gap is, and (4) what to do next. Read the
linked files IN FULL before acting — do not work from this summary alone.

- **Repo:** `/Users/apunt/repos/KIRBy` · **Branch:** `similarity-metrics-study`
- **Master record (read first, in full):** `SIMILARITY_METRICS_ROADMAP.md`
- **Literature report (read in full):** `~/Downloads/compass_artifact_wf-e97ce77d-c674-5f49-ba05-d22d72e6f544_text_markdown.md`
- **Code:** `tests/experiments/similarity_metrics/`

---

## 1. What the study is, in plain terms

A **similarity metric = a representation + a comparison function** (e.g. ECFP4
fingerprint + Tanimoto coefficient). The field standard is **ECFP4 + Tanimoto**.
The question: can we build a metric that beats that standard?

"Beats it" is judged by **two independent yardsticks**, on the SAME molecules:

1. **Activity coherence (read-across).** Do molecules the metric calls "close" actually
   have similar *measured* activity (pKi / logD)? Measured by k-nearest-neighbour
   read-across Spearman correlation, with **scaffold-grouped** cross-validation (never
   random splits). Code: `simeval.eval_knn_readacross`.
2. **Structural fidelity.** Does the metric's distance track an *independent* structural
   truth — **maximum common substructure (MCS)**? Measured by Spearman(distance, MCS).
   Code: MCS pairs precomputed in `mcs_pairs_<dataset>.npz`.

The point of two yardsticks is **anti-circularity**: we grade against measured activity
or against MCS — never against another fingerprint. Grading a fingerprint by its
agreement with another fingerprint is circular and proves nothing.

**The ultimate purpose (still not done):** use a better similarity metric to **clean
noisy activity labels**. This is the actual research contribution and has been deferred
the whole time. See §4.

---

## 2. Where the code and results stand (verified)

**Datasets used locally:** `herg_ki` (real ChEMBL pKi), `logd` (real OpenADMET measured
LogD). Both real, not synthetic. `adme_solubility` was used by mistake in some runs; the
user's real local set is herg_ki, logd, **caco2, adme_hlm** — but caco2/adme_hlm won't
load locally yet (`load_data` hangs; rep cache is content-hashed). OPEN.

**Main finding — the "winning recipe."** A multi-block hybrid metric:
`count (ecfp4_count) + physchem (RDKit desc, with saboteur features dropped) + 3D shape
(pharmacophore3d) + fine-2D (top atompair/torsion features by MCS-correlation)`.

- It beats the count-FP baseline on **both** yardsticks. On structure that comparison
  rests on the two numbers in the next bullet, which no molecule-level held-out run has
  produced.
- **The 0.410 and 0.687 came from a PAIR split, which does not hold molecules out**
  (`winning_recipe_reselected.py`, 2026-08-11). The feature-selection and the structure
  scoring were split across disjoint halves of the 8000 cached MCS pairs. Each molecule
  sits in about eleven of those pairs, so nearly every molecule landed in both halves and
  the selection had already seen every molecule it was graded on; the per-feature z-score
  mean and sd came from all n molecules on top of that. So herg structure 0.410 vs
  baseline 0.036 and logd 0.687 vs 0.380 are pair-split numbers, not held-out-molecule
  ones. The file was rewritten on 2026-09-04 to split MOLECULES by whole Bemis-Murcko
  scaffold group and states that reasoning in its own docstring
  (`winning_recipe_reselected.py:7-12`), but it has not been run since —
  `winning_recipe_reselected.json` is not in the directory. **No held-out-molecule structure
  number has been measured**, and none is written here in place of these two. The activity
  gains (herg +0.017, logd +0.071) come from the same run; the feature selection reads no
  labels, so the activity half was never the circular one.
  ⚠ ADELAIDE: this paragraph is yours. The "circularity resolved" wording was removed on a
  read of the code on 2026-09-12, not on a new measurement.
- **Blunt truth on activity:** the *big* wins are all structural. On ACTIVITY, nothing
  robustly beats the crude count-FP baseline across both datasets — the deep label-free
  activity signal is shallow/absent. Structure and activity are **decoupled** (this is
  the activity-cliff phenomenon; see literature §3). The multi-block hybrid buys a
  *modest* activity gain (+0.02..0.07), and 3D shape is what buys it.

**Feature blacklisting ("saboteurs").** Some features actively HURT similarity; dropping
them raises structural fidelity. Molecular saboteurs consistent across both datasets:
`fr_Ar_OH`, `fr_HOCCN`, `fr_nitrile` (sparse functional-group counts). ~40% of RDKit
`desc` columns sabotage structure. The same pattern reproduces on antibodies (charge,
isoelectric point sabotage the antibody physchem rep) — the method transfers across
domains. Finder: `blacklist.py` (`identify_saboteurs`), demo notebook
`saboteur_finder_demo.ipynb`.

**Engine change (committed, needs server verification).** `find_best_hybrid` and
`hybrid.py` were extended with a similarity objective + feature blacklisting:
- `OBJECTIVE_SIMILARITY = 'similarity'`, `_is_geometry_objective()` gate,
  `_readacross_objective_score()`, `_structural_fidelity_score()`.
- New param `feature_blacklist: Optional[Dict[str, Sequence[int]]] = None`, applied only
  when set — the default `predictive` path is untouched.
- **Caveat:** runs locally ONLY with lightgbm args (`importance_methods=['lightgbm_gain']`,
  `internal_evaluators=['lightgbm']`, `models=['lightgbm']`,
  `candidate_ranking='lightgbm_gain'`). Default SHAP/treeshap importance SEGFAULTS on this
  Mac (exit 139, native crash) — unrelated to the objective/blacklist edit, which is
  verified to run on all 3 engine paths. Needs a clean server run to confirm.

**Key files in `tests/experiments/similarity_metrics/`:**
- `_data.py` — shared loader; `load_prepared(name)` returns y/groups/smiles + rep matrices.
- `mcs_cache.py` — builds `mcs_pairs_<ds>.npz` (I, J, mcs, dy; 8000 pairs).
- `winning_recipe_reselected.py` — the honest, circularity-fixed winning-recipe run. **Most
  important result file.**
- `creative_hybrids_v2.py` — the winning-recipe block algebra (helpers reused above).
- `blacklist.py` + `saboteur_finder_demo.ipynb` — saboteur finder + demo.
- `ranker_experiment.py`, `aggregate.py` — Experiments 1 & 2 (feature-ranking, fusion).
- Avenue scripts: `triplet_similarity.py`, `nary_similarity.py`, `shape_similarity.py`,
  `extra_fingerprints.py`, `feature_ablation.py`, `augmented_descriptor_reps.py`.

---

## 3. What the literature actually says (from the deep-research report)

Full report: `~/Downloads/compass_artifact_wf-e97ce77d-...text_markdown.md`. The specific,
cited points that matter for our direction:

- **ECFP + Tanimoto is the defensible default and hard to beat.** Multiple rigorous
  head-to-head benchmarks (Riniker & Landrum 2013; O'Boyle & Sayle 2016; van Tilborg /
  MoleculeACE 2022; Praski et al. 2025 arXiv:2508.06199, 25 models × 25 datasets,
  Bayesian Bradley-Terry) find pretrained neural embeddings and GNNs do NOT reliably
  beat ECFP for similarity; only CLAMP (itself fingerprint-based) beats it. **So don't
  expect learned embeddings to win — expect them to lose, and test them fairly anyway.**
- **Coefficient choice is basically solved / low-stakes.** Tanimoto ≈ Dice ≈ Cosine ≈
  Soergel (Bajusz 2015; Todeschini 2012). Don't agonise over the coefficient.
- **Counts vs bits matters more than the coefficient** — count / unfolded fingerprints
  improve structural fidelity (Huber & Pollmann 2026). Consistent with our finding that
  count-FP is a strong cheap baseline.
- **MoleculeACE (van Tilborg 2022) = 30 ChEMBL targets** (e.g. dopamine D3/D4, mu/kappa/
  delta opioid, thrombin, JAK1/2, orexin OX2R), 48,707 molecules; 24 approaches (KNN/RF/
  GBM/SVM × ECFP/MACCS/WHIM/physchem, plus ~7 deep models); metric = RMSE and RMSE on
  activity-cliff compounds. Finding: **classical ML on descriptors beat deep learning on
  cliffs.** This is the study our two-yardstick framing is closest to.
- **Structure/activity decoupling (activity cliffs) is well-established** — it is
  *diagnosed* extensively (SALI, SARI, MoleculeACE, iCliff, ROGI). It is only just
  starting to be *exploited* for metric design (a small 2024–26 triplet/contrastive wave).

**The two gaps that are genuinely OUR openings (report Part D):**

1. **Better metric → measurably cleaner labels: essentially unclosed.** ChEMBL noise is
   quantified (Kramer 2012: SD 0.54 pKi; Kalliokoski 2013: σ ≈ 0.69 pKi / 0.87 pIC50 for
   *paired differences*). Similarity-based label cleaning (edited-nearest-neighbour,
   confident learning) is standard in general ML but **nobody has cleanly closed the loop
   "adopt a better metric → clean noisy activity labels → measurably better data",
   benchmarked against that ~0.5-log noise floor.** This is the clearest publishable gap
   and is exactly the study's deferred purpose.
2. **Grading metrics against an independent structural truth (MCS) is rare.** Huber &
   Pollmann (J Cheminform 2026, "Count your bits") is the first to benchmark fingerprints
   against MCS/MCES at scale to *avoid circularity* — but on mass-spec compounds, NOT
   bioactivity targets. **Nobody has jointly graded metrics on BOTH MCS fidelity AND
   activity coherence on the same targets and asked whether the two axes agree — which is
   precisely our two-axis design.** That joint framing is novel.

**Caveats to respect:** Praski and "Geometry of Activity Cliffs" are non-peer-reviewed
preprints — strong but provisional. MCS is NP-hard / slow and can over-weight a shared
scaffold (a methyl→nitro swap barely changes MCS but can change activity a lot), so MCS
truth is necessary but not sufficient — always keep the activity axis alongside it. Cite
the ChEMBL noise numbers precisely (Kramer's 0.54 is a single-measurement SD; Kalliokoski's
0.69/0.87 are paired-difference sigmas — different quantities).

---

## 4. What to do next (open, user decides — do not self-direct)

The literature confirms the direction but sharpens where the value is:

- **A. Close the noise-cleaning loop (the real contribution, still not started).** On
  ChEMBL targets with replicate measurements, test whether the winning-recipe metric,
  used for edited-nearest-neighbour / confident-learning label cleaning, produces label
  sets whose residual noise approaches the ~0.5-log floor AND improves scaffold-split
  prediction — versus an ECFP-Tanimoto cleaning baseline. This is Deliverable 3 of the
  original brief and the clearest novelty.
- **B. Build the main-suite similarity experiment.** The user explicitly wants a
  similarity experiment integrated into the MAIN experimental suite (like the `c*`
  complexity / `opig_*` experiments), NOT another standalone script. This was blocked on
  picking a cleanest example experiment file to match conventions — ask the user which
  one.
- **C. Verify the engine change on the server** (the objective + blacklist edit), since it
  segfaults locally only due to the SHAP importance path.
- **D. Test learned embeddings fairly** (MolFormer / ChemBERTa / D-MPNN / MolCLR) with
  Bayesian Bradley-Terry — expecting them to lose, per the literature. Lower priority.
- **E. Thread B (antibody "small-perturbation similarity")** is a separate candidate paper
  with its own handoff: `SCAFFOLD_DIFF_HANDOFF_PROMPT.md`. Keep it separate.

---

## 5. Hard constraints (non-negotiable — from user memory)

- **No random splits — ever.** Scaffold-group-aware CV only.
- **Baselines are individual reps, never averaged.** No mean baselines.
- **No server access for Claude.** The user runs ALL server ops (ssh/scp/sbatch) and
  pastes output. JSON-only rsync. Never ssh/scp.
- **Never bulk scancel / rm / git reset** without triple-checking — the user lost a week
  of queue priority to a reckless scancel.
- **Reps are NEVER in the cache key.** Rep-list change → wipe + rerun; do not optimise for
  pool changes. No in-cache versioning.
- **Plain English only.** Spell out all jargon. No "poetic" writing.
- **The user is an engineer — no solo design.** Propose options, the user decides. Do not
  change gates / search space / find_best_hybrid without sign-off.
- **Local compute:** ML in the similarity study is sanctioned locally, but
  `find_best_hybrid` needs the lightgbm args above to avoid the SHAP segfault. No heavy
  local compute otherwise — import/syntax checks only.
- **Don't call a thread "done" after experiments.** Experiments are step one; the
  implementation change / decision is the point.
