# Similarity Experiments (s01–s13) — Deep Verification & Fix Handoff

> **Paste everything below the line into a fresh Claude Code session in the KIRBy repo.**
> It is self-contained. It mirrors the process already done for the complexity (c01–c24)
> experiments. A separate chat is handling complexity fixes — you own **similarity only**.

---

## Your job

Deeply verify, then fix what is code-fixable, in the KIRBy **similarity** experiment suite
(`s01`–`s13`). The first batch of results is on disk. Some of it is real signal; a lot is
degenerate, run on datasets too small to be valid, or already known-broken. Your output:
(1) every claim re-verified from the raw JSON yourself, (2) code-level fixes for what's broken,
(3) real-data smoke tests for each fix, (4) updated analysis docs marking what's fixed vs pending.

This is a high-effort task. Be exhaustive and skeptical. Do not trust prior summaries —
**re-derive every number from the JSON before asserting it.**

## Hard constraints (from the user's standing rules — do not violate)

- **NEVER touch the server.** No ssh/scp/rsync/sbatch. The user deploys (`git push` → `git pull`
  on ARC → `setup.sh`) and runs all jobs. You only edit code locally. Your fixes do NOT change any
  result JSON until the user re-runs on ARC — say so honestly, never claim a result is "fixed."
- **No heavy local compute.** Import/syntax checks and small real-data smoke tests only. Never run
  `find_best_hybrid` or full ML workloads locally.
- **Anti-hallucination (the user has lost trust here before): re-grep / re-read before EVERY factual
  claim.** Write small Python to load the JSON and print exact numbers. Quote them. If you can't
  verify it, say "unverified," don't guess. Trust agent *breadth*, never agent *specifics*.
- **No random splits, ever.** All splits are scaffold/group-aware (`cv_split`, `groups`). Don't
  introduce a random 80/20 anywhere.
- **No mean/averaged baselines.** Report individual reps. (An "ensemble = average predictions" arm
  was deliberately removed elsewhere for this reason — don't re-add that pattern.)
- **No Ridge as a "sensitivity check"; no PCA-reduced rep variants** (importance filtering handles
  high-dim). These are explicit user prohibitions.
- **Keep SETUP.md in sync** with any dependency/setup change. Large files → `/Volumes/seagate`.
- Plain English in prose; spell out any abbreviation. Minimal output, no bloat.

## Orientation — where everything is

- **Result JSONs:** `tests/results/similarity/s01_similarity_baseline/<dataset>.json`, …, `s13_*`.
  (s09 cross-dataset-summary was NOT in this batch — no directory.)
- **Experiment source + docstrings:** `tests/experiments/similarity/s01_*.py` … `s13_*.py`, plus
  shared helpers `tests/experiments/similarity/similarity_helpers.py` and `tests/experiments/_shared.py`.
- **Read these two analysis docs first, in full:**
  - `COMPLEXITY_SIMILARITY_RESULTS_2026_06_02.md` — full results analysis (§2 and §6 are similarity).
  - `COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md` — the "what's broken" doc with a FIX STATUS table.
- **Smoke tests:** `tests/experiments/smoke_tests_broken_fixes.py` — already has MI + both cliff
  tests. ADD your new similarity smoke tests here; run with
  `python tests/experiments/smoke_tests_broken_fixes.py` (must stay green).
- You can load small real datasets locally with the `aac` rep (sequence-derived, no foundation
  model / no server data needed):
  ```python
  import sys; sys.path.insert(0, 'tests')
  from experiments._shared import load_data
  d = load_data('flab_polyreact_jain_bvp', rep_names=['aac'], rep_preset='fast')
  # keys: 'reps' (dict), 'labels' (ndarray), 'heavy_seqs', 'light_seqs', 'groups',
  #       'task_type', 'domain', 'augmentations'
  ```

## ⚠️ Validity gate — most datasets are too small (apply to EVERY conclusion)

5-fold CV on n<200 is essentially noise; sign-flips there are variance, not signal. n training rows:

```
94  ginkgo_heparin            217 flab_immunogenicity_marks   564 humanness_therapeutic (clf)
130 flab_pk_jain_heprt        239 ginkgo_titer                627 abagym_lysozyme
132 flab_pk_jain_fcrnrt       242 ginkgo_acsins               659 abagym_EGFR
137 flab_polyreact_jain_cic   242 ginkgo_colloidal            672 flab_thermo_tresanco
137 flab_aggregation_jain_hicrt 242 ginkgo_hic                3955 buchwald_hartwig (reaction)
137 flab_thermo_jain          397 abagym_Ang2                 5000 tz_her2_binary / multiclass (clf)
137 flab_aggregation_jain     418 abagym_VEGF                 ~   tcga_luad_vs_lusc (clf)
137 flab_polyreact_jain_bvp   429 ddg_combined
141 abcov_kd                  ----- below = n<250 (UNRELIABLE) -----
146 abagym_HER2               ----- powered (n>=~400): Ang2, VEGF, ddg, humanness, lysozyme,
                                    EGFR, tresanco, buchwald, tz_her2, tcga -----
```
Regression metric is Spearman ρ (higher better) unless noted. **Gate every claim on whether it
holds on the powered datasets, not the n<150 antibody ones.** The deep similarity experiments
(s06–s13) ran mostly on the small set — flag this.

## Already FIXED (do NOT redo — verify they still pass, then build on them)

These three were patched in this session's complexity work because they share code with similarity:

1. **MI estimator** (`_shared.py: estimate_mutual_information_knn`) — the old high-dim KSG collapsed
   to 0 everywhere (killed s03's MI mechanism). Replaced with a de-biased sklearn estimator +
   classification detection. **This unblocks s03.** Smoke-tested.
2. **s07 cliff detection** (`similarity_helpers.py: identify_cliff_compounds_editdist`) — absolute
   `max_edits=5` was degenerate (~0% cliffs on distinct-antibody data, ~100% on mutational-scan
   data). Added a distribution-relative `similar_percentile` (default 2%, closest X% of pairs).
   Smoke-tested. **Percentile default is a research knob — review prevalence after the rerun.**
3. **s12 cliff detection** (`s12_cross_domain_cliffs.py: identify_cliff_pairs_euclidean`) — absolute
   `sim>=0.55` found 0 cliffs in 13/16 datasets; same `similar_percentile` fix. Smoke-tested.

## The two GREEN similarity results (don't "fix" — these are the wins to protect)

- **s01 → s02 (CORE):** standard single-rep distances track activity poorly (sac_rho ~0.05–0.29,
  mostly non-significant); property-aware weighting **roughly doubles** the similarity-activity
  correlation, and this holds on powered data (humanness 0.289→0.515, buchwald 0.143→0.239,
  abagym_VEGF 0.098→0.226, tz_multiclass 0.077→0.189). Strongest positive result in the batch.
  Caveat to keep in mind: absolute correlations stay modest (~0.2–0.5), and the constructed
  "hybrid" `best_sac` is sometimes BELOW the simple property-weighted best single rep (`wtd`) —
  i.e. the weighted single rep is often the real winner. Verify this per dataset.

## What's wrong per experiment — verify each, then fix what's code-fixable

For EACH below: (a) read the `.py` docstring + metric code, (b) write Python to extract the exact
numbers across ALL its datasets, (c) confirm/correct the defect, (d) if code-fixable without server
or a user decision, fix it + add a real-data smoke test, (e) if it needs a rerun or a user decision,
mark it clearly. **Beware "by-design removals with stale docstrings"** — in the complexity suite,
THREE supposed "bugs" (c11 ensemble, c13 E−B, c19 raw-concat) turned out to be intentional removals
per user directive, mislabeled by earlier summaries. Read the surrounding source comments before
calling something a bug.

### s03_feature_concentration 🧪 — UNBLOCKED by the MI fix; needs rerun
- Tests WHY hybrid similarity works (MI-density ratio of selected vs random/PCA dims).
- Was degenerate ONLY because `mi_full=0` everywhere (the broken KSG). The MI estimator is now
  fixed, so on rerun the MI-density comparison should produce real values. The Gini-of-importance
  signal (~0.84–0.98) was the only thing that survived before.
- **Action:** confirm s03 calls the fixed `estimate_mutual_information_knn`; add a smoke test that
  the MI-density ratio (selected vs random dims) is now non-zero and ordered as expected on real
  data. No further code fix likely needed beyond the estimator — but verify the s03-side logic
  (selected/random/pca_match subsets) is sound.

### s04_activity_cliffs ☠️ — NEEDS USER DECISION
- All 23 files are 124-byte `{"skipped": true, "reason": "molecular-only experiment, domain=..."}`.
  The metric (ECFP4-Tanimoto cliffs) is molecular-only; no molecular dataset was in this batch.
- **Action:** this is a scope decision, not a bug. Either (a) add molecular datasets (caco2, herg,
  logd, qm9, drd2) to the s04 run list, or (b) formally mark s04 N/A for antibody/reaction/TCGA.
  Surface the decision to the user; don't silently pick one.

### s05_knn_prediction ⚠️ — selection-optimistic; weighting adds nothing
- kNN with hybrid distance vs best single-rep distance. Wins 17/22 overall, but powered gains shrink
  to ~0–0.07 (abagym_VEGF +0.049, tz_multiclass +0.066, humanness +0.004; **EGFR −0.004, Ang2
  −0.007 reverse**); the big wins are all n<200.
- **Two real defects to verify and address:** (1) "best hybrid" is the MAX over 6 budgets × 4
  conditions — post-hoc selection optimism. Consider reporting a held-out / nested-CV choice, or at
  least flag it. (2) the importance-WEIGHTED hybrid does NOT beat the unweighted hybrid (weight step
  is inert downstream). Verify both from the JSON; decide if a code change (honest selection) is
  warranted or if it's an analysis-only caveat.

### s06_applicability_domain ⚠️ — near-zero, inconsistent
- Does hybrid distance track prediction error? Distance–error Spearman ρ (k5, hybrid) is near zero
  and sign-inconsistent: abagym_VEGF −0.006, EGFR +0.006, Ang2 −0.108, tz_multiclass +0.246
  (verified). Hybrid gives tighter conformal intervals but lower coverage.
- **Action:** likely a genuine negative result, not a bug — but verify the distance–error correlation
  is computed correctly (sign convention, standardization) and that conformal PICP/MPIW aren't
  degenerate. If correct, it's a negative finding to report, not fix.

### s07_noise_bridge ✅ FIXED (cliff percentile) — needs rerun + prevalence review
- Already patched (distribution-relative cliff threshold). After the rerun, check that cliff
  prevalence is a sane minority on each dataset and that the contamination-rate / delta-vs-best-rep
  numbers are now meaningful (they were degenerate at 0%/100% before). Tune `similar_percentile` if
  prevalence is still off.

### s08_optimal_dimension ⚠️ — only 2 budget points
- Claims the optimal dimension differs for similarity vs prediction. But most datasets swept **only
  2 budgets (50, 500)**, so "optimal" is barely meaningful; `optima_differ` is 7 True / 9 False and
  the SAC ρ is ~0.1.
- **Action (code-fixable):** widen the budget grid so an optimum is identifiable (verify what grid
  the script requests vs what landed). This is a config/sweep fix you can make locally; it needs a
  rerun to take effect. Confirm the budget list in the source.

### s10_importance_downstream ⚠️ — weighting neutral-to-negative
- Does importance-weighted distance help GP/SVR/qRASAR? On powered sets it's neutral-to-negative
  (abagym_VEGF GP −0.005, Ang2 GP −0.015); the one big positive is n=137. **Watch for an RMSE-scale
  artifact**: buchwald "gp delta +13.077" / "svr +0.718" are RMSE-scale (lower-is-better) deltas
  misread as gains — verify the sign/scale handling in the script and FIX if the delta isn't
  normalized for metric direction (this IS a likely real bug).
- Cross-cutting: this + s05 show the property-weighting step adds nothing downstream — see tension
  note below.

### s11_bayesian_optimization ☠️ — NEEDS USER DECISION
- All 15 files are 121-byte `{"skipped": true, "reason": "domain=antibody"}` — molecular-only, same
  as s04. Same decision: add molecular datasets or mark N/A.

### s12_cross_domain_cliffs ✅ FIXED (cliff percentile) — needs rerun + prevalence review
- Already patched. After rerun, verify cliff pairs are now non-zero and a sane minority across
  datasets; tune `similar_percentile` if needed.

### s13_clustering_retrieval ⚠️ — clustering loses the majority
- Retrieval: hybrid wins 14/16 (small margins; VEGF/EGFR reverse). **Clustering coherence: hybrid
  wins only 5/16** — loses on most powered sets (EGFR 0.249 vs 0.381, VEGF 0.362 vs 0.393).
- **Action:** verify both metrics from JSON. Likely a mixed/negative finding to report honestly
  (retrieval modestly positive, clustering negative), not a code bug — but confirm the clustering
  metric (label coherence / k-means) is computed sensibly.

## Cross-cutting tension to resolve in the writeup (important for the paper)
In s05 AND s10 the importance-**weighting** step adds nothing downstream (weighted ≈ or < unweighted),
yet in s01/s02 weighting is exactly what doubles the similarity-activity correlation. Reconcile this
explicitly: weighting helps the SAC *metric* (how well the distance space orders activity) but not
downstream *prediction* (kNN/GP/SVR). State this clearly; it currently reads as a contradiction.

## Also check (shared infrastructure)
- Confirm s01–s13 that compute MI now import the fixed `estimate_mutual_information_knn` (grep).
- Check whether any similarity experiment has a stale docstring describing a removed arm (the
  complexity suite had three). Grep each `.py` for removed/deprecated comments before trusting the
  docstring.
- Watch every regression metric for the RMSE/lower-is-better sign trap (s10 buchwald is the known
  example). Any "delta" must be sign-normalized for metric direction.

## Methodology (follow the same loop used for complexity)
1. Map the grid: which datasets × which s-experiments produced output; flag empties/stubs by file size.
2. For the CORE claims (s01/s02 already done), and for each follow-up, extract exact numbers with
   deterministic Python — never eyeball, never trust a prior summary's specifics.
3. Classify each: ✅ real-on-powered / ⚠️ weak-or-reverses / 🧪 degenerate-metric / ☠️ dead /
   🔧 by-design (stale docstring). Read source comments before calling anything a bug.
4. Fix what's code-fixable WITHOUT the server or a user decision. For each fix: add a real-data
   smoke test to `tests/experiments/smoke_tests_broken_fixes.py` and keep the suite green.
5. For anything needing a rerun, add a self-diagnosing field to the output (the complexity suite did
   this for c16/c21/c24) so the rerun reveals the real cause without hunting slurm logs.
6. Update `COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md` (the FIX STATUS table + inline entries) and,
   if a finding changes, `COMPLEXITY_SIMILARITY_RESULTS_2026_06_02.md` §6. Be honest: code-fixed and
   smoke-tested ≠ verified; nothing is verified until the user re-runs on ARC.
7. Save a memory pointer if you discover something durable; don't collapse research into a checklist.

## Deliverables
- Code fixes (similarity only) + green smoke tests, each tested on a small real-data subset.
- Self-diagnosing fields for rerun-dependent failures.
- Updated BROKEN doc (status table + per-experiment) and RESULTS doc §6 where findings change.
- A short final summary: per experiment, the verdict (✅/⚠️/🧪/☠️/🔧), what you changed, and what
  still needs the user (s04/s11 scope decision; deploy + rerun; cliff-percentile tuning).
