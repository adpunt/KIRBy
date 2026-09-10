# Handoff: is the scaffold-diff representation publishable on its own?

## Goal (one line)
Establish whether a tiny physicochemical-change representation — "scaffold-diff": ~20
features encoding what each variant changed relative to its wild-type — rivals or beats
standard representations on multi-mutation variant-effect prediction, both as a standalone
representation and as features worth grafting onto other representations, across BOTH
antibodies and small molecules, with statistics a reviewer would accept.

This is NOT about the hybrid allocator's internals. The core claim under test is the
STANDALONE one: the ~20 features on their own outperform standard reps on multi-mutation
tasks. The hybrid/graft experiments are a second, separate result.

## What already exists — MEASURED, do not redo
- **Representation code:** `tests/experiments/similarity_metrics/scaffold_reps.py` —
  `encode_sequences`, `rep_dphys` (position-agnostic property change), `rep_dphys_positional`
  (position-tagged). numpy only. Lives on branch `similarity-metrics-study`, not main.
- **Comparison experiment:** `tests/experiments/opig_scaffold_hybrid.py` — builds the two
  reps in-file from the sequences `load_data` returns (row-aligned), injects them into the
  real rep pool, runs honest nested CV under cache tier `'scaffold_cmp'` (does NOT touch
  existing result DBs, no framework registration). Reads per-rep baseline means +
  per-model hybrid deltas.
- **Submit script:** `tests/slurm_scripts/submit_scaffold_hybrid.sh` (server, branch-aware pull).
- **Measured result (SLURM job 12830905, 4 antibody DMS datasets):**
  - `dphyschem_pos` (position-tagged) ranked **1st of 12 reps on all 6 models** on
    flab_binding_adams (AUC ~0.72–0.85 vs best embedding esm2_mutsite ~0.63–0.66).
  - 1st on 3/6 models (rf, mlp, svm) on flab_expression_adams (Spearman); weaker on
    boosters/knn.
  - LOSES on flab_binding_koenig (single mutation): ranks 7th–12th.
  - flab_expression_koenig: **EMPTY** — no models/baselines produced. Cause unknown;
    check the SLURM log.
  - Plain `dphyschem` mostly loses; the position tag is what carries it.
  - Read-out locations: standalone score = `baselines_by_model[model][rep]['mean']`;
    hybrid headline = `by_model[model]['delta_mean_of_folds']`.

## Known weaknesses to fix before any of this is publishable
1. **Only one independent multi-mutation regime.** The two koenig datasets share the same
   sequences (different labels) → not independent. koenig-expression failed. So the clean
   win rests on the adams pair alone. **Breadth is the load-bearing gap.**
2. **Degenerate folds.** Binding sets are binarised → AUC undefined on several outer folds
   (0–2 usable paired folds). Multi-mutation Spearman on tiny folds swings wildly. The
   per-rep baseline means are more trustworthy than the hybrid deltas; both need
   bigger/cleaner folds.
3. **Concept is not novel** (established 2026-08-28): DDGun (differences in hydrophobicity +
   substitution score between WT and mutant), AAindex-difference features, Gelman et al.
   PNAS 2021 (position-specific physicochemical featurization), matched molecular pairs /
   PADRE on the small-molecule side. The paper's claim is therefore COMPARATIVE — "cheap
   delta features rival protein-language-model embeddings" — never "new representation," and
   MUST benchmark against these named methods.

## Experiments to run (Adelaide's design — confirm scope with her first)
All via the existing standalone injection pattern (tier `scaffold_cmp`, no framework
registration), honest nested CV. Report `baselines_by_model[model][rep]['mean']` and
`by_model[model]['delta_mean_of_folds']`.

- **E1 — standalone, at breadth.** scaffold-diff alone vs each base rep vs best base rep,
  per model, across MANY independent multi-mutation datasets (antibody + molecule). The
  headline test.
- **E2 — grafted on.** scaffold-diff concatenated onto a standard representation. Does
  adding ~20 features to an embedding beat that embedding alone?
- **E3 — 2-rep hybrid.** Feature-selection hybrid over a pool of exactly {one standard rep,
  scaffold-diff}. Does it beat either alone, and do the scaffold features get picked?
- **E4 — molecular analogue.** Build the matched-molecular-pair delta descriptor; run E1–E3
  on molecular datasets with pair structure.

Two possible final results, both worth having: (a) the ~20 scaffold-diff features are useful
on their own; (b) the methodology of selecting the best features from a standard rep and
combining them with these 20.

## Research to do FIRST, in a dynamic multi-agent workflow
- **R1 — delta-feature baselines.** Enumerate existing difference/delta representation
  methods and which are runnable as head-to-head baselines: DDGun / DDGun3D,
  AAindex-difference encodings, mCSM-style signatures, matched molecular pairs / mmpdb /
  PADRE, position-specific physicochemical featurization (Gelman). For each: is there code,
  does it fit sequence-only antibody DMS, does it emit a per-variant feature vector we can
  drop into the pool. Output: a shortlist of baselines to implement.
- **R2 — trustworthy statistics.** What a reviewer accepts for "method A beats method B
  across many datasets and models with small folds": paired tests across datasets,
  critical-difference / Nemenyi diagrams, per-dataset win counts, effect sizes with
  confidence intervals, how to handle degenerate folds. Output: the exact statistical
  protocol to adopt.
- **R3 — breadth / datasets.** Enumerate independent multi-mutation antibody DMS datasets
  AND molecular matched-pair datasets, flagging which are already loadable in this framework
  vs need an adapter. Do NOT prune for compute. Output: the dataset roster for E1/E4.

## Decisions to put to Adelaide before running
- Which datasets go in the first breadth run (from R3).
- Molecules in scope for v1, or antibody-first with molecules as a second wave.
- Which single standard representation to pair in E2/E3 (strongest embedding per dataset,
  or a fixed one).
- Whether to fix the degenerate-fold / metric problem by changing the fold scheme or the
  dataset set.

## Hard constraints
- Don't prune datasets/axes/models for compute. Cuts need a stated reason: degenerate,
  broken, or undefined.
- Keep it standalone — inject into the pool, don't register into MUTATION_REPS/DATASET_REPS.
- Report `delta_mean_of_folds`, never `.all_results`/`search_mean`.
- Everything stays on branch `similarity-metrics-study`.
- Nothing local over ~3h; estimate wall time and route to a SLURM job.
- Adelaide decides; present options with numbers, don't design alone.
