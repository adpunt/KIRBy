# Pre-trained Representations Roadmap

**Status:** DRAFT (2026-05-14). Awaiting user review.
**Domain priority:** Antibody first (small-n pain point), molecular second.
**Relationship to existing work:** Adjacent to OPIG suite, not strictly OPIG-derived. Cross-referenced from `OPIG_COLLAB_ROADMAP.md`.

---

## 1. Motivation

KIRBy hybrids underperform on small datasets (n < 200). Root cause is well-characterised:
`find_best_hybrid` allocates feature budgets and selects importance-filtered slices using a noisy internal validation split. With ~22 internal-val samples (per F10 analysis), the search overfits to validation noise.

This is the same regime where deep models — in any other field — would use **pre-training**.

KIRBy has tried one form of PT already: **C4 (Imrie)** pre-trains the *downstream DNN* on ChEMBL, fine-tunes on target. That's PT of the **model**.

**This roadmap explores a different axis: PT of the representation itself.** The hybrid feature combiner — or the base reps it consumes — gets shaped by a larger related dataset before being applied to the small FT target.

---

## 2. Distinction from existing experiments

| Item | What's PT'd | What's frozen at FT time | Status |
|---|---|---|---|
| **C4 Imrie TL** | Downstream DNN weights | Reps are fixed (ECFP etc.); DNN body or all layers frozen | DONE |
| **C5 Blueprint Transfer** | *Recipe* (allocation/importance method, budget, evaluator) | Meta-params transfer; feature selection recomputed on target | DONE |
| **This roadmap (PT-Rep)** | The hybrid transformation and/or base reps | Hybrid feature indices, learned projections | NEW |

C5 says "the recipe generalises." PT-Rep asks "do the **specific learned features** generalise?" These are genuinely different claims — recipe transfer is about meta-learning; PT-Rep is closer to classical PT in the deep-learning sense.

---

## 3. Core question

> Does pre-training the representation (hybrid transformation and/or base reps) on a larger related dataset improve KIRBy performance on small FT targets, and does the choice of PT data matter more than the choice of PT method?

Sub-questions:
- (Q1) Does PT-Rep help where vanilla hybrid fails (small n) and stay neutral where it succeeds (large n)?
- (Q2) Which of the four PT flavors (§4) actually transfers?
- (Q3) Which PT dataset properties predict transfer success? Size? Task similarity? Modality match?
- (Q4) Is there negative transfer, and can we detect it before evaluating on FT?

---

## 4. PT flavors in scope (all four)

User explicitly asked for all of them — "I'm not sure if it will work at all but I would love to see what methods are beneficial."

### P1. Hybrid-weights PT (freeze transformation)
Fit `find_best_hybrid` on PT dataset with PT labels. Freeze the resulting transformation:
- allocation weights per base rep
- importance-filtered feature indices
- (optionally) the trained evaluator

Apply the frozen transformation to FT molecules → get FT hybrid features. Train downstream model on FT features only.

**Pros:** Hybrid sees orders of magnitude more samples for feature selection. Clean separation of PT (rep) and FT (downstream).
**Cons:** Hybrid optimises PT task, may not align with FT task.

### P2. Base-rep PT (learn a new base rep on PT)
Pre-train a learned representation on PT molecules:
- option (a) autoencoder over concat of existing base reps
- option (b) supervised projection (linear/MLP head trained on PT labels, return penultimate layer)
- option (c) contrastive (SimCLR-style) on molecules alone, no labels needed

Apply the trained encoder to FT molecules → new base rep. Add it to the FT hybrid search alongside ECFP/MACCS/etc.

**Pros:** Doesn't constrain hybrid structure; the hybrid gets to decide if the PT'd rep is useful. Closest to standard PT.
**Cons:** New training infrastructure. Most exploratory.

### P3. Joint PT+FT hybrid (pooled-data hybrid)
Fit one hybrid on the union of PT + FT, with task-aware sample weighting or multi-task labels.

**Pros:** Maximally simple. Tests whether *more data per se* > task alignment.
**Cons:** PT labels may dominate; FT signal may be drowned. Needs careful weighting.

### P4. PT-then-FT (warm-start hybrid)
Fit hybrid on PT (P1-style). Then re-run `find_best_hybrid` on FT initialised from PT's solution (warm-start allocations, importance scores).

**Pros:** Classical PT+FT semantics. Best of both worlds in principle.
**Cons:** Requires `find_best_hybrid` to accept a warm start. Implementation cost.

---

## 5. Phase plan

### Phase 0 — Scaffolding (1 session)
- [ ] Decide on PT dataset registry: a single Python module that exposes `get_pt_dataset(domain, source, task)` returning (X_pt, y_pt, groups_pt).
- [ ] Audit which existing `find_best_hybrid` internals expose enough state for P1 freezing (allocation, indices, evaluator). Likely needs a `dump_blueprint()` / `apply_blueprint()` pair — check whether C5 (`opig_blueprint_transfer.py`) already wrote these and reuse if so.
- [ ] Decide whether P4 warm-start is in v1 or deferred (most expensive to implement).

### Phase 1 — Antibody pilot
**FT targets (small-n where hybrid hurts most):**
- Ginkgo small subset properties (n ≈ 94–150)
- Other small Ab datasets that surface in F10 / EXPERIMENT_STATUS_PER_DATASET review

**PT candidates (need research §7 to refine):**
- OAS (Observed Antibody Space) — bulk, unlabelled, biggest
- SAbDab — structural Ab database
- Ginkgo full GDPa1 (n=242) — same property family, larger
- Therapeutic Ab corpora (IMGT/mAb-DB via BioPhi)

**Experiment matrix (per FT target × PT source):**
| Condition | Description |
|---|---|
| A | FT-only hybrid (baseline; current production) |
| B | P1 hybrid-weights PT |
| C | P2 base-rep PT, autoencoder variant |
| D | P2 base-rep PT, supervised-projection variant (if PT has labels) |
| E | P3 joint PT+FT hybrid |
| F | P4 warm-start hybrid (if Phase 0 ships it) |
| G | PT-only sanity (apply PT hybrid to FT *without any FT signal*) |
| H | Best single base rep on FT (rep-baseline reference) |

5-fold scaffold/cluster splits. Standard DOWNSTREAM_MODELS. Metrics by domain (R² + Spearman ρ for Ab regression, accuracy/F1w/AUROC for Ab classification).

### Phase 2 — Molecular replication
Same matrix on:
- FT targets: herg_ki (1.4K), small Polaris ADME, small flab_thermo splits
- PT candidates: ChEMBL bulk IC50, Polaris bulk, OpenADMET, ZINC (for unsupervised P2c)

### Phase 3 — Analysis (cross-cutting)
- (A1) Which condition wins per (FT, PT) pair?
- (A2) Does win-rate correlate with FT size? (expected: yes, PT helps more as n_FT shrinks)
- (A3) Does win-rate correlate with PT–FT task similarity? Define a similarity metric (label-correlation on shared molecules / Tanimoto distribution overlap / shared targets).
- (A4) Negative-transfer detection: can we predict ahead of time that a given PT source will hurt? (Distance from FT in some metric.)
- (A5) Does PT change rep ranking (C4-style analysis)?

### Phase 4 — Cross-modality stretch (deferred decision)
Does molecular ChEMBL PT help Ab targets? (Probably no, but cheap to check given Phase 2 infra.)

---

## 6. Open design questions (need user decision before Phase 1 runs)

1. **PT task-label requirement.** P1/P3/P4 need *labels* on PT. Antibody PT corpora are mostly unlabelled (OAS) or structural (SAbDab). Options: (a) restrict P1/P3/P4 to GDPa1-as-PT for Ginkgo-subset-as-FT; (b) use proxy labels (e.g., germline distance, length); (c) use only P2c (unsupervised contrastive) for unlabelled PT.
2. **PT dataset size cap.** ChEMBL bulk has 943K rows (per C4 work). Capping at e.g. 50K keeps compute tractable but throws away the main argument for PT (more data). Sweep across sizes?
3. **Holdout discipline.** PT and FT may share molecules / antibodies. Need an explicit dedup step (InChIKey for molecular, sequence identity threshold for Ab — what threshold?).
4. **P4 warm-start scope.** Implementing in v1 forces `find_best_hybrid` API changes. Defer to v2?
5. **Compute budget.** Phase 1 alone is ~6 FT targets × ~3 PT sources × 8 conditions × 5 folds × 7 models. Survey-scale (per [survey scope feedback]). May need to cut to "best single PT source per FT target" after a pilot.
6. **Cross-species track integration (added 2026-05-14).** The cross-species pretraining track (`CROSS_SPECIES_ROADMAP.md` §X-PT, tasks T-PT.1 through T-PT.8) is an instance of PT-Rep — but it was designed before this roadmap and doesn't commit to a P-flavor. Concretely:
    - `tests/experiments/cross_species/foundation_hybrid_vet.py` currently runs `find_best_hybrid` *only on the FT data* (vet ChEMBL targets, 27–1541 compounds). That's the **no-PT baseline** in PT-Rep terms, not a PT experiment. Its job in the cross-species track is "cheap anchor" — the cheapest thing the heavier T-PT.3/4/5/6/7 variants must beat.
    - `same_target_transfer` (T-PT.3), `rodent_to_vet` (T-PT.4), `species_token_multispecies` (T-PT.5), `microsome_transfer` (T-PT.6), `joint_multitask` (T-PT.7) each need to declare which P-flavor they instantiate before they're built. Mapping at first glance:
      - `same_target_transfer` ≈ **P1** (fit hybrid on human ChEMBL for same target, freeze, apply to vet).
      - `rodent_to_vet` ≈ **P1** with a different PT corpus (mouse/rat ChEMBL).
      - `species_token_multispecies` ≈ **P3** (joint hybrid on multi-species pool with species token as augmentation).
      - `microsome_transfer` ≈ **P1** (fit on human DLM/FLM, apply to vet microsome FT) — gated on X-Data.5/6 loaders.
      - `joint_multitask` ≈ **P3** strongest form (pooled molecular-only).
      - No T-PT.* maps cleanly to **P2** (learned base rep) or **P4** (warm-start) yet.
    - **Action:** before T-PT.3 starts, pick the P-flavor mapping explicitly. Either (a) confirm the mapping above and proceed with implementations that share PT-Rep utilities (blueprint dump/apply, dedup helpers), or (b) decide cross-species is its own thing and skip the shared scaffolding.
    - **Cross-link:** when the cross-species track gets there, `CROSS_SPECIES_ROADMAP.md` §X-PT entries should reference the chosen P-flavor explicitly and point back to this roadmap §4. The shared infrastructure (blueprint dump/apply, dedup helpers) lives here, not in the cross-species tree.

---

## 7. Research directions (what we don't know yet)

Before launching Phase 1 we need answers on:

1. **PT dataset taxonomy for antibodies.** What corpora exist beyond OAS/SAbDab/GDPa1? Sizes, label types, splits.
2. **Task-similarity metrics for PT-FT transfer.** Literature on predicting whether PT helps FT — what works for small-molecule property prediction? Does anything analogous exist for antibodies?
3. **Negative transfer in small-molecule QSAR.** Known cases where PT *hurt* on small downstream sets. What were the diagnostic signals?
4. **Comparable prior work — "pre-training feature combiners."** Is anyone publishing on PT'ing the feature-selection stage rather than the encoder? Closest priors: meta-learning (MAML on tabular), AutoML transfer, hyperparameter transfer (PFNs).
5. **OAS / SAbDab access mechanics.** Sequence dedup thresholds, splits used by existing benchmarks, licensing.
6. **Molecular PT corpora beyond ChEMBL.** OpenADMET coverage, Polaris bulk usage rights, ZINC subsets for contrastive PT.

A Claude-research prompt to fill these gaps is in §9.

---

## 8. Success criteria

This roadmap is "successful" if Phase 1 + Phase 2 produce a publishable answer to any of:
- **Positive:** "PT of the hybrid transformation reduces error on small datasets by X% when PT source satisfies Y."
- **Negative:** "PT of the hybrid transformation does **not** help once n_FT > Z, and on smaller targets only flavor Q works." (Also publishable — informs when to stop trying.)
- **Conditional:** "Flavor P2 (learned base rep) is the only consistent winner; flavors P1/P3/P4 fail because of [diagnosed cause]."

Failure mode to avoid: running all 4 flavors × all PT sources without a hypothesis about *why* each should work. Phase 0 design review must surface a mechanistic prediction per flavor before SLURM submission.

---

## 9. Claude research prompt (drop into a fresh session)

> I'm scoping a representation pre-training experiment for a molecular/antibody ML pipeline called KIRBy. KIRBy combines multiple base representations (fingerprints, descriptors, foundation-model embeddings, etc.) into a single "hybrid" representation per dataset by running a feature-selection + allocation search optimised against the dataset's own labels. The hybrid is the *input* to a downstream regressor or classifier; the downstream models are standard (RF, SVM, LightGBM, MLP, DNN, etc.).
>
> Hybrids underperform on small datasets (n < 200) because the internal validation split is too noisy to drive feature selection reliably. I want to test whether *pre-training the hybrid representation* on a larger related dataset can recover performance — analogous to how foundation models pre-train, but applied to the feature-combination stage rather than to an encoder.
>
> I have four candidate "flavors" of PT-Rep already mapped out: (P1) freeze the hybrid transformation trained on PT data and apply to FT, (P2) learn a new base rep on PT and add it to the FT hybrid pool, (P3) train a single hybrid on pooled PT+FT data, (P4) warm-start the FT hybrid search from a PT solution.
>
> Please research and report on the following, with citations:
>
> 1. **PT corpora for antibodies.** Beyond OAS (Observed Antibody Space), SAbDab, and GDPa1, what bulk antibody datasets exist that could serve as PT sources for downstream tasks like developability, humanness, affinity, polyspecificity? For each: size, sequence type (paired/unpaired, VH-only/VH+VL), label availability, license, and what existing benchmarks have used them as PT corpora. Particular interest in datasets > 10⁴ sequences with at least *some* labels (any property) so they can serve as supervised PT for our P1/P3/P4 flavors.
>
> 2. **PT corpora for small molecules (beyond ChEMBL bulk IC50).** What are the standard PT sources for small-molecule property prediction? OpenADMET coverage and licensing. Polaris bulk leaderboards. ZINC subsets used in contrastive pre-training (e.g. MolCLR, GraphMVP). Anything more recent (2024-2026) that has displaced these.
>
> 3. **Task-similarity metrics that predict PT→FT transfer success.** What does the literature say about predicting *ahead of time* whether a given PT dataset will help a given FT task? I'm specifically interested in: (a) molecule-level similarity metrics (Tanimoto distribution distance, scaffold overlap), (b) label-level similarity (correlation on shared molecules, label-distribution shifts), (c) embedding-space distance (e.g. CKA on frozen-encoder embeddings of PT vs FT molecules). Has anyone shown these predict transfer well enough to skip negative-transfer experiments?
>
> 4. **Negative transfer in small-molecule QSAR / antibody property prediction.** Documented cases where pre-training *hurt* performance on the downstream task. What were the diagnostic signals (size mismatch, task mismatch, distribution shift)? Is there a "PT helps when..." rule that's emerged?
>
> 5. **Pre-training the feature-selection stage specifically.** I'm pre-training a feature *combiner*, not an encoder. Closest priors I can think of: (a) hyperparameter transfer learning / PFNs (Müller et al.), (b) AutoML transfer (e.g. Auto-Sklearn warm starting), (c) meta-learning on tabular data (MAML variants), (d) "blueprint transfer" literature (if any). Is there published work on transferring *which features to use* across datasets, as opposed to transferring encoder weights? If yes, what worked and what didn't?
>
> 6. **Antibody sequence-identity thresholds for PT/FT dedup.** When using OAS or SAbDab as PT and a small Ab benchmark as FT, what sequence-identity threshold should we use to ensure no leakage? Standard practice in protein PLM literature (e.g. 30% / 50% / 95%)? Does CDR-only identity matter more than full-VH identity?
>
> 7. **Compute-budget realism for PT runs.** Rough cost estimates for PT runs in (a) — autoencoder PT on 50K molecules of concatenated base reps, (b) supervised PT of a small MLP on 50K labelled molecules, (c) contrastive PT (SimCLR-style) on 100K molecules. Single-GPU wall-clock order of magnitude.
>
> 8. **Anything I'm missing.** Recent papers (2024–2026) on the general theme of "improving small-dataset model performance via representation pre-training in chem/bio ML" that I should know about before designing experiments.
>
> Deliver as a structured report with one section per question, ~150–400 words per section. Prefer 2024–2026 citations where available. Don't pad — if a section is "no relevant prior work," say so. End with a 5-bullet "what I'd do if I were running this experiment" recommendation.

---

## 10. Status checklist

- [x] Roadmap drafted (this doc)
- [ ] User review of §6 open design questions
- [ ] Claude research prompt (§9) run and digested
- [ ] Phase 0 scaffolding
- [ ] Phase 1 antibody pilot
- [ ] Phase 2 molecular replication
- [ ] Phase 3 analysis
- [ ] OPIG_COLLAB_ROADMAP.md cross-link entry
- [ ] Memory entry in MEMORY.md
- [ ] Cross-species T-PT.* ↔ P-flavor mapping confirmed (§6 Q6); CROSS_SPECIES_ROADMAP.md updated to reference P-flavors
