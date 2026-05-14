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

### P4. PT-shortlist (PT-curated candidate set)
Run the PT search once. Pull the top-N best-performing blueprints out of its
`SearchResult` (the `best_by_model_budget` winners are a natural pool — sort by
score, dedup, take N). Evaluate those N blueprints on the FT set with normal CV
(`build_fold_hybrid` per fold — recipe transfer) and let FT pick the winner.

The `find_best_hybrid` search is a **grid search**, not an iterative optimiser,
so there is no "solution to warm-start from". P4 instead lets PT *narrow the
search space* — from 9×3×5×budgets down to N curated candidates — and FT chooses
among them.

**Pros:** Zero core-pipeline change — all existing machinery (`get_best_for_downstream`,
`build_fold_hybrid`, `Blueprint`). PT does the expensive search once.
**Cons:** FT can only pick from PT's shortlist; if the right recipe for FT isn't
in the top-N, P4 can't find it.

**Distinction from P1:** P1 freezes PT's *single* best transformation. P4 hands FT
PT's *top-N* and lets FT choose. P1 asks "does the best PT recipe transfer?"; P4
asks "does FT do better choosing from a PT-curated shortlist?"

---

## 5. Phase plan

### Phase 0 — Scaffolding (1 session)

Code lives in `tests/experiments/pretrain_rep/` (standalone experiment package).

- [x] **PT dataset registry** — `pt_registry.py`. `get_pt_dataset(domain, source, task, max_samples, exclude_keys)` returns a `PTDataset` (raw entities + labels + groups + dedup keys; `.as_tuple()` gives the `(X_pt, y_pt, groups_pt)` form). `X_pt` is raw entities, not a feature matrix — base-rep computation is a separate heavy step the Phase 1 harness owns. 8 real sources (5 molecular via `pretrain_corpora`, 3 antibody: `oas_species`, `ginkgo_gdpa1`, `sabdab`) + 2 offline `synthetic_*` sources for bug-catching.
- [x] **PT corpus quality pipeline** — `pt_quality.py`. Both layers mandatory; per-corpus criteria written down in `QUALITY_CRITERIA` (publication-facing) and emitted as a JSON `QualityReport`:
  - [x] Basic dedup hygiene — exact-duplicate removal (antibody: exact sequence; molecular: exact `(InChIKey, group, label)` row, so the same compound across targets and divergent measurements both survive to Layer 2), empty/invalid strings, alphabet check.
  - [x] Explicit quality filter — label-range sanity window, contradictory-measurement removal (drop `(compound, target)` clusters whose label spread exceeds a tolerance), sequence-length window, ambiguous-residue rejection.
- [x] **PT/FT leakage dedup** — `pt_leakage.py`. Molecular: InChIKey exact match (`inchikey_leakage_filter`). Antibody: `sequence_leakage_filter` — exact-sequence match is the default safe floor (`identity_threshold=1.0`); sub-1.0 thresholds are supported but emit a warning and are flagged `threshold_validated=False`, because the cutoff itself is still TBD (§9 research prompt Q6).
- [x] **PT base-rep computation bridge** — `pt_reps.py`. `compute_pt_reps(pt_ds, rep_names)` runs the *same* rep functions the FT pipeline uses (`ANTIBODY_REPS` for antibody, `get_molecular_representations` for molecular — confirmed by reading `_shared.load_data`: rep functions take raw entity lists, only the cache folder is dataset-named). Reuses `_shared`'s `_cache_rep` / `_cache_rep_dict` for an identical cache format. `pt_cache_key()` encodes source + task + a content hash of the ordered dedup keys, so any cap / quality / leakage change yields a fresh key (no stale-cache reuse). A `rep_fns` injection hook makes the orchestration offline-testable; the real foundation-model reps run on the server exactly as for FT.
- [x] **`find_best_hybrid` blueprint dump/apply audit** — DONE, findings below. P1 (freeze transformation) AND P4 (PT-shortlist) both need **no new core code** — the C5 path already covers them.

#### Phase 0 audit findings — blueprint dump/apply & P4

`find_best_hybrid.py` already exposes everything P1 needs:
- `Blueprint` dataclass with `to_dict()` / `from_dict()` — JSON round-trippable recipe (reps, allocation, importance, evaluator, budget, step_size, `rep_scores`).
- `apply_blueprint(base_reps, blueprint, labels)` — the product entry point; rebuilds a hybrid from a blueprint with no search.
- `get_precomputed_search(...)` → `get_best_for_downstream(search, model)['blueprint']` extracts a per-model `Blueprint` from a (cached) search.
- `build_fold_hybrid(train_reps, val_reps, y_train, blueprint)` (in `_shared.py`) applies a blueprint per CV fold — recipe-style transfer.
- `apply_feature_selection(base_reps, feature_info)` (in `hybrid.py`) applies exact feature indices — frozen-index transfer.
- **C5's `opig_blueprint_transfer.py` already chains all of these** for recipe and full-index transfer. **P1 reuses this directly — no core change.**

P4 (PT-shortlist): the search is a grid search (`SPEED_TIERS['full']` enumerates
9 allocation × 3 importance × 5 evaluator × budgets), not an iterative optimiser
— there is no solution state to "warm-start". Internally it has a two-stage
shape: Stage 1 scores each base rep on the dataset's internal CV folds
(`alloc_cache`) — the noisy small-n step — then Stage 2 grid-searches over
configs. P4 sidesteps both: PT runs the full search, the top-N `best_by_model_budget`
winners become a candidate shortlist, and FT evaluates that shortlist with
`build_fold_hybrid`. **No core change** — same machinery as P1.

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
| F | P4 PT-shortlist (FT picks from PT's top-N blueprints) |
| G | PT-only sanity (apply PT hybrid to FT *without any FT signal*) |
| H | Best single base rep on FT (rep-baseline reference) |

**Splits:** 5-fold scaffold/cluster splits — group-aware, non-random, same convention as the rest of the codebase. Rows shuffled before the group split runs (removes ordering artifacts; the split itself stays non-random). No random splits anywhere.
**Downstream models:** `rf, lightgbm, svm, mlp` — fixed for this experiment. These are standalone experiments, deliberately not bound to the codebase's auto-selection / size guardrails; this set was chosen on performance + runtime from molecular results.
**Job structure:** one SLURM job per FT target dataset. The expensive PT-side searches (P1 frozen hybrid, P3 joint hybrid) are computed once per PT source and reused across targets, so cost does not multiply by target count.
**Metrics:** R² + Spearman ρ for Ab regression; accuracy / F1w / AUROC for Ab classification.

**B vs G — the two ways to transfer a PT hybrid to FT (user asked to test both, 2026-05-14):** a hybrid run produces two things — a configuration (allocation method, importance method, budget) and the concrete selected feature indices that configuration produced on the data. **B** transfers the configuration and re-runs feature selection on the FT training data, so FT produces its own selected indices. **G** transfers PT's concrete selected feature indices and applies them to FT with no re-selection. The user was undecided which is better, so both are run as separate conditions.

**Harness:** `tests/experiments/pretrain_rep/pretrain_rep_experiment.py` — built and import/unit-tested. Implements conditions **A, B, F, G, H**; C, D, E (P2/P3) are deferred and explicitly not in the harness. `run(ft_data, pt_source, pt_task, ...)` does: prep PT corpus (quality → leakage → `compute_pt_reps`) → cached PT + FT hybrid searches → per-model per-fold evaluation of each condition. B uses `build_fold_hybrid` with the PT configuration; G uses `apply_feature_selection` with PT's exact indices (built once on the full PT corpus); F runs PT's top-N configurations B-style and picks the best on FT CV mean. `run()` itself is a server workload; the pure helpers are offline-tested in `smoke_test.py` (73 assertions, all passing).

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

## 6. Design decisions

§6.1–6.5 resolved with the user 2026-05-14. §6.6 (cross-species integration) descoped from v1.

### 6.1 PT data labels — RESOLVED

P1, P3, P4 pre-train by predicting a property, so the PT dataset needs a measured label. The largest antibody corpora (OAS, SAbDab) don't have one.

**Decision:** test all three approaches and compare — (a) GDPa1-only labelled PT, (b) stand-in labels from the sequence itself (germline distance, length), (c) contrastive-only (P2c) for unlabelled corpora. The pilot shows which are worth keeping.

### 6.2 PT data size — RESOLVED

The premise of PT is "more data," but the biggest corpora are slow and quality-suspect.

**Decision:** use fixed caps for v1, but test **several cap sizes** (not a single 50K) so we can see how the benefit scales with data volume. A full fine-grained sweep is deferred to a later version. The quality concern about large corpora is exactly why both quality layers in §6.3 are mandatory.

### 6.3 Leakage dedup + data quality — RESOLVED

If the same molecule/antibody is in both PT and FT sets, results are inflated. Separately, public PT corpora (ChEMBL etc.) aggregate noisy measurements from thousands of papers.

**Decision:** both quality layers are mandatory and the criteria must be **written down per corpus** (publication-facing):
- **Basic dedup hygiene** — remove exact-duplicate compounds/sequences and obvious junk.
- **Explicit quality filter** — keep only high-confidence assay flags, drop compounds with contradictory measured values, sanity-check value ranges/units.

PT/FT leakage removal: InChIKey match for molecular. Antibody sequence-identity cutoff still **TBD** — see §9 research prompt Q6.

### 6.4 P4 mechanism — RESOLVED (revised 2026-05-14)

Original plan was a "warm-start" API on `find_best_hybrid`. The Phase 0 audit
showed the search is a grid search, not an iterative optimiser — there is
nothing to warm-start. Revised to the **PT-shortlist** design (§4 P4): PT
produces a top-N blueprint shortlist, FT evaluates and picks among them.

**Decision:** build P4 as PT-shortlist in v1. Needs **no core-pipeline change** —
uses existing `get_best_for_downstream` / `build_fold_hybrid` / `Blueprint`.
Recipe transfer (re-select features on FT) is the default; frozen-index transfer
is a cheap optional second pass once the N blueprints are extracted.

### 6.5 Phase 1 scale — RESOLVED

The grid (~6 targets × 3 PT sources × 8 conditions × 5 folds × models) looked survey-sized as one monolith, but it isn't run that way.

**Decision:** run the full grid, **split into one SLURM job per FT target dataset** (normal practice here). Downstream models cut to `rf, lightgbm, svm, mlp` (§5 Phase 1). PT-side searches amortize — computed once per PT source, reused across targets. A 1–2 dataset pilot still runs first, purely to catch bugs before fanning out — not for compute safety.

### 6.6 How does the cross-species work fit in? — DESCOPED (2026-05-14)

Cross-species integration is **out of scope for PT-Rep v1**. An earlier draft of
this section proposed a P-flavor mapping for the cross-species T-PT.* tasks, but
that mapping was inferred from task *names* without reading
`CROSS_SPECIES_ROADMAP.md` — it is not reliable and has been removed to avoid
misrepresenting that experiment. The cross-species PT track stays on its own
roadmap. If sharing scaffolding ever makes sense, it needs a proper read of the
cross-species roadmap and a deliberate decision with the user first — not an
inferred mapping.

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
> I have four candidate "flavors" of PT-Rep already mapped out: (P1) freeze the hybrid transformation trained on PT data and apply to FT, (P2) learn a new base rep on PT and add it to the FT hybrid pool, (P3) train a single hybrid on pooled PT+FT data, (P4) run the search on PT, take the top-N best configs as a shortlist, and let FT pick among them.
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
- [x] User review of §6 design decisions (§6.1–6.5 resolved; §6.6 descoped from v1)
- [ ] Claude research prompt (§9) run and digested
- [x] Phase 0 scaffolding — registry + quality pipeline + leakage dedup + blueprint audit all done; P4 confirmed as PT-shortlist, no core change needed
- [~] Phase 1 antibody pilot — harness built + offline-tested (`pretrain_rep_experiment.py`, conditions A/B/F/G/H); not yet run on real data (needs server: PT rep precompute + hybrid searches)
- [ ] Phase 2 molecular replication
- [ ] Phase 3 analysis
- [ ] OPIG_COLLAB_ROADMAP.md cross-link entry
- [ ] Memory entry in MEMORY.md
- [x] Cross-species integration — descoped from v1 (§6.6); not pursued without a proper read of CROSS_SPECIES_ROADMAP.md
