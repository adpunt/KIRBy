# Hand-off prompt — "small-perturbation similarity" experiment suite (Thread B)

Copy everything below the line into a NEW chat. It is self-contained. It assumes the
KIRBy repo + the project memory, but NOT this conversation.

---

## What you are building

A **methodology**, tested across a variety of data (do NOT frame this as
"molecules vs antibodies" — the method is the unit; domain is just which data it is
tried on). The methodology:

Many important prediction problems are **small perturbations of a reference**: a
wildtype antibody with one point mutation, a protein mutation with a ΔΔG, a pair of
near-identical molecules (a matched molecular pair / activity cliff). Ordinary
whole-object representations collapse here — every variant looks ~99% identical, so
the fingerprint/encoding is nearly constant and useless. The idea is to represent
each variant by **what changed relative to the reference (scaffold)** — a
"scaffold-diff" representation — and then study those representations **as similarity
spaces**: do the pairwise distances they produce reflect true structural difference,
graded against an **independent** structural ground truth (never one built from the
same features — that circularity is the trap the whole study avoids).

This is a **separate experiment suite** that MIMICS the existing similarity
experiments (structural-yardstick / feature-ranker / aggregation) but focuses on
scaffold-diff representations and **distance metrics tailor-made for perturbations**.
Candidate home: a new experiment in the OPIG/curiosity suite, in the same vein as
`tests/experiments/opig_cdr_flexibility.py`. The general similarity experiments may
ALSO pick up the winning scaffold-diff reps as extra representations.

## Why this is worth doing (documented)

A 106-agent literature scan (2026-07-18) found the specific intersection is an OPEN
gap: delta/difference representations evaluated **as similarity spaces**, graded by
**independent** structural truths, with **circularity as the organizing frame**, and
extended to **protein/sequence** perturbations — none of the surveyed work hits it.
- Structure/activity decoupling for small perturbations = the known "activity cliff"
  phenomenon (Maggiora 2006, SALI, MoleculeACE) → do NOT claim as novel; use as framing.
- Closest competitor: arXiv:2605.30831 "The Geometry of Activity Cliffs"
  (Dabrowski-Tumanski et al.) — evaluates representations as similarity spaces, BUT
  uses STANDARD embeddings (Morgan/MACCS/MolFormer), molecules-only, circularity only
  implicit. Read + position against it.
Full verdict: `SIMILARITY_METRICS_ROADMAP.md` → "LITERATURE SCAN VERDICT" section.

## What has ALREADY been proven (koenig antibody DMS, n=4275) — build on this

Code: `tests/experiments/similarity_metrics/scaffold_reps.py` (reps + graders + the
new metric) and `scaffold_experiment.py` (orchestrator). Results:
`scaffold_experiment.json`. Write-up: `SIMILARITY_METRICS_ROADMAP.md` → "DONE —
scaffold-mutation representation investigation (koenig)". Commit `a26220a`. Memory:
`project_similarity_ranker_results_2026_07_18` (scaffold section).

**Representations that WORKED (carry these forward):**
- `dPhysChem` (vector of physicochemical property CHANGES caused by the mutation) with
  **euclidean** distance: structural fidelity **0.322** (independent BLOSUM grader) +
  activity read-across **0.47**.
- `dPhysChem_pos` (same, tagged by position): structural **0.390** (BLOSUM), ~0 activity.
- **`position_gated` — a NEW distance metric** (same mutated position ⇒ Grantham
  distance between the two mutant residues; different positions ⇒ summed perturbation):
  structural **0.519** (BLOSUM) — the best structural fidelity of anything tested;
  ~0 activity.
- `blosum_mean` (a whole-sequence base rep, kept as baseline): structural 0.408
  (Grantham) + activity 0.48 — the only "all-rounder". Not a scaffold-diff rep, but the
  bar to beat.

**Representations that UNDER-performed (refine or drop — needs more experimenting to
be sure):**
- `MutID` (one-hot of position×mutant-AA): 0.000 — one-hot vectors are orthogonal, so
  the distance is near-constant. Failed as-is.
- `BaseDelta` (full one-hot minus wildtype = "difference of fingerprints"): 0.118 —
  captures only Hamming identity, blind to substitution drasticness.
- `SubScore` (BLOSUM+Grantham scalar of the swap): CANNOT be cleanly graded — it is
  built from BOTH rich graders, so no independent grader remains. Design flaw to fix.

**Key methodological facts (reuse exactly):**
- **euclidean ≫ cosine** for sparse delta reps (cosine collapses to constant).
- **Three graders / independent structural ground truths:** Grantham (physicochemical,
  rich — 3422 distinct values), BLOSUM (evolutionary, 27 values), identity=Hamming
  (**DEGENERATE on single-mutation DMS — 99% of pairs sit at Hamming 2, std 0.07;
  exclude from any headline**).
- **Anti-circularity rule (the core method):** grade a representation ONLY with graders
  NOT built from its own basis (e.g. a physicochemical rep is graded by BLOSUM, never by
  Grantham). Cross-check across graders; where they agree the structure is real, where a
  rep scores high only on its own-basis grader it is circular inflation.
- **CV:** scaffold-grouped by mutated position (NO random splits — non-negotiable);
  verified zero train/test position overlap; constant- and random-distance controls give
  read-across ≈ 0, so activity numbers are honest.
- **Finding to carry:** structural similarity and activity/fitness are **decoupled** —
  structural winners (position_gated) have ~0 fitness signal; activity is a separate axis.

## Datasets you ALREADY have, and exactly what each needs to be integrated

Try the methodology on ALL of these (BOTH molecular and protein — methodology-first):

1. **Antibody DMS (single wildtype + point mutations)** — load via
   `kirby.datasets.flab.load_flab_dataset(name)`. Already returns sequences, labels,
   per-variant mutations, edit distances. `scaffold_reps.encode_sequences` already
   handles the encoding.
   - Ready to run through the EXISTING code: `flab_expression_koenig` (DONE),
     `flab_binding_koenig`, `flab_expression_adams`, `flab_thermo_hie`.
   - Need censored→classification handling (already binarised in the loader):
     `flab_binding_adams`, `flab_binding_phillips_h3`.
   - Too small, skip: `flab_binding_rosace` (n=14).
   - **Integration = run the other 6 through `scaffold_experiment.py`** (add the
     classification-label path for the censored ones). This is the cheapest next step.

2. **Protein mutation → ΔΔG** — loaders EXIST but are NOT yet wired to the scaffold
   format: `graphinity` (single-point Ab-Ag mutations, FoldX ~943K / Rosetta ~21K ΔΔG),
   `ddg_combined` (AB-Bind + SKEMPI ~754), `abbind` (~1101 mutations / 32 complexes).
   - **Integration = write an adapter** that turns each into (wildtype sequence/structure,
     mutation(s), effect=ΔΔG). ΔΔG is the "activity" axis; the structural ground truth is
     Grantham/BLOSUM of the mutation (already implemented in `scaffold_reps`). New code.

3. **Molecular matched pairs / activity cliffs** — `MoleculeACE` (30 cliff-rich targets);
   data builder `tests/experiments/similarity_metrics/build_moleculeace.py` exists, and
   the MCS structural ground truth is already wired (from the molecule similarity study,
   `structural_yardstick.py`).
   - **Integration = write the MOLECULAR scaffold-diff representation** (the analog of the
     mutation-diff: a matched-molecular-pair / transformation encoding — what atoms/bonds
     changed between two similar molecules). This does NOT exist yet. New work, and the
     part that connects to the crowded-but-established molecular-cliff literature.

## What the new experiment suite should contain

- **Scaffold-diff representations:** carry forward the winners (`dPhysChem`,
  `dPhysChem_pos`); refine or replace the under-performers (`MutID`, `BaseDelta`,
  `SubScore`) — more experimentation to decide which survive. Add a MOLECULAR
  scaffold-diff rep for MoleculeACE.
- **Tailor-made distance metrics for perturbations:** `position_gated` is the first and
  best; design more that are specific to this use case (e.g. substitution-weighted
  distances, position-kernel distances, transformation-aware molecular distances). This
  is explicitly where case-specific algorithms belong.
- **Experiments mimicking the similarity suite, but for scaffold-diff:** a structural-
  yardstick analog (rank reps×metrics by independent structural fidelity), a ranker
  analog, an aggregation analog — reusing the anti-circularity grading throughout.
- **Run across ALL the datasets above (both molecular and protein).** Let results decide
  which reps/metrics/datasets are the story.

## Constraints (locks — do not violate)
- Do NOT modify `find_best_hybrid.py` / `hybrid.py` without signoff (rebuild-roadmap lock).
- Scaffold-grouped CV, NEVER random splits. Baselines reported individually, never
  averaged. Anti-circularity grading is mandatory.
- Plain English in all explanations (spell out jargon). Document every experiment +
  result into a roadmap (do not let scratchpad-only results exist).
- Methodology-first framing; when asked "which domain first" the answer is BOTH.

## Where everything is documented
- `SIMILARITY_METRICS_ROADMAP.md` — sections: "WHERE WE ARE", "DONE — scaffold-mutation
  representation investigation (koenig)", "POSSIBLE STANDALONE PAPER", "LITERATURE SCAN
  VERDICT".
- Code: `tests/experiments/similarity_metrics/scaffold_reps.py`,
  `scaffold_experiment.py`; results `scaffold_experiment.json`.
- Commits: `a26220a` (koenig investigation), `df03789` (literature verdict),
  `2672813` (paper note + datasets), `fc45802` (re-orientation).
- Memory: `project_similarity_ranker_results_2026_07_18`,
  `project_similarity_metrics_study`, `feedback_methodology_first_framing`.
- Branch: `similarity-metrics-study`.

## First steps (cheapest → most novel)
1. Run the other 6 antibody DMS through `scaffold_experiment.py` (add classification
   path) — confirms koenig wasn't a one-off. Cheap; existing code.
2. Read arXiv:2605.30831 in full; write the precise differentiation.
3. Wire one ΔΔG dataset (graphinity or ddg_combined) into the scaffold format — the
   protein white-space pillar.
4. Build the molecular scaffold-diff rep on MoleculeACE — the molecular pillar.
Document results into the roadmap as you go, and STOP to discuss scope with the user
before large builds.
