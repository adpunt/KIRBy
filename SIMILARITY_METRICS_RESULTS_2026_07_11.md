# Similarity metrics & how to evaluate them — investigation

**Date:** 2026-07-11 · **Status:** local pilot complete + verified; server package
staged (not run); **no pipeline changes made**.

**Question (yours):** build a representation whose *pairwise comparison*
(Tanimoto/cosine/…) tells you whether two molecules are genuinely similar — **not
predictive performance** — because you suspect the field standard, ECFP4 +
Tanimoto, is not actually good at this. These experiments test that suspicion,
test alternatives, and — because label-noise mitigation is a major use case —
test each metric's value for **noise detection and recovery**. Methods follow the
research brief (`compass_artifact_wf-e835d62c…`, Deliverables 1–3).

## The idea that makes this non-circular

"Are two molecules really similar?" is answered by their **measured activity**,
never by ECFP4 Tanimoto. A metric is good if the molecules it places close
together behave alike. ECFP4 Tanimoto competes as **one metric among 47, judged by
the same activity yardstick as everything else — never the judge.** (Judging by
agreement-with-ECFP4 was the circularity the first attempt fell into; excluded
here, kept only as a labelled diagnostic `ecfp_agree`.)

## What was run (all local, verified)

- **Datasets (2 domains):** hERG Ki binding (n=1415, pKi) and LogD lipophilicity
  (n=5039). Each capped to a 1400-molecule **scaffold-grouped** subsample (whole
  Bemis–Murcko scaffolds; no random split, no leakage).
- **47 metrics** = {ECFP4, ECFP6, FCFP4, atom-pair, torsion, RDKit, MACCS,
  ECFP4-count, atom-pair-count, descriptors} × {Tanimoto, Dice, Cosine, Soergel,
  Tversky, Euclidean} (coefficients applied where meaningful).
- **Evaluation battery (9 methods, all activity-grounded), each cited:**
  kNN read-across / continuous-MODI (Golbraikh 2014), MODI (Golbraikh 2014),
  Patterson neighbourhood-behaviour (1996), SALI cliff-AUROC (Guha & Van Drie
  2008), BEDROC enrichment (Truchon & Bayly 2007), violation rate,
  **noise detection AUROC** and **noise recovery corr** (compass §3.4; recovery =
  corr(applied correction, −injected ε), the same validity metric as
  `kirby.utils.noisy_labels.noise_recovery_corr`).
- **Deferred to the server (compass Stage 3):** learned/triplet-MGNN embeddings
  (ChemBERTa/MolFormer + a triplet metric-learner — the "molecules as triplets"
  idea) and the full **MoleculeACE 30-target** battery with paired Wilcoxon/SRD.
  Code staged under `tests/experiments/similarity_metrics/` (see its README).

`simeval.py` passed a planted-signal self-test on every method; Tanimoto matches
RDKit exactly; binary Soergel ≡ Tanimoto (so it is dropped for binary FPs).

---

## The headline: ECFP4 Tanimoto's rank is *task-specific*

Rank of **ECFP4:Tanimoto** among the 47 metrics, per evaluation (1 = best):

ECFP4:Tanimoto rank of 47 — **mean ± SD over 5 scaffold-grouped subsamples**
(single-seed ranks in brackets show how misleading one draw can be):

| evaluation | hERG Ki | LogD | reading |
|---|---|---|---|
| **kNN read-across** | 27 ± 5 | 30 ± 4 | **weak & STABLE** — mid-pack every draw |
| **noise detection** | 20 ± **16** | 29 ± 9 | mid, but **UNSTABLE** (swings 6↔41) |
| **noise recovery** | 15 ± 7 | 25 ± 9 | mid, moderately stable |
| BEDROC enrichment | [15] | [21] | mid (single seed) |
| MODI | [3] | [42] | wildly task-dependent |
| cliff detection* | [12] | [8] | mid (*ECFP-anchored eval) |

**The honest, corrected picture (multi-seed):**

1. **Your suspicion holds for read-across / retrieval, and it is ROBUST.** ECFP4
   Tanimoto is reliably mid-pack (27±5, 30±4); ~21 of 47 metrics beat it on both
   datasets every draw. Δρ ≈ 0.02–0.09.
2. **For noise — your major use case — ECFP4 Tanimoto is ALSO mid-pack, not
   near-best.** An earlier single-seed run put it 5th–8th on noise; that was a
   lucky draw. Across 5 subsamples its detection rank is 20±16 on hERG — it swings
   between 6th and 41st — because the metrics are **nearly tied** on noise
   detection (AUROCs in a narrow band), so tiny differences reshuffle the ranking.
   **No metric, including ECFP4, is decisively best at noise detection here** — and
   that is itself the finding: on 2 datasets you cannot separate them; the
   30-target run is required.
3. The metrics that *do* edge ahead (when the ranking holds still) are **count
   fingerprints and FCFP4**, by small margins — not exotic ones.

### kNN read-across detail

| | ECFP4:Tanimoto | best alternatives (ρ, ECFP-agreement) |
|---|---|---|
| hERG | 0.684 (32nd) | atompair:euclid 0.732 (0.27); atompair:cosine 0.724 (0.45) |
| LogD | 0.598 (29th) | desc:cosine 0.688 (0.49); desc:euclid 0.674 (0.43); fcfp4 0.650 (0.74) |

**Honesty check (a claim I had to correct):** "the less a metric resembles ECFP4
Tanimoto, the better" is **false** as a global trend — Spearman(ECFP-agreement,
kNN-ρ) ≈ 0, n.s. on both datasets. The *winners* happen to depart from ECFP4, but
some low-agreement metrics (RDKit-topological) are also poor. The correct claim is
narrow: **ECFP4 Tanimoto is not the best choice; not that anything unlike it is
good.**

## The one robust, actionable finding

Across the whole battery (noise included), the top group by overall mean rank is
**count fingerprints and FCFP4**:

    ecfp4_count:soergel 14.4 · atompair_count:soergel 14.5 · ecfp4_count:dice 14.8
    ecfp4_count:tanimoto 15.0 · fcfp4:{tversky,dice,tanimoto} ~15.5 · desc:euclid 15.6
    …  ecfp4:tanimoto 20.2 (of 47)

**The cheapest, safest lever is binary → count ECFP4.** `ecfp4_count:tanimoto`
beats `ecfp4:tanimoto` on nearly every evaluation — same fingerprint, same
coefficient, counts instead of bits. **FCFP4** (functional-class) and **atom-pair**
are the next tier. None of these is a learned or exotic metric; all are one-line
RDKit changes. (This is a *finding*, not a change — nothing has been applied.)

---

## Robustness (5 scaffold-grouped subsamples) — `run_grid_robust.py`

ECFP4:Tanimoto rank (of 47), mean ± SD, and #metrics beating it on BOTH datasets:

| | hERG rank | LogD rank | beat-both (per-seed range) |
|---|---|---|---|
| kNN read-across | 27.0 ± 5.1 | 30.2 ± 3.7 | 20.8 (16–25) |
| noise detection | 20.0 ± **16.3** | 29.2 ± 9.0 | 12.8 (2–32) |
| noise recovery | 15.2 ± 6.6 | 24.6 ± 9.2 | 8.6 (1–16) |

**Read-across is stable** (tight SD, consistent beat-both). **Noise detection is
not** (SD 16 on hERG; beat-both ranges 2–32) — the metrics are near-tied on that
task, so a single-seed rank is unreliable. This is why the earlier "ECFP4 5th on
noise" claim was withdrawn.

## Noise σ × corruption-rate sweep — `run_noise_sweep.py`

σ ∈ {0.3, 0.6, 0.9} × rate ∈ {0.10, 0.20, 0.30}, 3 injection seeds each, shared
realisation per setting. "Ranking stability" = mean pairwise Spearman between
settings' per-metric score vectors (1 = ordering unchanged across regimes).

| dataset | detection stability | recovery stability |
|---|---|---|
| hERG | ρ = 0.57 | ρ = 0.92 |
| LogD | ρ = 0.44 | ρ = 0.89 |

- **The best noise-DETECTION metric depends on the noise level** (Cortés-Ciriano
  2015 confirmed). hERG: FCFP4 (low σ) → torsion (mid) → count-fingerprints
  (high σ). LogD: MACCS/RDKit (low σ) → FCFP4 (high σ). Strikingly,
  **RDKit-topological — the worst metric for read-across — is a top noise-detector
  on LogD at low σ.** The metric you want depends on the job *and* the noise level.
- **Noise RECOVERY ranking is stable** (ρ ≈ 0.9): the same metrics correct labels
  well across all regimes (count fingerprints, FCFP4).
- ECFP4:Tanimoto detection rank across the 9 regimes: hERG 10–31, LogD 22–41 —
  mid-pack throughout, never dominant.

**Implication:** there is no single "best noise metric"; the choice is regime- and
task-specific. The server run sweeps this across 30 targets so the regime-map is
statistically grounded rather than read off 2 datasets.

---

## Experiment 3 — swapping the KIRBy search objective

*(Does building the hybrid to optimise activity-coherent neighbourhoods beat the
standard? `hybrid.py` NOT modified — this only tests whether it's worth a
signed-off change. Numbers are within-outer-fold, comparable only within this
table.)*

| kNN read-across ρ | hERG | LogD |
|---|---|---|
| hybrid — triplet_activity | 0.542 | 0.556 |
| hybrid — predictive *(current default)* | 0.524 | **0.560** |
| hybrid — sac *(invented)* | 0.516 | 0.513 |
| best single base rep | 0.490 | 0.523 |
| ecfp4 (Tanimoto) baseline | 0.484 | 0.448 |

1. **The hybrid beats every single base representation** as a similarity space,
   both datasets — supports "use the similarity-hybrid in place of ECFP4 Tanimoto."
2. **The search *objective* barely matters** (all within ~0.03 ρ; the current
   predictive objective is *not* beaten, and wins on LogD). So **"replace the
   ranking metric" is not supported** by this evidence — the lever is the
   representation/coefficient, not the inner-loop scalar. **No `hybrid.py` change
   is justified** on these two datasets.
3. **SAC is the worst objective** on both datasets — on merit, not just because it
   was invented without consent (which it was; that is on the record).

---

## Verification (independent recompute, `verify_self.py`)

- **Non-circular design — confirmed.** All activity-grounded evals use zero ECFP4
  reference; only the cliff *definition* and the labelled diagnostic touch it.
- **Ranks & "22/47 beat on both" — confirmed** exactly.
- **BEDROC** closed form == brute-force to 4 dp.
- **No leakage** — independent kNN recompute matches to 1e-6; a scaffold-group
  assertion (fires if any scaffold spans train+test) never triggered.
- **One overclaim corrected** (the "less-like-ECFP4" sentence).

*(A 4-agent adversarial workflow died on a network outage mid-run; verification
was redone inline — same content.)*

---

## Limits

- **2 datasets** (local pilot). The 30-target MoleculeACE server run with paired
  Wilcoxon/SRD is the confirmation step; per-task winners differ (MODI: ECFP4 is
  3rd on hERG, 42nd on LogD), and noise-detection ranks are unstable on 2 datasets,
  so aggregate claims need the battery.
- **Classical metrics only** locally; learned/triplet embeddings are Stage 3 (GPU).
- Noise σ × rate sweep DONE locally (above); the server driver now runs the same
  sweep across 30 targets with paired Wilcoxon per regime.
- **SAC** appears only as a competitor, never a judge.

## Bottom line (for your review — nothing applied)

1. For **similarity/read-across**, ECFP4 Tanimoto is reliably not the best;
   **count-ECFP4, FCFP4, atom-pair** beat it cheaply and stably.
2. For **noise mitigation (your major use case)**, no metric is decisively best on
   2 datasets — ECFP4 Tanimoto is mid-pack, and the best noise-DETECTOR changes
   with the noise level (regime-dependent). The 30-target sweep is needed to draw
   a real recommendation. Noise RECOVERY is more stable (count/FCFP4 edge ahead).
3. The **KIRBy hybrid** already beats any single fingerprint as a similarity space
   — the strongest support for "use the hybrid in place of ECFP4 Tanimoto."
4. The **search-objective swap is not worth a `hybrid.py` change** on this evidence.
5. Next: run the staged 30-target + learned-metric server battery (both), which now
   includes the noise σ×rate sweep, then decide.
