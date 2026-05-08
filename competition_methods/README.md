# Competition methods — cross-comp reusable guidelines

A library of validation and fusion/ensembling techniques, each in its own file with a consistent template:
- **What it is** + when to use
- **Why it helps** (failure mode it addresses)
- **Implementation** (code references / packages)
- **Trade-offs** (cost, when it fails)
- **Empirical results** — filled in as we test it across competitions

## Why this directory exists

The `cherry_pick_hep_anomBlend_death_wapp` failure on ANNITIA (2026-05-08) — predicted LB 0.93–0.94 with 99% bootstrap confidence, actual LB 0.898 — proved that single-OOF, single-seed K-fold CV with ad-hoc fusion is a broken evaluation protocol for small-data, rare-event regimes. This library catalogs the literature-backed alternatives, so we don't reinvent the wheel each competition.

## Validation methods (`validation/`)

How to *measure* whether a model is actually better. Failure mode addressed: mistaking selection winner's curse for real lift.

| ID  | Name                                   | Best for |
|---|---|---|
| V01 | Repeated stratified K-fold             | Rare events, single-seed instability |
| V02 | Harrell bootstrap optimism correction  | Small-sample survival (Harrell RMS standard) |
| V03 | .632+ bootstrap estimator              | Small-sample, light-regularization regime |
| V04 | Uno IPCW concordance                   | Survival with moderate-to-high censoring |
| V05 | Integrated Brier Score (IPCW)          | Calibration + discrimination |
| V06 | Cumulative-dynamic AUC (Heagerty)      | Multi-horizon discrimination |
| V07 | Adversarial validation                 | Train→test distribution shift sanity check |
| V08 | Held-out fusion-test gate              | Stop selection-bias-on-OOF |
| V09 | Nested cross-validation                | Stacking / meta-learner weight fitting |
| V10 | Permutation null distribution for ΔC   | Significance of inter-candidate gaps |
| V11 | Per-patient Brier residual diversity   | Selecting variants by complementary errors |
| V12 | Royston-Sauerbrei D / Schemper-Henderson V | Survival effect-size as second opinion |

## Fusion / ensembling methods (`fusion/`)

How to *combine* base predictions. Failure mode addressed: greedy max-OOF picks overfit, simple averages plateau.

| ID  | Name                                         | Best for |
|---|---|---|
| F01 | Caruana bagged ensemble selection (KDD 2004) | Library of 50+ candidates, want robust selection |
| F02 | Geomean of rank-transformed predictions       | Heterogeneous bases, no fitting wanted |
| F03 | Hodges-Lehmann pseudomedian                   | Robust location, noisy bases |
| F04 | Bates-Granger variance-weighted combination   | Independent bases, hold-out variance available |
| F05 | Stacking with regularized meta-learner + nested CV | Many bases, calibrated probabilistic outputs |
| F06 | Diversity-penalized ensemble selection (ICDM 2006) | Pool collapses onto one base family |
| F07 | Bayesian Model Averaging                      | Theoretical robustness to selection |
| F08 | Frequentist Model Averaging (Hjort & Claeskens) | AIC/BIC-friendly model classes |
| F09 | Trimmed / winsorized mean of predictions      | One rogue base risk |
| F10 | Snapshot averaging within base models         | Reduce base-level variance pre-fusion |

## How to use this library

1. **Pick the right level.** Validation = "is X better than Y?" Fusion = "how do I combine X and Y?"
2. **Combine validation methods** — at least one discrimination metric (V04), one calibration metric (V05), and one selection-bias control (V08/V09).
3. **Combine fusion methods** — never submit a single fusion variant; submit multiple methodological families and check they converge.
4. **Empirical results section in each file** — record what worked / what didn't per competition. Over time this becomes the actual decision rule, not the textbook recommendation.

## Cross-references

Each file links back here. When a method proves to dominate in a competition, mark it in the result table with the LB delta — that's the evidence other comps will use.

## Conventions

- One file per method
- Consistent header: name, type, domain applicability, citation
- Code snippet or package reference where possible
- Results table entries: `competition · date · primary metric · LB before → after · notes`
