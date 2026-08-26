# New-chat prompt — build the distance coefficients and sweep them

Paste everything below the line into a fresh chat.

---

## The situation

KIRBy can compute a distance eight ways: euclidean, manhattan, cosine, chebyshev,
canberra, correlation, braycurtis and mahalanobis. All eight are continuous-feature
formulas. **Not one set-overlap coefficient exists in `src/kirby/`** — no Tanimoto, no
Dice, no Soergel, no Tversky. Tanimoto appears in the repository only as a Gaussian
process kernel, as the definition of an activity cliff, and in the conformal prediction
module. Unweighted Tanimoto, Dice, Cosine, Soergel and Tversky exist in the standalone
study at `tests/experiments/similarity_metrics/simeval.py`, and nothing imports them into
the framework.

Meanwhile the 2026-07-11 pilot ran 47 representation-by-coefficient combinations, and
Adelaide's locked decision of 2026-08-25 is that the coefficient is a real grid axis,
swept at all three scoring points: the per-representation baseline, the candidate score
inside the search, and the outer held-out fold. That axis cannot be swept while the
coefficients do not exist.

## What was built on 2026-08-26 — read this first, do not redo it

`src/kirby/similarity/preprocessing.py` is new and holds the preprocessing every
coefficient family needs:

- `feature_kind(X)` returns 'binary', 'count' or 'continuous' from the values, since a
  representation's name does not settle it
- `require_binary` and `require_nonnegative` raise rather than coerce
- `standardise(X, mu, sd)` z-scores with the training fold's own numbers
- `block_divisor(width, coefficient)` gives sqrt(width) for the Euclidean family and
  width for the Manhattan family; `stack_blocks` applies it
- `normalise_weights` clips negatives and rescales to average 1
- `fold_weights(Z, w, coefficient)` applies weights to already-standardised features with
  the correct power per coefficient, and returns `{'w': ...}` for scipy when the weight
  cannot be folded in at all
- `COEFFICIENT_FAMILY` maps each coefficient name to its family, and raises on an unknown
  name rather than guessing a power

`compute_distance_matrix` in `src/kirby/utils/noise_mitigation_methods.py` was fixed the
same day. Before, every metric got the same `X * sqrt(w)`. Now each family gets its own
handling, negative weights are clipped, weighted mahalanobis raises, braycurtis on
standardised features raises, and a rank-deficient mahalanobis covariance raises.

To add a coefficient you register its name in `COEFFICIENT_FAMILY` with its family, and
the weighting and block-scaling follow automatically.

## The research behind it

`SIMILARITY_COEFFICIENT_RESEARCH_2026_08_26.md` in the repository root. Read it before
designing anything. **Its header says so and it bears repeating: nothing in it is a KIRBy
measurement.** Formulas and duplicate relations were verified against primary sources; the
numeric claims come from small scripts the research agents wrote, not from this
repository's datasets. Re-verify any number before citing it.

What it settles:

- 58 named coefficients exist, but most are the same ranking under a different name.
  Eleven are strictly monotone in Hamming distance. Twelve are monotone in Tanimoto.
  Roughly 19 are independent on binary fingerprints, 4 on count fingerprints, 7 on
  continuous features.
- **Dice is rank-identical to Tanimoto** (`Dice = 2T/(1+T)`, Spearman exactly 1.000).
  `simeval.py` ships both and its Dice docstring claims they differ. They do not. Running
  both is a duplicate cell.
- **Soergel, Ruzicka and RDKit's count Tanimoto are one formula under three names.**
- **Two inequivalent count Tanimotos exist** and RDKit and `simeval` use different ones.
  On a real Morgan count pair they differed by 0.26. Ship both, named apart; never ship
  one function called `tanimoto`.
- Set coefficients on standardised continuous features are **undefined**, not weak. Log
  those cells as not-applicable; never leave them blank.

## What to build

Section 5 of the research file is an ordered build list. The short version:

**Port from `simeval.py` into `src/kirby/`, with weights:** Tanimoto in both forms
(`tanimoto_dot` and `tanimoto_minmax`), Cosine, Soergel, Euclidean, Tversky. Fix
`simeval`'s count path for Tversky first — `_bit_counts` uses row sums for two of the
three counts and the Gram diagonal for the third, which on counts returns distances up to
174. Ship Tversky unsymmetrised with alpha and beta as a real swept axis; the current
post-hoc symmetrisation makes it depend only on their product and lands at Spearman 0.9996
with Dice.

**Write from scratch:** Kulczynski-2, Pearson phi, Simpson or Braun-Blanquet (one, not
both — their product is cosine squared), Canberra on counts, Jensen-Shannon on
L1-normalised counts, fractional Minkowski, and weighted Mahalanobis with the precision
matrix estimated once on unweighted training rows and held fixed.

**Do not build** the long list in the research file's Tier 3. Every one of them is a
duplicate cell on every representation family it would run on, and two rows agreeing
perfectly in a results table reads as corroboration when it is arithmetic.

## Then sweep it

A **framework** experiment, next free f-number in `tests/experiments/framework/`. Not the
noise suite, not the similarity suite. Do not edit f00 through f15.

`f18_distance_metric_sweep.py` already sweeps coefficients standalone and cache-free, on
herg_ki and logd, but it does not touch the engine and it has no weighting. Read it before
writing anything; extend the idea rather than duplicating it.

The sweep has to answer: does the coefficient change which representation wins the budget
split, and does it change which configuration wins the search? Those are different
questions. The budget split only cares about the ORDER of the representations, so a
coefficient can leave every score alone and still change the allocation, or the reverse.

Cross the coefficient with: weighted against unweighted, both weight sources (model
importance and the label-free structural ranker in `src/kirby/similarity/selection.py`),
equal-block scaling on and off, and the full budget list. Report per dataset, never
averaged. Log every not-applicable cell.

## Two things to fix while you are in there

`simeval.py` and `f18` both cite a self-test proving Soergel equals Tanimoto distance on
binary vectors. The identity is true. The self-test is not in either file.

`SIMILARITY_HYBRID_DESIGN_2026_08_24.md` lists braycurtis and mahalanobis as valid on
standardised stacked features. Braycurtis is undefined on signed input, and `f18`'s own
`needs_nonneg=True` flag already contradicts the document.

## House rules

- Report `delta_mean_of_folds` from `nested_cv_evaluate`. Never `search_mean`, never
  `.all_results`.
- No random splits. Scaffold-group-aware folds throughout.
- Baselines are individual representations, reported separately. No averaged baselines.
- Do not cut axes for compute. Server time is free; route long runs to a server job.
- Greedy allocators are CUT. If greedy comes up, say so in one sentence and move on.
- Never `ssh` or `scp` to the server. Adelaide submits jobs.
- Read `CLAUDE.md` at the repository root before writing any message.

## Definition of done

The coefficients exist in `src/kirby/`, weighted, each registered in `COEFFICIENT_FAMILY`.
Every one has an assertion that it reduces to its unweighted form at w = 1 and that it
moves at a deliberately non-uniform w — a uniform w passes under every wrong exponent, so
it proves nothing. The sweep is built and either run or submitted. The roadmap entry that
says no weighted set coefficients exist is updated.
