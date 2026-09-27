# F34 and F35: pick one neighbour-weighting scheme and one distance coefficient

Written 2026-09-27. Everything in the "what exists" sections was read from the
code on that date, not remembered.

## Why these two experiments exist

Two settings decide every nearest-neighbour number in this repo, and today each
folder answers them differently. Nobody chose that; it is what each file
happened to do.

**Setting A, how a neighbour's label is counted.** Take a molecule, find its k
nearest training molecules, average their activities. Either they all count
equally, or a close one counts more than a far one.

**Setting B, which coefficient decides "near".** manhattan, euclidean, cosine
or a fractional Minkowski.

The s10 sweep finished 2026-09-27 and says counting close ones more wins all 12
matched pairs. But it measured that while the neighbour search itself was fixed
to Euclidean, because s04, s05 and s10 hand sklearn raw features and pass no
metric. So setting B never reached the kNN and the two settings have never been
crossed. F34 and F35 fix that.

## What EXISTS today, read from the code

### Distance coefficients — four, and the plumbing is already written

`src/kirby/similarity/distance_config.py:66`

    DISTANCE_METRICS = ('manhattan', 'euclidean', 'cosine', 'minkowski_p0.5')
    DEFAULT_DISTANCE_METRIC = 'manhattan'
    ENV_VAR = 'KIRBY_DISTANCE_METRIC'

The module already maps each canonical name to BOTH a scipy cdist name and an
sklearn metric name, with kwargs:

    _SKLEARN = {'manhattan': ('manhattan', {}),
                'euclidean': ('euclidean', {}),
                'cosine':    ('cosine', {}),
                'minkowski_p0.5': ('minkowski', {'p': 0.5})}

So passing a coefficient into a KNeighbors model needs no new code. It needs
the call sites to use `_SKLEARN` instead of letting sklearn default.

`canonical()` RAISES on an unknown spelling rather than falling back, and the
docstring says why: "a silent fallback to Euclidean is exactly the failure this
module exists to end." That failure is live in s04, s05 and s10.

⛔ Tanimoto, Dice and Jaccard are NOT on this list and must not be added for
hybrid spaces. `hybrid.py` standardises every block, so a 0/1 fingerprint column
becomes roughly -0.67 to 17.29. Set-overlap ratios need set membership, which
standardised features do not have. They remain valid only on raw bit
fingerprints, which is where `similarity_metrics/` uses them.

### Neighbour-label weighting — sklearn gives two, and that is all

`weights=` on KNeighborsRegressor/Classifier accepts exactly:

    'uniform'    every neighbour counts the same
    'distance'   each neighbour counts as 1/d, with no choice of steepness
    a callable   an array of distances in, an array of weights out

There is no third built-in. Anything between or beyond those two needs a
callable, and none is written anywhere in this repo today.

### A near relative that is NOT the same thing

`s06_applicability_domain.py:132`

    AD_STATISTICS = ('mean', 'nn1', 'harmonic', 'kth')

These summarise the k DISTANCES into one number for an applicability-domain
score. They do not average the k LABELS. `harmonic` is k / sum(1/d), which is
the same arithmetic shape as `weights='distance'`, so s06 has effectively
explored part of setting A already — on a different quantity, and without
sharing the result. Do not merge the two lists; do cite s06's finding that
harmonic wins on caco2 and adme_mdr1 and the plain mean wins on herg_ki.

## The complete option list to sweep

### Setting A: six schemes, as one family plus two shapes

The natural family is w = 1 / d**p, which contains both sklearn built-ins:

    name             p       what it is                      status
    uniform          0       every neighbour equal           sklearn built-in
    inverse_sqrt     0.5     mild preference for close ones   callable, TO WRITE
    inverse          1       sklearn's 'distance'             sklearn built-in
    inverse_square   2       steep preference                 callable, TO WRITE

Two shapes outside the family, because a power law is not the only option:

    gaussian    w = exp(-(d/h)**2), h = median neighbour distance  callable, TO WRITE
    rank        w = 1/rank, magnitude ignored, order only          callable, TO WRITE

p = 0 and p = 1 pin the family to the two settings already in use, so the sweep
subsumes what the code does today rather than replacing it.

⛔ Every callable must handle d = 0. Two molecules with identical features give
a zero distance and 1/d is infinite. sklearn's own 'distance' handles this by
giving exact matches all the weight and the rest none; a callable must do the
same or the fold silently produces NaN. Write one shared helper, test it on a
vector containing a zero, and have all four callables go through it.

### Setting B: the four canonical coefficients, unchanged

    manhattan, euclidean, cosine, minkowski_p0.5

Use `distance_config.canonical()` and `_SKLEARN` so a typo raises.

### The grid

6 weighting schemes x 4 coefficients = 24 cells per (dataset, representation).
Both axes swept together, because they may interact: cosine discards magnitude,
so a weighting that reads magnitude may behave differently under it than under
manhattan. That interaction is the reason not to sweep them one at a time.

## Datasets and representations

### Datasets: the three with a full local cache

Confirmed present under /Volumes/seagate/kirby_simrun_cache on 2026-09-27:
result_db, rep_cache and searches all exist for

    herg_ki, caco2, adme_mdr1

Use all three. Do not average across them — report each separately. If a
scheme wins on all three that is the answer; if each prefers a different one,
that is also an answer and it is the expensive one.

### Representations: top base reps plus the hybrids the searches already found

The local pool is seven:

    ecfp4, maccs, avalon, atom_pair, top_torsion, pdv, ecfp4_count

Test, per dataset:

  1. The three best base representations, taken from `top3_base_reps` in that
     dataset's own stored s02 result — do NOT hardcode a list. On caco2 that
     field currently reads pdv, maccs, atom_pair.
  2. The hybrid built by the stored search at budget 100, both lineages
     (`general` and `general_distance`). These come out of the cache; do not
     re-run a search.
  3. ecfp4 as a fixed reference, because it is the field default and gives a
     line every reader recognises.

That is about 6 representations per dataset, so roughly 24 cells x 6 reps x 3
datasets = 432 scored fits. Each fit is a kNN on a cached feature matrix, so
this is minutes, not the 318 hours the s10 sweep cost. That is the whole point
of doing it as a framework experiment: no search runs.

## The two experiments

Highest existing framework number is f33, so these are f34 and f35.

### F34 — the crossed sweep, on a fixed neighbour count

Cross 6 weighting schemes x 4 coefficients on each (dataset, representation).
Hold k at the values s05 and s10 already sweep: 5, 7, 10, 15, 25, and report
per k rather than collapsing, because s10 found the voting margin GROWS with k
(0.0035 at k=5 to 0.0230 at k=25 on herg_ki).

Report per dataset:
  - the winning cell, and by how much over second place, in the dataset's own
    metric
  - whether the winner is the same across representations
  - whether the two axes interact: is the best coefficient the same under every
    weighting scheme?

Held-out scoring only, scaffold-grouped folds, `delta_mean_of_folds`. Never
`.all_results`.

### F35 — does the scorer choice change the answer

F34 uses one scoring path. F35 checks whether the conclusion survives the other
one, because the repo has two and they disagree structurally:

  - `similarity_metrics/simeval.py:161` fits
    `KNeighborsRegressor(..., weights='distance', metric='precomputed')` on a
    distance matrix the caller computed. The coefficient is honoured.
  - `s04:372`, `s05:416`, `s10:586` fit on raw features with no metric, so
    sklearn uses minkowski p=2. The coefficient is ignored.

Run F34's winning cell through both paths on the same folds. If they agree, the
scorer split is cosmetic and can be unified at leisure. If they disagree, every
number in `similarity_metrics/` and every number in `similarity/` sit on
different footings and that has to be said out loud in the methods.

## What these results affect — every file that builds a kNN

Read from the tree 2026-09-27. Twenty files:

  ENGINE          src/kirby/find_best_hybrid.py, src/kirby/hybrid.py,
                  src/kirby/similarity/distance_config.py,
                  src/kirby/utils/noise_mitigation_methods.py

  SIMILARITY      s04_activity_cliffs.py, s05_knn_prediction.py,
                  s10_importance_downstream.py

  METRICS         simeval.py, base_rep_comparison.py, scaffold_experiment.py,
                  triplet_similarity.py

  NOISE           n40_model_side_regularisation.py, _local_run.py

  COMPLEXITY      c04_importance_as_metric.py, c12_modality_ablation.py

  FRAMEWORK       f17, f18, f19, f22

  SHARED          tests/experiments/_shared.py

Current state of setting A across the three folders, counted 2026-09-27:
`similarity_metrics/` uses 'distance' in 3 places; `noise/` writes 'uniform'
explicitly in 3; `similarity/` passes a variable in 2 (s05, s10, both added this
week) and lets s04 default to uniform.

⛔ f18 already swept the distance coefficient at the engine's six internal
scoring sites and is standalone and cache-free. F34 must NOT redo that. f18
covers where the ENGINE computes a distance; F34 covers the DOWNSTREAM kNN that
scores a finished space. Cite f18, do not repeat it.

## The plan to make this uniform

Do not change any experiment until F34 and F35 have answered. Then, in order:

1. **Add the weighting schemes as one shared helper.** One module, six named
   schemes, one zero-distance guard, resolved from an env var the way
   `distance_config` resolves the coefficient. Nothing else changes yet.

2. **Make the coefficient reach the downstream kNN.** s04, s05 and s10 pass
   `**_SKLEARN[canonical(metric)]` into their KNeighbors constructors. This is
   the change that invalidates stored numbers, because the neighbours
   themselves move. Do it once, then re-run those three on the server.

3. **Set the two defaults from F34.** If one cell wins across datasets and
   representations, make it the default in `distance_config` and the new
   weighting module, and delete the per-file choices. If two cells are close,
   keep both as a swept axis and say so.

4. **Bring `noise/` and `complexity/` into line last**, because they have the
   most stored results and the least to gain — their kNN is a downstream model,
   not the object of study.

5. **Leave `similarity_metrics/` alone until step 3.** It is the only group
   already honouring the coefficient, so it is the reference, not the patient.

Each step is a separate commit with its own re-run, so a bad answer at step 3
does not require unpicking steps 1 and 2.
