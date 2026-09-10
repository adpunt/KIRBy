# Does a better similarity metric clean noisy labels well enough to win back accuracy?

Branch `similarity-metrics-study`. Read every file named here in full before changing
anything. Everything below was read in the repo on 2026-09-02; re-read it, because a
document is a dated claim.

## The question

Adelaide, 2026-08-19: *"is that not the whole point of my noise experiments? ... I
thought this entire mechanism gets wired into those experiments, or at least it
should"*

Adelaide, 2026-08-21: *"Can you just stop this? I just need a baseline. I need clean
rmse, noisy rmse, and cleaned rmse"*

So: corrupt some training labels, clean them using neighbours, retrain, and report three
numbers per cell — RMSE on clean labels, RMSE on corrupted labels, RMSE after cleaning.
The thing under test is which space defines "neighbour".

This is a noise experiment. It goes in `tests/experiments/noise/`. Do not build a new
standalone script in `tests/experiments/framework/`.

## What already exists

`tests/experiments/noise/n11_estimation_remediation_grid.py` is the experiment. It
already:

- injects noise on training labels only, through `noise_helpers.inject_train_labels`
- writes `clean_score` and `noisy_score` per sigma (lines 592-593) and a `denoised`
  score plus a `recovery` percentage per config (lines 612-618)
- carries three space arms: one plain base representation, a predictive hybrid built by
  `create_hybrid`, and a similarity arm from `_similarity_hybrid_space.build_similarity_hybrid`
  (line 444)
- crosses noise estimator × remediation × alpha × distance metric × representation

The three RMSEs Adelaide asked for are already in its output. What is wrong is the
similarity arm.

## What is wrong with the similarity arm

`tests/experiments/noise/_similarity_hybrid_space.py` builds one fixed matrix:
count fingerprint + physchem-minus-saboteurs + 3D shape + fine-2D, z-scored, equal
blocks, count block halved. That recipe was chosen by the July standalone study. It is
not built by `find_best_hybrid`, no allocation method splits a budget across it, and no
objective selects it.

Its own docstring says it only returns a matrix when a `sim_<dataset>.npz` cache exists
and covers every molecule — herg_ki, logd and adme_solubility today. Otherwise the arm
is skipped. So six of the molecular datasets have no similarity arm at all.

One thing about it is right and must be kept: block selection never sees activity
labels, so unlike the predictive hybrid this arm is not oracle-biased. n11's own
docstring flags the predictive-hybrid arm as oracle-biased because its blueprint is
chosen once on clean labels.

## What to build

Replace the hardcoded similarity arm with a space the search built.

The engine takes `objective='similarity'` (each molecule's activity predicted from its
five nearest neighbours), `objective='struct_fid'` (Spearman between pairwise distance
and maximum-common-substructure distance), and a dict blending those two. All three are
in `_VALID_OBJECTIVES` in `src/kirby/find_best_hybrid.py` and in `_GEOMETRY_OBJECTIVES`
in `src/kirby/hybrid.py`. `struct_fid` raises without `structural_pairs` and `row_index`,
so both must be threaded through.

MCS pairs come from `kirby.similarity.mcs.get_mcs_pairs(dataset, smiles)`, which hashes
the molecule list it was given and caches per hash. Do not use `mcs_cache.build` — it
indexes the July study's row ordering, and row 900 is a different molecule there.

The representation pools per dataset are in
`tests/experiments/complexity/complexity_helpers.py`: `general` holds the signed-off ten,
`similarity` holds eleven `sim_` recipes, and `similarity_rep_names()` returns both.

## Decisions that are Adelaide's, not yours

Put these to her with the numbers attached. Do not pick one and present it as settled.

1. **Which space arms n11 carries.** The base representation and the predictive hybrid
   are already there. The candidates to add are a hybrid built under the nearest-neighbour
   objective, one under `struct_fid`, and one under a blend. Adding all three multiplies
   n11's grid by the number of arms.

2. **Whether the search that builds each arm runs on clean or corrupted labels.** Clean
   is oracle-biased in the same way n11 already flags for the predictive hybrid. Corrupted
   is honest and costs one search per noise level.

3. **Which datasets.** The eight molecular datasets have `similarity` recipes registered.
   The current arm reaches three. Say which are in.

Measure the cost of each option before asking, and show the arithmetic in the question.

## What f22 and f23 were

`tests/experiments/framework/f22_similarity_feature_sources.py` and
`f23_noise_cleaning_metric.py` are the wrong-question versions of this work. Neither calls
`create_hybrid`; both rewrite the feature ranking, the budget split and the concatenation
inside the script. f23 reports detection AUROC and a recovery correlation, not the three
RMSEs. Adelaide has said to delete both. Do not resurrect their design and do not carry
their numbers forward.

`tests/experiments/framework/f27_objective_and_budget.py` is the experiment that asks
whether the similarity representations help, three pools against five objectives through
the real search. It is separate from this work and stays.

## One live defect to check first

`src/kirby/noise_spec.py:344` calls `NoiseInjectorRegression.from_condition`, added in
NoiseInject 1.0.0. Array 12906522 lost six of seven tasks on 2026-08-29 because the
server's checkout of that package predated it. `setup.sh` was fixed in `c3b47da` to
fast-forward the checkout every run and print the version it loads. Confirm the server
prints a current version before submitting anything that injects noise — every n-series
experiment goes through this path.
