# The similarity-objective hybrid is asked for in 18 experiments and built in 2

Written 2026-09-19. Paste the prompt at the bottom into a fresh session.

## What is wrong, in one paragraph

`POOL_SUBSETS['predictive_and_similarity']` is `('predictive', 'similarity')`
and the tuple is in PREFERENCE ORDER. `get_best_search_for_downstream` returns
the FIRST family that has an entry, because the two families' scores are in
units that cannot be compared — one is the dataset's own metric, the other a
correlation between each held-out molecule's activity and what its five nearest
neighbours predict. A predictive recipe always exists. So passing
`pool_subset='predictive_and_similarity'` reads like "test both families" and
means "prefer the predictive one, and record that the other existed". The
geometry search is built by the precompute, paid for once per dataset, and then
used by almost nothing.

## Measured, 2026-09-19, by reading every call site

18 experiment files pass `pool_subset='predictive_and_similarity'`:

    similarity/  s02 s04 s05 s06 s08 s10 s13
    noise/       n02 n06 n07 n11 n12 n13 n17 n18 n19 n21 n23

Of those, the number that BUILD FEATURES from the similarity family's recipe:

    n11   yes — the reference implementation, see below
    n13   yes
    n18   yes — fixed 2026-09-19
    n21   yes — fixed 2026-09-19
    every other one   NO

`by_objective` appears in most of them, but only to WRITE the other family's
tier, aug_condition and score_mean into the result file. Recording that a
second search exists is not measuring in it.

### The sharpest case: n06 says in its own output that it did

`n06_bindingdb_ground_truth.py` writes `pool_subset_note` containing the words
"Both families' best recipes are scored here as separate spaces." It calls
`_blueprint_for` ONCE (n06:657) and takes `meta['best_config']['blueprint']` —
the preference winner. `SPACE_FOR_FAMILY` maps both families to space names and
only one is ever populated. EXPERIMENTS_FINALIZED.md's n06 entry also says it
scores "the hybrid whose features were chosen by distance". It does not.

### Why it matters most where a distance is the subject

Every detector in n02, n07, n11, n12, n17, n18, n19, n21 and n23 scores a
molecule by comparing it to its NEAREST NEIGHBOURS. The method rests entirely on
the distance those neighbours are found in. The similarity-objective search
picks features so that exactly that distance predicts activity from the five
nearest neighbours. It is the feature space built for the question those files
ask, and until 2026-09-19 it was the one candidate space never measured in.

The same holds in similarity/: s04 (activity cliffs), s05 (five-nearest-
neighbour prediction), s06 (applicability domain) and s13 (clustering and
retrieval) are all distance questions asking the pool for a family they never
build.

## The reference implementation — copy this, do not invent a second one

`n11_estimation_remediation_grid.py:731-762`. It iterates EVERY family the pool
reports and builds one space per family:

    hybrid_recipes = []
    _by_obj = meta.get('by_objective') or {meta.get('objective', 'predictive'): {...}}
    for _obj, _entry in _by_obj.items():
        _bp = (_entry.get('best_config') or {}).get('blueprint')
        if _bp is None:
            print(f"  pool family {_obj!r}: no blueprint on its best entry")
            continue
        hybrid_recipes.append({'space': f'hybrid_{_obj}', 'objective': _obj,
                               'blueprint': _bp, ...})

and then passes each recipe's OWN objective to the builder
(n11:329 `objective=_recipe.get('objective', 'predictive')`), because the
allocators must split the budget by the quantity the search ranked by.

`noise_helpers.hybrid_for_fold` already handles the rest: it reads the
objective off the blueprint and drops the model, the metric and its direction
under a geometry recipe, where `build_fold_hybrid` refuses the pair. n18 and n21
now call it for both recipes from one loop.

## Three things that are NOT the fix

1. Do not compare the two families' search scores to pick one. They are in
   different units. `_shared.py:5997-6018` refuses to and says why.
2. Do not change the preference order in POOL_SUBSETS. Every caller that wants
   ONE recipe should keep getting the predictive one, so nothing already
   reported moves.
3. Do not read the similarity recipe out of `data['precomputed_searches']`.
   Those dicts are filled by a predictive precompute and record no objective;
   `get_best_search_for_downstream` forces the disk path for a geometry request
   for that reason (`_shared.py:5800-5812`).

---

## THE PROMPT

> Read `SIMILARITY_HYBRID_NOT_BUILT_HANDOFF_2026_09_19.md` in full, then
> `tests/experiments/_shared.py` around `POOL_SUBSETS` (:5571) and
> `get_best_search_for_downstream`'s family-ranking block (:5973-6049), then
> `tests/experiments/noise/n11_estimation_remediation_grid.py:731-762` and
> `:326-347`, then `tests/experiments/noise/n18_clean_the_real_labels.py`
> around `SPACES` and its per-fold build, and
> `tests/experiments/noise/n21_neighbour_distance.py` `_specs` and
> `_spaces_for_pass`. n11, n13, n18 and n21 are the four that already do this;
> the last two were fixed on 2026-09-19 and are the smallest examples to copy.
>
> THE JOB. Fourteen experiments ask the pool for
> `pool_subset='predictive_and_similarity'` and never build a feature space from
> the similarity family. Give each of them the space, as an ADDITIONAL arm
> beside the predictive one, so that nothing already reported moves.
>
>     similarity/  s02 s04 s05 s06 s08 s10 s13
>     noise/       n02 n06 n07 n12 n17 n19 n23
>
> For each file, in this order and stopping to report after each:
>
> 1. Say what the file currently does with `by_objective` — builds from it,
>    records it, or ignores it. Quote the line.
> 2. Say what the second space would be called and what the file's existing
>    arms are called, so the result file reads as a list of spaces and not as a
>    hybrid plus an afterthought.
> 3. Add it with the n11 pattern. One loop over the families the pool reports,
>    one space per family, each recipe's own objective passed to the builder.
>    A family whose best entry carries no blueprint is SKIPPED with the reason
>    written to the result file, never silently.
> 4. Say what it costs: how many more `build_fold_hybrid` or `create_hybrid`
>    calls per dataset, and whether the similarity search is already on disk for
>    the datasets this experiment runs on. `noise_helpers.run_all` precomputes
>    it once per dataset; `tests/precompute_hybrids.py` is where the similarity
>    pass has to exist for the similarity/ family, and if it does not, the first
>    experiment to ask pays for a full search inside its own job.
>
> ⛔ n06 FIRST, and treat it as a correctness fix rather than a new arm. Its
> `pool_subset_note` already tells the reader "Both families' best recipes are
> scored here as separate spaces", `SPACE_FOR_FAMILY` already maps both family
> names to space names, and `EXPERIMENTS_FINALIZED.md` already describes three
> spaces including "the hybrid whose features were chosen by distance". One
> blueprint is fetched (n06:657). Every stored n06 result claims a comparison it
> did not make.
>
> ⛔ DO NOT rank the two families against each other by their search scores, do
> not reorder POOL_SUBSETS, and do not read a geometry recipe out of
> `data['precomputed_searches']`. The handoff says why for each.
>
> THE SMOKE MUST SHOW, for every file changed: a run where the pool holds both
> families produces a result file with both spaces populated and their scores in
> the dataset's own metric; a run where the similarity family is absent produces
> every predictive number unchanged plus a stated reason for the missing space;
> and the two spaces are not byte-identical, because a geometry recipe that
> silently resolved to the predictive one is exactly the failure this is fixing.
>
> Then update `EXPERIMENTS_FINALIZED.md` for each file you change, and add one
> line to `NOISE_AUDIT_2026_09_10.md` recording that n06's stored results
> claimed a two-space comparison they never ran.
