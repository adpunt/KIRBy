# Framework finalization

**What this file is.** The list of things still open in the framework — `src/kirby/` —
before it is finished. Not the experiments. Every line below was read in the code on
2026-09-03 and §3 was re-read against the code on 2026-09-04; nothing in §1-§3 is a
measured result. §4 is where measured results go, and it now has one.

**To work through it, go to §3 and answer the `>>` lines.** Ten items, thirteen answers.
Each one says what it is, what changes either way, the numbers I could count, and what I
would do. Nothing outside §3 needs anything from you.

**How the §1 fixes were checked.** Every objective path — predictive, `similarity`,
`struct_fid`, and a blend — was run end to end on synthetic data through both
`find_best_hybrid` and `nested_cv_evaluate`, and the predictive and caching paths were run
on real caco2 (2158 molecules, three fingerprint representations, 3 folds). The six
existing framework unit tests pass. Nothing was run on the cluster.

**Who writes here.** Chats working on framework *experiments* add their findings and
their decisions to §3 and §4. This file is the one place open framework questions live.
When something is decided, move it to §5 with the date.

**Scope.** `find_best_hybrid.py`, `hybrid.py`, `result_cache.py`, `registry.py`,
`screen.py`, `noise_spec.py`. Experiment files are somebody else's list.

---

## 1. What was fixed on 2026-09-03

These are done and committed — they went in as part of `47e4b9e`, which a parallel
session made on the same working tree. Each one was a defect with one right answer, so
none of them is a decision waiting on you. The four marked **INVALIDATES** change a number
that is already stored somewhere.

### Which recipe wins

1. **A configuration scoring NaN could win.** `max()` compares NaN as False against
   everything, so a NaN sitting first in the list was never displaced. The all-NaN case
   already crashed; a partly-NaN list handed back a winner picked by list order. Every
   `best_by_*` now chooses out of the configurations that scored a finite number.
   `all_results` still holds all of them.

2. **A configuration that scored 3 folds was ranked against one that scored 5.** A fold
   that raises is skipped and the mean is taken over what survived. Every result now
   carries `n_folds_scored` and `n_folds_asked`, and a short configuration prints a line
   saying so. What to *do* about it is still open — §3.1.

3. **The two arms averaged folds differently under a geometry objective.** A fold whose
   correlation is undefined comes back NaN. The per-representation baselines have skipped
   those folds since 2026-08-27; the hybrid arm used a plain mean, so one undefined fold
   voided the whole configuration while the representation it was being compared against
   kept its other four folds. Both sides now skip undefined folds. **INVALIDATES** any
   stored geometry result that had an undefined fold.

4. **The result database handed back configurations the run never asked for.** Every
   stored row for the data + folds + tier was appended to the results and ranked, so a
   call asking for one budget was reported over every budget any earlier call had stored.
   The run now reports what it asked for. The rows all stay in the database.

### The honest number (`nested_cv_evaluate`)

5. **Under a blend, the comparator was the WORST representation.** A blend containing
   prediction flips `lower_is_better` to False, because the blend is a mean of
   correlations. But `SearchResult.baselines` still holds the dataset's own metric — RMSE
   — so taking the max of them picked the highest RMSE. The hybrid was compared against
   the worst single representation on every RMSE dataset under every blend.
   **INVALIDATES** every blend result, f28 included.

6. **The baseline arm swallowed its failures.** The hybrid arm has printed loudly since it
   was written; the baseline arm turned any exception into a silent NaN, so a dead
   comparator read as `delta = NaN` — "no effect" rather than "the arm never ran". It now
   prints, and says at the end which (model, representation) pairs never scored on any
   fold. The per-budget rebuild's bare `except: pass` prints too.

7. **A top_k hybrid with augmentation lost its fold from one arm only.** When a top_k
   allocation zeroes a representation, `create_hybrid` moves that representation's top
   columns into the augmentation pool. The test side had no entry for it, the rebuild
   raised, and the fold was dropped from the hybrid arm and kept in the baseline arm. The
   test side now takes the same columns, which `feature_info['demoted_reps']` records.

8. **`n_features` was the requested budget, not the width.** The rebuilt hybrid's real
   column count is recorded per outer fold and reported. A representation narrower than
   its quota gives fewer columns; augmentation extras are appended outside the budget and
   give more. Where no width was recorded the field is now `None` rather than the budget.

9. **A budget scored on one outer fold sat beside five-fold means.** When a requested
   budget exceeds the row count, the search appends the row count itself as a boundary
   budget — and every outer fold has a different training row count, so each fold can add
   a budget of its own. The run now prints which budgets scored on fewer folds than the
   rest. Aligning them is still open — §3.2.

### Building the recipe

10. **The full-data refit rebuilt a different recipe.** `_compute_fitted_result` called
    `create_hybrid` with no model and no metric, which takes the other branch of
    `_evaluate_allocation` — internal evaluator, r2/accuracy — and resolves
    `internal_evaluator='match'` to lightgbm whatever the search matched it to. The real
    estimator and metric are passed now. **INVALIDATES** every `fitted_results` feature
    list (s03 reads these).

11. **A model with no metric split the budget equally, in silence.**
    `metric_fn(y_val, y_pred)` raises TypeError on None, the except clause counted that as
    an infeasible allocation, so every representation scored minus infinity, every weight
    became zero, and the split fell through to equal — filed under whatever allocation
    name the run was launched with. `_evaluate_allocation` refuses the pair now.
    **Eight federated experiment scripts pass a model with no metric; all eight now
    raise** until they pass their dataset's metric alongside the model. That is the point: they have been reporting equal splits.

12. **A degenerate geometry slice was a tie, not a failure.** Under ten held-out rows, one
    scaffold group, or constant activity, `activity_readacross` returns 0.0 — which is
    also what a real correlation of zero looks like. Every representation scored 0.0, every
    weight became 0, and the budget split equally. The budget-split path now asks for NaN
    on those, which the allocator drops. `activity_readacross` gained a `degenerate`
    argument; its default is unchanged, so no other caller moves.

13. **The equal-split fallback is now always said out loud.** Whenever every weight is
    zero or non-finite the allocator divides equally and reports itself under the
    requested method's name. It prints and warns, naming the method.

14. **`min_floor` could spend more than the budget.** A floor of 0.05 across 25
    representations asks for 1.25 times the budget, and the trim loop below could not take
    any of it back — it only ever reduces down to the floor. The floor is capped so it
    fits.

15. **An unknown objective name was scored as prediction.** A typo, or the cut
    `similarity_sac`, fell through every geometry branch and ran a prediction search under
    the name that was asked for. `create_hybrid` refuses an unknown name, with a specific
    message for `similarity_sac`.

16. **`struct_fid` baselines held the wrong quantity.** Under `objective='struct_fid'`
    every per-representation baseline was a nearest-neighbour activity correlation while
    the hybrid arm was an MCS correlation, and the two were subtracted from each other.
    The baselines now score MCS fidelity; `_evaluate_baselines` raises if the pairs and
    row numbers are missing rather than substituting. Verified on a 300-row synthetic set:
    the informative representation's baseline moved from 0.65 (nearest-neighbour activity)
    to 0.012 (MCS against random pairs).

17. **`apply_blueprint` never passed `is_clf`**, so a classification recipe was rebuilt
    under regression scoring. Passed now, from `task_type`.

### Caching

18. **The one-sided cache check covered four settings, not ten.** Cache validation walks
    the *expected* metadata, so a key written only when non-default is never checked when
    it is absent: a run without the setting matched a file written with it. The check
    already existed for `tune`, `tune_n_iter`, `tune_knn_k` and the MCS pair-set id — it
    went in on 2026-09-03 with the tuned-k work. It now also covers the per-source
    augmentation cap and the whole pruning schedule, none of which are in the search
    pickle's filename either. A run that caps augmentation columns per source, or prunes
    on a schedule, wrote its result where a run doing neither then read it.
    **INVALIDATES** nothing stored, but a run that had been silently reading one of those
    files will now recompute.

19. **Two arms of one experiment shared database rows.** `condition_suffix` reached the
    search pickle's filename and the search metadata but neither the database filename nor
    the row identity, so two arms differing only in it shared rows and whichever ran first
    answered for both. It rides on `tier` now, the same way the noise condition and the
    objective already do — no schema change and no wipe. A run passing no
    `condition_suffix` leaves `tier` byte-identical.

20. **A blend's `tier` depended on dict key order.** `f'{tier}__obj-{objective}'`
    interpolated the dict, braces and all. It uses the stable sorted name now.
    **INVALIDATES** stored rows for blend runs, which were already unreliable for this
    reason.

21. **`cache_dir` rerooted half the cache.** It moved the search pickle and the result
    database; per-representation baselines, fold splits, per-fold hybrid evaluations and
    fine-tuned embeddings kept reading the default root. Every cache path resolves through
    `_cache_base()`, which now honours `cache_dir` for the duration of the call. The search
    pickle moved from `<cache_dir>/name.pkl` to `<cache_dir>/searches/name.pkl` to match
    the database's layout — files written flat under an old `cache_dir` are now a miss.
    `KIRBY_CACHE_DIR` is unchanged and still reroots everything.

### Failures that were swallowed

22. **A per-fold representation build that raised left that fold without it**, silently
    unless verbose, and the search then averaged folds built from different pools. It
    refuses now.

23. **Fold-0 pruning ate its own exceptions.** A configuration that raised on fold 0
    contributed no score, so its allocation method lost a placement in the end-of-fold-0
    vote and could be pruned out of the whole run on the strength of a crash nobody saw.
    Each failure prints, with a summary line.

24. **A resumed run pruned a different set than an uninterrupted one.** The
    checkpoint-crossing test sat after the two `continue`s that skip an
    already-scored or already-dropped configuration, so a resume whose last configuration
    at a checkpoint budget was already done never fired that checkpoint's prune. The test
    runs on every iteration now.

25. **A budget policy that raised looked exactly like one that changed nothing.** It prints
    what it dropped, and prints the exception when it raises.

26. **`MODEL_MAX_BUDGET` removed budgets in silence.** It now says which model lost which
    budgets, and refuses rather than searching a model at zero configurations and leaving
    it absent from every table with nothing said.

27. **A representation wider than 5000 features reported a full-width baseline it never
    had.** It is scored on the top 500 (200 under 1000 rows) by a per-fold LightGBM
    filter. `dimension` still said 5120. A `dimension_scored` field is recorded beside it,
    and the run prints that this is not a full-dimension baseline.

28. **`screen_reps(metric='auroc')` scored accuracy** for any classifier without
    `decision_function` — random forest, LightGBM, KNN, i.e. most of them. It uses
    `predict_proba` now, and raises if neither exists.

29. **A validation-side augmentation with no honest source took the wrong columns.** The
    fallback took the full base representation, and the stored indices are positions into
    the demoted block, so the bounds check passed and the wrong columns were selected. It
    raises.

30. **`find_best_hybrid` never consults the registry for the metric.** `metric` defaults
    to `'r2'`, so a caller who forgot to pass one ran an RMSE dataset under r2 — and
    `registry.lookup`'s docstring claims a guard that does not exist. It now prints when
    the default r2 is used on a dataset the registry gives a different metric. Whether the
    registry should become the default is open — §3.6.

---

### From the 2026-07-15 rebuild roadmap

Read against the code on 2026-09-03. Four things that roadmap decided or asked for were
never finished, and none of them showed up in the usage audit.

31. **`create_hybrid` split the budget with an SVM.** The 2026-07-16 decision was that the
    internal evaluator defaults to lightgbm. The ridge-removal commit swept every function
    whose default was `'ridge'` — and `create_hybrid`'s said `'svm'`, so it was missed. It
    is the entry point every caller that does not pass an evaluator goes through, and its
    own module docstring advertised svm while every allocator it calls documented lightgbm.
    Now lightgbm. **Changes what a bare `create_hybrid(...)` does** — the fifteen experiment
    sites that pass no evaluator have been splitting the budget with an SVM.

32. **No run has ever recorded how the budget was actually split.** `create_hybrid` computes
    `feature_info['allocation']` — how many features each representation got — and it was
    thrown away every time. The augmentation side has been recorded on the Blueprint since
    2026-08-26 (`kept_augmentations`, "so a winning run can be explained"); the base split
    never was. So the question the whole rebuild roadmap turns on — which representations
    fed the hybrid, how many features each got, whether the best one was cut to dust — could
    not be answered from any stored result, only from an arithmetic demo. It now rides on
    the Blueprint as `allocation_per_fold` (data, not a cache key, JSON round-trips), and
    the search prints the winner's split with a line when it comes out flat.

33. **Nothing said the two arms are not the same width.** The hybrid is held to `budget`
    features — the default list tops out at 300. The representation it is measured against
    is scored at its full native width, which on caco2 runs to 4800. That asymmetry is the
    root cause found on 2026-07-14 and diagnosis (b) of the roadmap, and no output has ever
    printed the two numbers together. `nested_cv_evaluate` now records
    `best_single_n_features_per_fold` beside `n_features_per_fold` and says so when the
    comparator is at least twice as wide.

34. **Phase 0.2's blocking gate was never built.** It asked for one test proving that a
    genuine win comes out POSITIVE on both a lower-is-better and a higher-is-better dataset.
    `test_metric_direction_persisted.py` checks that a saved result FILE records its
    direction — it runs no search, so it cannot see a sign flip inside one. Built as
    `tests/test_metric_direction_end_to_end.py`: six tests on synthetic data, no cache, no
    cluster, about two minutes. It covers rmse, spearman, a geometry objective, and the
    blend case that was backwards (fix 5). All six pass.

### Three of these now stop a run that used to keep going

These are the fixes with teeth. Each one used to produce a number; now it raises.

- **A model with no metric** (11). Eight federated experiment scripts hit this.
- **A model capped out of every budget** (26). If you ask for budgets above
  `MODEL_MAX_BUDGET`'s ceiling for gp, knn, svm, mlp, bnn, dnn or cnn, the search stops
  instead of dropping the model. It names the model, the cap, and the three ways out.
- **An unknown objective name** (15), `similarity_sac` included.

Nothing in `src/` triggers any of them; every objective, metric and budget path was run
end to end after the change. If an experiment starts raising on one of these, it was
producing a number it should not have been.

---

### From the similarity roadmap's open threads — read against the code 2026-09-04

`SIMILARITY_METRICS_ROADMAP.md` ends in nine lettered sections of open threads. Six things
in them that touch `src/kirby/` were re-read today. Four are closed and the roadmap has not
caught up; two are open and are now §3.10 and part of §3.9.

35. **"The budget split ranks representations by the objective measured to be the worst of
    the four" (§I) — no longer true.** Under a geometry objective
    `_score_rep_for_budget` returns whatever `_evaluate_allocation` computes for the
    objective that was asked for. `similarity_sac` was cut, so nothing ranks by it. A run
    under `objective='similarity'` splits its budget by read-across; a run under
    `struct_fid` splits it by MCS fidelity.

36. **"Cached structural pairs cannot be reused across loaders — a fourth fault in this
    layer, known since July, still unfixed" (§H) — fixed.** `get_mcs_pairs` stores the
    SMILES ordering hash and refuses a file whose hash contradicts its own name;
    `assert_pairs_match_smiles` is the check a caller runs on pairs it was handed. The
    noise suite's `_similarity_hybrid_space.py`, named in §D as the one place still loading
    a July file blind, calls both. Same for item 6 of `SIMILARITY_OPEN_WORK_2026_08_27.md`.

37. **"`similarity_optimized_hybrid.py` defaults to the cut `similarity_sac`, so running it
    bare gives a predictive search under the similarity label" — fixed.** The default is
    read-across, and `_check_sim_objective` raises on `similarity_sac` with a message
    naming what to pass instead.

38. **"No importance method is redundancy-aware" (roadmap Phase 4.1) — the bake-off exists.**
    `f30_scorers.py` registers mRMR, JMI, CMIM, ReliefF, MultiSURF and Boruta, and carries a
    redundancy-awareness flag per scorer into every result file. This is §3.8; nothing to
    decide until f30 reports.

Two of the 2026-07-15 roadmap's own asks were also re-checked. **Ridge as a measurement
device** (its two remaining uses, the L1 linearity descriptor and c15's linear model family)
was resolved on 2026-07-18 to `LinearRegression` and `Lasso`; both are in
`tests/experiments/_shared.py` today and no Ridge remains there. **The internal-evaluator
axis**, which the 2026-07-16 log said must be re-verified on the new datasets before it
could be dropped from the search, is `f31_internal_evaluator_tuning.py`, written 2026-09-04.

---

## 2. Verified and left alone

Read on 2026-09-03, real, and deliberately not changed.

- **The auto-demote mutation.** `create_hybrid` writes a demoted representation into the
  augmentation dict it was handed, and `_evaluate_hybrid_config` depends on that to build
  the validation side. It is load-bearing there. What was fixed is the leak:
  `apply_blueprint` now rebuilds against a copy, so one model's demotion no longer follows
  the next model into its rebuild.

- **`augment_uncapped` scores in rank order and ships in donor order.** Same feature *set*
  either way — only the column order differs, and nothing downstream depends on column
  order beyond tree tie-breaking. Not worth a fix.

- **Two-phase refinement** applies only to greedy allocations. Greedy is cut, so the path
  is dead. It also does not forward `objective`, which would matter if it were ever
  revived.

- **Ridge is gone except for nucleotide**, exactly as decided on 2026-07-18: it sits at
  tier `extra` so nothing auto-selects it, and `_resolve_models` injects it for `nt_*` and
  `gb_*` only. Checked because the roadmap's Phase 0.1 is the same sweep that missed
  `create_hybrid`'s evaluator default — this half of it is complete.

- **The global feature ranking the roadmap asked for exists.** Phase 4.3 said feature
  selection is a per-representation quota with no global ranking. `weighted_global` /
  `wglobal_power2` pool candidates across representations and rank them once, and
  `wglobal_power2` is in the default set. It sizes each representation's slice of the pool
  by its weight rather than restricting to a top-N first, which is not quite Phase 5.3's
  shape — a design question, not a defect.

- **`feature_blacklist` is not in the result database's identity.** Dropping columns is a
  representation change, and "reps never in the cache key — change them, wipe and rerun"
  is a locked decision. Consistent with it, not a separate defect.

---

## 3. Open — needs your decision

**How to answer.** Every item below ends with a `>>` line. Write your answer after it and
nothing else has to change; I read the `>>` lines and implement them. If an item needs a
number I have not measured, say so on the `>>` line and I will go and measure it.

Thirteen decisions, in ten items, cheapest first. Each says what it costs to change.
3.9 to 3.11 came from a 2026-09-04 sweep of the 2026-07-15 rebuild roadmap and the
similarity roadmap's open threads; the rest were already here.

---

### 3.1 `create_hybrid` on its own splits the budget with `top_3`

*Cost to change: nothing. Nothing stored, nothing re-run.*

**What happens.** If you call `create_hybrid(...)` and do not say how to split the budget,
it uses `top_3` — give the whole budget to the three best-scoring representations, nothing
to the rest. `top_2/top_3/top_5` were taken out of the searched set on 2026-09-01 because
they weld "how many representations" into the same slot as "how to weight them", and the
number of representations is a separate axis you sweep separately.

**Numbers.** I counted the calls. Every live caller passes an allocation method
explicitly: the five framework experiments that reach `create_hybrid` through a
`hybrid_kwargs` dict all put `allocation_method` in it. The 29 calls that take the default
are all under `tests/archive/`. (An earlier version of this file said fifteen live sites
take the default. That was wrong — I had counted `**kwargs` calls as if they passed
nothing.)

So today this changes no result. It is a trap for the next caller, not a live defect.

**Your options.**
- **Change the default to `performance_weighted`** — the plainest method that is actually
  searched.
- **Keep `top_3`.**
- **Make `allocation_method` a required argument** — no caller can get one by accident,
  and the 29 archive calls start raising.

**My recommendation:** `performance_weighted`. Required-argument is cleaner but breaks the
archive for no gain.

**>>** I wasn't aware the top_n is still in place, I thought that was replaced by the knobs...
It feels like this document is horribly out-of-date. I modified the allocation methods, I believe power_2/quota was the top but I could be wrong. I want whatever was top of those framework experiments to be the default allocation method. 

---

### 3.2 Under `objective='similarity'` a classification dataset is scored as a rank problem

*Cost to change: nothing stored is affected; it changes the score on classification
datasets only.*

**What happens.** `objective='similarity'` scores a representation by predicting each
molecule's activity from its five nearest neighbours in that representation's space, then
correlating the prediction with the truth (Spearman). The function takes an `is_clf` flag
and never reads it. On a classification dataset it therefore runs the neighbour prediction
on the class codes 0/1 and Spearman-correlates those. That is a monotone relative of
AUROC — it will rank representations much the same way — but it is not AUROC and nothing
says so.

**For contrast:** the *prediction* component of a blend does handle classification —
`_score_component` computes `2 × (AUROC − 0.5)` when `is_clf` is set. So one component
respects the flag and the other ignores it.

**Which datasets this touches.** Every classification dataset: the whole nucleotide domain,
the TCGA pair, `tz_her2_*`, `humanness_therapeutic`.

**Your options.**
- **Score classification as AUROC of the neighbour prediction** — matches what the blend's
  prediction component already does, and makes the two consistent.
- **Drop the `is_clf` parameter and document that class codes are treated as a rank
  target** — no number changes, the lie goes away.

**My recommendation:** AUROC. Two functions in the same file taking the same flag and
doing opposite things with it will cost someone a day later.

**>>**
Well I did extensive research on the datasets so what I need you to do is look online and look at what metrics other studies that use these datasets use. Do NOT hallucinate this I need to see proof from actual papers before I decide. 

---

### 3.3 The metric default is `r2` and the dataset registry is never consulted

*Cost to change: one line. It moves nine calls onto a different metric and a different
cache file.*

**What happens.** `find_best_hybrid(metric=...)` defaults to `'r2'`. The registry knows
each dataset's real metric — rmse for the molecular sets, spearman for antibody, mcc for
nucleotide — and the search never looks it up. It now prints a warning when the default
disagrees with the registry.

**Numbers.** 59 live calls to `find_best_hybrid`. 45 pass `dataset_name`. Nine pass no
metric at all, and all nine are in one file, `tests/complexity_theory_experiments.py`. The
rest either pass `metric=` explicitly or route through `tests/experiments/_shared.py`,
which reads the registry itself.

The change would be: `metric=None` resolves to the registered metric; an explicit
`metric=` always wins, so f18, the metric sweep, is untouched by design.

**Your options.**
- **Make the registry the default.** Those nine calls silently move from r2 to the real
  metric, and onto a different cache file, so they re-run.
- **Leave the warning and fix the nine calls by hand.**

**My recommendation:** make the registry the default. One legacy file is the whole blast
radius, and leaving it means every new caller has to remember.

**>>** Make metric a required argument. I don't want the registry as a default. You need to think of find_best_hybrid like a product that other people will use. They may not have a registry or want to use the same dataset as me. This brings up a larger question of what the user experience would be like for someone who wants to input their own dataset? They do have to specify a metric, right? Can they use their own dataset at this point? 

---

### 3.4 A blend weights representations by rank; a plain objective weights them by score

*Cost to change: the budget split changes on every blend run, so every f28 row re-runs.
Nothing else moves.*

**What happens.** With a blend like `{'predictive': 0.5, 'similarity': 0.5}`,
`blended_rep_scores` scores every representation under each component, then **ranks** each
component across representations and combines the ranks. With a plain objective the raw
score goes straight to the weighting scheme.

So the corner of f28's triangle — all the weight on one component — does not give the same
budget split as running that component on its own. The same representation still wins,
because ranking preserves order. What changes is the size of the gaps: ranks make the
best representation's lead over the second the same size regardless of whether it won by a
mile or a hair, and that gap is exactly what the weighting scheme turns into a share of the
budget.

**What makes the alternative possible.** `_score_component`'s own docstring says every
component comes back "on the SAME scale, bounded in [−1, 1] and higher-is-better". I read
all three branches: prediction returns a Spearman correlation, or `2 × (AUROC − 0.5)` for
classification; similarity returns a Spearman correlation; structural fidelity returns a
correlation. If that holds, the scores can be blended directly, the corner becomes the
plain objective, and the space is continuous.

**Your options.**
- **Blend on scores.** The corner equals the plain objective. Every stored blend row
  re-runs.
- **Keep ranks** and state in the write-up that a corner of the triangle is not the plain
  objective.

**My recommendation:** scores — but this is f28's design and the call is yours. The reason
I lean that way: if the corner is not the plain objective, the triangle cannot be read as
"how much does mixing help", which is what the experiment is for.

**>>** Blend on scores

---

### 3.5 A configuration scored on fewer folds than its rivals still competes

*Cost to change: nothing to compute. It changes which configuration wins in searches where
a fold failed.*

**What happens.** If one fold raises, that fold is skipped and the configuration's score
is the mean over the folds that survived. Only a configuration where *every* fold failed
raises. So a 3-fold mean and a 5-fold mean sit in the same ranking and `max()` picks
between them. Every result now records `n_folds_scored` and `n_folds_asked`, and a short
configuration prints a line saying so, but the ranking still treats them alike.

**What I have not measured.** How often a fold actually fails on real data. It needs a
full run to answer and I have not done one.

**Your options.**
- **Refuse a configuration that did not score every fold.** A search can then finish with
  fewer configurations than it started with; the reason is already printed.
- **Keep it and exclude short configurations at ranking time** — they stay in the stored
  rows, they just cannot win.
- **Leave it and read the warning.**

**My recommendation:** exclude at ranking time. Refusing throws away a row you might want
to look at; excluding keeps it and stops it winning.

**>>** Refuse a configuration that did not score on every fold. Throw a warning to the user to double check the groups 

---

### 3.6 A budget above the row count is dropped, and each fold invents its own replacement

*Cost to change: the affected datasets re-run. Fifteen of them at least.*

This was two items and is really one, so I merged them.

**What happens.** If you ask for a budget larger than the number of rows, the search drops
it, and instead appends the row count itself as a substitute budget. Under nested
cross-validation each outer fold holds a different number of training rows, so each fold
appends a *different* substitute. Those per-fold budgets then land in the same results
table as the real ones, and a substitute that only one fold produced gets a "mean over
folds" that is a mean over one fold.

There is a switch, `allow_budget_exceeds_n`, that turns the whole thing off and searches
the budget you asked for. It defaults to False. One experiment sets it.

**The provenance.** Roadmap Phase 5.1, decided 2026-07-15, says: "Uncap the budget. Remove
`find_best_hybrid.py:4816`." What shipped is this opt-in switch, defaulting to off. That is
softer than the decision, and I do not know whether the softening was deliberate.

**Numbers.** The experiment suite's standard budget list is
`[10, 25, 50, 100, 250, 500, 1000, 2000]`. With 5 folds a dataset's outer training block is
about 80% of its rows, so a budget of 2000 is dropped on any dataset under about 2500 rows.
Fifteen of the 38 registry entries that record a row count are under 2500 — every ginkgo
set, every abagym set, `humanness_therapeutic`, both TCGA sets, `herg_ki`, `caco2`. (32
registry entries record no row count, so the true number is higher.) Seven of them are
under 375 rows, where even the framework's own default of 300 is dropped.

**Your options.**
- **Flip `allow_budget_exceeds_n` to True.** The budget you ask for is the budget searched;
  the per-fold substitute disappears entirely, because it is only created on the dropping
  path. This is what Phase 5.1 decided. A budget above the row count means more features
  than molecules, which some models will not fit, and those configurations will fail —
  see 3.5, which decides what happens to them.
- **Keep the drop, but use one substitute for every fold** — the smallest outer training
  block, computed once. The folds become comparable; the substitute is still not a budget
  you asked for.
- **Leave it and read the warning.**

**My recommendation:** flip the default. It is the decision that was already taken, and it
makes the second option unnecessary. But it will produce failing configurations on the
small antibody sets, so answer 3.5 first or in the same breath.

**>>**

---

### 3.7 Three cuts inside the search, none of them measured

*Cost to change: removing any of them enlarges the search. Nothing stored becomes wrong.*

Your standing rule is that cutting an axis needs a stated reason — degenerate, broken, or
undefined — and that a theory about why something will lose is not a reason. These three
cut without one. All three now print what they removed. None of them is removed.

**(a) A per-model cap on the budget.** `MODEL_MAX_BUDGET` caps gp at 1000, knn at 2000, svm
at 500, and mlp, bnn, dnn and cnn at 1000. Trees are uncapped. The comment gives the curse
of dimensionality and wall time. The effect is that a budget row in the results table
compares a different set of models than the row above it — at budget 2000 there is no svm,
no gp and no neural network, so the "best model at 2000" is chosen from trees only.

**(b) Fold-0 pruning of allocation methods.** After the first fold, allocation methods are
ranked and only the best three survive; anything within 15% of the bottom is dropped. It
fires whenever more than three methods are searched. The default set has five
(`performance_weighted`, `power_2`, `softmax_T1`, `rank_exp_d0.3`, `wglobal_power2`), so
**two of the five default methods are deleted after one fold, on every default search**,
and leave no row anywhere.

**(c) A per-dataset budget cap.** `resolve_budget_cap` drops any budget above 500 on the
whole nucleotide domain and on thirteen named datasets including caco2, logd, herg_ki, qm9
and the three adme sets. Against the suite's standard list that removes 1000 and 2000 —
two of eight budgets. This one lives in `tests/experiments/dataset_policy.py`, the
experiment layer, not in `src/`.

**Three separate answers, please.** For each: keep it with a reason written down, measure
whether it costs anything, or remove it.

**My recommendation:** (b) is the one I would remove now — it deletes 40% of the default
allocation set on the strength of one fold, and one fold is the noisiest evidence in the
run. (a) and (c) are wall-time cuts on real cost; they want a stated reason more than they
want removal.

**>> (a) per-model budget cap:**

**>> (b) fold-0 allocation pruning:**

**>> (c) per-dataset budget cap:**

---

### 3.8 Waiting on an experiment, not on you

**No importance method is redundancy-aware.** All eight score each feature on its own, so
across representations carrying the same signal they pick that signal repeatedly. The
default set is `['deeplift', 'lightgbm_gain', 'treeshap']`, unchanged since July. Roadmap
Phase 4.1 required at least one redundancy-aware survivor. **f30 owns this** — 31 arms.
Listed here so the framework side is not forgotten when f30 reports: a survivor has to be
registered in `compute_feature_importance` and added to the default set. Nothing to decide
until then.

---

### 3.9 The budget split has a different shape on an RMSE dataset than on a Spearman one

*Cost to change: the budget split moves on every dataset whose scores are all positive, so
every stored search on those datasets re-runs. Nothing else moves.*

**Where this came from.** Phase 5.2 of the 2026-07-15 rebuild roadmap: "normalise per-rep
performance before allocation, so a low absolute metric doesn't starve the best rep. Test
BOTH schemes (min-max across reps AND rank-based) in the same sweep, then cut to one."
Rank-based shipped — `rank_linear` and `rank_exp_d0.1/0.3/0.5` are in the allocation set.
Min-max never did, and the sweep was never run.

**What happens instead.** Before the weighting scheme sees them, the per-representation
scores go through `_sanitize_baseline_scores`, which subtracts the minimum **only when the
minimum is negative**. So the same five representations, in the same order, with the same
gaps between them, split a budget of 100 two different ways depending on the sign of the
number the metric produced. Measured today by calling the two functions directly:

| representation scores | resulting split |
|---|---|
| 0.700 0.697 0.695 0.694 0.693 (Spearman) | 20 · 20 · 20 · 20 · 20 |
| −0.300 −0.303 −0.305 −0.306 −0.307 (−RMSE) | 50 · 29 · 14 · 7 · 0 |

The molecular datasets score RMSE, so `lower_is_better` negates and they take the second
row. Antibody datasets score Spearman and take the first. A representation's share of the
budget is being set by where zero happens to sit on the metric's scale.

Two consequences worth naming. On the negative side the worst representation gets **exactly
zero features**, always — the remainder loop hands leftovers to the best performers first,
so it never recovers. On the positive side the split is near-equal however far apart the
representations really are, and nothing prints, because the equal-split warning only fires
when every weight is zero.

This also corrects a line in the similarity roadmap's §B: it says that under a geometry
objective `entropy`'s worst representation "is at exactly 0 and the penalty cannot move it".
That holds only when some representation scored negative. Read-across correlations are
usually all positive, where no shift happens at all.

**Your options.**
- **Min-max: divide by the range as well as subtracting the minimum.** The worst
  representation is at 0 on every dataset and the split stops depending on the sign. For
  `linear`, `power_law`, `thresholded` and `score_gap` this is exactly the second row above
  on every dataset; `softmax` is the one scheme where dividing by the range is not just a
  rescale.
- **Subtract the minimum always, not only when it is negative.** Same as min-max for every
  scheme except `softmax`, and one character to change.
- **Leave it and read the scores as they come.** Then the honest description of `linear` is
  "proportional to the score's distance from zero", and the Spearman datasets are running a
  near-equal split under the name `performance_weighted`.
- **Register min-max as a fourth normalisation and sweep it against raw and rank**, which is
  what Phase 5.2 asked for. That is an f-numbered experiment, not a code change.

**My recommendation:** sweep it. This is a budget-split rule and every one of the others —
softmax temperature, power, rank decay — was chosen by measurement. Changing it by argument
is the thing the roadmap said not to do. If you would rather not spend the run, the second
option is the cheap one and makes the two domains behave alike.

**>>**

---

### 3.10 The objective is in no cache metadata and in no result-database column

*Cost to change: every `searches/*.pkl`, `baselines/` and `hybrid_evals/` entry stops
matching and recomputes. The result databases must be deleted — which `c74905b` already
requires for a different reason, so today that half is free.*

**Where this came from.** §H of the similarity roadmap, 2026-08-26, listed as "what to do
about it, not yet decided". Both halves were re-read in the code today and both still stand.

**What happens.** Every cache store writes a metadata block that is compared on read; a
mismatch turns a wrong hit into a miss. `_make_search_meta`, `_make_baseline_meta` and
`_make_hybrid_eval_meta` between them record 60-odd fields — representation names and
shapes, label and group hashes, budgets, folds, seed, model, evaluator, augmentation
settings, the pruning schedule. **None of the three records the objective.** The separation
between a predictive result and a geometry result rests entirely on the objective being
folded into the filename, one level up. That is why the two faults §A of the similarity
roadmap describes were silent rather than loud: the filename was the only thing standing
between them and a wrong answer.

The result database is the same story in a different form. `IDENTITY_COLS` is `tier`,
`labels_hash`, `groups_hash`, `n_folds`, `seed`, `cv_splitter_id`, `metric`, `task_type`.
The objective is not among them — it is appended into `tier`, which is one free-text string
now carrying four unrelated things: the representation pool, whether the training labels
were noised, whether this is the distance lineage, and the objective. Four separate pieces
of code each remember to append their own marker in the right order.

**The timing.** `metric` and `task_type` were added to `IDENTITY_COLS` on 2026-09-04, and
`_ensure_schema` already raises on every database written before that date. So every
`result_db/*.db` is already being deleted. Adding `objective` in the same breath costs
nothing extra; adding it later costs a second wipe.

**Two answers, please.**

**>> (a) objective in the metadata of the search, baseline and hybrid-eval stores:**

**>> (b) objective as its own identity column in the result database:**

**My recommendation:** both, and (b) now while the wipe is already happening. (a) converts a
whole class of fault from silent-wrong to loud-miss and costs one recompute.

---

### 3.11 Phase 5.4's probe — spend part of the budget on components of what was not selected

*Cost to change: nothing is stored either way. Building the probe is an experiment.*

**Where this came from.** Phase 5.4 of the 2026-07-15 roadmap, written as a side experiment:
split a budget of D into D1 individually selected features plus D2 principal components of
the features the selection did **not** take, from the same representations — so the weaker
signal is compressed rather than discarded. The roadmap's own words: "just test whether this
retains weaker signal usefully — do not properly implement it yet", and it noted the idea
runs against the standing preference for importance filtering over PCA.

**What exists.** Nothing. No allocation method, no experiment, no result file. Checked today
across `src/kirby/hybrid.py`, `find_best_hybrid.py` and `tests/experiments/framework/`.

**Your options.**
- **Build it as an f-numbered probe.** One arm per D1/D2 ratio against plain selection at the
  same total width. It is a scoring change inside `create_hybrid`, not a new axis in the
  search, so it does not enlarge anything already running.
- **Drop it and write the reason down.** It has sat untried for seven weeks and it competes
  for f-numbers with f26 through f31, which are all live.

**My recommendation:** drop it, and say so in the roadmap. Not because it would lose — I have
no evidence either way, and that is not a reason to cut. Because it is the one item on that
roadmap you wrote as optional, and every other open framework question in this file is
about a path that experiments are running through today.

**>>**

---

## 4. Framework experiments write here

One heading per experiment. Say what the experiment measured, what it decided, and which
of §3 it closes.

### f31 — tuning the internal evaluator does not pay — 2026-09-06

**Recommendation: do not tune the internal evaluator's hyperparameters.** No setting beat
its family's shipped setting on all three datasets, and on every dataset the fold drawn
moved the score more than the setting chosen did.

**Scope of that recommendation.** Every setting here was compared against its OWN family's
shipped setting — mlp settings against `mlp_default`, dnn against `dnn_default`, and so on.
The families were never compared against each other on equal footing, so this says nothing
about WHICH internal evaluator to use, and nothing about whether the evaluator axis can be
narrowed. That is a separate experiment.

**What the internal evaluator does.** It scores each representation on its own inside
`allocate_performance_weighted`, and those scores are what the quota allocators divide the
budget by. Its hyperparameters are literals in `hybrid._evaluate_allocation`. Nothing tunes
them: `find_best_hybrid(tune=True)` tunes the downstream model, and `_HP_SEARCH_SPACES` has
no `dnn` entry at all.

**Setup.** `tests/experiments/framework/f31_internal_evaluator_tuning.py`, local, run
2026-09-05 to 2026-09-06. Three datasets on their post-pruning ten representations —
buchwald_hartwig (3955 rows, r2), caco2 (2158, r2), flab_expression_koenig (4275,
spearman). Every metric higher-is-better, each dataset ranked on its own and never pooled.
Budget 100, `lightgbm_gain`, three quota allocations (`performance_weighted`, `power_2`,
`rank_exp_d0.3`), downstream lightgbm and rf, 3 outer folds. 20 settings across five
evaluator families, injected through the callable branch of `_evaluate_allocation`. Each
result file records every setting's resolved parameters.

**Checked before the grid ran, on real caco2 data.** Each `*_default` arm selects the
identical features as the shipped evaluator it reproduces, and precomputing the feature
importance once per fold gives the identical budget split and identical selected features
as letting `create_hybrid` compute it. `--verify-defaults` asserts both.

**Result — paired wins against each family's shipped setting, 18 cells per dataset
(3 folds x 3 allocations x 2 downstream models). Chance is 27 of 54.**

| setting | buchwald | caco2 | koenig | total |
|---|---|---|---|---|
| svm_default_allrows | 11/18 | 11/18 | 11/18 | 33/54 |
| mlp_512_allrows | 6/18 | 15/18 | 11/18 | 32/54 |
| rf_200_dNone_sqrt | 10/18 | 8/18 | 14/18 | 32/54 |
| dnn_wide_512_256_128 | 12/18 | 10/18 | 10/18 | 32/54 |
| svm_C10_allrows | 14/18 | 5/18 | 9/18 | 28/54 |
| mlp_256_128_a0.01 | 6/18 | 11/18 | 11/18 | 28/54 |
| mlp_default_allrows | 5/18 | 15/18 | 7/18 | 27/54 |
| dnn_do0.5_lr3e-4_e100 | 11/18 | 6/18 | 10/18 | 27/54 |
| dnn_small_128_64 | 10/18 | 8/18 | 7/18 | 25/54 |
| lgbm_200_d10_lr0.05 | 12/18 | 7/18 | 6/18 | 25/54 |
| mlp_256_128_allrows | 5/18 | 10/18 | 10/18 | 25/54 |
| dnn_e50_p10 | 8/18 | 10/18 | 7/18 | 25/54 |
| mlp_256_128_lr0.01 | 6/18 | 11/18 | 5/18 | 22/54 |
| svm_C100_gauto_allrows | 8/18 | 8/18 | 5/18 | 21/54 |
| lgbm_200_dfree | 10/18 | 3/18 | 7/18 | 20/54 |

The top score is 33 of 54, from a setting that is 11/18 on each dataset. Every setting with
a high count on one dataset has a low one on another: `mlp_default_allrows` is 15/18 on
caco2 and 5/18 on buchwald; `svm_C10_allrows` is 14/18 on buchwald and 5/18 on caco2.

**Fold noise against the size of the effect.** Standard deviation of one fixed setting
across the three folds, against the full span of that dataset's table of setting means:

| dataset | fold-to-fold sd | span of the table |
|---|---|---|
| buchwald_hartwig | 0.0437 | 0.0253 |
| caco2 | 0.0724 | 0.0179 |
| flab_expression_koenig | 0.0195 | 0.0085 |

On all three the noise is larger than the whole table, which is why the paired counts above
are the statistic and the means are not.

**The 500-row fit cap (`_SCORER_FIT_MAX_SAMPLES`) is not costing anything reliable.** The
svm and mlp scorers fit on at most 500 rows against 1438-3401 training rows on these
datasets. Lifting it on the MLP is +0.0156 on caco2, -0.0081 on buchwald and -0.0013 on
koenig. On the SVM it is positive on all three and never larger than +0.0165. Leave it.

**The setting always changes what gets selected.** 20 distinct budget splits from 20
settings, in nearly every fold-by-allocation combination on all three datasets. The
internal evaluator's hyperparameters do move which representations get features. The
downstream score does not follow.

**Three settings were cut on cost, 2026-09-05.** `rf_200_dNone`, `rf_200_d10` and
`lgbm_500_dfree_lr0.01` were 13.6 of buchwald's 21.0 hours and 15.0 of koenig's 20.4.
`rf_200_dNone_sqrt` — the same forest with `max_features` capped — scored 0.4750 vs 0.4727
(caco2), 0.6856 vs 0.6875 (buchwald) and 0.7179 vs 0.7201 (koenig) for between a fortieth
and a hundred-and-fortieth of the time. They are refused by name in the experiment file;
see `DROPPED_SETTINGS`. This is a statement about three settings inside two families, not
about the families. Absolute build seconds here are not a clean cost measure at all — the
three datasets shared one machine, so `mlp_default` reads 19s on buchwald and 1821s on
caco2. Nothing in this experiment measures the cost of one evaluator against another.

**Closes nothing in §3.**

**One defect found, not fixed.** `hybrid.py:2423`, the `dnn` internal evaluator, catches
every exception and runs the MLP instead while still reporting itself as `dnn`. f31 builds
`DNNRegressor` directly and does not go through that branch, so nothing here is affected,
but any run using `internal_evaluator='dnn'` can silently be an MLP.

**One thing to know about the numbers.** The first pass of this experiment finished on the
engine as it stood before `c74905b`, where `create_hybrid` computed importance over every
row it was given rather than the internal training part only. All three datasets were
re-run on the fixed engine; the tables above are the re-run. The pre-fix files are kept in
`tests/results/framework/superseded/`. The two runs disagree — under the old engine lifting
the MLP row cap on caco2 was +0.0004 and 9/18, under the new one it is +0.0156 and 15/18.

---

## 5. Decided

### The rebuilt hybrid now splits its budget the way the search did — 2026-09-04

Was the largest open item in this file. The honest evaluator's hybrid arm rebuilds a
recipe fold by fold; it used to inject `Blueprint.rep_scores` — one number per
representation, its best score across models at full width over the whole search — where
the search itself had used a per-fold score measured on that fold's own internal carve.
Two different quantities, so the recipe scored on the untouched fold was not the recipe the
search picked.

**Decided: score each fold's representations on that fold's own training rows.** Signed off
2026-09-04, committed in `c74905b`. Measured cost: one internal-evaluator fit per
representation per rebuild, 151 seconds for ten representations on a 1726-row caco2 fold.
The stored scores are read in one case only — a rebuild with no labels at all, inference on
unlabelled rows, where there is nothing to score with.

### The budget-split scores are measured on the rows `create_hybrid` actually holds out — 2026-09-04

The search precomputed those scores on a `GroupShuffleSplit` draw while `create_hybrid`
carved with the first fold of a `GroupKFold`, under a comment saying the two matched. There
was no leak — both sat inside the fold's training rows — but the injected scores were not
the ones `create_hybrid` would have computed. Both now call one function,
`hybrid.internal_holdout_split`. Committed in `c74905b`.

### The result database now keys on the metric — 2026-09-04

Two runs over the same rows and folds under different metrics used to share a row, and
whichever ran first answered for both. `metric` and `task_type` are now identity columns in
`result_cache`. **This widens the primary key: every `result_db/*.db` written before
2026-09-04 has to be deleted.** There is no migration; `_ensure_schema` raises and names
the file. Committed in `c74905b`.

### One noise spec can still produce two different sets of labels — no fix, watching

`make_noisy_labels` takes `groups`, `reference` and an explicit `magnitude`. None of the
three is carried in `spec_id`, and `spec_id` is what the database identity is built from,
so two callers passing different `reference` values would write different noisy labels
under the same identity. `find_best_hybrid` passes none of the three, and a spec that needs
groups raises without them, so the framework path cannot reach this today and no experiment
mixes the two. Left alone deliberately — the fix is to hash the realised labels into the
identity, which orphans every stored noisy row. Revisit if a caller starts passing
`reference`.

---

## Speed tiers — DECIDED AND IMPLEMENTED 2026-09-06

Set by the owner. `speed='fast'` on `find_best_hybrid` cuts four axes; `speed='full'`
is the default and is the existing behaviour unchanged.

**It was not actually callable until 2026-09-09.** The lists and the code that reads
`speed` went in, but no `speed` parameter did, so every call to `find_best_hybrid`
raised `NameError: name 'speed' is not defined` before doing any work. A second defect
went with it: the shortened lists were built for the cache key and then thrown away, so
`_run_search` fell back to the full lists and a fast run searched full while recording
that it had searched fast. Both fixed 2026-09-09, and `speed='fast'` now measurably
cuts the grid.

| axis | fast | full | extra (never auto-selected) |
|---|---|---|---|
| downstream model | `rf`, `lightgbm` | + `xgboost`, `svm`, `dnn` | `mlp`, `gp` |
| internal evaluator | `rf`, `match` | `match`, `rf`, `svm`, `lightgbm`, `mlp` | — |
| importance method | `deeplift`, `lightgbm_gain` | + `treeshap` | — |
| allocation | `performance_weighted`, `power_2`, `rank_exp_d0.3` | + `softmax_T1`, `wglobal_power2` | — |

Searched grid per (model, budget): **12 configurations at fast, 75 at full.**

### Which axis lives where

`find_best_hybrid` answers one question: *for this dataset, this downstream model and
this budget, which features?* So the two axes in the question are the ones the result
is reported against, and the three that answer it are collapsed to a single winner.

| axis | role | where |
|---|---|---|
| budget | part of the question | caller supplies `budgets=`; result keyed `best_by_model_budget[model][budget]` |
| downstream model | part of the question | caller supplies `models=`; same key |
| allocation method | searched, collapsed | inside the search grid |
| importance method | searched, collapsed | inside the search grid |
| internal evaluator | searched, collapsed | inside the search grid |
| objective | searched, collapsed — **under test now** | `objective=` defaults to `'predictive'` |

### Constants

`src/kirby/find_best_hybrid.py`:
`FAST_MODEL_TIER`, `FAST_INTERNAL_EVALUATORS`, `FAST_IMPORTANCE_METHODS`,
`FAST_ALLOCATION_METHODS`, against `DEFAULT_*` for full. An explicitly passed list
always beats the tier. The per-model `'tier'` label in `DEFAULT_MODELS` already
existed (set 2026-09-04) but nothing branched on it; `speed='fast'` is what makes it
select.

### Where the fast lists came from

**Allocations** — the top three by the c32 criterion: how often a weighting|mechanism
pair lands in the top three of its cell, over 54 cells (dataset x model x budget),
qm9 excluded, each pair scored by its best over N.

| pair | top-3 | top-1 |
|---|---|---|
| linear \| quota (`performance_weighted`) | 30/54 | 9/54 |
| power_2 \| quota | 27/54 | 11/54 |
| rank_exp_d0.3 \| quota | 26/54 | 7/54 |
| softmax_T1 \| quota | 25/54 | 11/54 |
| softmax_T1 \| global | 18/54 | 9/54 |
| power_2 \| global (`wglobal_power2`) | 14/54 | 3/54 |

**Importance** — the f30 bake-off on caco2 (28 methods, 3 scaffold-grouped folds,
budget 100, delta against the best single representation):
`deeplift` +0.0144 (271s), `lightgbm_gain` +0.0100 (42s), `treeshap` +0.0067 (542s).
The ordering held in all four cells; `treeshap` is last every time and the dearest of
the three, which is why it is in full and not in fast. One dataset, three folds — the
gap between the top two is inside fold noise.

### Cache

**A fast run can be topped up into a full one.** Corrected 2026-09-09. The paragraph
that stood here said the opposite, and it was wrong.

Each finished configuration is one row in `<cache>/result_db/<dataset>.db`. The
allocation method, the importance method, the internal evaluator, the downstream model
and the budget are all columns of that row. What decides whether two runs are the same
search is the data, the fold split, the seed, the metric, the task type and the tier,
and nothing else — `IDENTITY_COLS` in `src/kirby/result_cache.py`. None of the four
axes is in it.

**How to top one up.** Run it again with `speed='full'`, and the same dataset name,
tier, metric, fold count and seed. Every recipe the fast run stored is read back out of
the database and only the new recipes are computed. Nothing needs deleting first. The
fast lists are subsets of the full ones, so every fast row is reusable.

Measured 2026-09-09 on a 60-row synthetic set: a search over two recipes stored two
rows, and a later search over eight recipes on the same data reused both and computed
six.

The per-fold importance scores and allocation scores are stored under
`<cache>/fold_scores/<dataset>/`, one file per fold and per method, so adding an
importance method or an evaluator costs only the one that was added.
