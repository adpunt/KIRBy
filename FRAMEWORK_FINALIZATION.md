# Framework finalization

**What this file is.** The list of things still open in the framework — `src/kirby/` —
before it is finished. Not the experiments. Every line below was read in the code on
2026-09-03, §3 was re-read against the code on 2026-09-04, and §3 and §4 were re-read
against the code again on 2026-09-09. Nothing in §1-§4 is a measured result except where
a measurement is named with the files it came from. §5 is where experiment results go.

**All eight of §3 are closed as of 2026-09-13.** §8 is the read-through done immediately
before the hybrid precompute: what metric direction was checked against, the three things
that were wrong and are now fixed, and the one line left that needs you (§8.6).

**To work through it, go to §3 and answer the `>>` lines.** Five items, eight answers.
§4 is the implementation list from your 2026-09-09 pass and needs nothing from you.
Each one says what it is, what changes either way, the numbers I could count, and what I
would do. Nothing outside §3 needs anything from you.

**How the §1 fixes were checked.** Every objective path — predictive, `similarity`,
`struct_fid`, and a blend — was run end to end on synthetic data through both
`find_best_hybrid` and `nested_cv_evaluate`, and the predictive and caching paths were run
on real caco2 (2158 molecules, three fingerprint representations, 3 folds). The six
existing framework unit tests pass. Nothing was run on the cluster.

**Who writes here.** Chats working on framework *experiments* add their findings and
their decisions to §3 and §5. This file is the one place open framework questions live.
When something is decided, move it to §4 with the date; when it is implemented, move it
to §6.

**Scope.** `find_best_hybrid.py`, `hybrid.py`, `result_cache.py`, `registry.py`,
`screen.py`, `noise_spec.py`. Experiment files are somebody else's list.

---

## 0. What the framework is FOR

**⛔ THERE IS NO CACHE. It was wiped.** Every "this invalidates every stored search",
"every result_db must be deleted", "this re-runs 22 datasets" in this file is describing a
cost that does not exist. Nothing below should be weighed against the price of recomputing
something, because there is nothing on disk to recompute. If a change is right, make it.


Written at the top on 2026-09-10 because it keeps getting lost, in your words:

> "I care about finding the best set of features (best hybrid) for a given dataset,
> model, and budget. That is the goal. A good way to compare against a base rep is with
> the same budget. [...] If I find a lower-budget alternative for the hybrid that's even
> better, think of budget as the max."

**The budget is a MAXIMUM, not a target.** Three rules follow, and every one of them is
something the code has to keep:

1. **A hybrid may come back narrower than its budget.** It does whenever an importance
   method cannot justify the whole quota. `concatenate_features` stops at the last
   feature scoring above zero and says so, instead of padding the rest out by column
   order — which is what it did until 2026-09-10, and `lightgbm_gain` scores only 201 of
   ecfp4's 2048 features above zero on caco2.
2. **A hybrid must never come back wider than its budget.** Anything appended outside it
   breaks the comparison the budget exists to make.
3. **The comparison is at the SAME budget.** A hybrid of N features is measured against a
   base representation cut to N features by the same importance method on the same fold.
   Not against that representation at full native width. `nested_cv_evaluate` does the
   latter today — the baseline arm fits `Xr[tr]`, every feature the representation has —
   so every delta the framework has ever reported mixes the method with the width. That
   is the open item at §3.8.

**What this framing is NOT.** It is not a reason to raise the per-dataset budget cap. The
evidence that looked like one — the honest delta growing with budget in 161 of 211 stored
cells — was measured against full-width comparators, so a larger budget was partly just
closing the width gap. Under rule 3 that trend has to be re-measured before it means
anything.

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
   saying so. **What to do about it was decided on 2026-09-09 and implemented on 2026-09-10:
   the configuration is refused.** See §4. (This line used to point at §3.1, which is the
   allocation default and has nothing to do with it.)

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

**Eight decisions, in five items.** Everything you answered on 2026-09-09 has moved to §4, which is
the implementation list. Two of the five below are here because your answer asked for a
number I did not have; both numbers are now in the item. One is here because two things
this file told you were wrong. One is still blank.

**Two more, from the cache audit of 2026-09-10, live in `CACHE_AUDIT_2026_09_10.md` §4.**
That audit opened with nine; seven were closed on 2026-09-10 once the owner pointed out that
the stored-score cost keeping them open no longer existed, and they are implemented. What is
left is c10's recipe being chosen on the true labels it then corrupts, and whether the per-fold
score cache should exist at all. Both are experiment-side or design questions, not `src/`
identity questions.

**Three things in `src/` changed with them, and they widen what a stored score is filed under.**
`rep_pool` moved from recorded data into the matched columns, keyed on (name, feature count);
`tune` became a matched column; and an estimator that is not what the registry builds for its
name now carries a settings fingerprint in the per-representation score files and in the stored
row. Every score store therefore has to be deleted -- free, they were already empty.

**One thing that file establishes changes how §1 item 18 and §4's caching paragraph read.**
`searches/*.pkl` is not written any more — the path is derived and never saved to — so
`_make_search_meta` survives only as checkpoint metadata. Every guard that lives there alone,
including `rep_names` and `rep_shapes`, protects a fold-0 recompute and nothing durable.

---

### 3.1 Which budget split `create_hybrid` uses when it is not told — the top two tie

*Cost to change: nothing. Nothing stored, nothing re-run.*

**You asked for whatever was top of the framework experiments.** Here is the count, and it
is a tie.

**IMPLEMENTED 2026-09-10: `performance_weighted`.** On the signature and in the module
docstring. Checked on real caco2: at budget 100 it spread 100 features across 9 of the 10
representations, largest share 32; `top_3` would have used three.

**First, the part you doubted — true when this was written, not now.** `top_3` WAS still the
default, on the signature and stated again in the module docstring. It was never
replaced. What was replaced is the *searched* set — `top_2/top_3/top_5` came out of
`DEFAULT_ALLOCATION_METHODS` on 2026-09-01 because they weld a representation count into
the weighting slot. The default argument was left behind.

**Where the numbers come from.** The c32 cross sweep, 30 files on
`/Volumes/seagate/kirby_cache/c32_allocation_cross`. A cell is one dataset × model × budget
— 9 datasets × 3 models × budgets 500 and 1000 = 54 cells. Ranked by `r2_delta_mean`, which
is that strategy's R² minus the best single representation's R², averaged over 5 folds;
positive means it beat using the single best representation on its own. One row per
allocation method, each giving all ten representations a share:

| allocation method | cells won | cells 1st or 2nd | cells beating the best single rep | median R² margin when it wins |
|---|---|---|---|---|
| `performance_weighted` | 14 of 54 | 29 | 44 | 0.0009 |
| `power_2` | 14 of 54 | 28 | 43 | 0.0017 |
| `softmax_T1` | 11 of 54 | 19 | 44 | 0.0042 |
| `rank_exp_d0.3` | 8 of 54 | 17 | 43 | 0.0032 |
| `wglobal_power2` | 7 of 54 | 15 | 45 | 0.0023 |

**The tie.** `performance_weighted` and `power_2` win 14 cells each. Second place breaks it
29 to 28 for `performance_weighted`. Margin-when-it-wins breaks it the other way — `power_2`
is 0.0017 R² ahead of second place when it wins, against 0.0009.

**One number that points elsewhere.** c32 measured 32 strategies, because it also swept how
many representations get a share. Ranked over all 32, the single most-winning strategy is
`softmax_T1` at all ten representations, 6 of 54 cells. It only drops to third in the table
above because the five-way comparison hands its cells back to the shorter `power_2` and
`linear` variants that are no longer searched.

**Your options.** All four cost nothing — this is the default on a function argument, and
no stored result used it.
- **`performance_weighted`** — tied on cells won, ahead on second places, 29 to 28. The
  plainest of the five: a representation's share of the budget is its score divided by the
  sum of the scores.
- **`power_2`** — tied on cells won, and when it wins it wins by twice as much, 0.0017 R²
  against 0.0009. It squares each score before dividing, so the better representations take
  a bigger share.
- **`softmax_T1`** — top of all 32 strategies c32 measured, third of the five here. Worth
  knowing: it responds to the *scale* of the scores rather than the gaps between them, which
  is the property §3.5 is about.
- **Keep `top_3`** — you would be keeping a method that is deliberately not searched.

**>> which one becomes the default in `create_hybrid`:** performance_weighted

**DONE 2026-09-10.** Three doors into the same engine in `tests/experiments/_shared.py` —
`create_hybrid_cached`, `evaluate_cv`, `evaluate_cv_with_train` — still said `top_3`, so they
would have disagreed with the engine about what "no allocation method given" means. All
three now track it. Latent, not live: every current caller names the method. The method is in
`create_hybrid_cached`'s own fold cache key, so the change writes new keys rather than
reusing old ones.


---

### 3.2 Classification under `objective='similarity'` — CLOSED, 2*(AUROC-0.5) in `7663136`

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

**Your answer on the last pass:** "look online and look at what metrics other studies that
use these datasets use. Do NOT hallucinate this I need to see proof from actual papers
before I decide."

**What the source papers report — read online 2026-09-09, every row has a link below.**
One row per dataset family in the registry. "Registry metric" is what
`tests/experiments/dataset_policy.py` records today.

| dataset family | registry metric | what the source papers report |
|---|---|---|
| `nt_*` — 17 entries | `mcc` | MCC. The Nucleotide Transformer paper's own words: *"For a consistent performance evaluation across tasks, we used the binary or multi-class Matthews Correlation Coefficient (MCC) as metric as it is robust to uneven label ratios."* One metric across all 18 of its tasks. |
| `gb_*` — 4 entries | `mcc` | Accuracy and F1. Genomic Benchmarks: *"The accuracy and F1 score for PyTorch and Tensorflow CNN models on all genomic benchmark datasets are shown in Table 2."* It reports no MCC. |
| `tcga_luad_vs_lusc` | `auc` | AUC and accuracy. The gene-expression classification papers report per-gene and per-model AUC; a 2026 methylation study reports AUC-ROC 0.9981 with 96.92% accuracy. |
| `tcga_brca_pam50` | `f1_weighted` | Accuracy and **macro** F1 — one pipeline reports accuracy 0.9114 with macro F1 0.8964. The papers choose macro deliberately: *"metrics such as macro F1 score, which equally weights all classes, provide a more informative assessment in imbalanced datasets"*. Weighted F1 does the opposite of that. |
| `humanness_therapeutic`, `humanness_species` | `auc` | ROC AUC. Hu-mAb 0.977 and OASis 0.966 in the BioPhi paper; a later comparison puts OASis at 0.975 and Hu-mAb at 0.979. `humanness_therapeutic` is IMGT/mAb-DB via BioPhi, so this is the same benchmark. |
| `tz_her2_binary`, `tz_her2_multiclass` | `auc`, `f1_weighted` | **Not established.** Mason et al. 2021 (Nature Biomedical Engineering) is the source of the trastuzumab-HER2 variants, and I could not read a headline classification metric out of the abstract or the openly available text. It reports a screening outcome — 30 of 30 predicted variants retained binding — not a held-out metric. |

**What this settles and what it does not.** No single classification metric is standard
across these families. Two of them use a correlation-shaped metric (MCC), two use AUROC,
two use F1, and the F1 they use is not the F1 in the registry. So "score classification as
AUROC" is right for the antibody and TCGA-binary sets and wrong for the nucleotide domain,
where the whole benchmark is MCC.

**Two registry entries disagree with their own source**, found while doing this. Neither is
this item's decision, and neither is a defect until you say so:
- `gb_*` is registered as `mcc`; Genomic Benchmarks reports accuracy and F1. The four `gb_*`
  entries sit beside 17 `nt_*` entries that are genuinely MCC, which would explain it.
- `tcga_brca_pam50` is registered as `f1_weighted`; the papers on that exact task report
  macro F1, and say in as many words that weighted averaging hides the minority subtypes.

**What the function actually computes.** Read in the code on 2026-09-09,
`find_best_hybrid.py:1734`. This matters for the choice, because it decides which metrics
are even available.

Inside the held-out fold it builds the pairwise Euclidean distance matrix from the scaled
embedding, splits those held-out rows again by scaffold group (`GroupKFold`, up to 5
splits), and for each row predicts its label from its 5 nearest neighbours with
`KNeighborsRegressor(weights='distance')`. On a classification dataset the labels are 0 and
1, so **that prediction is the distance-weighted fraction of a molecule's 5 neighbours that
are in class 1** — a continuous number between 0 and 1, not a hard vote. It then Spearman-
correlates those fractions against the true 0/1 labels.

So the prediction is already a graded score. AUROC can be computed from it directly. MCC,
accuracy and F1 cannot — they need a hard label, so they need a cut-off, and 0.5 would be an
invented one.

**It is computed in two places, not one, and only one of them has an `is_clf` argument at
all.** `find_best_hybrid._readacross_objective_score` (line 1734) scores a whole
configuration under `objective='similarity'`; it takes `is_clf` and never reads it. The
budget split goes through a second copy — `hybrid._score_component`'s similarity branch at
`hybrid.py:270`, which calls `activity_readacross` in
`kirby/utils/noise_mitigation_methods.py`. That one returns Spearman too and has no `is_clf`
parameter at any point. Whatever is decided here has to land in both, or the score a
configuration is judged by and the score its budget is split by stop agreeing.

**This item changes classification only.** On a regression dataset `is_clf` is False and
both functions already do the intended thing — predict the activity from the 5 nearest
neighbours and take the Spearman correlation. Nothing about the regression path moves, and
no regression result recomputes.

**But the two branches then have to live on one scale.** These scores are what the budget
split is proportional to, and a blend can mix a similarity score with a prediction score.
A correlation on regression and a rate on classification are not the same scale: a
correlation runs −1 to 1 with chance at 0, a class-averaged rate runs 0 to 1 with chance at
0.5 for a binary set. `_score_component`'s docstring promises every component comes back
"on the SAME scale, bounded in [−1, 1] and higher-is-better", and it keeps that promise for
prediction by returning `2 × (AUROC − 0.5)` rather than the AUROC. A rate chosen here needs
the same treatment for that promise to survive.

**The second literature question — how does anyone score a space by whether neighbours share
a label?** This is not "what metric do papers report on these datasets", which is the table
above. It is "what is the published name for this measurement". Two families, and the
docstring already names the first one.

| where it comes from | what it measures | what it reports |
|---|---|---|
| **MODI**, the modelability index (Golbraikh, Muratov, Tropsha, *JCIM* 2014) — the docstring cites it | For each molecule, does its **1** nearest neighbour share its class | The fraction that do, **averaged across the classes** so an imbalanced set cannot inflate it. Calculated on 100+ datasets; 0.65 separated the datasets a QSAR model could be built on from the ones it could not |
| **RMODI**, its regression form (Luque Ruiz & Gómez-Nieto) | Same idea on a continuous activity | It does **not** correlate anything. It discretises the activity first — inside or outside 0.65 standard deviations — and then runs the classification measure. So in the published lineage the classification form is the primitive and regression is derived from it, which is the opposite of what this code does |
| **kNN probing**, the standard way to score a learned embedding in representation learning (DINO and its lineage) | Predict each held-out point's class from its k neighbours, with a distance- or similarity-weighted soft vote — the same soft vote `weights='distance'` performs here | **Top-1 accuracy.** DINO reports 78.3% on ImageNet this way |

Neither family correlates anything, and neither uses AUROC. Both report a **label-agreement
rate**: what fraction of the time the neighbours got it right. MODI adds the class-averaging
that stops an imbalanced dataset scoring well for free.

**One thing this decision drags in.** §3.5 is about whether a score has a meaningful zero,
because the budget split is proportional to it. Here is where each option sits: Spearman is
0 at chance, MCC is 0 at chance, AUROC is 0.5, accuracy is the majority-class rate, and
class-averaged MODI is 0.5 on a binary set. Only the first two are already anchored.

**Your options.**
- **Leave it — Spearman on the neighbour fractions, and delete the `is_clf` argument that
  is never read.** Nothing recomputes. Zero means chance, which §3.5 wants. The objection is
  that no published work scores a classification neighbourhood this way, and the write-up
  has to say what the number is rather than name it.
- **AUROC of the neighbour fractions**, as `2 × (AUROC − 0.5)` so chance sits at zero.
  Consistent with `_score_component`, which already does exactly this for the prediction
  half of a blend. Uses the graded fractions, needs no cut-off. It is the metric the
  antibody and TCGA-binary sources report, and not the one the nucleotide benchmark reports.
- **The dataset's registered metric.** MCC on nucleotide, AUROC on antibody and TCGA-binary,
  F1 on the two multiclass sets. `objective='similarity'` and `objective='predictive'` then
  report in the same units. Cost: MCC and F1 need a cut-off on the neighbour fraction that
  nothing in the code or the literature sets, and F1's zero is the open question in §3.5.
- **Class-averaged label agreement — MODI's own definition, extended to the 5 neighbours
  already computed.** The measurement gets a published name and a published threshold, and
  the class-averaging handles the imbalanced nucleotide sets. Cost: it is a rate, not a
  correlation, so it is on a different scale from the regression form of the same objective,
  and its chance level is 0.5 rather than 0.

**Sources for this half.**
- MODI — <https://pubs.acs.org/doi/abs/10.1021/ci400572x> and <https://pubmed.ncbi.nlm.nih.gov/24251851/> ; reference implementation <https://github.com/molecularmodelinglab/MODI>
- RMODI and the rivality index — <https://link.springer.com/article/10.1007/s10910-018-0972-8>
- kNN probing of learned embeddings — <https://arxiv.org/pdf/2408.07519> and <https://arxiv.org/pdf/2311.00562>

**>> how `objective='similarity'` scores a classification dataset:**
**CLOSED — `2*(AUROC - 0.5)`, committed in `7663136` on 2026-09-09. The line was left blank
because it had already been handled.** Moved here from the open list on 2026-09-13.

*(the original note:*
*`7663136`, 2026-09-09, chose*
*`2*(AUROC - 0.5)`. Its message says the choice was measured on herg, 1500 rows, 45*
*candidate spaces: rank agreement rho=+1.000 against the old Spearman form, 2 of 990*
*pairwise orderings differed and both were ties, and all four rules picked the same*
*winning space — so it reorders nothing and what it fixes is the zero point. Written in*
*all three places that must agree. If that is the answer, this line can be filled in and*
*the item moved to §6; if it is not, the commit has to be reverted.)*

---

### 3.3 The per-model budget cap, and what a full search actually costs

*Cost to change: removing or moving a cap enlarges the search. Nothing stored becomes wrong.*

**Your answer on the last pass:** the caps may be too aggressive, possibly different for
`fast` and `full`, and you wanted better numbers off the stored results. Here they are, and
they say the caps are pointed at the wrong models.

**What is capped.** `MODEL_MAX_BUDGET` at `find_best_hybrid.py:3046`: svm at 500 features,
gp/mlp/bnn/dnn/cnn at 1000, knn at 2000. rf, xgboost, lightgbm, catboost and ngboost are
uncapped.

**Measurement 1 — seconds per fit against feature width.** From the f29 base-representation
model survey, 410 stored fits over `buchwald_hartwig`, `caco2`, `flab_expression_koenig`,
`logd` and `tcga_brca_pam50`, untuned arm only. Each representation is fitted at its own
native width, so this reads cost against width directly. Median seconds for one fit:

| model | ≤128 features | ≤512 | ≤1024 | ≤2048 | ≤4096 | above 4096 |
|---|---|---|---|---|---|---|
| catboost | 20.6 | 15.6 | 26.6 | 14.1 | 114.6 | **184.1** |
| xgboost | 0.8 | 2.3 | 5.5 | 1.1 | 14.3 | 40.2 |
| lightgbm | 2.0 | 1.7 | 4.1 | 2.0 | 7.9 | 25.5 |
| mlp | 16.7 | 0.7 | 4.6 | 9.5 | 16.4 | 20.3 |
| dnn | 3.9 | 2.4 | 5.4 | 8.3 | 8.8 | 12.9 |
| rf | 0.3 | 0.4 | 1.6 | 0.7 | 2.5 | 5.1 |
| ridge | 0.0 | 0.0 | 0.0 | 0.1 | 0.3 | 0.7 |
| knn | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gp | 2.5 | 5.8 | 4.8 | 2.5 | — | — |

Fits behind each figure: 2 at ≤128, 15 at ≤512, 10 at ≤1024, 14 at ≤2048, 6 at ≤4096, 3
above 4096. The last column rests on three fits per model.

Above 4096 features the three most expensive models are catboost, xgboost and lightgbm, and
all three are uncapped. Every capped model except gp is cheaper there than lightgbm. dnn,
capped at 1000, costs 12.9 seconds a fit where lightgbm costs 25.5 and catboost 184.1.

**Measurement 2 — seconds per configuration in a real search.** One database on
`/Volumes/seagate/kirby_cache/result_db` is a genuine budget sweep:
`caco2__ocv1f0s42.db`, 2350 configurations, 27.5 hours of measured work. caco2, 2158
molecules, RMSE, 5 inner folds per configuration, one outer fold, the five default
allocation methods and all three importance methods. The other four databases are
budget-25 smoke runs and are not used here.

A configuration is one (allocation × importance × internal evaluator × model × budget). Its
time is the whole thing — building the hybrid, scoring the representations, computing the
importances, and fitting the model on all 5 inner folds. Median seconds:

| model | 10 | 25 | 50 | 100 | 250 | 500 | its own 250→500 factor |
|---|---|---|---|---|---|---|---|
| dnn | 16.7 | 22.3 | 24.4 | 24.2 | 24.3 | 34.4 | 1.42 |
| lightgbm | 7.5 | 7.9 | 10.0 | 12.1 | 22.7 | **51.2** | 2.26 |
| xgboost | 4.2 | 4.7 | 6.8 | 11.1 | 23.1 | **49.0** | 2.12 |
| svm | 5.3 | 8.1 | 8.0 | 9.9 | 11.2 | 15.7 | 1.40 |
| rf | 6.2 | 7.1 | 6.9 | 8.3 | 9.9 | 12.1 | 1.22 |

**Which models this decision is about.** With `models=None` — every precompute and every
default search — the model set comes from the tier table. Closed by the owner on 2026-09-09
and changed in the code the same day:

| tier | models | what it means |
|---|---|---|
| `fast` | rf, lightgbm | the whole of `speed='fast'`, and part of the default set |
| `full` | xgboost, svm, dnn | the rest of the default set |
| `extra` | mlp, gp, knn, ngboost | never auto-selected; available by name |
| `off` | catboost, cnn, bnn, ridge | never auto-selected, and not run in the suite. The code stays so a caller can still ask by name. ridge is called separately when it is wanted |

The membership test is now an explicit set, `AUTO_SELECTED_TIERS = {'fast','full'}`. It used
to be a rank comparison where an unrecognised tier scored 0 and was therefore **included**, so
a new label meant to hold a model out would have let it in.

Two things went with it. The ridge injection for `nt_*`/`gb_*` in `_resolve_models` is gone —
nucleotide searches no longer get a linear baseline unless someone asks for one. And the two
hardcoded model lists in `tests/experiments/_shared.py`, which named knn, mlp, gp, catboost
and cnn, now read `DOWNSTREAM_MODELS`.

So of the eight caps in `MODEL_MAX_BUDGET`, six are on models no search selects. **Two are
live: svm at 500 and dnn at 1000.**

**Estimates.** Hours for one outer fold of one dataset the size of caco2. Measured seconds
per configuration × the configuration count the defaults produce.

**Full search** — 5 allocation methods × 3 importance methods × that model's evaluators.

| model | 10 | 25 | 50 | 100 | 250 | 500 | 1000 | 2000 | configs per budget |
|---|---|---|---|---|---|---|---|---|---|
| rf | 0.10 | 0.12 | 0.12 | 0.14 | 0.16 | 0.20 | 0.25 | 0.30 | 60 |
| lightgbm | 0.13 | 0.13 | 0.17 | 0.20 | 0.38 | 0.85 | 1.93 | 4.36 | 60 |
| xgboost | 0.09 | 0.10 | 0.14 | 0.23 | 0.48 | 1.02 | 2.16 | 4.58 | 75 |
| svm | 0.09 | 0.14 | 0.13 | 0.17 | 0.19 | 0.26 | 0.37 | 0.51 | 60 |
| dnn | 0.35 | 0.46 | 0.51 | 0.50 | 0.51 | 0.72 | 1.02 | 1.44 | 75 |
| **total for that budget** | 0.75 | 0.95 | 1.07 | 1.24 | 1.72 | 3.05 | 5.72 | 11.20 | |
| **running total up to it** | 0.8 | 1.7 | 2.8 | 4.0 | 5.7 | **8.8** | **14.5** | **25.7** | |

**Fast search** — 3 allocation methods × 2 importance methods, rf and lightgbm only.

| model | 10 | 25 | 50 | 100 | 250 | 500 | 1000 | 2000 | configs per budget |
|---|---|---|---|---|---|---|---|---|---|
| rf | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.03 | 6 |
| lightgbm | 0.03 | 0.03 | 0.03 | 0.04 | 0.08 | 0.17 | 0.39 | 0.87 | 12 |
| **total for that budget** | 0.04 | 0.04 | 0.04 | 0.05 | 0.09 | 0.19 | 0.41 | 0.90 | |
| **running total up to it** | 0.0 | 0.1 | 0.1 | 0.2 | 0.3 | 0.5 | 0.9 | **1.8** | |

Multiply by 5 for a nested-CV run: a full search over the whole budget list is 129 hours per
dataset, 44 hours if it stops at 500, 73 hours if it stops at 1000. A fast search over the
whole list is 9 hours.

Both tables ignore `MODEL_MAX_BUDGET`, so they show what each model would cost if it ran at
every budget. Today svm stops at 500 and dnn at 1000, which removes 0.37 + 0.51 + 1.44 =
2.3 hours from the 25.7 — **9%**.

**What is measured and what is not.** Budgets 10 to 500 are measured, from
`caco2__ocv1f0s42.db` — 2350 configurations, 27.5 hours of real work, caco2 at 2158
molecules, RMSE, 5 inner folds per configuration, one outer fold. 1000 and 2000 are
extrapolated, by applying each model's own measured 250→500 factor again. That extrapolation
is the weakest number here and it is what separates 8.8 hours from 25.7. Closing it is one
run at budget 1000 on caco2, about 5 hours.

Outside these figures: the baseline arm `nested_cv_evaluate` runs alongside the search, and
each outer fold holds about 80% of the rows.

**What the estimate says about the caps.** `MODEL_MAX_BUDGET` saves 2.3 hours of 25.7 —
**9% of the wall**. It saves so little because the two live caps are on svm and dnn, which
cost 1.0 and 4.1 hours over the whole budget list, while 17.6 of the hours are lightgbm and
xgboost, which have no cap. If a full search is what worries you, the lever is the top budget
— 8.8 hours at 500, 14.5 at 1000, 25.7 at 2000 — not which models are capped.

**Your options.**
- **Keep `MODEL_MAX_BUDGET` as it is and write the reason down.** It costs svm the 1000 and
  2000 budgets and dnn the 2000 budget, and buys 9% of the wall.
- **Cap on measured cost instead.** Put lightgbm and xgboost under a cap and drop the caps on
  svm and dnn. That reverses which models can be compared at the top budget, and at 2000 the
  survivors would be rf, svm and dnn rather than the trees.
- **Cap the top budget rather than the models.** One number for every model, so every budget
  row compares the same five. This was `resolve_budget_cap`; §4 raised it to 1000 on
  2026-09-09 and it was removed outright on 2026-09-10 — see the superseding note there.
- **Remove `MODEL_MAX_BUDGET`.** Costs 2.3 hours per outer fold at the full budget list, and
  every budget row then compares all five models.

**>> (a) what to do with the cap:**
Remove cap

**>> (b) different caps for `fast` and `full`, or one set:**
Remove cap

**DONE 2026-09-10.** `MODEL_MAX_BUDGET` is an empty dict. The name stays — f15 imports it,
f27 and f28 read it into their `budget_caps` output, and a caller can put a cap back by name
— and every reader still honours whatever is in it. Checked: svm searched at budget 1000,
which it could not reach before.

**Everything else that still stated the old caps.** `f27` and `f28`'s docstrings and both of
their readers in `tests/analysis/`; `TIME_ESTIMATION_METHOD.md`, whose cost model encodes the
caps in its configuration count, so a SLURM wall set from it is now too short; and
`tools/build_thesis.py`, which wrote "the GP is capped at 1,000, SVM and the neural models
lower" straight into the thesis text. All corrected. **Not touched, yours to decide:**
`PAPER_OUTLINE.md`, `PAPER_DRAFT.md` and `PAPER_FACTCHECK_2026_07_10.md` state per-model caps
(and three different sets of numbers between them, none matching the dict even before this).
`PIPELINE_READINESS_TODO.md` and `AUGMENTATION.md` state the old `top_3` default; auto-demotion
in particular is gated on a top_k allocation, so the mechanism `AUGMENTATION.md` calls
automatic now fires on no fold of a default search.
---

### 3.4 Fold-0 pruning of allocation methods — CLOSED 2026-09-13, OFF by default

*Cost to change: nothing stored becomes wrong.*

**Your answer on the last pass:** "that's the point of pruning, sometimes it's turned on,
sometimes its not. nothing is broken here. I mean double check it but by default its turned
off." You also asked to note pruning over 2 folds instead of 1.

I double-checked it. Two things need correcting before that answer stands.

**Correction 1 — it is on by default, not off.** `find_best_hybrid` takes
`adaptive_pruning=True` at `find_best_hybrid.py:6594`. `_normalize_pruning_schedule` turns
`True` into `final_strategy_threshold=0.15` at line 3106. The prune then runs whenever the
search has more than one fold and more allocation methods than `strategy_top_n`, which is 3
(the condition is at line 5684). The default set has five methods, so it is enabled on every
default search of more than one fold. Turning it off is `adaptive_pruning=False`.

**Correction 2 — my claim that it deletes two of the five was wrong.** This file said "two
of the five default methods are deleted after one fold, on every default search". That is
not what the code does. At line 5911, fold 0's results are grouped by (importance method,
internal evaluator, model, budget), each group's top 3 allocation methods score a point, and
a method is dropped only if it scored in fewer than 15% of the groups. With five methods
competing for three places per group, a method has to place near-last in more than 85% of
groups before it goes. Nothing is deleted by count. I have not measured how often anything
is actually dropped on a real run.

So the cut is much weaker than this file claimed, which may be the whole of your answer. The
question left is whether it stays on by default.

**Pruning over 2 folds instead of 1 — noted, and it does not exist.** The schedule's
`checkpoints` are budgets, not folds; there is no field anywhere that says how many folds
feed the vote. The prune reads `fold0_results` and nothing else. Making it 2 folds is new
code, not a setting.

**Your options for (a).**
- **Leave it on.** It is a weak cut. It only removes a method that lost almost everywhere on
  fold 0, and it saves four folds of work on that method.
- **Turn it off by default** and let experiments switch it on. Every default search then
  scores all five allocation methods on all five folds. Nothing else changes.
- **Leave it on and count how often it fires** before deciding. That is a counter in the
  next full run, not an experiment.

**Your options for (b).** The vote reads fold 0 only.
- **Build the 2-fold version.** New code: collect two folds, then vote. Every search pays a
  second unpruned fold before the cut starts saving anything.
- **Leave it as a note** here, and revisit if the fold-0 vote turns out to be noisy.

**>> (a) leave `adaptive_pruning=True` on by default, turn it off, or count first:**
**OFF by default. Decided 2026-09-13, implemented the same day.** Your words: "Leave it off
by default - the main precompute won't use it but subsequent experiments that call
find_best_hybrid will."

`adaptive_pruning` defaults to `False` in `find_best_hybrid`. The precompute does not prune.
An experiment that wants it passes it explicitly, and is then asking for a different search,
so it is not served the precomputed one.

*(the earlier answer, kept for the record: "TBD - I need to re-test adaptive pruning and
keep track of the times when I do it.")* 

**>> (b) build the 2-fold version, or leave it as a note:**
I will be testing the two fold version. **Owner's, not a framework task** — 2026-09-13:
"I'm testing the feasibility of the 2 fold version right now. Not your wheelhouse." 
---

### 3.5 The worst representation always gets nothing — CLOSED 2026-09-13, leave it

**Decision: leave it as is.** Measured, not argued — see the numbers below.


**The problem, plainly.** The budget is divided among the representations in proportion to
their scores, but before dividing, the code shifts every score up so that none is negative —
and the amount it shifts by is set by the worst representation in the pool, so that one
always lands on zero and gets no features, and everyone else's share depends on how bad the
worst one happened to be.

**Your options.** Leave it as is, or convert each score into "how much better than a trivial
guess" before dividing, so that zero means genuinely no signal instead of "worst in this
pool".

**The second issue (this is §3.7).** That shift only happens when some score is negative,
which on a correlation dataset is usually never — so the same method behaves one way on an
RMSE dataset and another way on a correlation one, and the question is whether to clip a
negative correlation to zero so they match.

---

**Measured, so the choice has numbers.**

Ten representations, RMSEs in the range logd produces, budget 1000 features. Removing only
the worst representation from the pool — changing nothing about the others — moved the best
one from **178 features to 199**, and the second-worst from **27 features to 0**.

f32 counted how often a representation ends with nothing at all across 45 folds and 9
datasets: **0.57 per split under today's code, 0.13 under the "better than a trivial guess"
rule.**

`tests/test_budget_split_arithmetic.py` holds the property as an expected failure —
*adding a worse representation must not resize the others* — so it starts passing the day
this lands.

**What "better than a trivial guess" means per metric.** One line each, from the f32
write-up in §5:

| metric | datasets | the score becomes |
|---|---|---|
| `spearman`, `mcc` | 25 + 21 | unchanged — zero already means no signal |
| `rmse` | 9 | 1 − (rmse / rmse of predicting the training mean) |
| `mae` | 1 | 1 − (mae / mae of predicting the training median) |
| `auc`, `ci` | 8 + 4 | 2·(score − 0.5) |
| `f1_weighted` | 2 | (f1 − f1 of always predicting the majority training class) / (1 − that f1) |

**One thing that is settled and should not be re-derived.** The rule cannot be applied at the
last step alone. The per-fold scores used to be shifted once before they ever reached the
place the budget is divided, so what arrived on an RMSE dataset was *the worst
representation's error minus this one's* and the error itself was gone. That pre-shift was
removed on 2026-09-10 and the numbers now keep their meaning, which is what makes this
implementable at all.

**>> (a) leave it, or convert to "better than a trivial guess":**
**LEAVE IT.** f32 ran both rules on 9 datasets and 45 folds. Changing the rule moved which
features got chosen, but it did not improve how accurately the final models predicted
held-out molecules — accuracy got slightly worse about as often as it got better.

Read off the result files on 2026-09-13, over the **226 cells where the two rules actually
produced a different split** — the skill score predicted held-out molecules better in 85 and
worse in 133:

| | better | worse |
|---|---|---|
| rmse | 58 | 87 |
| auc | 15 | 25 |
| f1_weighted | 12 | 21 |
| buchwald_hartwig | 22 | 18 |
| herg_ki | 15 | 17 |
| humanness_therapeutic | 15 | 25 |
| caco2 | 9 | 26 |
| logd | 12 | 26 |
| tcga_brca_pam50 | 12 | 21 |
| budget 50 | 38 | 67 |
| budget 300 | 47 | 66 |

Every metric, every weighting scheme, both budgets, and five of six datasets came out worse.
The one exception, buchwald_hartwig at 22 against 18, is a margin that flips on a seed.

**And it does not pay even where the problem is worst.** Split the same cells by how many
representations today's rule leaves with nothing:

| representations starved today | better | worse |
|---|---|---|
| none | 33 | 56 |
| one | 22 | 43 |
| two | 16 | 15 |
| three | 10 | 13 |
| four or more | 4 | 6 |

Where today's rule starves two or more — the failure the skill score exists to fix — the two
rules are a coin flip. Where it starves none or one, the skill score is clearly worse. So it
buys no accuracy even at its own best case; it only stops costing any.

**>> (b) on a correlation dataset, clip a negative score to zero, or keep today's shift:**
**KEEP TODAY'S SHIFT.** f32 measured nothing here because it never saw a negative score: the
smallest across all 15 spearman and mcc folds was 0.1712, and the smallest mcc was 0.5683.
It almost never fires. Revisit if a negative correlation ever turns up.

**Recorded in the code** as
`tests/test_budget_split_arithmetic.py::test_adding_a_worse_rep_DOES_resize_the_others_and_that_is_the_decision`,
which asserts today's behaviour and carries these numbers, so nobody changes the
normalisation without evidence that beats f32's.

### 3.6 A degenerate held-out carve — CLOSED 2026-09-13, it costs the configuration

*Cost to change: nothing stored. It decides whether a search on a coarsely-grouped dataset
returns a table or returns nothing.*

**Where this comes from.** Two things you decided landed together. `create_hybrid` now splits
the budget with `performance_weighted`, which scores every representation on the fold's
internal held-out carve; and a configuration that lost a fold is refused. When that carve
holds under ten rows, one scaffold group, or one activity value, no representation can be
scored, `_require_scored_reps` raises, and the fold is lost — so the configuration goes too.

**Measured, synthetic, 2026-09-10.** 150 rows in 15 groups of 10: every configuration of a
geometry search was refused and the run returned nothing. The same 150 rows in 30 groups of
5: nothing was lost. The threshold is whether the internal carve gets at least two groups and
ten rows, which is a property of the grouping, not of the recipe.

**Which datasets are exposed.** Not measured. `_require_scored_reps`'s own docstring names
`tz_her2_binary`, whose held-out block always holds one CDRH3 edit-distance group. f32 found
`tcga_brca_pam50`'s five-fold split gives held-out blocks of 3, 79, 403, 142 and 194 rows.

**Your options.**
- **Leave it — the configuration is refused.** It is what your answer said: a fold that
  cannot be scored means the groups are wrong, and the message says so. On a dataset like
  `tz_her2_binary` the search returns nothing at all, loudly.
- **Cost it the fold, not the configuration.** Give `_require_scored_reps` its own exception
  type, record it apart from a real failure, and let the refusal fire on the second such fold
  rather than the first. A degenerate carve then behaves like a fold that was never asked
  for, and the configuration competes on the folds that worked — which is what you refused
  for a real failure.

**>> (a) a degenerate held-out carve costs the fold, or the configuration:**
**THE CONFIGURATION. Decided 2026-09-13, implemented the same day.** Your words: "Leave it
and throw a warning to the user to get better groups."

Both refusals now say `GET BETTER GROUPS` and name what is wrong with the block — one group,
under ten rows, or one activity value — and say in as many words that this is how the
dataset is grouped, not a property of the recipe.

---

### 3.7 — folded into §3.5

Both questions are now asked in §3.5, in plain language, with the same numbers. This
heading is kept so nothing that cites §3.7 dangles.

### 3.8 Comparing a hybrid against a base representation — CLOSED 2026-09-13

**This was written up wrongly and the owner corrected it.** The claim was that a delta
between a 100-feature hybrid and a 2048-feature representation "mixes method and width and
cannot be read as either", and that a same-width baseline arm was needed. That is not the
question.

**What is actually true**, in the owner's words, 2026-09-13:

> "I made the choice to not sweep over budget. I kind of do by running the search multiple
> times, but I wanted that to be intentionally left out. Then I look at a range of budgets
> and I select the best one. So even if you're using a hybrid with a budget of 100, if it
> has better performance than the larger base reps who cares, it's the same internal logic."

> "It really is just the best features for a dataset and model, but you can't let N go wild
> in this context. You could search it inside, but I chose not to. There's plenty of other
> representations that do this — even when using ECFP4 you can customise the size of the
> substructure and the number of bits."

So a hybrid that beats a wider representation on fewer features **is the result**, not a
caveat. The number of features is a setting of the method, the way ECFP4's bit count is a
setting of ECFP4. It stays out of the main search grid because searching it is heavy; the
range is covered by running the search at a range of budgets and reading across them.

**The one thing that would be wrong** is arbitrarily reporting hybrids at a single budget —
picking 100 and quoting it. Reading the range is what makes it honest.

**Three things done on 2026-09-13.**

1. **The width warning is gone.** `find_best_hybrid` used to print that the comparator was
   at least twice as wide as the hybrid. It was treating a deliberate choice as a fault.
   Both widths are still recorded per fold (`n_features_per_fold`,
   `best_single_n_features_per_fold`) for anyone who wants them.
2. **The same-width comparison already existed**, as the owner thought:
   `tests/experiments/complexity/c02_complexity_grid.py` runs `find_best_hybrid` at
   `budgets=[D]`, where D is a base representation's own native width, per downstream
   model, with its own cache namespace. It is a complexity experiment, not central.
3. **It now runs at the top TWO representations' widths**, not just the best one's —
   owner's call. Two representations give two points on the budget curve instead of one,
   and the second-best is often a very different width from the first. Each is compared
   against its own score, and the rank-1 pass keeps its old key so existing readers still
   find it.

**Two questions this section used to ask, and why they were wrong to ask.** "Which
importance method cuts the base representation" conflated two settled things: how to RANK
base representations (by their score in the dataset's own metric) and which feature
importance method to use (already an axis of the search). Neither was ever an open question.

---

## 4. Decided 2026-09-09 — IMPLEMENTED 2026-09-10

Your answers from the 2026-09-09 pass. Each says what changes and what it costs. **Every
item below is now in the working tree**, together with §3.1 (the `create_hybrid` default)
and §3.3 (the per-model budget cap). What each one is verified against is written under it.

**What was NOT implemented, and why.** §3.4 is yours to test. §3.5's rule — one
normalisation per metric — is answered but not built: it needs a metric name threaded into
the budget split, which nothing down there carries, and two sub-questions came out of
tracing it. Both are at the end of this section under "§3.5 — traced, not built".

**How the nine were checked.** Eight assertions on synthetic data covering every change
(`metric` refused, the default split, a blend's corner equal to its component, a lost fold
refused, a budget above the row count searched, the objective in the identity and in all
three metas, svm searched at 1000); a caching run in which a predictive, a geometry and a
blended search over one dataset wrote three separate database rows and six separately-named
baseline files, and a repeat added none; the six-test metric-direction gate; and a real
caco2 run — 2158 molecules, the signed-off ten representations — end to end through both
`find_best_hybrid` and `nested_cv_evaluate`.

### The metric becomes a required argument

**Your answer:** "Make metric a required argument. I don't want the registry as a default."

**The change.** `find_best_hybrid(metric=...)` currently defaults to `'r2'`
(`find_best_hybrid.py:6540`). It becomes required and raises when absent, the same way
`task_type` already does. The registry stays available to the experiment layer, which
already reads it in `tests/experiments/_shared.py`, but `src/` never looks a dataset up to
decide a metric.

**Blast radius — the count in this paragraph was wrong.** It said nine calls, all in
`tests/complexity_theory_experiments.py`. An AST pass over every `.py` outside `build/` and
`tests/archive/` gives **twelve**: those nine, plus
`tests/experiments/opig_ginkgo_developability.py`,
`tests/experiments/opig_kingsmith_component_splits.py` and
`tests/experiments/similarity_optimized_hybrid.py`. The three missing ones are exactly the
ones that now raise, and they were nearly left unfixed because the count said they did not
exist. `tests/archive/` holds 30 more and is excluded.

A thirteenth is worse than missing: `tests/similarity_experiments.py` — the standalone
similarity suite the cluster launches from `run_similarity_experiments.sh` — carried
`metric: str = 'rmse'` on its own helper's signature with all eight callers omitting it. No
metric sweep could see it, because it looks like a call that passes one. Every dataset in
that suite would have been searched under rmse, including the antibody sets the registry
records as spearman.

**DONE.** The nine take `**search_metric(data)`, which resolves metric and task type through
`experiments._shared._resolve_dataset_metric` — the one resolver, registry first. The two
opig files take `metric=_resolve_dataset_metric(<dataset>)[0]`; both were raising on every
fold into an `except Exception` that printed and continued, so they were writing result files
with no hybrid rows in them. `similarity_optimized_hybrid.py` needs no change: a geometry
objective is exempt (below). The similarity suite's helper now requires the `data` dict and
resolves the metric, the task type and the dataset name from it.

**One thing the requirement got wrong at first, found by re-reading the code.** The refusal
sat 58 lines ABOVE the geometry override that replaces `metric` with the objective's own
metric name, so a `objective='similarity'` run was forced to supply a metric it then
discarded — which killed the geometry arm of `similarity_optimized_hybrid.py` outright. The
check now sits after the override, and a geometry objective may omit the metric. A blend
containing `predictive` may not: a model is still fitted and scored with it.

**The question you attached to it — can somebody bring their own dataset today?** Read in
the code on 2026-09-09:

- **Required now:** `base_reps`, `labels`, and `task_type`. `task_type` is already a
  required argument in everything but its signature default — it is declared `= ''` and then
  raises at `find_best_hybrid.py:7009` unless it is `'classification'` or `'regression'`.
  There is no auto-detection on this path.
- **Not required, and this is the hole:** `metric`. It defaults to `'r2'`, so somebody
  bringing a classification dataset gets r2 on class codes and a warning only if they also
  passed a `dataset_name` the registry knows.
- **`dataset_name` is optional** unless `cache=True`, which raises without it
  (`find_best_hybrid.py:7158`). With no `dataset_name` the per-dataset budget cap and the
  model skip list both return their input unchanged, so an unknown dataset is not blocked
  or silently reshaped by policy.

So: yes, somebody can run their own data today with `cache=False`, and they must already
supply `task_type`. Making `metric` required closes the last gap between what the search
needs and what it demands.

### A blend combines representation scores, not their ranks

**Your answer:** "Blend on scores."

**The change.** `blended_rep_scores` at `hybrid.py:321` ranks each component across
representations through `_rank01` before combining. It stops ranking and combines the raw
component scores, which `_score_component` already returns on one scale, bounded −1 to 1 and
higher-is-better.

**Cost.** The budget split changes on every blend run, so every stored f28 row re-runs. The
corner of f28's triangle then equals the plain objective, which it does not today.

**DONE.** `blended_rep_scores` takes the weighted mean of the raw component scores;
`_rank01` is deleted. Checked: the one-component corner `{similarity: 1.0}` now equals
`_score_component`'s own similarity score to floating point, on every representation, and a
two-way blend equals the weighted mean of its two component scores exactly.

**A hole this opened, found by re-reading the allocators.** `blended_rep_scores` returns
-inf for a representation that failed any ONE component, and its per-component guard only
asks that each component separated something — so with representation A failing one
component and B failing another, every blended score can be -inf while no component looks
dead. The sanitiser then replaced every non-finite score with 1.0 and the budget was split
EQUALLY, filed under the requested allocation method's name. `_require_scored_reps` now runs
on the blend path of both allocators, BEFORE the sanitiser erases the -inf. In
`allocate_weighted_global` it was inside the `else` and now sits outside it.

### A configuration that missed a fold is refused

**Your answer:** "Refuse a configuration that did not score on every fold. Throw a warning
to the user to double check the groups."

**The change.** A configuration whose `n_folds_scored` is below `n_folds_asked` raises
instead of competing on a shorter mean. The message names the folds that failed and tells
the caller to check their groups.

**Cost.** Nothing to compute. It changes which configuration wins in any search where a fold
failed, and a search can now end with fewer configurations than it started with.

**DONE.** The message names each lost fold and what it raised, and ends "CHECK THE GROUPS".
Every dropped configuration prints — unconditionally, not under `verbose` — in the main
loop, the fold-0 pass and the refinement phase, and each of the three ends with a count of
how many it dropped and the first one's reason. Without that a quiet run would end with
fewer configurations than it listed and say nothing.

**It missed a whole class of lost fold at first.** `n_folds_scored` counted every appended
entry, and under a geometry objective or a blend a fold whose correlation could not be formed
appends **NaN** — still an entry. So a configuration defined on one fold of five passed the
test with 5 of 5 and was ranked against configurations defined on all five, while
`score_mean` quietly averaged the one. The count is now the number of FINITE folds whenever
the objective is geometry or a blend, and the message says how many were undefined.

**One thing to know before this runs on a small dataset.** `_require_scored_reps` raises
inside `create_hybrid` when no representation can be scored on a fold's internal held-out
carve — under ten held-out rows, one scaffold group, or constant activity. That is a lost
fold, so the refusal now discards the whole configuration rather than the fold. Measured on
synthetic data: 150 rows in 15 groups of 10 lost EVERY configuration of a geometry search;
the same 150 rows in 30 groups of 5 lost none. Whether a degenerate carve should cost one
fold or the configuration is not decided — see §3.6 below.

### A budget above the row count is searched, not swapped

**Your answer:** "flip the default."

**The change.** `allow_budget_exceeds_n` at `find_best_hybrid.py:6648` defaults to `True`.
The path that drops an oversized budget and appends the fold's own row count in its place
(`find_best_hybrid.py:6911`) stops firing, so no fold invents a budget of its own.

**Cost.** Measured on 2026-09-09 over the 38 registry entries that record a row count, with
the per-model caps applied: model-and-budget combinations go from 1284 to 1406, so 1.10×.
Only 14 datasets move — the five ginkgo sets, four abagym sets, `humanness_therapeutic`,
`abagym_lysozyme`, `abagym_EGFR` and the two TCGA sets. The largest single change is
`ginkgo_heparin` at 1.85×, from 20 combinations to 37, on 94 molecules. `herg_ki` does not
move because the per-dataset cap already holds it at 500.

**What it exposes — measured 2026-09-10, and it does not raise.** The worry was fitting 2000
features on 75 training molecules. Run at ginkgo_heparin's shape (94 molecules, ~75 training
rows per fold, 32 scaffold groups) on abcov_kd's real representations: rf, svm and lightgbm
all completed at budgets 100, 1000 and 2000, on 3 of 3 folds, with no raise. The scores are
poor — spearman between -0.21 and +0.03 — but the configurations exist and compete rather
than stopping the search. dnn could not be tested on this machine: torch and LightGBM in one
process segfault here, which is the known local fault, not this change.

**DONE.** The default is `True`. Checked: budgets `[10, 500]` on 120 rows stayed `[10, 500]`
instead of becoming `[10, 96]`.

**Three experiment files were asking for a shorter budget list than the pool now holds.**
`c16`, `c17` and `c11` each filtered `FULL_BUDGETS` by `b <= n_samples` under a comment
saying it matched the engine's internal filter. It no longer does, so on the small datasets
they were dropping budgets that exist on disk — three of eight on a 94-molecule set — which
is exactly the divergence that makes a blueprint lookup return `None` and silently skip a
column. The clause is gone from all three. `c02` skipped a per-model pass at `len(labels)`
because the engine used to append it; that clause is gone too, and with it a model whose
best representation is exactly as wide as the dataset has rows was being dropped from the
per-model table with nothing printed. `f15` has the same filter and was not touched — f00-f15
are not to be edited.

### The per-dataset budget cap rises from 500 to 1000

**Your answer:** "I think this is a bit too aggressive when some of the base reps are higher
than that. Increase to 1000."

**The change.** `resolve_budget_cap` in `tests/experiments/dataset_policy.py:180` keeps
budgets at or below 1000 instead of at or below 500. It applies to `caco2`, `logd`, `qm9`,
`herg`, `herg_ki`, the three adme sets, the five large `flab_*` sets, `davis`, and every
dataset whose name starts `nt_` or `gb_`.

**Cost.** Against the suite's standard budget list this restores the 1000 budget to those
datasets — one budget back out of eight, on the largest datasets in the suite. Every affected
search re-runs. This is the experiment layer, not `src/`.

**DONE.** `_BUDGET_CAP_500` is now `_BUDGET_CAPPED_DATASETS` with the ceiling in
`_BUDGET_CAP_MAX = 1000`; the cap is applied in one place, `resolve_budget_cap`. Checked:
caco2, `nt_*` and `flab_binding_koenig` keep budgets up to 1000 and drop 2000; an unlisted
dataset keeps all eight.

**The interaction this paragraph raised is gone.** It said the new 1000 budget would be
searched by trees, knn and nothing else because `MODEL_MAX_BUDGET` capped svm at 500. §3.3
was answered in the same pass — the caps are removed — so all five default models are
searched at 1000.

**⛔ SUPERSEDED THE SAME DAY — THE CAP IS REMOVED, NOT RAISED. 2026-09-10, owner's
instruction: "go in the code and remove the budget cap."** `_BUDGET_CAP_MAX`,
`_BUDGET_CAPPED_DATASETS` and `resolve_budget_cap` are deleted from
`tests/experiments/dataset_policy.py`, and nothing calls `register_budget_policy` any more,
so `_apply_budget_policy` is a pass-through for every dataset.

**What forced it.** c33 asks whether the search overfits at budgets above a dataset's own
training row count. The cap removed exactly those budgets on 5 of the 9 datasets with
representations cached locally: herg_ki lost 1412, 1695, 2000, 2260, 4000 and 8000; logd
lost 5028, 6033, 8000 and 8044; caco2 lost 2000, 2158, 2589, 3452, 4000 and 8000;
nt_promoter_tata lost 5274, 6328, 8000 and 8438; flab_expression_koenig lost 4000, 4275,
5130, 6840 and 8000. herg_ki is the one local dataset the 2026-07-14 power audit leaves
able to resolve a small drop above the line. The cap's stated justification was wall time
(the 2026-06-21 cost-cut plan), which is not one of the three reasons a cut needs.

**Cost.** The budget list is in the search cache key, so every search on those 35 datasets
now asks for a key it has never written. Nothing cached is wrong or lost; none of it is
reused. Those searches re-run, over a longer budget axis than before.

**Two other budget filters were found in the same pass.** `c06_within_family_gradient.py:67`
dropped budgets above the molecule count while calling itself "the canonical budget filter
(matches find_best_hybrid.py:3934)" — the engine filter it mirrored is gone, so the mirror
was the only thing still cutting c06's budget axis. Removed. `f15_adaptive_pruning_
validation.py:244` does the same and is left alone, f00–f15 not being open to edits.
`f27_objective_and_budget.py:165` reads `MODEL_MAX_BUDGET`, which is empty, so it reports
nothing.

### The objective goes into the cache metadata and into the database identity

**Your answer:** "All results have been deleted already, add both."

**The change, both halves.** `_make_search_meta`, `_make_baseline_meta` and
`_make_hybrid_eval_meta` each record `objective`. `IDENTITY_COLS` in
`src/kirby/result_cache.py:46` gains `objective` alongside `metric` and `task_type`.

**What it fixes.** Today the only thing separating a predictive result from a geometry
result is the objective being folded into the filename, one level up, and into the free-text
`tier` string, which already carries four unrelated things. A mismatch that should be a loud
miss is currently a silent hit.

**Cost.** Every `searches/*.pkl`, `baselines/` and `hybrid_evals/` entry stops matching and
recomputes — the check walks the EXPECTED meta, so a stored meta lacking the key is a stale
miss, rewritten in place. `~/.cache/kirby/baselines` held 3,942 such files on 2026-09-10.

**The result databases must be DELETED, and nothing does it for you.** `_ensure_schema`
raises on a file written under the older column list and names it, so the first
`find_best_hybrid(cache=True)` on any dataset with a stored database stops before the first
model fit. On this machine that is 9 files under `~/.cache/kirby/result_db` and 5 under
`/Volumes/seagate/kirby_cache/result_db`. `wipe_allocation_caches.sh` already lists
`result_db` and leaves `baselines/` and `rep_cache/` alone.

**DONE.** `objective` is an identity column and a key in all three metas. Checked end to
end: a predictive, a geometry and a blended search over one dataset wrote three separate
rows carrying `predictive`, `similarity` and `blend_predictive0.5_similarity0.5`, six
separately-named baseline files, and a repeat of the predictive run added no rows.

**Two tools were reporting those databases as readable.** `check_precompute_status.py` — the
script that decides whether a precompute is finished before a resubmit — and
`blueprint_trace.py`'s cache guard both probe the table with plain SQL that an older file
answers perfectly well. Both now read `PRAGMA table_info` first and report a stale schema
instead of a row count.

### The model tiers are closed — 2026-09-09

**Your decision:** "mlp, gp, knn, ngboost in extra. catboost, cnn, bnn, ridge REMOVE... The
code can exist, just don't run it... Ridge will get called separately."

**Done.** `DEFAULT_MODELS` in `src/kirby/find_best_hybrid.py`: catboost, cnn, bnn and ridge
carry `tier: 'off'`; mlp, gp, knn and ngboost stay `extra`; the default set is rf, lightgbm,
xgboost, svm, dnn. Selection is an explicit `AUTO_SELECTED_TIERS = {'fast','full'}` rather
than a rank comparison that let an unknown label through. The nucleotide ridge injection in
`_resolve_models` is removed. The two hardcoded model lists in `tests/experiments/_shared.py`
now read `DOWNSTREAM_MODELS`. No model code was deleted and every name still works when asked
for explicitly.

**Cost.** Every model name is in the search cache key, so this invalidates every cached
search. Nucleotide datasets lose the linear baseline the 2026-07-18 decision gave them, by
your instruction that ridge is called separately.

**Still naming these models, not changed — your call on each.**
- `tests/experiments/framework/f29_model_and_tuning_survey.py:76` — `MODELS` includes ridge,
  catboost and cnn. f29 has already run and its results are on disk.
- `tests/experiments/framework/f16_multistage_pruning.py:75` — `F16_MODELS` includes ridge.
- `tests/experiments/covariate_shift.py:59` — `SURVEY_MODELS = ['rf','ridge','xgboost']`.
- `tests/experiments/imrie_transfer_learning.py:103` and
  `tests/experiments/molecular_graph_survey.py:174` — ridge is a named condition in the study
  itself, not a default.
- `tests/experiments/framework/f07_internal_evaluator.py:50` and
  `f15_adaptive_pruning_validation.py:61` — both name ridge; f00-f15 are not to be edited.
- `tests/experiments/framework/f30_scorers.py:175` — catboost as a feature-importance scorer,
  which is a different axis from the downstream model.
- `tests/experiments/dataset_policy.py:106-124` — ridge appears in the per-dataset model
  *skip* lists, so it only ever removed something.

### Phase 5.4's probe is dropped

**Your answer:** "Drop it."

Splitting a budget into individually selected features plus principal components of what was
not selected. Nothing was ever built. Dropped, and the 2026-07-15 roadmap gets a line saying
so with today's date.

---

## 5. Framework experiments write here

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

### f32 — the budget split needs one normalisation rule per metric, not a search axis — 2026-09-10

**Decision: B. Closes §3.9.** One hardcoded rule per metric, keyed on the metric the
registry already supplies. Normalisation does NOT become an axis inside the search. The
`f1_weighted` null is the f1 of always predicting the majority training class.

**The rule, in full.**

| metric | registry entries | what the score becomes |
|---|---|---|
| `spearman`, `mcc` | 25 + 21 | unchanged — zero already means no signal |
| `rmse` | 9 | 1 − (rmse / rmse of predicting the training mean) |
| `auc` | 8 | 2·(auc − 0.5) |
| `ci` | 4 | 2·(ci − 0.5) |
| `mae` | 1 | 1 − (mae / mae of predicting the training median) |
| `f1_weighted` | 2 | (f1 − f1 of the majority training class) / (1 − that f1) |

All clipped at zero. Counts read out of `DATASET_REGISTRY` on 2026-09-10; it holds 70
datasets. 46 of them do not move.

**What the experiment measured.** Five ways of rescaling the per-representation score before
`create_hybrid` divides the budget by it — today's code, subtract-the-minimum-always, min-max,
the per-metric skill score above, and rank — plus a sixth on `tcga_brca_pam50` only, the skill
score against a permutation null instead of the majority class.

**Setup.** `tests/experiments/framework/f32_budget_split_normalisation.py`, local, run
2026-09-10. Result files `tests/results/framework/f32_<dataset>__fold<n>.json`, read with
`f32_analyse.py`. Nine datasets on their post-pruning representation pools — herg_ki (1412
rows, rmse), caco2 (2158, rmse), logd (5028, rmse), buchwald_hartwig (3955, rmse),
flab_thermo_tresanco (672, spearman), flab_expression_koenig (4275, spearman),
humanness_therapeutic (564, auc), nt_promoter_tata (5274, mcc, nine representations),
tcga_brca_pam50 (821, f1_weighted). Five outer folds, budgets 50 and 300 features, all five
shipped allocation methods, downstream lightgbm, internal evaluator lightgbm, importance
`lightgbm_gain`, seed 42. 45 folds, no build failures. Each arm transforms the score vector
and hands it in as `precomputed_alloc_scores`; nothing in `src/` was changed to run it.

**Checked before the grid was trusted, on real data.** On the spearman and mcc datasets the
skill rule is the identity, so it must give the allocation today gives — it did, feature for
feature, on all 15 of their folds. `rank_exp_d0.3` reads only the order of the scores, so its
split must not move under any arm — it did not, on 44 of 45 folds. The one move is expected:
on tcga_brca_pam50 fold 2 mirna_top200's f1_weighted is 0.25899 and the majority-class f1 is
also 0.25899, so its skill is exactly zero, wsi_ctranspath sits below the same reference and
is also zero, and two representations tied at zero take the roster order. They swapped 7
features for 5 at a budget of 300.

**Result — the split moves a great deal and the held-out score does not.** The allocation
differed from today's in 978 of 1,800 arm-against-today comparisons, reassigning 80,164
features in total. Representations left with no features at all: 0.57 per split under
today's code, 0.13 under the skill score.

Comparing each arm against today on the SAME fold, and counting only where the split actually
changed:

| arm | folds where the split changed | better than today | worse |
|---|---|---|---|
| `rank` | 351 | 171 | 180 |
| `minmax` | 230 | 104 | 126 |
| `skill` | 216 | 84 | 132 |
| `subtract_min` | 144 | 60 | 84 |

One row of that table counts single folds. Taken a whole dataset at a time: of the 206
arm-by-dataset-by-weighting-by-budget cases where the split changed, 8 were better on all
five folds, 29 were worse on all five, and 169 changed sign from one fold to the next.

**Why that settles A versus B.** The axis would only earn its 2.6x cost on every search if
some dataset wanted a normalisation its own metric does not already imply. Nothing here does.
The direction of the effect reverses between folds of the same dataset in most cases, so a
per-dataset search would be choosing on which molecules landed in the held-out block.

**What this does NOT say.** It never ran `find_best_hybrid`. Every cell is one build scored
on one untouched fold with the model, evaluator and importance held fixed, so nothing here
says whether a different split shape makes the search pick a different budget or allocation
method. That is a separate experiment and it is not run.

**Two things found on the way, neither fixed.**

- The AUROC that sizes the split is computed on hard class labels, because
  `_evaluate_allocation` returns `m.predict(...)`. That number is (sensitivity + specificity)
  / 2, not a ranked AUROC. Chance is still 0.5, so the rule above is unaffected. It applies to
  8 datasets. Recorded in every f32 result file as `auc_on_hard_labels`.
- `tcga_brca_pam50` cannot be split five ways. Its 19 TSS-site groups hold 133 down to 1
  samples each, so a group-respecting five-fold split of 821 samples gives held-out blocks of
  3, 79, 403, 142 and 194 rows. Fold 0 scored 28 of its 30 cells at f1_weighted exactly 1.0 on
  three molecules. `f32_analyse.py` drops folds under 20 held-out rows and names them.
  `tcga_luad_vs_lusc` is grouped the same way and its fold sizes were not checked.

**Not run: the second `auc` and second `mcc` dataset.** `tcga_luad_vs_lusc` and
`gb_human_enhancers_cohn` have no representation pool on this machine. `auc`, `mcc` and
`f1_weighted` each ran on one dataset; `rmse` ran on four and `spearman` on two. The SLURM
script `tests/slurm_scripts/run_f32_budget_split_normalisation.sh` runs all nine on the
cluster, and its `check` stage refuses if a pool is missing.

**Not implemented.** The rule above is a decision, not a code change. It lands at
`_sanitize_baseline_scores`, `src/kirby/hybrid.py:821`, and it re-runs every stored search on
the 22 datasets reporting rmse, auc, ci or mae. The 46 reporting spearman or mcc do not move.


## 6. Decided — earlier passes

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

**The table below is the 2026-09-06 setting and is superseded.** The current lists
are in §7. Kept here because the provenance paragraphs under it explain where the
2026-09-06 choices came from.

| axis | fast | full | extra (never auto-selected) |
|---|---|---|---|
| downstream model | `rf`, `lightgbm` | + `xgboost`, `svm`, `dnn` | `mlp`, `gp` |
| internal evaluator | `rf`, `match` | `match`, `rf`, `svm`, `lightgbm`, `mlp` | — |
| importance method | `deeplift`, `lightgbm_gain` | + `treeshap` | — |
| allocation | `performance_weighted`, `power_2`, `rank_exp_d0.3` | + `softmax_T1`, `wglobal_power2` | — |

Searched grid per (model, budget): **12 configurations at fast, 75 at full.**

The internal-evaluator column above never matched the code. `DEFAULT_INTERNAL_EVALUATORS`
has been `['match', 'rf', 'lightgbm', 'dnn']` since 2026-09-10; `svm` and `mlp` were
never in it, and `svm` and `xgboost` reach the search only through `match` resolving to
the model's own type.

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

---

## 7. The representation count, the fast tier and adaptive pruning — 2026-09-12

Three separate things in `find_best_hybrid` make a search cost less, and they were
being read as interchangeable. They are not, and the difference decides how each one
is set.

**The speed tier is for when compute is the constraint.** A shorter list on five axes,
fixed before the search starts from prior measurement, the same for every dataset. It
knowingly returns a worse answer for a much smaller grid. A member that is expensive
*and* wins belongs in full and not in fast. A member that is expensive and does not
win belongs in neither.

**Adaptive pruning is for finding the best answer.** It decides during the search from
this dataset's own fold scores. Hours saved is incidental. The only thing a threshold
is judged on is whether it ever deletes something that would have won, so a threshold
is chosen from the settings that lose no winner, and among those for the widest margin
either side — never for the largest saving. If `wglobal_power2` is what is winning on
this dataset, adaptive pruning keeps searching it.

**Augmentation is not a compute lever.** It asks whether feature blocks from outside
the representation pool earn a place, admitted only if they beat keeping nothing on the
internal split.

### 7.1 The representation count is now a searched axis — IMPLEMENTED

`top_2`, `top_3` and `top_5` restricted the hybrid to the best k representations. They
were `performance_weighted` with `weighting_scheme='top_k'`, so a count could only ever
be paired with one weighting. They were retired on 2026-09-01 for exactly that reason
— they moved two axes at once — and **nothing replaced them.**

`screen_threshold` was not a replacement. It differed from `top_k` in three ways at
once: it was a percentage gap rather than a count, so it kept a different number of
representations on every dataset; it fired once before the search rather than per fold;
and it ranked on whole-dataset single-representation baselines, which every fold's
validation rows had already contributed to. It also defaulted to `None`. **It is
removed as of 2026-09-12** — parameter, `_screen_reps`, cache-key entry and call site.

In its place, `rep_counts` is a list and is searched like `budgets`:

```python
DEFAULT_REP_COUNTS = [2, 3, 5, None]      # None = the whole pool
FAST_REP_COUNTS    = [3, None]
```

Each entry runs the whole grid restricted to the N representations **that fold's**
allocation scores rank highest — the per-representation score the chosen internal
evaluator gives under the chosen objective, computed on the internal carve inside the
fold's training rows. The choice never sees a validation row. A count at or above the
pool size collapses onto `None` rather than filing twice for one search. A
configuration that asks for a count in a mode with no per-fold allocation scores
(`objective='similarity'`, and blends) raises rather than silently running on the whole
pool and storing the row under N.

`rep_count` is a key column of the result database and is in the search cache key, so
every `result_db/*.db` written before 2026-09-12 is stale. The 17 that existed are in
`result_db/stale_pre_rep_count_2026_09_12/`. The count also joins the `(model,
evaluator)` group the mid-flight tail drop works inside, or a pair dropped at N=3 would
be dropped at N=10 as well.

It is a CONFIG column, not an IDENTITY column, so a fast run over `[3, None]` can still
be topped up into a full run over `[2, 3, 5, None]` — the same top-up property the
other four axes have.

**Cost.** The grid is `len(rep_counts)` times larger: 1275 configurations per dataset
became 5100 at five budgets. This is the case adaptive pruning has to carry.

**First measurement.** On the tcga_brca_pam50 smoke (lightgbm, budget 50, two folds,
45 configurations per count) the best score rose from 0.8226 to 0.8372 f1_weighted once
N=3 was reachable. One dataset, one model, two folds — a smoke, not a result.

### 7.2 The fast tier as it now stands — IMPLEMENTED

| axis | fast | full |
|---|---|---|
| downstream model | `dnn`, `lightgbm`, `rf` | + `svm`, `xgboost` |
| internal evaluator | `match`, `rf`, `dnn`, `lightgbm` | same |
| importance method | `lightgbm_gain` | + `deeplift`, `treeshap` |
| allocation | `performance_weighted`, `power_2`, `rank_exp_d0.3` | + `softmax_T1`, `wglobal_power2` |
| representation count | `3`, all | `2`, `3`, `5`, all |

Over three budgets: **162 configurations at fast, 3060 at full.** Fast is 5.3% of full.

`match` is now redundant in the fast evaluator list — all three fast models are already
explicit evaluators — but it is kept so the list does not have to change when the model
tier does.

Three changes on 2026-09-12, owner's decisions, each measured on the f33 record (five
datasets recorded unpruned: flab_thermo_tresanco, herg_ki, humanness_therapeutic, logd,
tcga_brca_pam50; 1275 configurations each, five folds, every fold score stored).

**`deeplift` left the fast importance list.** In the fast run it was 50.1%, 49.9%,
50.5%, 51.6% and 50.5% of the hours on the five datasets, and won 0 of the 6 (model,
budget) cells on four of them. It won 4 of 6 on logd. Removing it leaves every
dataset's answer unchanged except logd, which moves from 0.4905 to 0.4930 rmse.
`treeshap` was offered as the replacement and declined: it gains 0.0182 spearman on
flab_thermo_tresanco and 0.0010 rmse on herg_ki for the same hours, and gives up the
same logd 0.0024 rmse.

**`dnn` moved from the `full` model tier to `fast`.** Both labels are in
`AUTO_SELECTED_TIERS`, so this does not change what a full search runs. Adding one more
model to the fast list, one at a time, against the gap to the full grid's best in each
dataset's own metric:

| added | flab | herg | humanness | logd | tcga |
|---|---|---|---|---|---|
| nothing | 0.0273 | 0.0188 | 0.0011 | 0.0760 | 0.0031 |
| `dnn` | 0.0273 | 0.0188 | 0.0011 | **0.0103** | **0.0000** |
| `svm` | 0.0273 | **0.0010** | 0.0011 | 0.0263 | 0.0031 |
| `xgboost` | 0.0273 | 0.0188 | 0.0011 | 0.0760 | 0.0031 |

`dnn` was the largest single gain available on any of the four lists. `xgboost` closed
nothing on any of the five.

**`lightgbm` joined the fast evaluator list.** It closed flab_thermo_tresanco from
0.6357 to 0.6506 spearman — the full grid's best exactly.

After all three, four of the five datasets reach the full grid's best exactly or within
0.0001. What remains: herg_ki is 0.0188 rmse short because its best is an `svm`
configuration and `svm` is a full-tier model, and logd is 0.0103 rmse short because its
best configuration needs `wglobal_power2`, `deeplift` and the `lightgbm` evaluator
together — no one of them alone moves it.

**`wglobal_power2` stays out of the fast allocation list.** In the full grid it wins 60
of the 125 measured (model, budget) cells and is the most expensive allocation method
in the set — 71.0% of logd's 14.45 measured hours. Inside the fast tier it wins nothing
extra, because with one importance method and three models the cells it would win are
already reached by `rank_exp_d0.3`. Adding it costs 55% to 80% more hours and changes
one of five datasets' answers, by 0.0022 rmse on logd. That is the expensive-and-wins
case: it belongs in full, not in fast.

### 7.3 Adaptive pruning — OPEN, being re-measured

⛔ **Every number in `DEFAULT_PRUNING_SCHEDULE` and in `_shared.PRUNING_MODES` is
uncalibrated against the current search space.** They were fitted in April (f15, f16)
over 13 allocation methods of which two survive, on a grid with no representation-count
axis. §3.4 of this document treats the fold-0 vote as a setting to validate; it is
being rebuilt instead.

What is measured so far, on the five datasets recorded before the representation-count
axis existed. Stated as what it is: a reading of the old grid, kept because it is the
only evidence on the table until the re-runs land.

The shipped setting — top-3 of 5 at 15%, voting on fold 0 — deletes nothing on
flab_thermo_tresanco, herg_ki, humanness_therapeutic or logd. On tcga_brca_pam50 it
deletes `rank_exp_d0.3` and `wglobal_power2` and makes 18 of 25 (model, budget) winners
unreachable, at a 0.0181 f1_weighted mean gap. The cause is measured: 198 of that
fold's 255 groups have all five allocation methods scoring the identical number, and
241 of 255 are tied at the third-place boundary, so the vote ranks by the order the
methods are written in `DEFAULT_ALLOCATION_METHODS` and deletes the last two names.
Voting on fold 1, 2, 3 or 4 alone, or on folds 0 and 1 together in either form, deletes
nothing on any of the five.

Across the whole sweep — `strategy_top_n` in 1 to 4 crossed with the threshold from 5%
to 50% — **no setting is safe on all five datasets and deletes anything.** The only two
that are safe on all five delete nothing at all.

Which axis a prune should act on was measured the same way: delete the member fold 0
ranks last, then check the five-fold answer.

| axis | winners lost, worst dataset | mean gap there |
|---|---|---|
| allocation | 7 of 25 (tcga_brca_pam50) | 0.0123 f1_weighted |
| importance | 4 of 25 (tcga_brca_pam50) | 0.0010 f1_weighted |
| internal evaluator | 1 of 25 (flab_thermo_tresanco) | 0.0055 spearman |

The prune currently acts on the allocation axis, which is the one it damages most, and
there is no rule at all for importance, where fold 0 identifies the dead method
correctly on three of five datasets outright.

Sweeping a gap threshold rather than a place count, on the importance axis: at **4%
after one fold**, no winner is lost on any of the five datasets. The safe window is
one value wide — at 3% flab_thermo_tresanco loses 14 of 25, at 5% nothing is deleted.
**After two folds the safe window is 4% to 6%**, three values wide. That is the
one-fold-versus-two-fold question answered: one fold sits on a knife edge, two folds
has room either side.

**None of this is decided.** All five datasets are being re-recorded with the
representation-count axis in, 5100 configurations each. Thresholds are to be set on
that record, not on this one.

### 7.4 What is not measured

- Augmentation. Every f33 record was made with `augmentation_strategy='none'`.
- The budget axis as a prune target. "Winners" are defined per (model, budget) cell, so
  deleting a budget mechanically deletes its five cells and the count says nothing. It
  needs "lost" redefined as the best over the whole search before it can be answered.
- Four of the nine datasets with a complete representation pool: `buchwald_hartwig`,
  `caco2`, `flab_expression_koenig`, `nt_promoter_tata`. Not cut — not yet recorded.

---

## 8. Read against the code before the precompute — 2026-09-13

Eighteen passes over `find_best_hybrid.py`, `hybrid.py`, `result_cache.py` and
`tests/precompute_hybrids.py`, four of them on metric direction alone. Everything below
was read in the code on 2026-09-13 and the fixes are in the working tree. Ten of the
eighteen passes stopped part-way on a usage limit; the four they did not finish are
named at the end.

### 8.1 Metric direction

**No flipped comparison was found on either objective this pass runs.**

- The metric table at `find_best_hybrid.py:1496` marks `rmse` and `mae` as the only two
  where a smaller number is better. All thirteen entries were read against the function
  each one maps to.
- All 74 registered datasets were checked against that table: zero disagreements, and no
  registered metric is missing from it.
- Every winner is chosen with `max()` over `_score_to_comparable`, which negates the
  score when smaller is better, after every non-finite score has been removed.
- The per-representation scores are ranked with the representations' own direction —
  `min()` on rmse, `max()` on spearman — not with the blend's flipped flag. That is the
  comparator defect from 2026-09-03 and it is closed in both places.

**Measured, not read.** A representation of pure random numbers was planted into the real
representation pool of caco2 (2158 molecules, rmse, smaller is better) and of
`flab_expression_koenig` (4275 antibodies, spearman, bigger is better), and the search
run on each.

| dataset | metric | planted noise scored | best representation scored | where noise ranked | features it got of 100 |
|---|---|---|---|---|---|
| caco2 | rmse, smaller better | 0.4414 | 0.3169 (`pdv`) | last of 11 | 0 |
| flab_expression_koenig | spearman, bigger better | -0.0091 | 0.6770 (`prot_t5`) | last of 11 | 1 |
| herg | auc, bigger better | 0.5083 | 0.8351 (`ecfp4`) | last of 2 | 3 |

A flipped comparison anywhere would have put the planted noise first and given it the
largest share. All three pass on the fixed code. Script:
`scratchpad/smoke_direction_real.py`.

`herg` is a classification dataset and its metric is auc, so it is the arm that exercises
§8.2 on real molecules. One caveat on that row: this machine's representation cache holds
only `ecfp4` of `herg`'s pool of 10, so the planted noise competed against one real
representation rather than ten. The direction result stands — noise last, smallest share —
but the split had almost nothing to split. It is also the case the new refusal in §8.4
catches: run on this machine today, `herg` is refused with "9 of 10 pool representations
did not load" instead of being searched over one.

### 8.2 auc was scored off hard predicted classes — FIXED

Two things that disagree. The main search scores a fitted classifier through
`_predict_for_metric`, which returns the positive class's probability when the metric is
auc. The pass that computes the per-representation scores that **split the budget** calls
`_evaluate_allocation` with `return_predictions=True`, which returns `model.predict` —
hard 0s and 1s — and then applies every entry of the metric table to that one vector.

So on a dataset whose metric is auc, the number stored under the name `auc` and used to
size each representation's share was AUROC read off thresholded labels, which is
(sensitivity + specificity) / 2. Higher is still better and nothing raises; the direction
was never wrong. The quantity was.

Measured on synthetic data, one fitted random forest, 100 held-out rows: **0.7436 off the
hard classes against 0.8732 off the graded ones.** That is the number that sizes the
split.

Six of the 42 datasets in the roster have auc as their metric: `herg`,
`tz_her2_binary`, `tcga_luad_vs_lusc`, `humanness_therapeutic`, `flab_binding_adams`,
`cdr_flexibility`.

The fix takes two prediction vectors from the one fitted model. `accuracy`, `f1_weighted`
and `mcc` keep the hard classes — they raise on a graded vector — and `auc` is read off
the graded one. Both the search's precompute and the non-search path
(`_score_rep_for_budget`) now do this. A fold-score file written before today records the
old quantity, so its metadata gained a key and it is a miss, not a hit.

Gated by `test_the_budget_split_reads_auc_off_graded_predictions` in
`tests/test_metric_direction_end_to_end.py`. That file planted only regression labels
before today, so nothing in it exercised a classification dataset at all; it now also runs
the whole direction check on one.

### 8.3 Two configurations shared one fold score — FIXED

The in-memory reuse key sorted each representation's chosen feature positions. The matrix
is built with those positions in importance-descending order. So two configurations that
pick the same features in a different order got the same key, and the second was handed
the first's fold score instead of being fitted. The usual colliding pair is two importance
methods — an axis the precompute sweeps. `sorted` is gone from the key; two genuinely
identical matrices still share it.

### 8.4 A partly-loaded representation pool now refuses — FIXED

The loader drops a representation it could not build, prints a warning and returns the
rest. `precompute_hybrids.py` kept whatever came back and ran. Which representations a
search ran over is part of the result database's identity (`rep_pool`, stored as sorted
name-and-width pairs), so a search over 7 of the 10 is a different search and no
experiment asking for the full pool will ever read it — it finds nothing and rebuilds
the whole search inside its own wall. The log line read `general=7` and the run reported
success.

A dataset missing any of its 10 pool representations is now refused by name. A dataset
missing an augmentation representation keeps `none` and drops `uncapped` and `capped`.

### 8.5 Smaller things fixed in the same pass

- Three entries of the metric table are guards that raise — `sac`, `readacross`,
  `struct_fidelity`. The allocation-score pass ran all three for every representation,
  every fold and every evaluator, caught the exception each time, and stored three
  all-NaN vectors in every cached file. Nothing reads them. They are skipped now.
- A blend containing prediction stamped `metric='rmse'` next to `lower_is_better=False`
  onto every recipe. `apply_blueprint` resolved the direction from the NAME and rebuilt
  with `lower_is_better=True` — the opposite of the search that chose the recipe. It now
  takes the blueprint's own flag when the objective is such a blend. This pass runs no
  blends; f28 and f32 do.
- `augment_uncapped` overwrote the task type its caller passed with a guess made from the
  label values. `augment_capped` keeps the argument and names its guess separately. The
  same dataset under `uncapped` and under `capped` could be scored as two different task
  types. Renamed to match.
- Every fold of every augmented configuration tripped the warning saying the returned
  matrix is the base block alone and therefore a different feature set from the one that
  was fitted. The matrix is correct — the extras are appended two lines later. It was the
  loudest line in an augmented run and it said the opposite of what happened.
- `precompute_hybrids.py` could not fail. Every error was printed and swallowed, the last
  line read `All done: N datasets` and the process exited 0. A job in which every dataset
  failed to load looked exactly like one that wrote everything. It now prints searches
  written against searches asked for, names what was refused, and exits 1.
- `submit_precompute_extra_passes.sh` put `--pruning 0.25` on every capped job. It could
  never take effect — the pruning mode defaults to `no_pruning`, which always yields a
  schedule, and a schedule wins over the threshold. The number reached the search metadata
  and nothing else, where it read as though capped was pruned. Removed.

### 8.5b The representation-count axis — what it does and what it does not carry

`DEFAULT_REP_COUNTS = [2, 3, 5, None]`, and the precompute sweeps it: the whole grid runs
once per entry, each time restricted to the N representations that fold's allocation
scores rank highest, `None` meaning the whole pool. It replaced the `top_2`/`top_3`/
`top_5` allocation names, which welded a representation count into the weighting slot.
Those three names are still implemented and callable — what changed is that they are out
of the default set, and `create_hybrid`'s default when it is not told is
`performance_weighted`. The default allocation set is the five from the c32 cross sweep.
Nothing in the precompute path depends on a `top_N` name.

The restriction itself was read and is right: the scores it ranks by are
higher-is-better whatever the dataset's metric is, it sorts descending always, and a
configuration that asks for N and has no per-fold ranking raises rather than quietly
running on the whole pool.

**What the recipe did not say.** `Blueprint.reps` lists the pool the configuration was
chosen from, not the N it used, and the Blueprint had no `rep_count` field at all — so a
recipe that won on 2 of the 10 representations read back as a recipe over 10, and a
rebuild spread the same budget over all ten. `rep_count` is a field on the Blueprint now
and is stamped from the configuration, matching the column the result database already
had. **The rebuild side is not fixed**: `build_fold_hybrid` in `tests/experiments/_shared.py`
still takes whatever representations its caller hands it. The information is now on the
recipe for a caller to use; nothing uses it yet.

### 8.6 Open, and not fixed here

**`kiba` and `bindingdb_kd` — one line, your call.** `precompute_hybrids.py` drops the
`uncapped` condition for these two on a 2026-06-21 cost decision (`general/none` alone
was about 4 days on `bindingdb_kd`). `_shared._pool_conditions` builds `none`, `uncapped`
and `capped` for any dataset that has augmentation representations, with no dataset
exceptions — and both names are in the roster with 10 pool representations and 3
augmentation representations. So the read side asks for something the write side never
makes, falls through, and builds that search live inside the experiment's wall. Either
pay the uncapped pass on two 50k-row datasets, or add the two names to `_pool_conditions`
so it stops asking.

**Fingerprints are z-scored in the hybrid arm and raw in the best-single arm.** Still
open, still the 324-site handoff in `SCALING_MOVE_HANDOFF_2026_09_04.md`. One correction
to how it is usually stated: the honest outer fold rescales both arms, so the published
paired difference is not affected. What is affected is the per-representation scores
*inside* the search — the ones that rank single representations against the hybrid and
that get stamped onto every recipe. Trees are unaffected; svm, dnn and knn are not.

**`hybrid_registry.json` merges rather than replaces on a re-run**, so a model or budget
the new run did not score keeps the previous run's score and recipe beside the new run's
metric. Nothing reads the file — the lookup function has no callers in Python.

### 8.7 What was not read

Four of the eighteen passes stopped on a usage limit and their areas were not covered:
the delta's sign under blends and geometry objectives read end to end, the direction
inside the budget split as its own pass, whether this document matches the code line by
line, and a pass asking only what would force the precompute to be run twice. The
direction work in §8.1 and §8.2 was done by the passes that did finish, plus the planted-
noise run on real data.
