# Prompt for a separate Claude chat — KIRBy hybrid generation + search-space diagram

You are helping design a **methodology figure for a research paper** about KIRBy, a system that builds "hybrid" molecular/biological representations by mixing features from many base representations and then augmenting them, with a large automated search over how to do the mixing. Below is a complete, code-grounded background summary of the two subsystems the figure must depict: **(A) hybrid generation** and **(B) search-space optimisation**. It was extracted directly from the source. Your job: **produce one or more clear diagrams** (a top-level pipeline flow plus, if useful, expanded sub-panels for the search axes and the augmentation strategies) that a reader of a methods paper could follow. Ask me which diagram style/tooling I want (e.g. TikZ, Graphviz/DOT, Mermaid, or an SVG) if it matters; otherwise propose a layout and produce it. Preserve the exact vocabulary (strategy names, knob names, defaults) — these are the terms the paper uses.

Source files (for reference, in repo `/Users/apunt/repos/KIRBy`):
- `src/kirby/hybrid.py` (~3925 lines) — hybrid generation: allocation, feature importance, augmentation, `create_hybrid`.
- `src/kirby/find_best_hybrid.py` (~5473 lines) — the search engine, caching, blueprint, nested-CV.
- `tests/experiments/_shared.py` — experiment-layer constant lists (budgets, evaluator pools, downstream model set).

---

## Vocabulary (define these once in the figure)

- **Base representation (rep):** one feature matrix `X ∈ R^{n×d}` for a dataset (e.g. a molecular fingerprint or a protein language-model embedding). A dataset has many candidate reps.
- **Hybrid:** a single feature matrix assembled by selecting a subset of columns from several reps (the "base hybrid"), optionally grown/edited with extra features ("augmentation").
- **Allocation:** the rule deciding *how many* columns each rep contributes to the budget.
- **Importance:** the rule deciding *which* columns within a rep are picked.
- **Augmentation:** a post-step that treats extra reps as a candidate feature pool and appends/swaps features into the hybrid via model-scored A/B tests.
- **Budget:** the target total feature count of the base hybrid.
- **Blueprint:** the serialized, transferable recipe (reps + allocation + importance + budget + evaluator + step_size + objective) that reproduces a hybrid on new data.
- **Objective:** either `predictive` (score a fitted downstream model's held-out metric) or `similarity_sac` (score held-out geometry via similarity–activity correlation, *no model fit*). This objective swap is the conceptual novelty — everything scores through one chokepoint (`_evaluate_allocation`).

---

## SUBSYSTEM A — HYBRID GENERATION (`hybrid.py`)

Top-level entry point: **`create_hybrid(base_reps, labels, ...)`**. It ties together allocation → importance → augmentation. Real defaults (from the signature, not the docstring):
`allocation_method='top_3'`, `n_per_rep=100`, `budget=100`, `step_size=5`, `patience=20`, `validation_split=0.2`, `importance_method='lightgbm_gain'`, `augmentation_strategy='uncapped'`, `augmentation_budget=20`, `candidate_ranking='rf_importance'`, `internal_evaluator='svm'`, `objective='predictive'`.

### create_hybrid control flow (depict as the main hybrid-generation pipeline)
1. **Validate inputs** (shapes, sample-count consistency, positivity of budget/step_size, labels finite).
2. **Resolve `internal_evaluator='match'`** → maps the downstream model's class name to an evaluator (`ridge/rf/xgboost/lightgbm/mlp/svm/gp/knn`).
3. **Optional quality filters** (`apply_filters`): per-rep sparsity + variance pruning (records an index remap).
4. **Train/val split** — group-aware `GroupShuffleSplit` if `groups` given, else seeded random permutation (`random_state=42`).
5. **Named-allocation resolution:** if `allocation_method` is in `NAMED_ALLOCATIONS` (33 named variants), unpack to `(base_method, extra_kwargs)`.
6. **Allocation dispatch** → one of four base methods (see below): produces `allocation = {rep: n_features}` + per-rep selected indices.
7. **Optional budget floor** (`budget_fill=True`): if the hybrid is under budget, greedily append each rep's next best unselected features until full.
8. **Auto-demote for top-k:** for `top_2/3/5`, reps that got 0 features are pushed into the augmentation pool (their top-`augmentation_budget` features) so they can still be A/B-tested.
9. **Augmentation dispatch** → one of the strategies below (or `none`).
10. **Return** `(hybrid_features, feature_info)`. `feature_info` fully records the selection so `apply_feature_selection` + `apply_augmentation_selection` reproduce the exact hybrid on a held-out test fold.

### A.1 Allocation strategies (how many columns per rep)
Four **base methods** (dispatched inside `create_hybrid`):
- **`greedy`** (`allocate_greedy_forward`): forward selection in rep-chunks of `step_size`; each iteration tries adding a chunk from each rep, keeps the best, with plateau/patience early-stop (`greedy_tolerance` allows half-rate patience within tolerance). *Importance-insensitive* (uses an internal RF's `feature_importances_`).
- **`performance_weighted`** (`allocate_performance_weighted`): score each rep standalone, then allocate budget ∝ a transformed weight. *Importance-sensitive.*
- **`fixed`**: give every rep exactly `n_per_rep` columns.
- **`dominant_greedy`** (`allocate_dominant_greedy`): give the single best rep a large fixed share (`dominant_fraction`, default 0.7), then greedy-fill the rest. *Importance-insensitive.*

Two extra concat-space methods exist in code (`allocate_two_stage`, `allocate_stability_selection`) but are **not** dispatched by `create_hybrid`'s four-way branch — mention only if depicting the full library.

The **`NAMED_ALLOCATIONS`** registry (33 entries) parametrizes these bases into the named strategies the paper reports. Families and their knobs:
- **top_k** (over `performance_weighted`): `top_2` (k=2), `top_3` (k=3, the global default), `top_5` (k=5) — keep the k best reps.
- **softmax temperature:** `softmax_T0.5/T1/T2/T5/T10` (temperature 0.5…10).
- **power law:** `power_2` (p=2), `power_3` (p=3).
- **thresholded:** `thresh_0.3/0.5/0.7` (drop reps below α·max).
- **min_floor:** guarantee each rep ≥ `floor` (default 0.05) of budget, distribute the rest proportionally.
- **rank-based:** `rank_linear`; `rank_exp_d0.1/d0.3/d0.5` (exponential decay by rank).
- **score_gap:** `score_gap_0.2/0.3/0.5/0.7` (drop reps more than gap·best below the best).
- **entropy (redundancy-penalised):** `entropy_0.5/1.0/2.0` (down-weight reps correlated with already-accepted higher-scoring reps).
- **greedy forced / greedy fill:** `greedy_forced_s5/s10/s25` (no patience, fill budget); `greedy_fill_s5/s10/s25` (patience + budget fill) — the only `step_size`-sensitive family alongside `dominant_greedy`.
- **dominant greedy:** `dominant_greedy`, `dom_greedy_f0.7`, `dom_greedy_f0.9`.

All allocators ultimately score candidate matrices through the single chokepoint **`_evaluate_allocation`**, which is where the `predictive` ↔ `similarity_sac` objective swap happens (predictive = fit an internal evaluator / model + R²/accuracy; similarity = no fit, score held-out similarity–activity correlation).

### A.2 Feature importance (which columns within a rep) — `compute_feature_importance`
Default `lightgbm_gain`. Full method menu (all clean features, StandardScaler, return per-feature scores):
- **Fast:** `treeshap` (mean |SHAP| over a RandomForest; the library default), `xgboost_gain`, `lightgbm_gain`, `deeplift` (Captum DeepLift on an MLP).
- **Slower:** `permutation` (sklearn permutation importance), `kernelshap` (model-agnostic KernelExplainer).
- **Long-search only:** `integrated_gradients`, `gradientshap` (Captum, on an MLP).
- (`mutual_info` was intentionally removed.)
High-dim reps are variance-prescreened (e.g. `max_features_shap=2000`) before SHAP/gain, then scores mapped back to full space. Note: `greedy` and `dominant_greedy` ignore `importance_method` (they use an internal RF).

### A.3 Augmentation strategies (post-step; extra reps as candidate pool)
Shared machinery: pool all candidate columns → rank them (`candidate_ranking ∈ {rf_importance, residual_correlation}`) → trim to `max_candidates` (default 50) → sequential A/B test each candidate through `_evaluate_allocation`, accept iff strictly improves; hard-coded `patience=10` at the call sites.
- **`uncapped`** (DEFAULT): no budget ceiling, **append-only** — grows the hybrid beyond `budget`, never replaces base features.
- **`capped`**: fixed ceiling = `budget`. Phase A fills the slack between `n_base` and `budget` (append); once full, Phase B **swaps** — find the worst base feature by permutation importance and overwrite it with a better candidate (net count unchanged).
- **`greedy_ablation`**: whole-rep granularity — for each augmentation rep, keep its top-`augmentation_budget` features iff adding the block improves (or beats `augmentation_threshold`).
- **`all`**: unconditionally concatenate the top-`augmentation_budget` features of every augmentation rep.
- **`none`**: skip.

---

## SUBSYSTEM B — SEARCH-SPACE OPTIMISATION (`find_best_hybrid.py`)

Entry point: **`find_best_hybrid(base_reps, labels, groups=None, *, budgets=..., task_type=..., ...)`**. `task_type` is required (no auto-detect). It searches a large product space of hybrid configs, scores each by cross-validation, prunes aggressively, and returns the best **Blueprint(s)**.

### B.1 The search axes (the "search space" — depict as a product/grid)
A config is a point in:
`[ rep-subset × allocation × budget × importance × downstream-model × internal-evaluator ]`, at a fixed `objective` and (mostly) fixed `step_size`.

| Axis | Default list (source) |
|---|---|
| rep subset | `['all']`; `+ leave-one-out (no_<rep>)` only if `test_rep_subsets=True` and ≥3 reps |
| allocation | `SPEED_TIERS['full']`: `top_2, top_3, top_5, performance_weighted, entropy_1.0, power_2, rank_linear, entropy_2.0, rank_exp_d0.5` (9 of the 33 named) |
| budget | caller-supplied; canonical `FULL_BUDGETS = [10,25,50,100,250,500,1000,2000]` (also `SCREENING_BUDGETS`, `DISTANCE_BUDGETS`), per-model capped |
| importance | `['deeplift','lightgbm_gain','treeshap']` |
| downstream model | `DOWNSTREAM_MODELS = ['rf','xgboost','ridge','svm','mlp','lightgbm']` (full registry adds `knn,gp,catboost,dnn,cnn,bnn,ngboost`) |
| internal evaluator | `INTERNAL_EVALUATORS = ['match','rf','svm','lightgbm','mlp']` (distance lineage: `DISTANCE_EVALUATORS = ['knn','ridge','rf','mlp','lightgbm','svm']`) |
| step_size | fixed `5` in main loop; varied `[1,2,5,10,25]` only in refinement |
| augmentation strategy | caller: `uncapped` default |
| objective | `predictive` (default) or `similarity_sac` |

Per-model budget caps `MODEL_MAX_BUDGET = {gp:1000, knn:2000, svm:500, mlp:1000, bnn:1000, dnn:1000, cnn:1000}` (rf/xgboost/lightgbm/catboost/ridge/ngboost uncapped). Downstream-model registry has tiers (`fast`/`full`/`extra`); `speed='full'` uses fast+full, `extra` models only if named. **`tier`** (distinct from `speed`) is a cache/condition namespace — `general` (full rep pool) vs `top` (top reps) vs `general_distance` (distance-evaluator lineage). Reps are deliberately **not** in the cache key; `tier` covers general-vs-top.

### B.2 The search pipeline (depict as the top-level flow)
1. **Validate/normalize** inputs; apply budget policy; drop budgets `> n_samples` (append `n` as a boundary point).
2. **Resolve** metric, secondary metrics, downstream models, folds.
3. **Precompute fine-tuned reps per fold** (if `rep_factories`) — each factory runs once per fold, embeddings cached.
4. **Baselines:** score every (rep × model) at full dimensionality via CV (cached per-(rep,model)). Feeds screening, allocation ranking, and `Blueprint.rep_scores`.
5. **Optional rep screening** (`_screen_reps`, if `screen_threshold` set) — drop weak reps before the search.
6. **`_run_search`** — the core engine:
   a. Precompute the **importance cache** (per fold × importance method, once — identical across alloc/budget/evaluator/model).
   b. Precompute **allocation baseline scores** (per fold × evaluator × metric).
   c. Enumerate the flat config grid (with per-model evaluator + budget resolution and the `top_k ≥ n_reps` guard).
   d. Optional expensive-first ordering (`KIRBY_EXPENSIVE_FIRST`) — timeout-casualty control, result-neutral.
   e. **Adaptive pruning** (multi-stage): evaluate every config on **fold 0 only**, then (i) mid-flight tail pruning drops the bottom fraction of `(alloc, importance)` pairs within each `(model, evaluator)` group at each checkpoint budget, and (ii) end-of-fold-0 strategy pruning drops allocations that rarely land in the top-3. Survivors go to full CV.
   f. **Full-fold CV scoring** of survivors via `_evaluate_hybrid_config` → `create_hybrid` (allocation + selection) → StandardScaler + impute → fit `deepcopy(model)` → metric (predictive) *or* similarity-geometry score (similarity). Finite rows upserted to a per-dataset **SQLite ResultCache** (NaN not stored, retried next run). Checkpointing every 100 configs.
7. **Optional two-phase refinement** (`_run_refinement_phase`): re-evaluate the top configs whose allocation is `step_size`-sensitive across `step_sizes=[1,2,5,10,25]`.
8. **Aggregate** → best-by-model / -dimension / -budget / -(model,budget) + performance matrix.
9. Build **`Blueprint`** (reps, allocation, importance, budget, evaluator, step_size, objective, metric, `rep_scores`) and return a `SearchResult`.

### B.3 How one config is scored (sub-panel, `_evaluate_hybrid_config`)
Per CV fold: slice train/val → look up precomputed fold importance + alloc scores → optional persistent hybrid-eval cache probe (top_k configs, keyed by subset **name** not value) → `create_hybrid` on train fold, `apply_feature_selection` to val → scale + impute → **predictive:** fit downstream model, predict, compute the ranking metric + full classic metric set; **similarity:** no fit, score fold geometry (SAC). CV splitter is always **group-aware** (`StratifiedGroupKFold` for classification, `GroupKFold` for regression; `n_folds=1` → single 80/20; **raises rather than random-split** — no-random-splits is a hard rule).

### B.4 Downstream artifacts (depict as outputs)
- **`Blueprint` / `apply_blueprint`:** apply a recipe to new data with no search; can run **label-free** if `rep_scores` are stored (skips per-rep model evaluation).
- **`get_best_hybrid`:** registry lookup (`hybrid_registry.json`) keyed `dataset → tier/aug_condition → model → budget`.
- **`nested_cv_evaluate`:** honest outer evaluation; runs the full search on outer-train only, applies the winner to outer-test, and reports **bias = search_mean − honest_mean** (quantifies selection bias).

### B.5 Metrics
`BUILTIN_METRICS` (scorer, lower_is_better): `r2`(F), `rmse`(T), `mae`(T), `accuracy`(F), `auc`(F), `spearman`(F), `pearson`(F), `ci`(F), `f1_weighted`(F), `mcc`(F), `sac`(geometry, F). Everything is compared through a direction-aware "higher is better" convention.

---

## What I want from you
Produce the figure(s). At minimum:
1. **A top-level pipeline diagram** spanning: base reps → baselines → (screening) → search grid → adaptive pruning (fold-0 → survivors) → full-CV scoring → refinement → aggregate → Blueprint → {apply_blueprint / registry / nested-CV}.
2. **A "hybrid generation" sub-panel** showing `create_hybrid`: allocation (4 base methods + named families) → importance (which columns) → augmentation (uncapped/capped/greedy_ablation/all), all scored through the `_evaluate_allocation` chokepoint with the predictive/similarity objective swap highlighted.
3. **A "search space" sub-panel** showing the 6 grid axes with their default lists as a product, plus the pruning funnel.

Keep the named strategies, knob names, and default values verbatim. Call out the two conceptual pillars: (i) the **objective swap** (predictive vs similarity, model-free) and (ii) the **grid-enumerate → screen → prune → CV** funnel. Ask me for the output format if it changes what you produce; otherwise pick a clean layout (I'd suggest Graphviz/DOT or TikZ for a paper) and generate it.
