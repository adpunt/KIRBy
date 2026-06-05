# KIRBy Cache-Invalidation Contract

READ-ONLY audit, 2026-06-05. Every claim cites `file:line`. Files audited:
`src/kirby/find_best_hybrid.py` and `tests/experiments/_shared.py`.

Goal: state EXACTLY which changes the cache auto-detects (recompute, no action)
versus which slip through silently (same shape + same labels, different rep
values → stale until manually purged). This bounds the "what must be deleted"
question.

---

## 0. The five cache layers

| Layer | Path builder | Meta builder | Filename = key | Validated header |
|-------|--------------|--------------|----------------|------------------|
| search | `_search_cache_path` (`find_best_hybrid.py:89`) | `_make_search_meta` (`:277`) | dataset, tier, aug_strategy, speed, condition_suffix (`:95-98`) | yes |
| baseline | `_baseline_cache_path` (`:109`) | `_make_baseline_meta` (`:398`) | dataset, rep, model, metric, folds, seed (`:111-113`) | yes |
| folds | `_folds_cache_path` (`:116`) | `_make_folds_meta` (`:425`) | dataset, folds, seed, grouped/ungrouped (`:119-120`) | yes |
| hybrid_eval | `_hybrid_eval_cache_path` (`:445`) | `_make_hybrid_eval_meta` (`:552`) | **subset_hash (= content hash of rep values, `:467`)**, alloc, imp, evaluator, budget, fold, seed, model, aug_strategy | yes |
| rep_cache | `_cache_rep` / `_cache_rep_dict` (`_shared.py:168`,`:194`) | NONE | `<dataset>/<rep>.npy` (`:171`,`:215`) | **no header at all — existence only** |

All five validated layers go through the single function `_load_cache_with_meta`
(`find_best_hybrid.py:182`). The rep_cache (the `.npy` layer in `_shared.py`)
does NOT use it.

---

## 1. Search-cache load/save path and `_cache_meta` validation

### The validation function — `_load_cache_with_meta` (`find_best_hybrid.py:182-251`)

Validation loop (`:239-249`):
```python
stored = wrapper.get('_cache_meta') or {}
for k, v in expected_meta.items():
    if k in ignore_keys:
        continue
    if k not in stored or stored[k] != v:
        ... return None   # STALE → recompute
```
It iterates the **expected** meta's keys; for each non-ignored key, a hit
requires the stored value to be present AND equal. Note the asymmetry: extra
keys present in `stored` but absent from `expected_meta` are NOT checked — only
keys the caller currently expects must match. This is what makes
`AUG_LOGIC_VERSION` / `pruning_schedule` back-compatible: a legacy file lacking a
key the new caller expects → `k not in stored` → stale; a legacy caller not
expecting a key a newer file has → ignored.

### The `ignore_keys` — the CACHE-1 fix (`find_best_hybrid.py:198-199`)

```python
ignore_keys: Tuple[str, ...] = (
    'code_timestamp', 'rep_hashes', 'aug_hashes', 'rep_hash'),
```
This default is **never overridden anywhere** — confirmed by grep: the only two
occurrences of `ignore_keys` in either file are the definition (`:198`) and the
loop check (`:241`). So the contract below applies to every load call.

The four ignored keys and why (verbatim rationale, `:185-197`):
- `code_timestamp` — wall-clock stamp written on every save (`:393`, `:421`,
  `:441`, `:591`, `:624`); never a correctness signal.
- `rep_hashes` (plural; search meta, `:353`), `rep_hash` (singular; baseline
  meta, `:408`/`:675`), `aug_hashes` (search + hybrid_eval, `:364`/`:581`) — all
  three are **content hashes of the representation/augmentation arrays**. They
  are ignored because neural/transformer reps (ESM, prot_t5, geneformer, DNA-LMs,
  chembert, etc.) are NOT bit-reproducible across thread counts / GPU / library
  versions, so their value-hash drifts run-to-run and was wrongly rejecting valid
  caches (the infinite-recompute "churn" on molecular jobs). **This is the
  CACHE-1 fix and the crux of the silent-staleness exposure.**

The code's own stated safety net (`:193-197`): cache identity is "still guarded
by rep_names + rep_shape(s) + n_samples + seed + labels_hash + groups_hash +
budgets + tier + strategy + model/metric/fold config ... A genuine rep change
that alters dimensionality is still caught by rep_shape(s); a target change is
still caught by labels_hash + n_samples." The unstated gap: a rep change that
keeps the SAME shape and SAME labels is NOT caught. See §4.

### Fields VALIDATED on a search load — `_make_search_meta` (`:343-395`)

Every key below is in `expected_meta` and (except the ignored ones) must match:

- `kind` = 'search' (`:344`)
- `dataset_name` (`:345`)
- `tier` (`:346`)
- `aug_strategy` (`:347`)
- `speed` (`:348`)
- `n_samples` = `len(labels)` (`:349`)
- `rep_names` = sorted rep keys (`:350`)
- `rep_shapes` = sorted (name, shape) tuples (`:351-352`) — **catches dimensionality changes**
- `rep_hashes` (`:353`) — **IGNORED on load**
- `model_names` (`:354`)
- `metric_name` (`:355`)
- `task_type` (`:356`)
- `n_folds` (`:357`)
- `seed` (`:358`)
- `budgets` (`:359`)
- `labels_hash` = SHA1 of label bytes (`:360`) — **catches any target value change**
- `groups_hash` = SHA1 of group bytes, or None (`:361-362`) — **catches split-group changes**
- `aug_names` (`:363`)
- `aug_hashes` (`:364`) — **IGNORED on load**
- `augmentation_strategy` (`:365`)
- `internal_evaluators`, `allocation_methods`, `importance_methods` (`:366-371`)
- `adaptive_pruning` (`:372`)
- `augmentation_threshold`, `augmentation_budget` (`:373-375`)
- `candidate_ranking`, `greedy_tolerance`, `screen_threshold`, `two_phase`,
  `refinement_top_n`, `refinement_step_sizes`, `test_rep_subsets`,
  `dimension_tiers`, `weight_method`, `step_size`, `secondary_metrics` (`:376-390`)
- `cv_splitter_id`, `rep_factories_id` (`:391-392`) — caller-supplied identifiers
  for unhashable objects (custom splitter / rep factories)
- `code_timestamp` (`:393`) — **IGNORED on load**
- conditional extras `pruning_checkpoints` / `pruning_tail_drop` /
  `pruning_min_configs` / `model_pruning_overrides` (`:317-342`, only when
  non-default)
- `condition_suffix` — injected onto the meta after construction at the call site
  (`:4223`), so it is validated too.

Load call site: `:4225-4243` (plus an empty-payload guard at `:4232-4239` that
treats `best_by_model={}` + `all_results=[]` as a miss). Save: `:4528`+ via
`_save_cache_with_meta` (`:254`).

---

## 2. Baseline-cache and folds-cache layers

### Baseline — `_make_baseline_meta` (`find_best_hybrid.py:398-422`), load at `:2676-2678`

Validated keys: `kind`='baseline', `dataset_name`, `rep_name`, `rep_shape`
(`:407` — catches dim change), `rep_hash` (`:408`, set by caller at `:2675` from
`_hash_array(X)`, `:2648`) — **IGNORED on load**, `model_name`, `metric_name`,
`n_folds`, `seed`, `n_samples`=`len(labels)` (`:413`), `labels_hash` (`:414`),
`groups_hash` (`:415-416`), `hd_filter_version` (`:419` = `HIGH_DIM_FILTER_VERSION`,
constant `=1` at `:2098`), `hd_filter_threshold` (`:420` = constant `=5000` at
`:2097`), `code_timestamp` (`:421`) — **IGNORED**.

So the baseline layer auto-invalidates on: dataset/rep/model/metric/folds/seed
name changes (also in the filename, `:111-113`), rep dimensionality (`rep_shape`),
n_samples, label values, group values, and a bump of either high-dim-filter
constant. It does NOT auto-invalidate on a same-shape rep VALUE change
(`rep_hash` ignored).

### Folds — `_make_folds_meta` (`find_best_hybrid.py:425-442`), load at `:653-662`

Validated keys: `kind`='folds', `dataset_name`, `n_folds`, `seed`,
`n_samples` (`:435`), `groups_hash` (`:436-437`), `cv_splitter_id` (`:440` —
distinguishes default KFold/GroupKFold from a custom splitter), `code_timestamp`
(`:441`) — **IGNORED**. Folds depend only on (n_samples, groups, splitter, seed,
n_folds); none of those involve rep values, so this layer has no silent-rep
exposure. Note `_load_or_compute_folds` short-circuits to an uncached
`splitter.split(...)` when `dataset_name` is falsy (`:650-651`).

### Hybrid_eval — `_make_hybrid_eval_meta` (`find_best_hybrid.py:552-592`), load at `:2417-2419`

This is the one validated layer where rep VALUES feed identity even though the
hash is in `ignore_keys`:
- The **filename** embeds `subset_hash` = `_subset_hash_str(_subset_hashes)`
  (`:2388`,`:467`,`:595-601`), and `_subset_hashes` is built from
  `_hash_array(base_reps[name])` (`:2385-2387`) — i.e. the rep CONTENT hash.
- The meta also stores `subset_hashes` (`:567`), which is NOT in `ignore_keys`,
  so it IS validated on load.
- `aug_hashes` (`:581`) IS in `ignore_keys` → ignored.
Other validated keys: `subset_names` (`:566`), `alloc`/`imp`/`evaluator`/`budget`
(`:568-571`), `fold_idx`/`n_folds`/`seed`/`model_name` (`:572-575`),
`train_idx_hash`/`val_idx_hash` (`:576-577` — fold-split identity), `labels_hash`
(`:578`), `aug_strategy`/`aug_names` (`:579-580`), `step_size`,
`candidate_ranking`, `greedy_tolerance`, `augmentation_threshold`,
`augmentation_budget` (`:582-587`), `aug_logic_version` (`:588` =
`AUG_LOGIC_VERSION`, constant `=2` at `:162`), `metric_name`, `task_type`
(`:589-590`), `code_timestamp` (`:591`) — **IGNORED**.

Important consequence: because the hybrid_eval **filename and `subset_hashes`
both come from the rep content hash**, a same-shape rep VALUE change DOES rotate
the hybrid_eval cache to a new file (old file is simply never looked up again).
So hybrid_eval entries do not serve stale rep values — but they also do not get
cleaned up, and only apply to top_k allocations (`_predict_top_k_subset` returns
None otherwise, `:2378-2384`,`:511-518`). This makes hybrid_eval the EXCEPTION to
the silent-staleness rule that bites the search and baseline layers.

---

## 3. Representation cache `_cache_rep` / `_cache_rep_dict` (`_shared.py`)

This is the layer with the weakest contract.

- `_cache_rep` (`_shared.py:168-191`): the entire validation is
  `if cache_file.exists(): arr = np.load(cache_file); return arr` (`:173-176`).
  No header, no shape check, no hash, no n_samples check. **Existence is the only
  test.** Writes are atomic (tmp + `os.replace`, `:180-183`).
- `_cache_rep_dict` (`_shared.py:194-246`): same — for each wanted name,
  `if cache_file.exists(): reps[rn] = np.load(...)` (`:216-218`); only missing
  names trigger `compute_all_fn()` (`:222-227`). Filename is `<dataset>/<rep>.npy`
  (`:215`,`:230`). No content validation.

So the `.npy` rep cache, on its own, will silently reuse ANY file that happens to
sit at `<dataset>/<rep>.npy`, regardless of its contents, dimensionality, or row
count.

### The only guard on the rep cache: `_finalize_load_data` row-count check (`_shared.py:2054-2075`)

After load, every public `load_data` call (`:2080-2094`) runs
`_finalize_load_data`, which checks:
```python
_bad = {rn: int(arr.shape[0]) for rn, arr in _reps.items()
        if hasattr(arr, 'shape') and arr.ndim >= 1
        and arr.shape[0] != _n_labels}        # :2065-2067
if _bad:
    raise ValueError(... "stale rep cache from an older loader version"
                     ... "python tests/purge_rep_cache.py {dataset_name} --delete")
```
It checks ONLY `arr.shape[0] != len(labels)` — i.e. **row count vs label count**.
It does NOT check:
- the number of columns / dimensionality (`shape[1]`),
- the actual values in the array,
- any hash or version stamp.

It raises (non-destructive) naming the offending rep and the purge command;
`tests/purge_rep_cache.py` exists (confirmed). The check runs BEFORE
`_smoke_subsample` so it sees full pre-subsample arrays (`:2060`,`:2077`).

Net: the row-count guard catches the classic "old loader emitted a different
number of rows" misalignment, but a rep whose row count is unchanged and whose
columns are unchanged but whose VALUES changed will pass this guard silently.

---

## 4. Conclusion — what auto-invalidates vs. what silently persists

### A. AUTO-INVALIDATES (safe — cache self-heals, NO manual purge needed)

A change in any of these is detected; the affected layer recomputes on next run.

1. **Label / target value changes** — `labels_hash` in search (`:360`), baseline
   (`:414`), hybrid_eval (`:578`); plus `n_samples` everywhere.
2. **Sample count (n) changes** — `n_samples` in search (`:349`), baseline
   (`:413`), folds (`:435`); also the rep-cache row-count guard
   (`_shared.py:2065`).
3. **Group / split-membership changes** — `groups_hash` in search (`:361`),
   baseline (`:415`), folds (`:436`).
4. **Representation DIMENSIONALITY changes (column count)** — `rep_shapes` in
   search (`:351-352`), `rep_shape` in baseline (`:407`). A rep whose width
   changes is caught.
5. **Representation SET changes (adding/removing/renaming a rep)** — `rep_names`
   (search, `:350`); rep/model/metric/folds/seed are in the baseline FILENAME
   (`:111-113`) and search FILENAME (`:95-98`).
6. **Model / metric / task / folds / seed / tier / speed / aug_strategy /
   condition_suffix changes** — all in meta and/or filename (`:344-358`,`:4223`).
7. **Search-algorithm config changes** — allocation/importance/evaluator lists,
   pruning, candidate_ranking, two_phase, refinement, dimension_tiers,
   weight_method, step_size, thresholds, cv_splitter_id, rep_factories_id
   (`:366-392`).
8. **Augmentation NAME / strategy / threshold / budget changes** — `aug_names`,
   `augmentation_strategy`, `augmentation_threshold`, `augmentation_budget`
   (`:363`,`:365`,`:373-375`); plus the `AUG_LOGIC_VERSION` bump for hybrid_eval
   (`:162`,`:588`) and the no-aug strategy normalization to `'none'` (CACHE-2,
   `:4035-4040`).
9. **High-dim-filter policy changes** — `hd_filter_version` /
   `hd_filter_threshold` bumps invalidate baseline scores (`:419-420`).
10. **Hybrid_eval rep-value changes (top_k allocations only)** — because the
    hybrid_eval FILENAME and `subset_hashes` are derived from the rep content
    hash (`:467`,`:567`,`:2385-2388`), a value change rotates to a fresh file
    (old never reused). This is the lone layer where rep values DO affect identity.

### B. SILENTLY PERSISTS (UNSAFE — stale until MANUALLY purged)

The single dominant case, by deliberate design (the CACHE-1 fix):

1. **A representation's VALUES change while its shape AND the labels AND n_samples
   AND group membership all stay the same.**
   - SEARCH layer: stale. `rep_hashes` is in `ignore_keys` (`:198-199`); the only
     value-sensitive keys are `labels_hash`/`groups_hash`, which don't move.
     Result: an old `searches/*.pkl` is served with the new reps' identity but
     old reps' scores.
   - BASELINE layer: stale. `rep_hash` is in `ignore_keys`; same reasoning
     (`:408`,`:2675`).
   - REP_CACHE (`.npy`) layer: stale. `_cache_rep`/`_cache_rep_dict` validate
     nothing but existence (`_shared.py:173-176`,`:216-218`); the only post-load
     guard is row count (`:2065`), which a same-shape value change passes.
   - hybrid_eval layer is the exception (rotates by content hash, §A.10).

   Concrete triggers for this class (all produce same shape, same labels):
   - A loader/featurizer code change that alters how a rep is computed but keeps
     its width (e.g. a bug fix in a descriptor, a normalization tweak, a corrected
     PDB-sequence lookup) — e.g. the Graphinity case the project's own
     `tests/purge_graphinity_cache.py` exists to handle.
   - A non-reproducible neural/transformer rep that the design INTENTIONALLY does
     not re-validate (ESM/prot_t5/geneformer/DNA-LMs/chembert), to stop churn.
     This is accepted run-to-run noise, not corruption — but a genuine
     model/weights/version swap that keeps the embedding width is also NOT caught.
   - A corrupted or partially-written `.npy` that still loads at the right shape.

2. **`code_timestamp` is always ignored** (`:198`) — pure provenance, never a
   correctness signal. Listed only to be explicit: a code change by itself does
   NOT invalidate any cache.

### The rule the "malformed" determination depends on

A cached entry is at risk of being silently stale **iff** the representation's
numeric values changed but its **shape, the labels, n_samples, and group
membership are all unchanged**. Such an entry is invisible to the search and
baseline validators (rep hashes ignored) and to the rep-cache layer (existence-
only + row-count-only guard). It must be removed by hand:
`tests/purge_rep_cache.py <dataset> --delete` for the `.npy` layer, and the
`searches/` + `baselines/` files for the validated layers (purge_graphinity_cache.py
is the worked precedent). Any change that moves shape, labels, n, or groups is
self-healing and needs no purge.

### Uncertainties (stated, not guessed)

- I did not exhaustively audit `ft_embeddings` (`_make_ft_embedding_meta`,
  `:604-625`) beyond noting it keys on `factory_id` (caller-supplied, carries
  version per the docstring `:478-479`) plus fold-index hashes + `labels_hash` +
  `groups_hash`; its rep-value sensitivity therefore rides entirely on whether
  the caller bumps `factory_id`. Not central to the rep-staleness question but
  flagged for completeness.
- The `_hybrid_fold_cache_*` helpers in `_shared.py:47-72` (a separate
  experiment-side hybrid-fold cache) were not deep-audited here; the prompt
  scoped the rep cache, which is covered in §3.
