# Cache audit — 2026-09-10

**Scope.** `find_best_hybrid.py`, `hybrid.py`, `result_cache.py`, `screen.py`, and the five
experiment families: complexity, noise, similarity, similarity_metrics, federated, OPIG.
`tests/experiments/framework/` and `annitia/` were not audited.

**How this was checked.** Twelve readers over the cache layers and the experiment families,
then five runs that executed code, then one skeptic per finding told to refute it. 63 distinct
findings; 43 were sent to verification and every one came back with a verdict. Every number
below was read in the code or printed by a run in this session. Nothing here is taken from a
memory file or from another document.

**What ran.** A real suite experiment ran end to end: n02 on `humanness_therapeutic`, 1321
seconds cold, 614 seconds warm, byte-identical output both times. Everything else was
synthetic sets of 90 to 120 molecules.

---

## 1. Do this first

**Check the commit, not the date.** For most of this audit the 2026-09-10 engine changes were
uncommitted working-tree edits — `objective` as an identity column, the NaN encoding for an
unscored representation, and `create_hybrid`'s `performance_weighted` default. They were
committed during the session, in `7546bae` "The working tree as it stood" and `147293c`. Every
fix in §3 went in with them. Verified at HEAD: `result_cache.py` carries `'objective',` in
`IDENTITY_COLS`, `hybrid.py` says `allocation_method='performance_weighted'`, the wipe script
lists `fold_scores`, and `setup.sh` writes `${VIRTUAL_ENV:-}`.

**So the check the server needs is the hash, not the date.** A date check would have passed
throughout the window when none of this was committed, and the wipe would then have run under
code that still writes rows with no objective column. Step 0 below prints the hash and greps
for the column.

Two files still carry uncommitted changes, neither of them mine and neither cache-related:
`find_best_hybrid.py` (4 lines, an error message) and `hybrid.py` (48 lines, an
`apply_feature_selection` signature change). A parallel session is working on them. Push or
stash them before the server pulls, or the server runs a different `apply_feature_selection`
from this laptop.

---

## 2. The server wipe, in order

Run from the repo root.

```bash
# 0 — get the code the wipe is for. Check the HASH, not the date.
cd /path/to/KIRBy && git checkout similarity-metrics-study && git pull
git log -1 --format='%H'
grep -n "'objective'," src/kirby/result_cache.py   # must print a line

# 1 — point at the real cache and prove it before deleting anything
. ./setup.sh
echo "$KIRBY_CACHE_DIR"          # must be /data/stat-ecr/scat9264/.cache/kirby
ls -d "$KIRBY_CACHE_DIR"/baselines "$KIRBY_CACHE_DIR"/rep_cache

# 2 — inventory, so step 5 has something to compare against
ls -1 "$KIRBY_CACHE_DIR"
ls -1 "$KIRBY_CACHE_DIR"/result_db 2>/dev/null | wc -l
ls -1 "$KIRBY_CACHE_DIR"/fold_scores 2>/dev/null | wc -l
find "$KIRBY_CACHE_DIR"/rep_cache -name '*.npy' | wc -l   # must NOT change

# 3 — wipe, on a compute node. fold_scores is now in the target list.
sbatch tests/slurm_scripts/wipe_allocation_caches.sh

# 4 — the experiment JSONs survive a cache wipe. Force past the resume.
export KIRBY_FORCE_RERUN=1
python tests/check_precompute_status.py    # expect "no result cache" or all-missing
bash tests/slurm_scripts/submit_all_datasets.sh <category>

# 5 — verify. No database may survive with an old column list.
python - <<'PY'
import os, glob, sqlite3, sys
sys.path.insert(0, 'src')
from kirby.result_cache import ALL_COLS
root = os.environ['KIRBY_CACHE_DIR']
for p in sorted(glob.glob(os.path.join(root, 'result_db', '*.db'))):
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    have = {r[1] for r in c.execute('PRAGMA table_info(results)')}
    miss = [x for x in ALL_COLS if x not in have]
    n = 0 if miss else c.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    print(('STALE ' if miss else 'ok    ') + p + f'  rows={n} missing={miss}')
    c.close()
PY
```

### Before step 3: one command, and paste me what it prints

I cannot reach ARC (`REPRODUCIBILITY_LEDGER.md:451`). This reads and deletes nothing, and it
answers every server question this audit left open.

```bash
cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh
{
echo "== commit =="; git log -1 --format='%H %s'
echo "== the 09-10 changes are present? (3 lines, all must print) =="
grep -n "'objective'," src/kirby/result_cache.py
grep -n "allocation_method='performance_weighted'," src/kirby/hybrid.py | head -1
grep -n "fold_scores" tests/slurm_scripts/wipe_allocation_caches.sh | head -1
echo "== cache root =="; echo "$KIRBY_CACHE_DIR"; ls -1 "$KIRBY_CACHE_DIR"
echo "== which layers exist, and how big =="
for d in result_db fold_scores baselines hybrid_fold_cache hybrid_evals searches folds rep_cache; do
  printf '%-20s %s\n' "$d" "$(ls -1 "$KIRBY_CACHE_DIR/$d" 2>/dev/null | wc -l)"
done
echo "== do the libraries import on a COMPUTE node? (this is 4.4) =="
srun -p short -n1 -t2 python -c "import torch, lightgbm, xgboost; print('all import')" 2>&1 | tail -2
echo "== every result_db, and whether it can be opened =="
python - <<'PY'
import os, glob, sqlite3, sys
sys.path.insert(0,'src')
from kirby.result_cache import ALL_COLS
for p in sorted(glob.glob(os.path.join(os.environ['KIRBY_CACHE_DIR'],'result_db','*.db'))):
    c=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
    have={r[1] for r in c.execute('PRAGMA table_info(results)')}
    miss=[x for x in ALL_COLS if x not in have]
    if miss:
        print('STALE', os.path.basename(p), 'missing', miss); c.close(); continue
    n=c.execute('select count(*) from results').fetchone()[0]
    # 4.1: was any one search written by two different representation pools?
    bad=c.execute('''SELECT COUNT(*) FROM (SELECT 1 FROM results
        GROUP BY tier,objective,metric,task_type,labels_hash,groups_hash,
                 n_folds,seed,cv_splitter_id
        HAVING COUNT(DISTINCT rep_pool)>1)''').fetchone()[0]
    # 4.7: any row from an allocation that reaches the per-fold layer
    topk=c.execute("select count(*) from results where allocation_method like 'top!_%' escape '!'").fetchone()[0]
    print('ok   ', os.path.basename(p), 'rows',n, 'mixed_pool_searches',bad, 'top_k_rows',topk)
    c.close()
PY
} 2>&1 | tee /tmp/kirby_cache_check.txt
```

**Run against the two local caches, so you know what a clean answer looks like.** Every
database on `/Volumes/seagate/kirby_cache/result_db` came back `ok`, `mixed_pool_searches 0`,
`top_k_rows 0`: caco2 990 rows, humanness_therapeutic 2640 rows in each of its two outer-fold
files, tcga_brca_pam50 995 rows. `~/.cache/kirby/result_db` holds 14 stale-schema files, all
already moved aside into `stale_schema_2026-09-10/`. So neither local cache carries evidence of
§4.1 or §4.7 damage. The server is unchecked.

What each line decides. `mixed_pool_searches` above 0 on any file is proof of §4.1 on the
server, not a risk — those rows were written by two different representation pools and the
second pool's numbers are the first pool's. `top_k_rows` above 0 means §4.7's pool collision
could have touched that dataset. A `srun` line that does not say `all import` names which of
§4.4's substitutions actually happened. `STALE` on a file is expected — the wipe removes it.

---

**Do not run either `invalidate_*.sh`** until you have read §4 on them. They are now guarded,
but the two scripts overlap with the wipe script and disagree with it about which layers matter.

### What each recent change does to the cache, and whether you have to act

| change | mechanism | do you have to delete anything |
|---|---|---|
| `objective` in the three metas (2026-09-10) | every `baselines/` and `hybrid_evals/` file lacking the key is a stale miss and rewrites itself | no — measured: `meta drift on 'objective' (stored=None, expected='predictive')`, 3 stale, 0 hit |
| `objective` as an identity column | `_ensure_schema` refuses the file and names it | **yes** — every `result_db/*.db` |
| `metric` + `task_type` as identity columns (2026-09-04) | same refusal | **yes** — same files |
| model tiers closed | `model` is a per-row config column; an old catboost row is simply never selected | no |
| budget cap 500 → 1000 | `budget` is a per-row config column; 1000 is a new row | no |
| inner split GroupShuffleSplit → GroupKFold | in no cache key at any layer | **yes** — `result_db/`, `hybrid_fold_cache/`, and `fold_scores/` |
| unscored representation 0.0 → NaN | in no key and in no meta | **yes** — `fold_scores/` |

**`baselines/` and `rep_cache/` stay.** That is most of the compute and none of it is wrong.
The wipe script aborts if it cannot see both, which is the guard against a mispointed
`KIRBY_CACHE_DIR`.

**One change is still coming.** f32's per-metric normalisation rule is decided and not built
(FRAMEWORK_FINALIZATION §5). It lands at `_sanitize_baseline_scores` and re-runs every stored
search on the 22 datasets reporting rmse, auc, ci or mae. Wiping now and building it later
means wiping twice.

---

## 3. Fixed

The cache held no searched scores when these went in, so nothing was lost and no cost
argument applied to any of them. Seven were open decisions until 2026-09-10; the owner closed
six by saying the stored-work cost was the only thing keeping them open, and chose "refuse" on
the seventh.

### Closed by decision, then implemented

**Which representations were searched is now part of what a stored score is filed under.**
`rep_pool` moved from recorded data into the matched columns, keyed on sorted (name, feature
count) rather than name alone, and captured after any screening so a screened run keys on what
it actually searched. Two searches over the same rows, folds, seed, metric and tier — the
first over two representations, the second over four — now score 1.5261 and 1.7258 rmse and
store two rows. Before, the second read the first's score, its width and its budget split, and
fitted nothing.

**Whether models were tuned is now part of it too.** Stored as `''` when off, so an untuned run
writes what it always wrote. `find_best_hybrid.py`'s dead `row['tune']` assignment — which
`upsert_rows` silently discarded, because no such column existed — is gone.

**Two estimators can no longer share one name.** An estimator that is what the registry builds
for its name keeps the bare name, so every `models=['rf', ...]` caller is byte-identical. Any
other estimator gains a short settings fingerprint from its class and `get_params`. Tested: a
random forest and a Ridge both handed in as `setting4` now score 0.5113 and 2.0611 rmse and
write four separate per-representation score files; before, the second read the first's. The
name a result *reports* is unchanged — `best_by_model_budget` still keys on `setting4`.

**The similarity precompute stops discarding its model list.** `models = None` was assigned
unconditionally, throwing away both the argument and `config['models']`. The signature default
is already None, so a caller passing nothing is unaffected; a caller passing six models now
gets six. kNN starts getting a recipe, and the fold-0 allocation vote is taken over the same
model list the cluster precompute uses.

**A recipe pruned at fold 0 now leaves a row.** It is filed under a `rep_subset` no search asks
for, so it can never be read back as a finished result — it holds one fold, and a one-fold mean
must not sit beside a five-fold one. The fold-0 vote reads it deliberately. Measured on a
five-allocation search: cold 12.0 s, warm 0.0 s, and the warm run printed
`fold 0: 5/5 read from the result database instead of re-fitted`. Before the fix the warm run
was 9.4 s of a 16.4 s cold run and re-pruned the same two recipes every time.

**Two jobs writing one score store now wait instead of dying.** `sqlite3.connect` got no
timeout, so the wait was the library default of 5 seconds and neither the open nor the write was
wrapped — a whole search died inside `_ensure_schema` after paying for its per-representation
scores and its per-fold precompute. Now 120 seconds. Uncontended behaviour is unchanged and was
already clean: three concurrent writers, 360 of 360 rows, integrity ok.

**A missing library now stops the run.** Owner decision: refuse rather than substitute. Ask for
lightgbm on a machine without it and you got xgboost, or a random forest, written to a file
named lightgbm with the requested name in its metadata — undetectable afterwards. The four
library branches (xgboost, lightgbm, catboost, ngboost) now raise `EvaluatorLibraryMissing`
naming the module and what to do. For dnn, cnn and bnn the import is separated from the fit: a
missing torch raises, a fit that fails on one fold's rows still falls back and still says so out
loud, because that is a property of those rows and not of the machine. Tested: with lightgbm
made unimportable the call raises; `rf` is unaffected.

### Small, applied without asking

- `wipe_allocation_caches.sh` — `fold_scores` added to the delete list, with a note saying what
  the layer holds and why it cannot detect its own staleness. It was in neither list.
- `wipe_allocation_caches.sh` — `set -u` moved to after `. ./setup.sh`. On a machine where the
  venv is unreadable, `setup.sh` expanded `$VIRTUAL_ENV` bare and the job exited before reading
  `KIRBY_CACHE_DIR`, deleting nothing and logging one line that reads like a failed delete.
  `setup.sh` now writes `${VIRTUAL_ENV:-}` at both places.
- `invalidate_allocation_caches.sh`, `invalidate_refolded.sh` — both fell back to
  `$HOME/.cache/kirby` when `KIRBY_CACHE_DIR` was unset, and that directory exists on the login
  node, so they would delete a near-empty home directory and report success while the real store
  survived. Both now source `setup.sh` (with `KIRBY_SKIP_INSTALLS=1`), abort on an empty root,
  and abort unless `baselines/` and `rep_cache/` are both present. Tested both ways.
- `_shared.py` — `create_hybrid_cached`'s internal evaluator default was `svm` where
  `create_hybrid`'s is `lightgbm`. Now `lightgbm`.
- `_shared.py` — `create_hybrid_cached` now **refuses** any `create_hybrid` parameter its cache
  key cannot represent, instead of forwarding it. See §5 for the eighteen collisions this was
  hiding. `augmentation_margin` and `augmentation_fraction` are now passed into the key.
- `find_best_hybrid.py` — the per-representation score metadata gained `struct_pairs_id` and
  `tune_knn_k`, both written only when the feature is on, so nothing already stored is
  invalidated.
- `_run_precompute_handoff.py`, `_smoke_noisy_search.py` — both printed a row count off a score
  store that `ResultCache` refuses to open. Both now check the columns first.

**Checked after all of it:** 21 unit tests pass. An ordinary two-model, two-budget search
reports `['lightgbm', 'rf']`, stores those names bare, and reports both budgets.

---

## 4. Decided 2026-09-11 — both implemented

### 4.1 c10 picked its recipe on the true activity values

**The fault.** c10 runs two arms. One is a hybrid, built from a recipe. The other is each
representation on its own, with no recipe. The recipe came from a search that saw the true
activity value of every molecule. c10 then corrupts those values and asks how far each arm
falls. One arm had a look at the answers the rest of the experiment hides; the other did not.

**Decided: measure it at one level per strategy, on fast tier.** 11 searches per dataset rather
than the 56 it takes to redo every strategy at every level.

**DONE.** A new phase runs one search per noise strategy, at that strategy's strongest
corruption, with `speed='fast'` — 12 recipes per model and budget instead of 75. Per strategy
and budget it records the recipe chosen on true activity values, the recipe chosen on corrupted
ones, whether the two differ, and what the corrupted-label recipe scores. It adds a field and
moves nothing c10 already reports; the clean-recipe score at the same level is already in
`hybrids[budget][strategy]['scores_by_level']`.

### 4.2 The per-fold score store

**The fault, as two things that disagree.** It turned on only for an allocation method whose
name begins `top_`. The five methods a search uses are `performance_weighted`, `power_2`,
`softmax_T1`, `rank_exp_d0.3` and `wglobal_power2`. Forced on with `top_2` it still stored
nothing: it guesses which representations the allocation will pick, then refuses to store unless
the guess was right, and on every fold measured it guessed two and the allocation picked three.

**Decided: leave it off.** The one thing it would have bought is that a job killed by its time
limit keeps the folds it finished. The owner's call is that the suite's searches do not die that
way often enough to pay for three fixes.

**DONE.** `_hybrid_cache_enabled = False`, with both disagreements written above it. The path
and metadata builders stay, unused, so reviving it is one flag rather than a rewrite. Checked
that a `top_2` search still runs normally with it off.

---

## 5. Fixed, but you should know what it was hiding

`create_hybrid_cached` forwarded parameters to `create_hybrid` through `**kwargs` and hashed
three of them. **Eighteen parameters changed the selected features and left the cache key
byte-identical.** Demonstrated one at a time, comparing the bare call against the cached one:

| parameter | what the two bare calls did | through the cached door |
|---|---|---|
| `augmentation_margin` | 0.0 kept all 20 extra features; 5.0 kept none | one file, the 5.0 call got all 20 |
| `augmentation_threshold` | None kept all 20; 10.0 kept none | one file, the 10.0 call got all 20 |
| `n_per_rep` | None gave 20 features; 3 gave 9 | one file, the second got a 20-feature matrix |
| `objective` | prediction and geometry shared only 7 of ~20 features | one file, geometry got prediction's |
| `metric_fn` | MAE and MSE differed in the base features too, 16 vs 15 from repA | one file |
| `lower_is_better` | flipping it flipped which extras helped | one file |
| `model`, `validation_split`, `candidate_ranking`, `augmentation_fraction`, `augmentation_budget`, `aug_max_candidates`, `aug_per_source_budget`, `precomputed_importance`, `precomputed_alloc_scores` | all changed the selection | all one file |
| representation **values** at the same names and widths | completely different selection | one file |
| donor **values** at the same names and widths | 20 extras vs 5 | one file |

It is latent — `c09_formal_analysis.py:366` and `:545` are the only callers and pass none of
them. The function now raises on any parameter it cannot key, so it cannot come back.

**The two content rows are the locked decision, not a defect.** Representation values are
deliberately not in any key. What follows from that is the wipe rule in 6.3.

---

## 6. Dead, orphaned, or a rule rather than a fix

### 6.1 Two cache layers do nothing

- **`searches/*.pkl` is never written.** `_search_cache_path` is reached once from
  `_find_best_hybrid_inner` (`find_best_hybrid.py:7517`) and the path it returns is only used to
  derive the checkpoint name.
  Nothing saves to it. The search pickle was replaced by `result_db`. Consequence worth knowing:
  the one guard that keys on `rep_names` and `rep_shapes` is `_make_search_meta`, which now
  survives only as checkpoint metadata — so it protects a fold-0 recompute and nothing durable.
  That is why 4.1 is possible.
- **`folds/` is written only by `_precompute_fold_reps`**, which returns immediately unless
  `rep_factories` is passed. Five other places call `splitter.split()` directly. No answer is
  wrong: with groups present the splitter is `GroupKFold`, which has no shuffle, and three
  separate processes gave the same fold hash `9f99bd251eaaa3b4`. The layer's stated purpose is
  "a determinism guarantee more than a speed gain", and that guarantee is currently carried by
  `GroupKFold` rather than by the cache.

Three of the wipe script's seven targets are directories that no longer exist.

### 6.2 The stale-schema refusal costs a full precompute before it fires

`ResultCache` opens at `find_best_hybrid.py:5890`. The baselines print at `:4751` and the
importance precompute at `:5535`. On 120 synthetic rows the refusal came 22.3 seconds in, having
already written 3 baseline files and 6 fold-score files. On a real dataset with ten
representations that is the whole precompute. Nothing is stored wrongly and no result is
corrupted — the cost is cluster hours. Opening `ResultCache` once at the top of
`_find_best_hybrid_inner`, where `result_db_identity` is already built at `:7510`, would fix it.

### 6.3 The wipe rule for representations, which no code change can replace

`rep_cache/<dataset>/<rep>.npy` is keyed on the dataset name and the representation name. The only
guard is a row-count check, which passes when a loader change swaps one molecule for another. That
is the locked decision working as designed. What is not designed is that the remediation does not
work:

- `purge_rep_cache.py` deletes `rep_cache/<ds>/` and not `reps/`, so on any molecular, nucleotide
  or TCGA dataset the features survive the purge.
- The guard's own message (`_shared.py:3254`, `preflight_cache_check.py:95`) says `--delete` with
  no `--all-caches`, which removes the features and leaves `baselines/`, `hybrid_fold_cache/` and
  `result_db/` holding numbers computed from them.
- `result_db` is not in `pkl_dirs` at `purge_rep_cache.py:108`.

**The rule:** change a loader, a filter, a row order or a featuriser version, and
`rep_cache/<ds>/` plus `reps/` plus every derived layer for that dataset is deleted, on the laptop
and on the server. `fold_scores/` is the one exception — its key is a content hash of the fold's
matrices, so changed features give it a different key.

**Whether `result_db` joins `--all-caches` is your call.** Its rows are keyed on `labels_hash`, so
a molecule-list change already gives them a new key; a featuriser change does not. Adding it
throws away every searched configuration for that dataset, which is the expensive layer. **>>**

### 6.4 Fix 7 landed in three of four places

`FRAMEWORK_FINALIZATION` fix 7 says a top_k hybrid with augmentation no longer loses its fold.
`_shared.py:4312` and two places in `find_best_hybrid.py` read `feature_info['demoted_reps']` back
to rebuild the held-out side. The search's own per-fold loop does not: at
`find_best_hybrid.py:4385` it raises
`KeyError: augmentation 'c' has no validation-side source` — 3 of 3 folds with a predictive donor
pool, 1 of 3 with a noise pool, and the configuration reported FAILED with 0 result rows. The same
augmentations under `performance_weighted` returned a row both times. Only `top_*` plus an
augmentation dict reaches it, so nothing in the suite hits it today.

### 6.5 Two `find_best_hybrid` calls in threads do not keep their caches apart

`_CACHE_ROOT_OVERRIDE` (`find_best_hybrid.py:110`) is a module global. In a two-thread test one
root held **both** datasets' baselines and fold scores, and after both calls returned the global
still pointed at the first thread's temporary directory, so every later call in that process would
have used it. Separate processes and separate SLURM tasks are safe — each has its own module
state. The filenames carry the dataset name, so the files are in the wrong place rather than
holding the wrong content. A `contextvars.ContextVar` fixes it; `threading.local` would not.

### 6.6 An augmentation pool is identified by its names only

`aug_hashes` is in `_load_cache_with_meta`'s ignore list. Stored under one hash and read back
under another, a `hybrid_evals` entry returned `{'score': 0.111}` and counted as a hit. The
exclusion is right for representations — neural embeddings are not bit-reproducible — but a donor
block is usually a stored matrix. Unreachable today, since no current search arms that layer.

---

## 7. Checked and correct — do not re-audit these

Each was tested by running it, not by reading it.

- **`cache_dir=` reroots every layer.** With `KIRBY_CACHE_DIR` pointing at root A and
  `cache_dir=` at root B, root A stayed empty and root B held `baselines/`, `fold_scores/` and
  `result_db/`. The module global is restored after the call. (Single-threaded — see 6.5.)
- **The identity separates what it claims to.** Each of tier, metric, task_type, objective, seed,
  n_folds and a changed label vector added exactly one distinct identity tuple: 0→1→2→…→8.
- **Widening the budget list reuses the stored rows.** `budgets=[20]` then `[20,40]` printed
  `1 stored configs reused; computing only the remainder of 2`, and the budget-20 row's
  `created_at` did not move.
- **Fix 4 holds.** With budgets 20 and 40 stored, a run asking for `[20]` printed
  `1 further stored rows are other configurations and are not reported here` and did not rank them.
- **A tuned run does not read untuned baselines.** `__tuned` in the filename; both files coexist.
- **A different objective adds `__obj-` to the baseline filename**, and an old-format file is
  rejected: `meta drift on 'objective' (stored=None, expected='predictive')`, 3 stale, 0 hit.
- **`fold_scores/` is read back correctly and incrementally.** An identical rerun printed
  `3/3 successful, 3 read from disk`; adding one internal evaluator cost only that evaluator. The
  stored payload holds all 13 metrics, so reuse across a metric change is correct, not a collision.
- **The speed top-up works.** `speed='fast'` stored 6 rows; `speed='full'` over the same cache
  reached 45 and reused the 6. The 2026-09-09 claim is confirmed.
- **The result-database schema guard raises and names the file**, for a missing `objective` and
  for a missing `metric`/`task_type`.
- **The resume fingerprint catches drift.** Editing the stored `n_splits` printed
  `RERUN n02: existing output n_splits changed: 5 -> 2`; deleting the key printed
  `RERUN n02: the file records nothing`.
- **`--force-rerun` overrides the skip**, and `KIRBY_FORCE_RERUN=1` reaches a SLURM job through
  `submit_dataset.sh:77`.
- **After a cache wipe the experiment JSONs still skip.** Deleting `result_db/` made the five
  search configurations recompute and n02 still resumed. Only deleting the output JSON, or
  `--force-rerun`, forces the experiment itself.
- **Fold splits are stable without the folds cache.** Three processes, same fold hash; run 1 and
  run 2 of n02 were byte-identical.
- **Concurrent uncontended writers to one database are clean.** 360 of 360 rows, integrity ok.

---

## 8. The twelve smaller items — fixed 2026-09-10

All verified, all cheap, none needing a decision. The cache held no searched scores, so the
three that were only open because fixing them meant a deletion cost nothing.

**In the engine.**

- **A blend was writing its own name into the metric column**, so the dataset's rmse or spearman
  dropped out of what a score is filed under and two blends on one dataset asking for different
  metrics recorded the same thing. That workaround predates `objective` having a column; it is
  now redundant and it cost the metric. Tested: two blends now store `('rmse', blend…)` and
  `('spearman', blend…)`.
- **`rep_factories_id` is now a matched column.** It is a required argument when caching is on
  and reached nothing, so bumping it — the documented way to invalidate a changed per-fold
  representation builder — moved the fine-tuned embeddings and left every stored score readable.
- **A noisy search now records the corrupted label column, not just the condition's name.**
  `spec_id` says the strategy, the level and the seeds; it says nothing about the numbers those
  produced, so changing the injector under an unchanged name served the old scores. An eight-hex
  hash of the noisy labels rides on the noise id and on the tier. Tested both ways: two identical
  noisy runs write one tier and one row, and a changed injector under the same name writes two.
- **`AUG_LOGIC_VERSION`'s comment no longer claims "no deletion needed".** It reaches the
  per-fold metadata and nothing else; a finished configuration's score is a row that this
  constant is in no column of, so bumping it needs the result databases deleted with it.

**In the experiment layer.**

- **The in-process fold-search memo keyed on the training row *count*.** Two outer folds of one
  dataset are the same size by construction, so the second was handed the first's search. It
  keys on a hash of the rows now, and on the metric and task type, which were missing too.
  Representations are keyed by name and width rather than name alone.
- **The resume stamp records the metric, the task type and the noise grid.** Without them a
  dataset whose metric moves, or a noise sweep whose levels move, keeps every finished result
  file and the resume sees nothing changed. Tested: `metric changed: 'r2' -> 'rmse'` and
  `noise_grid changed: …['0','0.2'] -> …['0','0.5']`.
- **The transformer embedding file name carries the model id and the molecule list.** `which` is
  the alias `chemberta` or `molformer`; which checkpoint that means is an environment variable,
  so a second run under a different checkpoint scored the first one's vectors. The reader accepts
  the old name as a fallback, so nothing on disk is orphaned.
- **The CDR-string cache is keyed on the sequences it was built from.** It holds one string per
  antibody, in order, so it is valid only for that exact list — and a run on a filtered set read
  the older, longer list and paired the first N against its own N labels.
- **Three of the nine Ginkgo properties were unregistered** — `ginkgo_tm2`, `ginkgo_purity`,
  `ginkgo_aggregation` — so the metric lookup raised, the `except Exception` around the search
  swallowed it, and the run wrote a results file with no hybrid rows for those three. Registered
  as regression/spearman/antibody, matching the six already there and the loader they share.
  `n_samples` left unset rather than guessed. All nine now resolve.
- **fed01's centralised arm builds its hybrid with scaffold groups.** It was the only
  scaffold-blind `create_hybrid` call in the file — the other four pass them — and it is the arm
  every silo arm is measured against.
- **fed05 passes the metric with the model.** Without it each representation's share of the
  budget was decided by the internal evaluator's R² rather than by spearman, which is what the
  three antibody silos resolve to and what survives their per-silo z-scoring.
- **n09 records which labels chose each fold's recipe.** When the noisy search scores no budget
  it falls back to the clean-label recipe, which is reasonable and is a different measurement;
  nothing said which had happened. Each fold now appends `noisy_search` or
  `clean_pool_fallback` to `recipe_sources`, and the fallback prints. The `budget` field is the
  clean pool's, which it always was — now also stored as `clean_pool_budget` and said so.

**Checked after all twelve:** every touched file parses, 21 unit tests pass, and an ordinary
two-model two-budget search runs cold in 8.2 s, warm in 0.0 s, with identical answers, reporting
`['lightgbm', 'rf']` and storing those names bare.

---

## 9. The seven that were work rather than decisions — done 2026-09-11

- **Three rebuild doors disagreed about the recipe.** `build_fold_hybrid` forwarded seven
  extra-feature settings to a rebuild; `get_fitted_pool_hybrid` and `get_fitted_hybrid`
  forwarded none, so both took `create_hybrid`'s defaults whatever the search ran under. Five of
  the seven decide whether a block of extra features is kept at all, so the matrix either door
  handed back could hold different features from the one that won. All three forward all seven
  now — checked by reading the three functions back.
- **c23 and c24 handed in a made-up dataset name**, like `logd_fold0`. The per-dataset budget
  cap and the per-dataset model skips are looked up by name, so neither matched: on the capped
  datasets those two searched budgets the dataset is capped away from. They pass the real name
  with a `condition_suffix` now, which rides on `tier` and so reaches every layer and the stored
  row. Tested: two suffixes write two tiers inside one `caco2.db`. c23 passes its models as a
  dict, which skips the policy's model list, so no arm of c23 can be dropped by this.
- **c30, c31 and c32 could report two sweeps in one file.** A resumed run kept the dataset
  entries already there and overwrote the settings block with the current settings. All three
  now compare the two, name every field that moved, and drop the old entries.
- **The stored molecule pairs record which molecule list they came from.** `I` and `J` are
  positions into that list and the file said nothing about which list, so rebuilding a dataset
  re-paired every stored pair in silence. `mcs_cache.load_pairs` refuses a file whose hash
  disagrees with the molecules a reader holds, and four readers go through it.
- **Five `similarity_metrics` caches keyed on the dataset name alone** — same rule as §6.3. The
  purge now reaches the searched scores, which is the half that was missing.
- **s09 reads what its inputs were computed under.** It pools six files per dataset across
  datasets and read none of the coefficient, molecule count or label hash each file records. It
  collects all three now, names a disagreement within a dataset and across datasets, and stamps
  the summary `mixed` or `unknown` rather than asserting its own default.
  **One file moved, not deleted:** `s09_cross_dataset_summary/results.json` claimed manhattan,
  and of the 156 `s0*` files it aggregated, none records a coefficient at all. It is now
  `results.STALE_2026-09-11_asserts_manhattan_inputs_unstamped.json` — deleting a results file
  is the owner's call.
- **The noise precompute built three searches per dataset that nothing read.** It built the
  version that picks its winner on clean activity values; n15, the only consumer, always passes
  `score_on_noisy_labels=True` and asks for the version that picks on corrupted ones. Different
  tiers, so they could never meet. Removed. The dict stays, empty, for a caller that asks for a
  noisy search without that flag — there is none today.

---

## 10. Still open

**Nothing about the cache.** Every finding this audit raised is fixed, decided, or written down
as a rule with the code that enforces it.

**One thing I cannot do: check the server.** `REPRODUCIBILITY_LEDGER.md:451` says Claude cannot
reach ARC, and the key is refused. §2 holds one command that reads and deletes nothing and
answers every server question this audit left — the commit hash, which layers exist and how
large, whether torch and lightgbm import on a compute node, and per stored database whether any
one search was written by two different representation pools. Run it and paste the output.

**State of the tests after everything.** 21 unit tests pass. The six end-to-end
metric-direction tests pass, 170 seconds — they run real searches and check that a genuine win
comes out positive on a lower-is-better dataset, a higher-is-better one, a geometry objective
and a blend. Every changed file parses and imports.

**What a stored score is now filed under**, beyond the data, folds, seed, splitter, metric, task
type and tier it always had: the objective, the representation pool with widths, whether models
were tuned, the version of any per-fold representation builder, and — for an estimator that is
not what the registry builds for its name — a fingerprint of its own settings.
