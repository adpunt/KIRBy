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

## 3. Fixed. Small, one right answer each, nothing stored changes meaning

- `wipe_allocation_caches.sh` — `fold_scores` added to the delete list, with a note saying what
  the layer holds and why it cannot detect its own staleness. It was in neither the delete list
  nor the keep list.
- `wipe_allocation_caches.sh` — `set -u` moved to after `. ./setup.sh`. On a node where the venv
  is unreadable, `setup.sh` expands `$VIRTUAL_ENV` bare and the job exited before reading
  `KIRBY_CACHE_DIR`, deleting nothing and logging one line that reads like a failed delete.
  `setup.sh` now writes `${VIRTUAL_ENV:-}` at both places.
- `invalidate_allocation_caches.sh`, `invalidate_refolded.sh` — both fell back to
  `$HOME/.cache/kirby` when `KIRBY_CACHE_DIR` was unset, and that directory exists on the login
  node, so they would delete a near-empty home directory and report success while the real store
  survived. Both now source `setup.sh` (with `KIRBY_SKIP_INSTALLS=1`), abort on an empty root,
  and abort unless `baselines/` and `rep_cache/` are both present. Tested: with the variable
  unset the first now prints `/data/stat-ecr/scat9264/.cache/kirby`, and a root without
  `baselines/` aborts.
- `_shared.py:114` — `create_hybrid_cached`'s internal evaluator default was `svm`;
  `create_hybrid`'s is `lightgbm`. The 2026-09-10 pass tracked the allocation method through all
  three doors and left the evaluator behind. Now `lightgbm`. Latent: c09 is the only caller and
  passes one explicitly.
- `_shared.py` — `create_hybrid_cached` now **refuses** any `create_hybrid` parameter its cache
  key cannot represent, instead of forwarding it. See §5 for what this was hiding.
  `augmentation_margin` and `augmentation_fraction` are now passed into the key; the key function
  had accepted them since it was written and nothing ever supplied them. Tested: `objective=` now
  raises, and two margins now write two files where they wrote one.
- `find_best_hybrid.py` — the per-representation baseline meta gained two opt-in keys.
  `struct_pairs_id`: under `objective='struct_fid'` the baseline is a correlation against one MCS
  pair set and nothing said which. `tune_knn_k`: the kNN baseline fitted with the tuned neighbour
  count and the one fitted with the fixed five shared a file. Both are opt-in, so every baseline
  on disk keeps hitting — verified by a run: 2 hit, 0 stale, filenames unchanged.
- `_run_precompute_handoff.py`, `_smoke_noisy_search.py` — both printed a row count off a
  `result_db` that `ResultCache` refuses to open. Both now check `PRAGMA table_info` first, the
  way `check_precompute_status.py` already did.

---

## 4. Open — needs your decision

Each item says what the choice is, what it costs either way, and what I would do.

### 4.1 The result database does not record which representations were searched

**This is the largest one. Five of the twelve readers found it independently.**

`IDENTITY_COLS` is tier, labels, groups, fold count, seed, splitter, metric, task type,
objective. The representation pool is stored on every row as `rep_pool` and never compared —
`result_cache.py:96` says so in as many words, and a grep finds no `SELECT`, no assert and no
raise that reads it.

**Measured.** Two searches, same labels, same groups, three folds, budget 20, model rf. The
first over two representations, the second over four. The second printed
`ResultCache: 1 stored configs reused`, fitted nothing, and returned the first's score to the
last digit (1.3809760917581493), the first's width (16 features), and the first's split —
`20 features across 2 of 2 representations; a=12, b=8` — while reporting a four-representation
search. The database held one row, `rep_pool` recorded as `('a', 'b')`.

**Who reaches it.** `opig_scaffold_hybrid.py:234` runs at a fixed `tier='scaffold_cmp'` and
takes its pool from the command line: `--smoke` gives five representations, the real run gives
twelve, and everything else about the two is identical. `federated_helpers.py:2494` does the same
with `--reps`. So does any change to a dataset's entry in `DATASET_REPS` between two runs, and so
does a representation that simply fails to build on one node.

**On disk now.** The three databases I can read locally are clean. The check is per file:

```sql
SELECT tier, objective, metric, task_type, labels_hash, groups_hash, n_folds, seed,
       cv_splitter_id, COUNT(DISTINCT rep_pool)
FROM results GROUP BY 1,2,3,4,5,6,7,8,9 HAVING COUNT(DISTINCT rep_pool) > 1;
```
Any group it returns was written by two different pools. A clean result is not a clean bill of
health: an identity where the second run was served everything still reads as one pool.

**Your options.**

- **Put the pool in the identity**, built from the widths rather than the names so a
  same-name-different-width run separates too. Costs: the primary key widens, so every
  `result_db/*.db` is deleted — which the wipe in §2 does anyway. The real cost is afterwards:
  the precompute's ten-representation rows stop being reusable by any experiment that searches a
  subset, and those experiments start paying a full search each.
- **Leave the key alone and make the pool a namespace.** The mechanism exists and c13 already
  uses it — a `condition_suffix` that rides on `tier`, which is why the two clamp arms in
  `caco2.db` do not collide. Costs no compute. It is caller-side discipline and does not stop the
  next script that forgets.

**I would put it in the identity.** The subset-reuse it costs is the thing that is currently
wrong, not a saving being given up.

**>>**

---

### 4.2 A tuned search reads back an untuned score

`tune`, `tune_n_iter` and `tune_knn_k` are in the baseline filename and in the hybrid-eval
filename and are rejected by the opt-in check. They are in no `result_db` column. So a tuned run
recomputes its baselines correctly and then reads the untuned hybrid rows and reports them, with
no tuned model ever fitted. It collides in both directions.

`find_best_hybrid.py:5190` assigns `row['tune']`, and `upsert_rows` discards it — the value is
in no column. That line reads as though tuning were recorded and should go either way.

Live caller: `tests/find_best_hybrid_tcga.py:1250` passes `tune=bool(args.tune)` and
`tier='general'` into one call, so both arms land on one database.

**Size of what this hides:** tuning kNN moved 6 of 8 datasets, +8.6% on flab_thermo_tresanco,
and changed the winning budget on 4 of them.

**On disk now:** nothing is wrong. Every row in the three readable databases was written by an
untuned run.

**Your options.**
- **Ride it on `tier`**, the way the noise condition, the objective and `condition_suffix`
  already do. Three lines, no schema change, nothing deleted, every untuned row keeps matching.
  The row itself still records nothing, so a later reader tells the two apart by parsing a string.
- **Add `tune` to `IDENTITY_COLS`.** The row becomes self-describing. Widens the primary key, so
  every database is deleted — free if you are doing §2 anyway.

**I would add the column,** because you are deleting the databases in §2 regardless and it is the
version that survives being read six months from now.

**>>**

---

### 4.3 Two runs that name the same estimator differently share everything

A model is keyed by its **name**. `complexity_theory_experiments.py:1315` (experiment 6, full
mode) names the fourth variant of every model family `depth=None`. So the random forest's
`max_depth=None` and XGBoost's `max_depth=0` write and read the same baseline files and the same
database rows. XGBoost is never fitted; its capacity gradient is the random forest's.

Quick mode is clean — its families are RF, Ridge and MLP, which share no variant name. Only
`run_complexity_experiments.sh full_b` reaches it.

**On disk now:** `~/.cache/kirby/baselines` holds 9 files whose model name is the bare string
`depth=None` — 3 on buchwald_hartwig, 6 on flab_thermo_jain. Nothing says which estimator wrote
them. Today's c06 prefixes the family, so it will never read them again. Delete them. No database
holds a variant-named row.

**The one-line half is not a decision:** `complexity_theory_experiments.py:1315` should name the
model `f"{fam_name}:{var_name}"`, which is what c06 already does. I have not made that change
because it is an experiment file and it changes what that experiment computes.

**The engine half is yours.** Nothing stops the next caller repeating it. Refusing a dict `models`
under `cache=True` would break five legitimate callers (c02:328, c06:183, c11:462, c15:633,
c16:285).
- **Require a caller-supplied `models_id`**, matching the `cv_splitter_id` and `rep_factories_id`
  pattern already there, and put it in the baseline filename and the row's model column.
- **Derive the identity from each estimator's class name plus `get_params()`.** No caller changes.
  Changes every baseline filename a dict caller writes, so it costs a wipe of `baselines/` and of
  those callers' databases.

**>>** (a) the one-line c06-style rename in `complexity_theory_experiments.py`:
**>>** (b) `models_id`, or derive from `get_params()`:

---

### 4.4 A missing library is substituted silently and the file keeps the requested name

Ask for `internal_evaluator='lightgbm'` on a node where lightgbm cannot be imported and
`_evaluate_allocation` catches the ImportError and recurses into xgboost, then into rf. The
per-representation scores that split the budget come back from the substitute and are written to
`fold_scores/<ds>/alloc__f0__lightgbm__<hash>.pkl` with meta `evaluator='lightgbm'`. The same
run's downstream `lightgbm` model is a `RandomForestRegressor` for the same reason, and its score
goes to a baseline file named for lightgbm.

A later run on a node that has lightgbm matches on filename and on every meta key. It splits its
budget by XGBoost scores and reports a random forest's number as its lightgbm baseline. Nothing
on disk records the substitution.

`dnn` is worse: it catches bare `Exception`, so it also substitutes when a fit fails on one
fold's rows. One file can hold a real dnn for fold 0 and a random forest for fold 1.

**On disk now:** `/Volumes/seagate/kirby_cache/fold_scores` holds 35 files named `__lightgbm__`,
25 `__dnn__`, 25 `__xgboost__`; its `baselines/` holds 67 lightgbm and 44 dnn;
`~/.cache/kirby/baselines` holds 1447 across the four names. **How many hold a substitute cannot
be read off the files.** On this laptop everything imports, so nothing written locally was
substituted. One `python -c 'import torch, lightgbm'` on the partitions the precomputes ran on
settles the cluster.

**Your options.**
- **Record what was actually fitted.** Have `_evaluate_allocation` return the evaluator it used
  and put that name, not the requested one, in the filename and the meta. A substituted file then
  misses for a run that has the real library.
- **Refuse rather than substitute** when a caller names the library explicitly. Costs a re-submit
  and buys the guarantee that a file named lightgbm holds lightgbm.

They are not exclusive: recording protects the cache, raising protects the run. **I would do
both.** Either way `_announce_evaluator_fallback` should cover the lightgbm, xgboost, catboost and
ngboost branches, which today substitute with no warning at all.

**>>**

---

### 4.5 c10 picks its recipe on the true labels and then reports noise robustness

`c10:156` reads a recipe — allocation method, importance method, internal evaluator, budget —
from a search scored on the true labels of every row. It then corrupts the training labels of
each fold and rebuilds under that recipe. The hybrid arm carries a choice made with the labels
the run goes on to corrupt; the single-representation arm it is compared against had no such
choice made for it.

Not a cache collision — passing `label_noise` renames the tier and the noisy search recomputes
correctly. c10 just never asks for the noisy lineage, and its output records nothing about which
labels chose the recipe. Because the summaries are ratios to each arm's own score at level zero,
the advantage partly cancels; what was left in my runs was roughly half a point to two and a half
points of retention.

**On disk:** nothing locally — both c10 result directories are empty. Any c10 file on the server
carries the bias.

**Your options.**
- **Resolve the recipe under each cell's own noise.** The honest comparison the `label_noise`
  parameter was written for. Costs 56 additional hybrid searches per dataset — 6 levels for the
  10 dose-matched strategies and 7 for censoring, minus level zero, which is already on disk.
- **Keep the clean recipe and say so in the output**, one field, the way c15, c16, c17, c19 and
  c22 already do. A few lines, no compute. The advantage stays in the number but stops being
  silent.
- **One level per strategy, at the top of the grid where the gap is largest**, reported beside
  the clean-recipe number. 11 searches per dataset, and it measures the size of the effect on
  real data instead of on my synthetic sets.

**I would take the third.** It is the only one that tells you how big the problem is.

**>>**

---

### 4.6 The similarity precompute throws away its model list, so kNN never gets a recipe

`_shared.py:4641` sets `models = None` unconditionally, discarding both the argument and
`config['models']`. The distance search therefore scores rf, xgboost, svm, dnn and lightgbm, and
never knn.

Three consequences. `s10:970` asks the pool for `downstream_model='knn'`, finds it absent, and
writes `hybrid_evaluated=False` — the kNN hybrid arm of the distance study is never computed.
Any distance call that does not pass `data=` (s06:149, s07:513, s08:161, c03:107) reaches
`find_best_hybrid` with six models while the in-process precompute reaches it with five, and the
two discard each other's checkpoint forever. And the fold-0 allocation vote is taken over five
models rather than six, so the winners the in-process search reports are not the winners the
SLURM precompute reports for the same dataset and model — those two numbers are not comparable
side by side.

**Nothing on disk is wrong.** The rows that are there are correct; knn rows are simply absent.
Do not delete `result_db`. Delete `searches/*__general_distance__*.ckpt.pkl` after the fix so the
allocation vote is retaken, and re-run s10 wherever `hybrid_evaluated=False`.

**The fix is one line** — keep `None` as the default and let an explicit list through, which is
what the docstring at `_shared.py:911` already says the distance lineage requires and what
`get_fold_best_search` and the SLURM precompute already do. **I have not applied it** because it
changes which models the whole similarity suite searches, and that is a compute decision.

**>>**

---

### 4.7 The per-fold hybrid cache is dead, and it is wrong where it is alive

Two separate things about `hybrid_evals/`.

**It is unreachable.** `_hybrid_cache_enabled` requires `allocation_method.startswith('top_')`
(`find_best_hybrid.py:4130`). `top_2/top_3/top_5` came out of `DEFAULT_ALLOCATION_METHODS` on
2026-09-01. Neither the default nor the fast list has a `top_` name. Measured: a 60-configuration
run printed `hybrid_eval: 0 hit / 0 miss / 0 stale` and created no directory. Forced to `top_2`
the layer armed and recorded 3 misses.

**And it never writes even then.** The save at `:4542` requires the subset the allocator actually
picked to equal the subset `_predict_top_k_subset` guessed. On all three folds it printed
`ACTUAL picked=['repA','repB','repC'] predicted=['repA','repC']`. A monkeypatch on the save
function logged only `baselines/` and `fold_scores/` writes — never a `hybrid_evals` path.

**And where it does key, it keys on the wrong thing.** The filename names the k winners, not the
pool they were chosen from — but the pool sets the shift applied to the scores, so the same three
winners get different feature counts out of different pools. Measured: five representations,
`top_3`, budget 60. Full pool gives A/B/C 22/20/18 features; drop one representation and the same
three should get 25/20/15. Sharing one cache directory, the second run returned the first's fold
scores `[0.15817, 0.08981, 0.08095]` instead of its own `[0.16202, 0.08273, 0.08426]`. The
comment at `:789` saying two runs may share one base subset is wrong: the split among the
winners is not the same across pools.

**On disk:** 38,708 files under `~/.cache/kirby/hybrid_evals`. None can be served to current code —
five meta keys added on 2026-09-10 are absent from all of them. The seagate cache has no such
directory. The wipe in §2 covers it.

**Your options.**
- **Fix and revive it.** Put the whole pool — sorted (name, width) for every representation in
  `base_reps` — into the meta and into the filename, and fix the write cross-check. Then the layer
  works for `top_*`. It still will not arm on the five default allocations.
- **Widen the gate so the default allocations reach it too**, which is the version that actually
  saves compute, and fix the pool key at the same time.
- **Delete the layer.** It has never written a file under any allocation the suite searches, and
  the same reuse now lives in `result_db`.

**I would delete it.** Reviving it means fixing a subset predictor, a write cross-check and a pool
key, for a layer whose job `result_db` already does. Note one real cost of deleting: `result_db`
stores one row per finished configuration, and a configuration that is pruned at fold 0 leaves no
row — see 4.8.

**>>**

---

### 4.8 A search that prunes never warms up

Measured. Five allocation methods asked for; fold 0 pruned two
(`Pruned 2/5 strategies (top-3 in <15% of groups): ['softmax_T1', 'wglobal_power2']`); three rows
stored. The second identical run printed `ResultCache: 3 stored configs reused` — and then printed
the same three pruning lines again, re-fitting fold 0 for all five recipes and pruning the same
two. 9.4 seconds of a 16.4-second cold run, recurring on every call forever.

A pruned recipe leaves no row and no marker, so nothing records that it was tried. On a real
dataset the recurring cost is one fold's fits for every pruned recipe, on every resumed or
extended search. The 45-row full run showed the other side: nothing was pruned there and its warm
re-run took 1.1 seconds.

The two pruned names are dropped in silence when `verbose=False`.

**Your options.**
- **Store a pruned recipe as a row** carrying its fold-0 score and a `pruned` marker, so the next
  run reads the vote instead of retaking it. Widens what the table holds; does not widen the key.
- **Leave it and print the pruned names unconditionally**, so at least the missing rows are
  explained.

**I would store the row.** It is the difference between a resumed search being cheap and a
resumed search paying a constant tax.

**>>**

---

### 4.9 The SLURM arrays break the single-writer rule on one SQLite file

`result_cache.py` says single-writer. Three uncontended concurrent writers were clean: 360 of 360
rows, `PRAGMA integrity_check` ok, no corruption in any of three tests.

Contended is not. `sqlite3.connect` is called with no `timeout`, so the busy wait is the library
default of 5 seconds, and neither the open at `:5890` nor the write at `:6288` is wrapped. A
forced lock killed a whole search after 24.3 seconds with `database is locked`, raised inside
`_ensure_schema` — which needs a write lock even when the table already exists. So a task can die
at database open **after** paying for its baselines and its importance precompute.

Two array scripts put many tasks on one database: `run_f26_redundancy_penalty.sh` sets 24 tasks
per dataset with `--array=0-239%40`, and the 12 arms ride on `tier`, so all 24 land in one file.
`run_f24_augmentation_grid.sh` does the same with 16. Both are framework experiments, so they are
out of the audit's scope — but the lock behaviour is in `src/` and any suite array with the same
shape hits it. `run_f28_objective_weight_space.sh` is one task per dataset and is safe.

**How often the 5-second window is hit, I did not measure.**

**Your options.**
- **Pass a `timeout` to `sqlite3.connect`** — 60 seconds costs nothing when uncontended and turns
  most of these into a wait rather than a dead job.
- **Give each array task its own database file** and merge afterwards. No contention at all; needs
  a merge step.

**I would set the timeout now** — it is one argument and it converts a dead job into a pause —
and treat the per-task file as the answer if it still bites.

**>>**

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

## 8. Smaller open items, listed so they are not lost

Verified, real, and each latent or cheap. None needs a decision before the wipe.

| what | where | status |
|---|---|---|
| The blend's own name is written into the `metric` column, so the dataset's metric drops out of the key | `find_best_hybrid.py` result_db identity | latent — only f28 reaches it |
| `rep_factories_id` is demanded when caching is on and reaches no database column | `:7377` raises, `:7485` has no slot | latent |
| Raising `AUG_LOGIC_VERSION` invalidates `hybrid_evals/` and cannot invalidate a `result_db` row | `result_cache.py` | the comment claiming "no deletion needed" is wrong |
| The fold-search memo keys on the training row **count**, not the rows | `_shared.py` | latent |
| `get_fitted_pool_hybrid` rebuilds a blueprint without five of its extra-feature settings | `_shared.py` | latent, eight callers |
| c23 and c24 hand `find_best_hybrid` a made-up dataset name, so the budget cap and model skips never match | `complexity_helpers.py:1128` | affects 14 of 70 datasets |
| c30/c31/c32 keep a dataset's old scores and rewrite the settings block to the new settings | `run_c32_allocation_cross.sh:69` | one file reports two sweeps |
| The resume fingerprint records no metric and no noise grid | `_shared.py:5657` | a metric or level-grid change leaves finished results on disk |
| The MCS pair file records no molecule ordering | `similarity_metrics/mcs_cache.py` | rebuilding `sim_<dataset>` silently re-pairs |
| Five `similarity_metrics` caches key on the dataset name alone | `shape_similarity.py:168` and others | same wipe rule as 6.3 |
| The transformer embedding file name omits the model id | `run_battery_moleculeace.py:90` | a second transformer scores the first's vectors |
| The CDR-index cache is keyed on the dataset name alone | `s01:97`, `s02:540` | a shorter run gets the longer list |
| n09 falls back to the clean blueprint and the output cannot tell | `noise_helpers.py:187` | records the clean budget as the noisy one |
| Every noise run builds three searches nothing reads — n15 asks for a different tier | `noise_helpers.py:375` | wasted compute only, no wrong number |
| A noisy search records the noise **name**, never the corrupted label column | `result_db` rows with `__noise_` in tier | trap for the next injector change; no rows on disk today |
| s09 pools six files per dataset without reading their coefficient or fingerprint | `s09:770` | `s09_cross_dataset_summary/results.json` says manhattan and its inputs are Euclidean — delete it |
| fed01's centralised arm builds its hybrid with no scaffold groups | `fed01:381` | the other five federated calls pass groups |
| fed05 ranks representations by the internal R² instead of spearman | `fed05:533` | passes a model with no metric |
| Three Ginkgo properties are not in `DATASET_REGISTRY`, so the hybrid arm is dropped | `opig_ginkgo_developability` | metric lookup raises |
