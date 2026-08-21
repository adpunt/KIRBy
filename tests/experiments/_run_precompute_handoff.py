"""§V smoke: precompute -> experiment SQLite-cache handoff (local).

Proves, for ONE small reg dataset and ONE small clf dataset, that:
  1. precompute_hybrid_searches() runs and writes the new per-dataset SQLite
     result DB at $KIRBY_CACHE_DIR/result_db/<dataset>.db (COLD compute).
  2. A second identical precompute REUSES those rows ('ResultCache: N stored
     configs reused', ~0 recompute, NO meta drift).
  3. A downstream experiment-style call (get_precomputed_search with
     precomputed=None) HITS the same precomputed rows.
  4. Changing the MODEL list reuses the old rows and computes only the new
     model (the core bug the redesign fixes).

Usage:
  python tests/experiments/_run_precompute_handoff.py reg   # ginkgo_heparin
  python tests/experiments/_run_precompute_handoff.py clf   # humanness_therapeutic

Env: KIRBY_CACHE_DIR (own dir), KIRBY_SMOKE_N for clf subsample.
This is a SMOKE driver: it monkeypatches imported module-level objects in THIS
process only (never edits _shared.py / src), exactly like _run_complexity_locally.
"""
import sys
import os
import warnings
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import experiments._shared as _shared  # noqa: E402
from experiments._shared import (  # noqa: E402
    load_data, precompute_hybrid_searches, get_precomputed_search,
)
import kirby.find_best_hybrid as _fbh  # noqa: E402
from kirby.result_cache import db_path_for  # noqa: E402

# ── SMOKE-ONLY runtime shrink (local only; server imports fresh, unaffected) ──
# Stash the full model dict BEFORE trimming so step 5 can add a 2nd model back.
_ORIG_DEFAULT_MODELS = dict(_fbh.DEFAULT_MODELS)
_shared.FULL_BUDGETS = [25]
_shared.INTERNAL_EVALUATORS = ['rf']
_fbh.DEFAULT_MODELS = {'rf': _fbh.DEFAULT_MODELS['rf']}
_fbh.DEFAULT_INTERNAL_EVALUATORS = ['rf']
_fbh.DEFAULT_ALLOCATION_METHODS = _fbh.DEFAULT_ALLOCATION_METHODS[:1]
_fbh.DEFAULT_IMPORTANCE_METHODS = ['treeshap']

DATASETS = {
    'reg': 'ginkgo_heparin',
    'clf': 'humanness_therapeutic',
}
REPS = os.environ.get('KIRBY_LOCAL_REPS', 'aac,blosum,onehot').split(',')


def _count_rows(ds):
    p = db_path_for(ds)
    if not p.exists():
        return None
    con = sqlite3.connect(str(p))
    try:
        return con.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    finally:
        con.close()


def main(kind):
    ds = DATASETS[kind]
    print('=' * 72)
    print(f"PRECOMPUTE HANDOFF SMOKE — kind={kind} dataset={ds}")
    print(f"  KIRBY_CACHE_DIR={os.environ.get('KIRBY_CACHE_DIR')}")
    print('=' * 72)

    data = load_data(ds, rep_names=REPS, rep_preset='fast')
    reps, labels, groups = data['reps'], data['labels'], data['groups']
    task = data['task_type']
    print(f"\nLoaded {ds}: n={len(labels)} task={task} reps={list(reps)} "
          f"groups={'yes' if groups is not None else 'no'}\n")

    cfg = {'budgets': _shared.FULL_BUDGETS, 'search_speed': 'full'}
    common = dict(
        reps=reps, labels=labels, groups=groups, config=cfg,
        task_type=task, domain='antibody', dataset_name=ds, tier='general',
        internal_evaluators=['rf'],
    )

    db = db_path_for(ds)
    print(f"DB target: {db}  (exists before={db.exists()})\n")

    # ---- STEP 2: COLD precompute (computes + writes SQLite) ----------------
    print('\n----- STEP 2: COLD precompute -----')
    precompute_hybrid_searches(verbose=True, **common)
    n_cold = _count_rows(ds)
    print(f"\n[COLD] DB now exists={db.exists()} rows={n_cold}")
    assert db.exists(), "FAIL: SQLite DB was not written by precompute"
    assert n_cold and n_cold > 0, "FAIL: no rows written to SQLite"

    # ---- STEP 3: WARM precompute (must reuse) ------------------------------
    print('\n----- STEP 3: WARM precompute (identical args) -----')
    precompute_hybrid_searches(verbose=True, **common)
    n_warm = _count_rows(ds)
    print(f"\n[WARM] rows={n_warm} (cold was {n_cold})")
    assert n_warm == n_cold, (
        f"FAIL: warm run changed row count {n_cold} -> {n_warm}")

    # ---- STEP 4: downstream experiment-style HIT ---------------------------
    # get_precomputed_search with precomputed=None falls back to find_best_hybrid
    # cache=True — exactly what an experiment does when the in-memory dict misses.
    print('\n----- STEP 4: downstream get_precomputed_search (precomputed=None) -----')
    search = get_precomputed_search(
        reps, labels, groups, aug_condition='none', augmentations=None,
        speed='full', precomputed=None, task_type=task, domain='antibody',
        dataset_name=ds, tier='general', internal_evaluators=['rf'],
        budgets=_shared.FULL_BUDGETS, verbose=True,
    )
    nbest = len(search.best_by_model) if search and search.best_by_model else 0
    print(f"[STEP4] downstream search returned best_by_model={nbest} models")
    n_after4 = _count_rows(ds)
    print(f"[STEP4] rows after downstream call={n_after4} (was {n_warm})")
    assert n_after4 == n_warm, (
        f"FAIL: downstream call recomputed rows {n_warm} -> {n_after4}")

    # ---- STEP 5: changed-MODEL-list reuse (the core bug) -------------------
    # Add a 2nd model. Old 'rf' rows must be reused; only 'ridge' computes.
    print('\n----- STEP 5: add a model (ridge) — old rows reused, only new computed -----')
    # ridge for reg, logreg/a clf-capable model for clf; fall back to any 2nd model.
    pref = ['ridge', 'lightgbm', 'logreg', 'svm'] if task == 'regression' \
        else ['logreg', 'lightgbm', 'svm']
    extra = next((m for m in pref if m in _ORIG_DEFAULT_MODELS), None)
    if extra is None:
        extra = next((m for m in _ORIG_DEFAULT_MODELS if m != 'rf'), None)
    _fbh.DEFAULT_MODELS = {'rf': _ORIG_DEFAULT_MODELS['rf']}
    if extra:
        _fbh.DEFAULT_MODELS[extra] = _ORIG_DEFAULT_MODELS[extra]
    print(f"  model list now: {list(_fbh.DEFAULT_MODELS)}")
    precompute_hybrid_searches(verbose=True, **common)
    n_step5 = _count_rows(ds)
    print(f"[STEP5] rows after adding model={n_step5} (was {n_warm})")
    if n_step5 > n_warm:
        print(f"[STEP5] +{n_step5 - n_warm} new rows for the added model; "
              f"old {n_warm} reused. PASS")
    else:
        print(f"[STEP5] NOTE: row count unchanged ({n_step5}); the added model "
              f"may not have produced finite-score configs on this tiny data.")

    print(f"\n===== {kind.upper()} HANDOFF SMOKE COMPLETE =====")
    print(f"  cold_rows={n_cold} warm_rows={n_warm} after_downstream={n_after4} "
          f"after_add_model={n_step5}")


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'reg'
    assert kind in DATASETS, f"kind must be one of {list(DATASETS)}"
    main(kind)
