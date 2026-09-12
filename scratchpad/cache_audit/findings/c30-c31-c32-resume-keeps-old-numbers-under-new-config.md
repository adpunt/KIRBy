# c30-c31-c32-resume-keeps-old-numbers-under-new-config

**title:** c30/c31/c32 resume on dataset name alone while rewriting the config block, so one file mixes two sweeps

**area:** tests/experiments/complexity c20+ (c20, c21, c22, c23, c24, c30, c31, c32, c33, exp_b, c43_signal, c43_step3) plus tests/experiments/_alloc_harness.py and _run_complexity_locally.py

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/complexity/c32_allocation_strategy_cross.py:533 and 551-553; c31_perf_weighted_global_selection.py:294 and 316-318; c30_balanced_global_selection.py:448 and 462-464

## scenario as reported
Run 1: `python c32_allocation_strategy_cross.py herg_ki` with N_VALUES = (50, 100, 200). The checkpoint gets datasets.herg_ki (cells at three N values) and config.n_values = [50, 100, 200]. Someone then edits N_VALUES to (50, 100, 200, 400) — these files are sweep scripts whose axes live as module constants — and runs `python c32_allocation_strategy_cross.py herg_ki logd`. herg_ki has a prior entry with 'by_model' and no 'error', so line 551 skips it; line 533 overwrites config.n_values with [50, 100, 200, 400] regardless. logd is computed at four N values. The single JSON now claims a four-value sweep while herg_ki's cells were computed at three, and nothing in the per-dataset block records which sweep produced it. The same holds for MODELS, BUDGETS, SCHEMES, MECHANISMS, POOL_MAX and NFOLDS in c32; MODELS, IMPS, POOL, BUDGETS, POOL_TOTAL_MULTS, SCHEMES, NFOLDS, N_INNER, N_BOOT in c31; MODEL, IMP, BUDGETS, NCAPS, POOLS, NFOLDS, N_INNER in c30. The same skip also cannot see a changed label vector, group vector or representation pool: the checkpoint stores no input_fingerprint at all, unlike the complexity run_all resume (complexity_helpers.py:1093-1107).

## evidence as reported
c32:533 `all_out["config"] = {"models": MODELS, "importance": IMP, "pool_max": POOL_MAX, "budgets": BUDGETS, "n_values": N_VALUES, "schemes": list(SCHEMES), "mechanisms": MECHANISMS, ...}` — unconditional, every run. c32:551-553 `prior = all_out["datasets"].get(ds)` / `if prior and "error" not in prior and prior.get("by_model"): print(f"[skip] {ds} already complete in checkpoint."); continue` — the only completeness test is the presence of a key. c31:316 and c30:462 are the same two lines with 'by_model' / 'target'. Compare _shared.py:5657-5710 input_fingerprint and _shared.py:5713-5727 fingerprint_mismatch, which the run_dataset.py path uses and these three do not.

## fix as proposed
Stamp the config dict and an input_fingerprint INSIDE each `all_out['datasets'][ds]` entry when it is written, and make the skip at c32:551 / c31:316 / c30:462 compare the stored per-dataset config and fingerprint against the current ones — recompute on any difference, naming the field that moved the way fingerprint_mismatch does. Leave the top-level config block as a copy of the last run's settings, but stop letting it stand for the per-dataset cells.
