# c30-c31-c32-gate-not-rerun-for-new-datasets

**title:** The blocking metric-direction gate is skipped whenever any earlier checkpoint recorded a pass, including for datasets it never covered

**area:** tests/experiments/complexity c20+ (c20, c21, c22, c23, c24, c30, c31, c32, c33, exp_b, c43_signal, c43_step3) plus tests/experiments/_alloc_harness.py and _run_complexity_locally.py

**class:** A-wrong-answer

**severity as reported:** small

**location:** tests/experiments/complexity/c32_allocation_strategy_cross.py:517-535; c31_perf_weighted_global_selection.py:275-293; c30_balanced_global_selection.py:441-446

## scenario as reported
Run 1: `python c32_allocation_strategy_cross.py herg_ki`. _gate_dsets is EXPECTED_DIRECTION | {herg_ki} = the four anchors, assert_metric_direction runs and passes, and the pass is written to all_out['gate'] keyed by those four names. Run 2: `python c32_allocation_strategy_cross.py tcga_brca` against the same checkpoint. gate_ok is True because all_out['gate'] is non-empty, so the gate never runs and gate_rec is copied forward. tcga_brca's registry metric and direction are never checked end to end, yet the file it writes carries a 'gate' block that reads as a pass and lists only the four molecular/antibody anchors. Every direction-corrected delta c32 then computes for tcga_brca rests on an unverified direction.

## evidence as reported
c32:517 `gate_ok = bool(all_out and all_out.get("gate"))`; c32:519-527 the gate only runs `if not gate_ok`, and the dataset list it would have used is built inside that branch at c32:523 `_gate_dsets = sorted(set(H.EXPECTED_DIRECTION) | set(datasets))`; c32:532 `all_out["gate"] = gate_rec` copies the old record forward. c31:275 and c30:441 are the same test. _alloc_harness.py:171-186 shows assert_metric_direction loads no representation matrices — it builds a synthetic label ladder — so re-running it for the new datasets costs nothing.

## fix as proposed
Replace the emptiness test with a coverage test: run the gate whenever `set(_gate_dsets) - set(all_out.get('gate') or {})` is non-empty, and merge the new records into the stored ones rather than copying the old dict forward wholesale.
