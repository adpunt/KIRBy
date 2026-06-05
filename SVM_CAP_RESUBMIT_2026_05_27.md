# SVM Cap Resubmit — 2026-05-27 ~13:39 BST

Code change: `MODEL_MAX_BUDGET['svm']` lowered from 1000 → 500 in
`src/kirby/find_best_hybrid.py` (commit `20f436c`). Drops SVM b=1000
cells — the precompute chokepoint.

## Cancelled (11 jobs, state CG)

| Job | Name | Reason |
|---|---|---|
| 12198459 | cx_qm9 | stuck on SVM b=1000 (+10 cells/17h) |
| 12198494 | sim_qm9 | same |
| 12198463 | cx_herg | same |
| 12198458 | cx_logd | in SVM zone, would hit b=1000 |
| 12198495 | sim_logd | same |
| 12198460 | cx_adme_hlm | stuck on single SVM b=1000 cell 17h |
| 12198493 | sim_adme_hlm | same |
| 12198461 | cx_adme_rlm | same |
| 12198492 | sim_adme_rlm | same |
| 12198469 | cx_flab_expression_koenig | same |
| 12198484 | sim_flab_expression_koenig | same |

NOT cancelled: 12198490 sim_herg — already past SVM zone, retained.

## Resubmitted with svm=500 (11 jobs, PD)

| Job | Category | Dataset |
|---|---|---|
| 12208163 | complexity | qm9 |
| 12208164 | complexity | herg |
| 12208165 | complexity | logd |
| 12208166 | complexity | adme_hlm |
| 12208167 | complexity | adme_rlm |
| 12208168 | complexity | flab_expression_koenig |
| 12208169 | similarity | qm9 |
| 12208170 | similarity | logd |
| 12208171 | similarity | adme_hlm |
| 12208172 | similarity | adme_rlm |
| 12208173 | similarity | flab_expression_koenig |

## Missing-from-original-submission backfill (3 jobs, PD)

| Job | Category | Dataset |
|---|---|---|
| 12208174 | complexity | humanness_therapeutic |
| 12208175 | similarity | humanness_therapeutic |
| 12208176 | similarity | abcov_kd |

## Scheduled snapshots (4 jobs, PD with --begin)

| Job | Name | Begin time |
|---|---|---|
| 12208177 | snap_2h | 2026-05-27 15:39:29 |
| 12208178 | snap_6h | 2026-05-27 19:39:29 |
| 12208179 | snap_12h | 2026-05-28 01:39:29 |
| 12208180 | snap_20h | 2026-05-28 09:39:29 |

Snapshot output lands in `slurm-<JID>.out` (default location, no `--output`).

## Outstanding

The doc-vs-submitted comparison only matched 14 lines against the doc — the
grep pattern misses many table rows. PIPELINE_READINESS_TODO.md totals 81
datasets, 66 active (15 dropped). With 30 cx_ and ~27 sim_ submitted, there
may still be ~36 datasets uncovered. Better grep needed to find them.

---

## 2026-05-27 evening — Round 3+4+5 corrections

**Round 3 (cx 33 missing backfill, forward):** 12208320–12208353 (33 jobs)

**Round 4 (cx 4 svm-affected, forward):** 12198464/65/68/72 cancelled,
resubmitted as 12208377–12208380.

**Round 5 (--reverse flag removed; full sim nuclear resubmit with PROPER reverse):**

- Removed `--reverse` from `submit_all_datasets.sh` (commit `4af8de8`).
  It was reversing dataset submission order; the actual goal is reversing
  CONDITION ORDER inside each precompute job via the env var
  `KIRBY_PRECOMPUTE_REVERSE_ORDER=1` (read by
  `tests/precompute_hybrids.py:360` and `tests/experiments/_shared.py:2195`).
- All old sim jobs cancelled (16 running + 7 PD from earlier rounds).
- 63 sim jobs resubmitted in forward dataset order with env var set:
  job IDs **12208512–12208574**. Each should print
  `KIRBY_PRECOMPUTE_REVERSE_ORDER=1 → reversed condition order` once started.

## Final job state (2026-05-27 ~14:30 BST)

- cx_ jobs: 63 (19 R from yesterday past precompute + 44 PD from rounds 1/3/4)
- sim_ jobs: 63 (all PD, Round 5)
- snap_* scheduled: 12208177–12208180
- exp_%j: 12198500 (legacy, still running)
