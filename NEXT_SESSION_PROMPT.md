# Next session — KIRBy (handoff 2026-06-04, after the overnight fix-fleet)

**READ FIRST, IN FULL:** `DECISIONS_FOR_USER.md` (plain-English what-needs-your-eyes), `COVERAGE_AUDIT_2026_06_04.md` (proof every issue was addressed → disposition), `CACHE_OPERATIONS_2026_06_04.md` (cache-removal commands), and `SYNTHESIS_2026_06_04.md` (every agent's full report). Then this file. Corpus: `MASTER_ISSUES_2026_06_04.md` (now has a "NEW ISSUES FOUND" live list, NEW-F1..5 fixed / NEW-O1..14 open). Estimates: `ESTIMATES_INVESTIGATION_2026_06_04.md` (corrections folded in at top).

**ROUND 2 (applied + committed this session):** 3e c23 size gate (n>1000), 3h c05 dedup, 3i screening→FULL (cx/sim/noise helpers), 4a n10 budgets reachable, 4b s11 Tanimoto proper kernel, 4c 8 datasets registered (+ precompute submission), 4d c05/c07/s11 groupless companion, 4f noise clf estimators dropped, ORD/3c opt-in expensive-first (`KIRBY_EXPENSIVE_FIRST=1`), ISS-84 StratifiedGroupKFold for clf (production engine), ISS-29 antibody near-dup leakage fix, find_best_hybrid random-fallback→raise, ISS-85 Butina (as option). **Before re-running: (1) the cache fix is committed now; (2) run `tests/purge_graphinity_cache.py --delete` on the server (Graphinity corrupt cache); (3) decide the open NEW-O items + cost levers in DECISIONS §3-4.** NOT applied (analysis/your-call): GP cap (small-finisher throughput only), capped cuts (n≤250), c17 MI subsample, s03 MI subsample.

**All results live on the ARC server — never look locally; the user pastes/uploads them. JSON-only rsync. The user runs all server ops.**

## What happened this session
A fleet of 12 file-partitioned agents worked EVERY issue in `MASTER_ISSUES_2026_06_04.md` — reading the real code + audit docs (not memory) and smoke-testing with the project Python (`/Users/apunt/anaconda3/bin/python`, 3.11). Lane discipline held: 56 files changed, each owned by exactly one agent, no collisions, no noise-file edits (noise was already clean), no framework (F00–F15) edits, no federated deletions. **Nothing is committed** — the user is the sign-off gate (commit → push → resubmit is the only path to the cluster).

Key outcome: a large fraction of the audited bugs were **already fixed in committed history** (verified against current code). The new work = a few fresh bug fixes, ~6 results-affecting fixes to confirm, several brand-new bugs caught, and the cost levers (all written up, none applied).

## ⭐ THE #1 ACTION — commit the cache fix (deploy-blocker)
The neural-rep cache-churn fix (the months-lost mechanism) is **uncommitted** — `HEAD:src/kirby/find_best_hybrid.py` still has the old churning code. A clean checkout loses it. **Commit `src/kirby/find_best_hybrid.py` before any re-run.** The agent also completed two gaps in it: the baseline-layer hash (neural reps were still churning the per-rep baseline cache) and the no-augmentation collision in the core (was only patched in callers). Details: DECISIONS Section 0.

## DECISIONS WAITING ON THE USER (none applied; see DECISIONS_FOR_USER.md)
**Results-affecting fixes ALREADY in the working tree — confirm or revert (Section 2):**
- c09 internal CV 3→5 fold (fixes ~120 recomputes + an inconsistency; changes 9b/9c numbers)
- s11 rebuild now matches its blueprint's augmentation strategy (bites only with `--augment`)
- Ridge removed from the distance-tier evaluator list (busts the distance cache → recompute)
- c18 budget sweep unlocked from [50,100] to the configured budgets
- random-split fallback now RAISES (enforces no-random-splits; needs companion fix 4d)
- federated now uses a classifier on classification data

**Cost levers NOT applied — approve to apply (Section 3):**
- ★ GP budget cap 1000→250 = the top lever now (20–40% per-search cut on n<1000). **SVM is NOT the monster anymore — it's already capped; GP is.**
- loop ordering expensive-first (keep the costly results when a job times out)
- capped-augmentation cuts (compute the importance pass once; cap 50→20) = 3–5× on the capped tier
- c23 gate the CatBoost/DNN/CNN tier behind a flag
- c17 MI subsample + smaller prefilter (the free once-slice fix is already applied)
- s01 large-N skip, s03 MI cap at 2000 rows
- c05 repeats 3→1 for screening + the free full-data 3× dedup
- 7 experiments: screening→full budgets for finer curves

**New bugs needing a decision (Section 4):**
- 4a n10 will silently produce EMPTY hybrid results (its budgets [25,100,250] aren't precomputed) — fix the budgets
- 4b s11 crashes on molecular datasets with an ECFP4 rep (Tanimoto kernel vs sklearn) — fix or disable that condition
- 4c register 8 OPIG dataset names in DATASET_REPS (the "stuck in survey stage" root cause): chembl_qsar, cdr_flexibility, humanness_species, ginkgo_tm2, ginkgo_aggregation, ginkgo_purity, graphinity_foldx, graphinity_rosetta
- 4d companion fix so c05/c07/s11 reduce folds (or skip+flag) on group-poor subsamples — prevents crashes/empty results from the no-random-splits enforcement
- 4f drop noise estimators that fit a regressor on class labels
- 4g fed05 single-split → k-fold (a real restructure)

## ⚠️ CACHE — the user said pay close attention here (more opportunities)
1. **Commit the churn fix (above) — without it everything else is undermined.**
2. **Estimate corrections that SHRINK the waste (DECISIONS Section 5):** the "SCREENING forces a cache rebuild" and "cx+sim each re-run precompute 2×" claims are BOTH FALSE — the precompute always builds full budgets and is shared by filename across cx/sim. Only a *concurrent* cx+sim race double-computes → fix by running the precompute job to completion before the experiments. The distance tier IS precomputed (audit was stale).
3. **Cache hygiene already verified solid:** all 13 search-meta params keyed, cv_splitter keyed, checkpoint validates meta on load, atomic writes are NFS-safe (temp-then-rename same dir). One residual: neural-rep content hashes are in the `hybrid_evals` *filename*, so those configs miss the dedup across runs (optimization loss, not churn/wrong) — deeper fix = key on rep names.
4. **Cache-key consequences of approved changes:** removing Ridge from distance evaluators (2c) and the no-aug normalization (Section 0) both shift cache keys → expect a one-time recompute of affected tiers on the next run.

## ESTIMATES — still the #1 long-term priority
The committed per-dataset table + the c17-dominated cx model are in `ESTIMATES_INVESTIGATION_2026_06_04.md` (read in full). Carry forward:
- Node CONTENTION (3–48× wall variance) is still the dominant uncertainty — anchor on un-contended runs.
- Per-rep DIMENSIONALITY (item 1.1, server dump) is still the master unlock for tight per-dataset cx; the forbidden `(n/anchor)^k` form is still banned — the cx hour column is order-of-magnitude/ordering only.
- TODO: fold Section 5 corrections into the estimates doc (downgrade CACHE-3/COST-11/TIME-1/TIME-7; mark COST-6 SVM stale, promote GP). Not yet done — offered to the user.

## OFFER for next session
On the user's word: (1) apply approved cost levers, (2) do the 4d companion fix, (3) register the 4c datasets + precompute submission, (4) fold Section 5 into the estimates doc, (5) fix/disable 4a n10 budgets and 4b s11 Tanimoto, (6) commit + push the approved set. Begin by confirming you've read DECISIONS_FOR_USER.md + SYNTHESIS, then ask which Section 3 levers and Section 4 fixes to apply.
