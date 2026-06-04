# UPDATED + VERIFIED wall-time estimates — KIRBy wave (2026-06-04)

Every number cites a measured anchor in `WAVE_SNAPSHOT_2026_06_04_0207.md` / `WAVE_STATUS_2026_06_04.md` / `CHECK_PRECOMPUTE_TIMING_2026_06_02.md`, cross-checked against source (`find_best_hybrid.py`, `_shared.py`, c05/c17/c23, s01/s03). NO memory. Banned-form (`(n/anchor)^k`) used ONLY as order-of-magnitude/ordering; anywhere n is outside a measured range = EXTRAPOLATED, bounded from the nearest measured neighbour. Node contention inflates measured walls 3–48× → estimated cells are brackets, not points.

**Wall clock:** wave started 2026-06-01 ~06:00; 7-day limit ≈ 2026-06-08 06:00 → at the 02:07 snapshot **~68h elapsed, ~100h left**; one partition ≈168h.

## ★★★ CRITICAL CODE-STATE FINDING (verified via git timeline)
The running wave is on commit **`374bb08` (06-01 05:43)**. The fix-fleet commit **`3cadb95` (06-04 16:12)** is **NOT in the running wave** — it only takes effect on a **restart/resubmit**. Consequences:
- IN the running wave (helping now): c23 extra-tier gate (`1474bd1`, 04-02), capped n≤250 gate (`374bb08`, 06-01).
- NOT in the running wave (working tree only, restart-gated): cache-drift fix (CACHE-1), c17 mRMR once-slice, c05 full-data dedup, s01 OOM guard.
- 32-CPU bump (`0a6b746`, 06-02) is post-wave-start → **running jobs are 8-CPU**.

## Corrections applied (verified)
1. **GP, not SVM, is the per-search driver IN THE CAPPED/SMALL REGIME** — `MODEL_MAX_BUDGET['svm']=500` (:1723) caps SVM to 0.1–5% on antibody `none` rows; GP is 28–72%. **BUT SVM IS the driver in the large-n UNCAPPED precompute** (herg gen/none svm=19.31h/65.6%, snapshot :772) because there it runs to `_max_samples=10000` (:1199). Both true per regime.
2. No SCREENING cache-rebuild (precompute always FULL_BUDGETS).
3. No double-precompute unless cx & sim run concurrently before either writes (shared filename-keyed cache).
4. Distance tier IS precomputed by run_all.

## cx model validation (the only firm cx data — 5 measured antibody pipelines, WAVE_STATUS §8)
| dataset | n | cx MEASURED | c17 meas | c17 law | residual |
|---|---|---|---|---|---|
| flab_polyreact_bvp | 137 | 38.98h complete | 9.09h | ~16h | law over-predicts ~1.8× (rep-poor) |
| flab_polyreact_cic | 137 | 37.11h complete | 8.32h | ~16h | ~1.9× over |
| abagym_HER2 | 146 | 34.94h complete | 6.74h | ~16h | ~2.4× over |
| flab_immunogenicity_marks | 217 | c17=27.6h | 27.6h | 26.0h | **+6% (anchor)** |
| abagym_Ang2 | 397 | ≥63.87h through c19 | 51.54h | 51.5h | anchor |
| abagym_VEGF | 418 | ≥63.59h through c19 | 53.40h | 53.4h | anchor |

Law holds ±6% at n=217–418 anchor; over-predicts ~2× for rep-poor (total_dim<2000); 2-point interpolation. **Every cx cell at n>418 is EXTRAPOLATED.** Also measured cx/c17 = 4.3–5.2× (not the ×1.5 the table assumes) → cx-total biased LOW at large n.

## SECTION 1 — per-dataset table (precompute LEFT | cx total | sim total)
Confidence: MEASURED / NEAR / EXTRAPOLATED. "dist" = distance-lineage precompute (structurally derived, UNVALIDATED — zero `*_distance` rows in timing).

| dataset | n | dom | precompute LEFT | cx total | sim total | conf(cx) |
|---|---|---|---|---|---|---|
| ginkgo_heparin / flab_pk_* / flab_aggregation_* / flab_thermo_jain | 94–137 | ab | 0 (+~3–6 dist) | 35–45 | ~0.5 | NEAR |
| flab_polyreact_bvp | 137 | ab | 0 | **38.98 MEAS** | ~0.5 | MEASURED |
| flab_polyreact_cic | 137 | ab | 0 | **37.11 MEAS** | ~0.5 | MEASURED |
| abagym_HER2 | 146 | ab | 0 | **34.94 MEAS** | ~0.5 | MEASURED |
| ginkgo_pr_cho..ginkgo_hic | 197–242 | ab | 0 (+~5–6 dist) | 40–60 | ~0.7 | NEAR |
| flab_immunogenicity_marks | 217 | ab | 0 | ~45–55 (c17 meas 27.6) | ~0.7 | NEAR |
| abagym_Ang2 | 397 | ab | 0 (+~2 dist) | ≥64 MEAS→c19; ~75–95 full | ~1.5 | MEAS(partial) |
| abagym_VEGF | 418 | ab | 0 (+~2 dist) | ≥64 MEAS→c19; ~75–95 full | ~1.5 | MEAS(partial) |
| ddg_combined | 429 | ab | 0 (+~10 dist) | ~80–110 (c17→55) | ~1.5 | EXTRAP |
| humanness_therapeutic | 564 | ab | 0 | ~95–130 (c17→74) | ~1.5 | EXTRAP |
| abagym_lysozyme | 627 | ab | 0 (+~13 dist) | ~110–150 (c17→83) | ~1.5 | EXTRAP |
| abagym_EGFR | 659 | ab | 0 | ~115–160 (c17→87) | ~1.5 | EXTRAP |
| flab_thermo_tresanco | 672 | ab | 0 (+~23 dist) | ~115–160 (c17→89) | ~1.5 | EXTRAP |
| tcga_brca_pam50 | 821 | tcga | 0 (+~12 dist) | ~145–200 (c17→110, +c12 2–7) | ~3–5 | EXTRAP |
| tcga_luad_vs_lusc | 1017 | tcga | 0 (+~12 dist) | ~180–250 (c17→139) | ~3–5 | EXTRAP |
| herg_ki | 1412 | mol | 0 (+~40 dist) | ~260–360 (c17→198) | ~3–5 | EXTRAP |
| caco2 | 2158 | mol | 0 on disk; **re-running cache-drift ~+1h cx/+25h sim** | ~400–560 (c17→314) | ~3–5 | EXTRAP |
| adme_mdr1 | 2641 | mol | 0; **+7h cx/+35h sim drag** | ~500–700 (c17→389) | ~3–5 | EXTRAP |
| adme_rlm | 3053 | mol | 0; **+43h cx/+170h sim drag** | ~590–820 (c17→455) | ~3–5 | EXTRAP |
| adme_hlm | 3086 | mol | 0; **+22h cx/+73h sim drag** | ~600–830 (c17→461) | ~3–5 | EXTRAP |
| buchwald_hartwig | 3955 | rxn | 0 (+~12 dist) | ~780–1080 (c17→602) | ~15–25 | EXTRAP |
| flab_binding_koenig | 4275 | ab | **top/uncapped ~7h** (90.8% done) | ~850–1180 | ~15–25 | EXTRAP |
| flab_expression_koenig | 4275 | ab | 0 (+~30 dist) | ~850–1180 | ~15–25 | EXTRAP |
| bh_doyle_plates | 4312 | rxn | 0 (+~50 dist) | ~860–1190 | ~15–25 | EXTRAP |
| flab_binding_adams/phillips_h3, flab_expression_adams, tz_her2_* | 5000 | ab | 0 | ~1000–1400 | ~15–25 | EXTRAP |
| logd | 5028 | mol | **~460** (gen/unc+top/none+top/unc missing) | ~1000–1400 (c17→777) | ~15–25 | EXTRAP |
| nt_promoter_tata | 5274 | nuc | **~29** (top/uncapped) | ≫wall | ~15–25 | EXTRAP |
| herg | 7772 | mol | **~660** (gen/unc+top/none+top/unc) | ~1600–2250 (c17→1247) | ~15–25 | EXTRAP |
| qm9 | 9978 | mol | **~450** (gen/unc+top/unc) | ~2100–2900 (c17→1632) | ~15–25 | EXTRAP |
| nt_H3K4me3 | 18244 | nuc | ~46 | ≫wall | ~150–225 (n², OOM-risk) | EXTRAP |
| gb_human_enhancers_cohn | 27791 | nuc | ~0 (small) | ≫wall | ~150–225 | EXTRAP |
| bindingdb_kd | 30000 | dta | **~2165** (gen/none was 96.23h) | ≫wall | ~150–225 | EXTRAP |
| kiba | 30000 | dta | **~2741** (gen/none was 121.83h) | ≫wall | ~150–225 | EXTRAP |
| gb_human_nontata_promoters / gb_demo_* | 36k–100k | nuc | small–ALL5 | ≫wall | ~150–225 | EXTRAP |

**Molecular cache-drift drag total: ~73h cx + ~303h sim of pure recompute across the 4 jobs** — the single largest recoverable sink, removed only by the cache fix + restart.

## SECTION 2 — ACTIVELY-BLOCKED datasets (loop-position method; no log tail = "dead" is almost-certain not confirmed)
**(a) DEAD / stalled (idle ≫ plausible wall):** ALL sim jobs idle at `[S02 1done]` (s02=cache-reuse, should be minutes) — herg_ki 10.1h, ddg_combined 51.6h, flab_thermo_tresanco 41.8h, abagym_lysozyme 67.8h, tcga_luad_vs_lusc 50.1h, tcga_brca_pam50 24.8h. cx jobs idle ≫ c05 wall — lysozyme C05 67.4h, flab_aggregation_jain C05 56.5h, ddg C05 52.5h, tcga_luad C05 51.5h, +9 more. DTA both stalled at precompute boundary. → all DEAD.
**(b) RUNNING but WON'T FINISH in ~100h:** herg (~660h left), logd (~460h), qm9 (~450h) — each single missing uncapped condition alone >168h partition; all molecular/reaction/large-ab/TCGA cx never started; DTA cx/sim never started; nucleotide ~40-job queue still PENDING. HIGH confidence on verdict.
**(c) RUNNING FINE:** abagym_Ang2 (C21, 3 recent saves), abagym_VEGF (C21), humanness_therapeutic (C13), abagym_EGFR (C05) — the closest to finishing.
**(d) DONE:** precompute 38/65 ready; 5 complete cx pipelines (bvp/cic/HER2 full; Ang2/VEGF through c19). Caveat: "DONE on disk" ≠ "not re-running" for the 6-neural-rep molecular/DTA/reaction under cache drift.

## SECTION 3 — LEVER vs ACTIVELY-BLOCKED (does it help the blocked set? + code state)
1. **GP cap 1000→250** — NOT applied. **Helps NONE** — GP is gated OFF at n>1000, so contributes 0 to every blocked large dataset; it's 28–72% only on small antibody that already finish.
2. **c23 extra-tier gate** — ALREADY in running wave (04-02). Helps none further (already active for blocked; blocked cx never starts anyway).
3. **Capped cuts** — capped ALREADY gated to n≤250 in running wave (06-01). **Helps NONE** — every blocked dataset is n≥397, so capped is already skipped for all of them.
4. **c17 MI row-subsample to ~2000** — NOT WRITTEN. **The single biggest cx lever** — makes c17 ∝const not ∝n (c17@9978 1632h→~310h-equiv; @30k 5400h→~310h). Helps all cx-blocked molecular/reaction/large/DTA/nucleotide — but ONLY after their precompute blockers clear (cx hasn't started).
5. **c17 mRMR once-slice** — applied in working tree, NOT in running wave. On restart: mRMR ÷4–8, ~100–250h/molecular. Compounds with #4 (weaker — cuts budget-loop redundancy, not the n-dependence).
6. **s01 OOM guard (n>15000)** — applied in working tree, NOT in running wave. Helps sim of DTA/nucleotide (n≥15k) — converts an ~35GB/rep OOM crash into a recorded skip. Helps nothing ≤15k; doesn't touch the n² s03/s05/s10 that also block those sims.
7. **s03 kNN-MI subsample** — NOT WRITTEN. Helps sim of n≥30k blocked (s03 ~111h@30k is the biggest sim cost). Helps nothing ≤5000.
8. **screening→FULL** — NOT applied in running wave (committed in 3cadb95 working tree). ⚠️ **CORRECTED FRAMING (user, 2026-06-04): screening budgets were NEVER intended for publication. FULL is a PUBLICATION REQUIREMENT, not an optional throughput lever.** The earlier "cost-increase / harmful" wording applied the wrong lens. The FULL cost IS the real cost of a publishable result; you canNOT "rescue" a blocked dataset by keeping screening, because a screening result (budget-biased best-budget over only 4 points) is unpublishable — running a blocked dataset cheaply at screening just yields a throwaway number. Implication: **the blocked datasets' true (publishable) cost is the FULL cost — which makes the blocked-set problem MORE severe, not optimizable away.** Magnitude note: the PRECOMPUTE already runs FULL regardless (it ignores config['budgets']); screening only coarsened the internal budget sweep of 7 experiments (c06/c07/c09/c10/c15/c18/s03), so the added cost is those experiments' extra downstream model-fits, NOT a precompute rebuild. The "+20–40%" figure (ESTIMATES III.0) is an upper bound on those 7 experiments' own search cost, not the whole wave; the helpers-agent read it as "modest." Either way it is REQUIRED, not weighed against throughput.
9. **expensive-first loop ordering** — partial (the committed flag reorders conditions, not inner budget/model loops). Helps the blocked PRECOMPUTE set as fail-fast/completes-or-rules-out + lets per-condition arrays each take a partition; does NOT reduce total work (herg gen/uncapped ~302h still >168h regardless of order).
10. **c05 full-data dedup** — 1.0-fraction dedup applied in working tree, NOT in wave. On restart: removes 2 of 3 full uncapped subsample searches (~90h caco2-scale); n_repeats 3→1 on other fractions (not applied) = further ÷3. Helps cx-blocked molecular/reaction/large (only after precompute clears).
11. **CACHE-1 churn fix** — applied in working tree, NOT in wave, UNTESTED on server. **The single largest help to the blocked set — restart-gated.** Stops the 4 molecular jobs re-computing on-disk precompute (~376h: 73h cx + 303h sim). Scope = 6-neural-rep datasets (molecular+DTA+reaction); does NOT help antibody/TCGA/nucleotide (deterministic reps, low drift). NEW hazard: value-only rep changes (Graphinity) won't auto-invalidate → manual purge.
12. **Sequence cx-before-sim** — organizational. **Cleanest pure win for the blocked precompute set** — per the snapshot every dataset runs cx & sim concurrently, doubling the shared precompute; sequencing ≈halves precompute for the large blocked datasets. Compounds with #11.

**Levers that HELP the blocked set:** 4 (biggest, not written), 5, 10, 11, 12 (restart-gated), 6+7 (sim only, large-n), 9 (scheduling). **Help NONE:** 1, 2, 3, 8 (8 is a cost increase). **#1 actual precompute blocker for herg/logd/qm9 = SVM-uncapped O(n²) at large n** — a model gate there (SVM `_max_samples` 10000→lower at large n) was NOT in the user's 12 levers but is the real killer; flag for discussion.

## Could NOT verify (flagged)
No server access → can't confirm dead-vs-slow (no log tail), distance-precompute state (zero `*_distance` rows), or that the staged fixes pass on server (cache fix UNTESTED). Zero finished sim pipeline → every sim cell + sim lever is STRUCTURAL/UNVALIDATED. No cx measured above n=418 → all cx ≥564 is EXTRAPOLATED (banned-form, order-of-magnitude). `total_dim` per dataset not in code = the master unlock (item 1.1). Large-partial precompute hours (herg ~660, DTA ~2165–2741) rest on the none→uncapped multiplier (biased LOW where SVM O(n²) grows) — ordering + "won't finish" verdict HIGH-confidence, specific hours SOFT.

**BOTTOM LINE (verdict HIGH-confidence, hours soft):** only small/mid antibody ≤~670 that are still ALIVE (Ang2, VEGF, EGFR, humanness, surviving small ones) can finish cx in ~100h. Everything n≥~1400 will not finish — precompute incomplete and/or cx never started, and much of the fleet is already dead (Section 2a) or wasting cycles on cache-drift recompute. **The two highest-leverage interventions are both restart-gated: deploy the cache fix + restart molecular (~376h recovered), and sequence cx-before-sim (≈halve large-dataset precompute). But a restart abandons the ~68h of alive progress (Ang2/VEGF) — that trade-off is yours.**
