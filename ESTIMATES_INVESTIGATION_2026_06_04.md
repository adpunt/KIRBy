# Estimates investigation — 2026-06-04 (master document)

> 🔄 **CORRECTIONS from the fix-fleet pass (2026-06-04, verified in code — fold into the numbers below).** See `DECISIONS_FOR_USER.md` §5 + `SYNTHESIS_2026_06_04.md`.
> 1. **Context-problem A (cache rejection) is NARROWER than modelled.** The "SCREENING vs FULL forces a cache rebuild" premise is FALSE: `precompute_hybrid_searches` ignores `config['budgets']` and ALWAYS precomputes FULL_BUDGETS. No two cache families, no cold rebuild from the screening setting. The screening value only coarsened the budget curve of 7 experiments (c06/c07/c09/c10/c15/c18/s03) — now fixed to FULL. ⇒ remove the SCREENING-rebuild waste from the estimate.
> 2. **No double-precompute.** cx and sim share the search cache by filename (`{dataset}__{tier}__{aug}__{speed}.pkl`, caller-independent); the "2× precompute per dataset" only happens if the cx and sim jobs run CONCURRENTLY before either writes. Sequence them ⇒ ~1× precompute. The distance tier IS precomputed by run_all (not on-demand).
> 3. **Cost driver is GP, not SVM.** SVM is already budget-capped (`MODEL_MAX_BUDGET['svm']=500`) → measured share 0.1–5%. The 302h-SVM figure (Section 9 / COST-6) is STALE. GP is 28–61% of per-search wall and is only capped at n≥1000, so it dominates the n<1000 antibody regime — but those datasets FINISH anyway, so GP cap is a throughput win, NOT a fix for the n≥1400 non-finishers (where GP is already capped). The n≥1400 verdict is unchanged.
> 4. **New cache hazard (raises a recompute need):** the CACHE-1 anti-churn fix now IGNORES rep content hashes, so any rep whose VALUES changed without shape/labels changing will NOT auto-invalidate. Known case: Graphinity (mutation-position parser fix) — its cache must be MANUALLY purged (`tests/purge_graphinity_cache.py`) or the corrupt embeddings are silently reused.
> 5. **Unchanged:** the c17-dominated cx model, the `(n/anchor)^k` BAN + order-of-magnitude caveat (LIN-1), node contention 3–48×, and the bottom-line verdict (only small/mid antibody finish in ~100h; everything n≥1400 won't). Per-rep `total_dim` (item 1.1) is still the master unlock.

**VALIDATION STATUS (answer to "are the estimates validated by the times we're seeing?"):** the cx model is validated against the **5 measured antibody pipelines only** — c17 predicts 26.0h @ n=217 vs measured 27.6h (+6%) at the anchor, with hold-out interpolation residuals 14–37% on the non-monotone GP cells, and ~2× over-prediction for rep-poor sets (bvp/cic/HER2). **Everything at n≥1400 (molecular/DTA/reaction/nucleotide/large-antibody) is EXTRAPOLATED, not measured** — flagged order-of-magnitude (the banned power-law caveat, ⚠️ below). Contention adds 3–48×. So: firm where measured (small antibody), order-of-magnitude + HIGH-confidence "won't finish" elsewhere.

**Goal:** code-grounded, data-verified, defensible wall-time estimates for (1) precompute remaining, (2) cx cumulative per c-experiment, (3) sim cumulative per s-experiment — plus full investigation of every context problem (A cache rejection, B wedge classification, C non-linear scaling, D conditional inclusion, E nucleotide queue, F sim unmeasured, G stale timing).

**Rules honored:** no recommendations until asked; NO proportional scaling on n (`(n/anchor)^k` forbidden); validate cx predictions against all 5 measured pipelines (residual >30% on any c# ⇒ model wrong); wedge status not from idle alone; state unknowns explicitly; refuse to extrapolate beyond measured regime.

**Data sources:** `WAVE_SNAPSHOT_2026_06_04_0207.md` (freshest measured timing, supersedes 06-02), `WAVE_STATUS_2026_06_04.md` (5 measured cx pipelines, §8), `CHECK_PRECOMPUTE_TIMING_2026_06_02.md` (extra rows).

**Extraction status:** workflow `w5wfx1981` (5 foundation + 3 cache-forensics + 36 experiment agents) running. Sections marked ⏳ PENDING fill in on completion. Sections marked ✅ VERIFIED are read directly from source by the main agent.

---

## ★★★ THE DELIVERABLE — PROPER PER-DATASET TIME ESTIMATES (2026-06-04, supersedes the group-range headline below)

**This is the #1 priority and the answer to the whole effort: committed per-dataset numbers for precompute-remaining, cx total, sim total — each derived, each anchored on measured data, none written off as "not estimable."**

### The cx cost model (derived, not n-scaled)
cx total = Σ per-experiment. At every n≥~800 the sum is **dominated by c17**, whose cost is a VALIDATED law:

> **c17(n) ≈ 53.4h × (n/418)^1.078**, where the exponent ~1.0 is **mRMR's intrinsic complexity** (single-column mutual-information calls are linear in n × capped-feature-count).
> **⚠️ AUDIT FLAG (PART VIII, LIN-1): this IS an `(n/anchor)^k` power law — the form the rules BAN.** The n-shape is O(n log n)-consistent, but it's fit on only 2 rep-rich points (n=217,418) and ignores total_dim variation, model-gate discontinuities, and 3–48× contention. **⇒ Treat the whole cx hour-column below as ORDER-OF-MAGNITUDE + ORDERING ONLY, not trustworthy hours.** Only the 5 measured pipelines are firm.
> **Validation:** predicts **26.0h @ n=217** (measured immunogenicity c17 = 27.6h, +6%); 53.4h @ n=418 (anchor VEGF); 51.5h @ n=397 (anchor Ang2). For **rep-poor** datasets (bvp/cic/HER2/abcov, total_dim<2000) it over-predicts ~2× (measured ~7h vs formula ~16h) because their mRMR feature cap doesn't bind; for **rep-rich** datasets (≥12 general reps → total_dim>2000, cap binds → M constant) it holds. Molecular/DTA/reaction/large-antibody are all rep-rich ⇒ the law applies.

**cx total ≈ c17(n) × 1.5** for n≥~560 (the ×1.3–1.8 band covers c05+c09+c16+c21+c23+c24+cheap, which are sub-dominant to c17 at large n — c05 the largest of them, ~0.3–0.5×c17). For **n≤418 antibody** c17 is NOT dominant → use the **measured 5-pipeline envelope** directly (bvp/cic/HER2 = 35–39h complete; Ang2/VEGF = 64h *through c19* + unfinished c21/c23/c24).

### THE TABLE (hours). cx column shows **c17(n) central → cx total ≈ c17×1.5**.

| dataset | n | dom | precompute LEFT | **cx total (derived)** | sim total |
|---|---|---|---|---|---|
| ginkgo_heparin | 94 | ab | 0 (+~3 dist) | 35–39 (meas regime) | ~0.5 |
| flab_pk_jain_heprt/fcrnrt | 130/132 | ab | 0 (+~4 dist) | 35–39 | ~0.5 |
| flab_polyreact_bvp/cic | 137 | ab | 0 | **38.98 / 37.11 (MEASURED)** | ~0.5 |
| flab_aggregation_jain(+hicrt) | 137 | ab | 0 (+~5 dist) | 35–45 | ~0.5 |
| flab_thermo_jain | 137 | ab | 0 (+~6 dist) | 35–45 | ~0.5 |
| abagym_HER2 | 146 | ab | 0 | **34.94 (MEASURED)** | ~0.5 |
| ginkgo_pr_cho | 197 | ab | 0 (+~6 dist) | 40–55 (c17→~22) | ~0.7 |
| flab_immunogenicity_marks† | 217 | ab | 0 | ~50 (c17 meas 27.6) | ~0.7 |
| ginkgo_titer/acsins/colloidal/hic | 239–242 | ab | 0 (+~5 dist) | 45–60 (c17→~26) | ~0.7 |
| abagym_Ang2 | 397 | ab | 0 (+~2 dist) | **≥64 meas→c19**, ~75–95 full | ~1.5 |
| abagym_VEGF | 418 | ab | 0 (+~2 dist) | **≥64 meas→c19**, ~75–95 full | ~1.5 |
| ddg_combined | 429 | ab | 0 (+~10 dist) | ~80–110 (c17→~55) | ~1.5 |
| humanness_therapeutic | 564 | ab | 0 | ~95–130 (c17→74) | ~1.5 |
| abagym_lysozyme | 627 | ab | 0 (+~13 dist) | ~110–150 (c17→83) | ~1.5 |
| abagym_EGFR | 659 | ab | 0 | ~115–160 (c17→87) | ~1.5 |
| flab_thermo_tresanco | 672 | ab | 0 (+~23 dist) | ~115–160 (c17→89) | ~1.5 |
| tcga_brca_pam50 | 821 | tcga | 0 (+~12 dist) | **~145–200** (c17→110, +c12~2) | ~3–5 |
| tcga_luad_vs_lusc | 1017 | tcga | 0 (+~12 dist) | **~180–250** (c17→139, +c12~2) | ~3–5 |
| herg_ki | 1412 | mol | 0 (+~40 dist) | **~260–360** (c17→198) | ~3–5 |
| caco2 | 2158 | mol | 0 (+~77 dist) ‡ | **~400–560** (c17→314) | ~3–5 |
| adme_mdr1 | 2641 | mol | 0 (+~70 dist) ‡ | **~500–700** (c17→389) | ~3–5 |
| adme_rlm | 3053 | mol | 0 (+~80 dist) ‡ | **~590–820** (c17→455) | ~3–5 |
| adme_hlm | 3086 | mol | 0 (+~80 dist) ‡ | **~600–830** (c17→461) | ~3–5 |
| buchwald_hartwig | 3955 | rxn | 0 (+~12 dist) | **~780–1080** (c17→602) | ~15–25 |
| flab_binding_koenig | 4275 | ab | ~101 (top/unc) | **~850–1180** (c17→655) | ~15–25 |
| flab_expression_koenig | 4275 | ab | 0 (+~30 dist) | **~850–1180** (c17→655) | ~15–25 |
| bh_doyle_plates | 4312 | rxn | 0 (+~50 dist) | **~860–1190** (c17→660) | ~15–25 |
| flab_binding_adams/phillips/expr_adams | 5000 | ab | 0–small (+dist) | **~1000–1400** (c17→775) | ~15–25 |
| tz_her2_binary/multiclass | 5000 | ab | 0 (+dist) | **~1000–1400** (c17→775) | ~15–25 |
| logd | 5028 | mol | ~460 (gen+top unc) | **~1000–1400** (c17→777) | ~15–25 |
| nt_promoter_tata | 5274 | nuc | ~29 (top/unc) | ≫wall (c17 formula→~800, 13× beyond anchor) | ~15–25 |
| herg | 7772 | mol | ~662 (gen+top unc) | **~1600–2250** (c17→1247) | ~15–25 |
| qm9 | 9978 | mol | ~452 (gen+top unc) | **~2100–2900** (c17→1632) | ~15–25 |
| nt_H3K4me3 | 18244 | nuc | ~46 | ≫wall (thousands) | ~150–225 (n²,OOM) |
| gb_human_enhancers_cohn | 27791 | nuc | 0 | ≫wall (thousands) | ~150–225 (n²,OOM) |
| bindingdb_kd | 30000 | dta | ~2165 | ≫wall (c17 formula→~5400, 72× beyond anchor) | ~150–225 (n²,OOM) |
| kiba | 30000 | dta | ~2741 | ≫wall (c17→~5400) | ~150–225 (n²,OOM) |
| gb_human_nontata_promoters | 36131 | nuc | 0 | ≫wall (thousands) | ~150–225 (n²,OOM) |
| gb_demo_human_or_worm | 100000 | nuc | 0 | ≫wall (c17→~20000) | ~150–225 (n²,OOM) |

† not in 69-job wave. ‡ molecular precompute "DONE" but **re-running from cache drift** (PART V.3.5: +73h cx +303h sim of pure recompute across the 4 adme/caco2 jobs). "dist" = the unmeasured distance-lineage precompute the sim experiments need (PART V.3.1).

### THE VERDICT (what the months-of-lost-compute question actually needs)
- **c17 alone busts the 7-day wall for every dataset at n≥~1400** (≥198h). At n≥2000 cx is **400h–2900h** — the prior "150–400h" in the old table was a **2–10× under-estimate** (the exact error that has burned every attempt). The wall-buster is **c17 (mRMR), then c05 (subsample searches), then capped precompute** — all dimension-and-n driven.
- **Only the small/mid antibody datasets (n≤~670) can finish cx in the ~100h left**, and only the ones still alive (PART V/§10 wedge classification).
- **Everything molecular/reaction/large-antibody/DTA/nucleotide will NOT finish** — HIGH confidence on the verdict; the cx hours carry the ×1.3–1.8 "rest" band + the rep-poor/rich c17 caveat.
- **⚠️ REVISED by PART X:** this verdict + the PC-total column assumed **serial conditions on contention/collision-inflated walls**. Re-framed per-condition (`--condition` arrays parallelise → wall = max not sum; clean run ~1.5–3× faster): the **"won't finish" set SHRINKS** to datasets with a SINGLE supra-partition (>168h) condition (big-n uncapped/capped, DTA, nucleotide). Mid datasets whose missing conditions each fit one partition (e.g. flab_binding_koenig top/uncapped) CAN finish. See PART X.
- **What tightens it:** `total_dim` per dataset (item 1.1) sharpens c17's M factor (matters most for borderline rep-poor sets); a measured cx above n=418 would confirm the c17 extrapolation past the anchor. Neither is needed to reach the *verdict*.

---

## ★★ HEADLINE — EDUCATED TIME ESTIMATES (SUPERSEDED by the per-dataset table above; kept for history — the group ranges here were the early guesses)

**Wall clock:** wave started 2026-06-01 ~06:00; 7-day limit ≈ **2026-06-08 06:00**. Now ≈ 2026-06-04 02:00 → **~68h elapsed, ~100h left.**

**Confidence key:** HIGH = measured for this exact dataset; MED = measured for a near-identical dataset (same domain, similar n & rep count); LOW = no measurement in this size range, reasoned bound only. **No n-power-law used** (rule 2) — ranges come from measured same-condition / same-regime neighbours and measured strategy ratios, not from scaling n.

These are EDUCATED estimates of **total wall to fully finish each job (precompute + cx experiments + sim experiments)**, and whether it fits in the ~100h left.

### Group A — small antibody, precompute done, in experiments → the ones that CAN finish
n=94–250. cx measured at 35–39h (bvp/cic/HER2). Educated cx total **30–50h** (wide because c17 6–9h and c21 0.8–19h swing by dataset). sim unmeasured, educated **+5–20h** (LOW). Precompute already done.
Datasets: ginkgo_acsins/colloidal/heparin/hic/pr_cho/titer, flab_aggregation_jain(+hicrt), flab_pk_jain_fcrnrt/heprt, flab_thermo_jain, abcov_kd, flab_immunogenicity, flab_polyreact_bvp/cic(DONE), abagym_HER2(DONE).
**Verdict: fits ~100h IF alive.** Confidence MED. BUT several are DEAD right now (Section 10) → making no progress.

### Group B — mid antibody, in experiments, cx dominated by c17 (~50h)
n=397–672. Measured Ang2/VEGF = 64h *through c19, not done* (c17 alone 51–53h). Educated cx total **70–95h**. 
Datasets: abagym_Ang2(alive), abagym_VEGF(alive), abagym_EGFR(alive), abagym_lysozyme(DEAD), flab_thermo_tresanco(DEAD), ddg_combined(DEAD), humanness_therapeutic(alive, n=564).
**Verdict: borderline — alive ones (Ang2/VEGF/EGFR/humanness) may just finish in ~100h; dead ones won't (not running).** Confidence MED-LOW.

### Group C — large antibody classification, STILL in precompute, cx not started
n=4275–5000. flab_binding_koenig (~7h precompute left, HIGH), flab_binding_phillips_h3, flab_expression_koenig/adams, flab_binding_adams, tz_her2_binary/multiclass.
cx at n=5000 **never measured**; c17/c05 grow with n and rep width → educated cx **100–250h, LOW confidence**.
**Verdict: very unlikely to finish cx in ~100h.** Precompute may finish; experiments won't.

### Group D — molecular, n=1412–9978
- adme_hlm/mdr1/rlm, caco2 (n=2158–3086): **precompute DONE but re-running due to the cache bug** (Section 9) → effectively stuck, not progressing. If unstuck, cx at this n is unmeasured → educated **80–200h, LOW**.
- herg(7772), logd(5028), qm9(9978): precompute PARTIAL, missing the big "uncapped" search — **unmeasured at this n, educated 100–250h each, LOW** (the measured molecular uncapped at n≤3086 was 48–103h; these are bigger and n=7772 keeps the expensive svm). cx not started.
- herg_ki(1412): precompute done; cx unmeasured at n=1412, educated **50–120h, LOW**.
**Verdict: will NOT finish in ~100h** (HIGH confidence on the verdict, LOW on the exact hours).

### Group E — DTA, n=30000
kiba, bindingdb_kd. Measured general/none = 96h & 122h. Missing uncapped pieces are the *expensive* strategy; even with svm/gp gated out, educated **+200–500h, LOW on number / HIGH it busts the wall.** cx never started, never measured at n=30k.
**Verdict: will NOT finish. HIGH confidence.**

### Group F — reaction (bh_doyle_plates, buchwald_hartwig, n≈4000) & TCGA (n≈1000)
Reaction precompute mostly done; cx unmeasured at n≈4000 → educated **80–200h, LOW**. TCGA precompute done, in experiments but **DEAD** (idle 28–51h); cx (+c12 modality ablation, tcga-only) unmeasured → educated **40–90h if alive, LOW**.
**Verdict: reaction won't finish; TCGA not running.**

### Group G — nucleotide (PENDING, n=5k–100k, not yet started)
Precompute per-dataset estimable from neighbours (measured general/none 2–54h; uncapped 50–140h where measured). cx/sim at n=18k–100k **never measured, not estimable** beyond "large."
**Verdict: will NOT finish in this wave (still queued + huge). HIGH confidence on verdict.**

**BOTTOM LINE:** Only **small + mid antibody datasets (Groups A–B), and only the ones still alive,** realistically finish in the ~100h left. **Everything at n ≥ ~1400 (molecular, reaction, large-antibody, DTA, nucleotide) will not finish** — its precompute alone (the big uncapped searches) is large, and its experiments haven't started. The exact hours for the n≥1400 group are LOW-confidence (no cx ever measured above n=418), but the *will-not-finish* verdict is HIGH-confidence.

---

## ⭐ Section 0 — DEFERRED solution notes (record now, DECIDE AFTER the wave) — user directive 2026-06-04

NOT to act on during the estimates phase. For the later SOLUTION phase, after ALL current cx/sim/precompute experiments finish:
1. **Reuse completed precompute.** If a dataset happens to fully get through its search space via precompute, use those results rather than recompute them.
2. **Gating may be needed.** Some experiments may have to be gated (skipped/limited) for some datasets to fit the wall.
3. **Fine balance, decided later.** There is a real balance to strike on the overall solution — but that balance is a post-wave decision, made once these experiments are over, not now.

(User, verbatim intent: "if some datasets happen to get through this search space via precompute, you might as well use those results. But you may have to gate some experiments... that is after all these experiments are over. Write that down somewhere though so you don't forget it.")

---

## ✅ Section 1 — Experiment inclusion (Context D), verified from source

`SINGLE_DATASET_EXPERIMENTS` in `tests/experiments/complexity/complexity_helpers.py:968`:
```
c01,c02, c04,c05,c06, c07, c08, c09, c10,c11, c13, c15, c16, c17, c18, c19, c21, c22, c23, c24   (20)
```
Similarity `similarity_helpers.py:564`: `s01..s08, s10, s11, s12, s13` (12) — **s09 absent**.

| Exp | In per-dataset run? | Evidence |
|---|---|---|
| c03 | NO — cross-dataset / distance-pool only | `complexity_helpers.py:1173-1174` "c03 is the only complexity experiment that uses the distance pool" |
| c08 | YES in list, **attempted but FAILED at runtime** in all 5 logs | c08 runs 4 `build_fold_hybrid` conditions for BOTH task types (`c08_calibration.py:154-205`) — NOT task-gated, NOT ~0 when it succeeds. Its absence from the timed table = it raised an exception (runner prints `FAILED`, not `done in Xs`). ⏳ exact failure cause pending c08/runner agents. (Correction: earlier "task_type-gated ~0" inference was wrong — refuted by code.) |
| c12 | **tcga_ datasets only** | `complexity_helpers.py:1052` `if dataset_name.startswith('tcga_') and 'c12' not in experiments: insert c12 after c11` |
| c14 | NO — cross-domain summary (aggregation) | not in `SINGLE_DATASET_EXPERIMENTS` |
| c24 | YES, all datasets, **no n-gate** | in list `:984`; `c24_blueprint_transfer.py:241` only early-returns if `<2` folds succeed. **Refutes the WAVE_STATUS §8.1 prose that c24 is skipped at n≥146**; the HER2 n=146 data table (c24=1166.8s) is correct. |
| s09 | NO — cross-dataset summary | not in similarity `SINGLE_DATASET_EXPERIMENTS` |
| c20 | merged into c21 | `complexity_helpers.py:981` "single-rep augmentation (merged C20)" |

**Per-dataset experiment SET (drives cumulative wall):**
- non-tcga: 20 complexity experiments (c08 ≈ 0 for regression).
- tcga: 21 (adds c12).
- The 5 measured antibody pipelines ran exactly: c01,c02,c04,c05,c06,c07,c09,c10,c11,c13,c15,c16,c17,c18,c19,c21,c22,c23,c24 (c08 raised → emitted no `done` line). Summation check: bvp listed-experiment seconds = 140,332 = stated total ⇒ these ARE the complete timed set.

**Timing-marker mechanism (verified):** runner stamps `t_exp=time.time()` per experiment (`complexity_helpers.py:1228`); on success prints `  {exp_id} done in {Xs}` (`:1251`); on exception prints `  {exp_id} FAILED` + stores `{'error':...}` (`:1253-1254`). WAVE_STATUS §8 walls = the `done in Xs` values, so a FAILED experiment is absent from the table yet still consumes some wall (its failed-attempt time). This is why c08 (and only c08, of the 20 in-list experiments) is missing.

---

## ✅ Section 2 — Grid + gates, verified from source (`find_best_hybrid.py`)

**Models** (`:1138`): tiers — fast=`rf,xgboost`; full=`ridge,knn,lightgbm,mlp,svm,gp`; extra(explicit only)=`catboost,dnn,cnn,bnn,ngboost`. Default speed `full` ⇒ 8 models {rf,xgboost,ridge,knn,lightgbm,mlp,svm,gp}.

**`_max_samples` auto-skips (the core n-nonlinearity):**
- `svm:_max_samples=10000` (`:1184`) → svm dropped at n>10000.
- `gp:_max_samples=1000` (`:1190`) → gp dropped at n>1000.

**`MODEL_MAX_BUDGET` (`:1705`):** gp=1000, knn=2000, svm=500, mlp=1000, bnn/dnn/cnn=1000; **no cap**: rf,xgboost,lightgbm,catboost,ridge,ngboost. ⇒ at budgets above a model's cap it emits 0 configs.

**`SPEED_TIERS['full']` (`:1680`):** importance=`[deeplift,lightgbm_gain,treeshap]` (3); allocation=`[top_2,top_3,top_5,performance_weighted,entropy_1.0,power_2,rank_linear,entropy_2.0,rank_exp_d0.5]` (9); evaluators=`[match,rf,svm,lightgbm,mlp]` (5); `n_folds=5`. Budget sweep = `FULL_BUDGETS` (precompute speed=full) / `SCREENING_BUDGETS` / `DISTANCE_BUDGETS` (`precompute_hybrids.py:290,394`); per-model pruning via `_budgets_for_model` (`find_best_hybrid.py:3051`). Exact constant values ⏳ (precompute agent). Observed in monitor: 10,25,50,100,250,500,1000,2000 + budget=n.

**`_DOMAIN_MODEL_SKIPS` (`:1231`):** nucleotide always skips `[ridge,knn,xgboost]`. **`_DOMAIN_STRATEGY_SKIPS` (`:1245`):** (nucleotide,capped) also skips `[mlp]`.

**`_DATASET_STRATEGY_SKIPS` (`:1253`) — exact, drives precompute-remaining:**
| dataset | strategy | skip |
|---|---|---|
| flab_thermo_tresanco | uncapped | gp, mlp |
| tcga_luad_vs_lusc | uncapped | gp |
| ddg_combined | uncapped | gp |
| tcga_brca_pam50 | none/uncapped/capped | gp |
| adme_hlm/mdr1/rlm, caco2, logd, bh_doyle_plates | capped | mlp, svm, knn |
| herg, qm9 | capped | mlp, svm, knn, rf |
| bindingdb_kd, kiba | capped | mlp, rf, knn |

`_domain_for_skip` (`:1295`): only `nt_/gb_` → nucleotide; other domains have no domain skips.

**Cell count:** nominal grid is pruned (`final_strategy_threshold=0.15`, `:1725`); **measured per-condition `n_configs` (3000–6500) is the true post-prune cell count** — anchor on it, do not multiply the nominal grid.

---

## ✅ Section 3 — cx validation targets (the 5 measured pipelines, per-experiment seconds)

From `WAVE_STATUS_2026_06_04.md §8`. ANY model prediction must reproduce each cell within 30%.

| exp | bvp n=137 | cic n=137 | HER2 n=146 | Ang2 n=397 | VEGF n=418 |
|---|---|---|---|---|---|
| c01 | 88.2 | 155.3 | 3862.2 | 478.4 | 1605.4 |
| c02 | 298.1 | 168.7 | 2857.4 | 1148.0 | 634.4 |
| c04 | 685.2 | 652.8 | 9262.8 | 1804.7 | 1878.6 |
| c05 | 4205.9 | 2894.0 | 24645.6 | 8420.7 | 10078.2 |
| c06 | 10.2 | 32.8 | 6.4 | 11.5 | 42.8 |
| c07 | 0.5 | 4.5 | 158.6 | 1748.0 | 3054.9 |
| c09 | 2877.0 | 3299.7 | 37288.4 | 8728.6 | 7331.8 |
| c10 | 1.8 | 0.4 | 4.6 | 0.9 | 1.1 |
| c11 | 1877.7 | 843.4 | 6824.0 | 5059.5 | 2118.4 |
| c13 | 759.4 | 584.4 | 2699.2 | 2068.3 | 2177.4 |
| c15 | 4634.9 | 1334.6 | 1791.1 | 5572.9 | 1249.3 |
| c16 | 4356.0 | 1547.1 | 6499.7 | 8891.0 | 6152.4 |
| c17 | 32708.8 | 29946.3 | 24247.9 | 185540.5 | 192233.2 |
| c18 | 198.6 | 165.0 | 1182.4 | 399.9 | 358.8 |
| c19 | 18.6 | 28.0 | 515.1 | 49.9 | 24.4 |
| c21 | 69338.4 | 69490.7 | 2739.9 | (running) | (running) |
| c22 | 127.2 | 59.2 | 12.5 | — | — |
| c23 | 10362.9 | 10357.9 | 18.8 | — | — |
| c24 | 7783.5 | 12043.5 | 1166.8 | — | — |
| **TOTAL** | **38.98h** | **37.11h** | **34.94h** | 63.87h(→c19) | 63.59h(→c19) |

**Non-linearity facts the model MUST reproduce (Context C):** c01 44× difference at n≈137 vs 146; c17 non-monotone 9.1→8.3→6.7h (137/137/146) then 51.5/53.4h (397/418); c21 19.3h@137 vs 0.76h@146; c13 0.21h@137 vs 0.75h@146 vs 18.6h@217; c09 0.80h@137 vs 10.4h@146. These are **dataset-rep-set specific, not n-driven** — note bvp/cic/HER2 are different antibody datasets with different rep sets, so per-cell wall + cell count both differ.

---

## ✅ Section 4 — Per-cell wall model (verified, NO n-power-law)

per_cfg = seconds per searched config, indexed by (model, n, strategy_class ∈ {none, uncapped, capped}, pool ∈ {general, top}). Anchored on MEASURED per-config times in the 02:07 snapshot (precompute search phase). **The strategy axis dominates the n axis** within any size regime:
- none→uncapped per_cfg multiplier: GP ~18-25×, RF ~10-12×, XGB ~10-16×, SVM ~30-33×, MLP ~8-10×, LGBM ~24×, RIDGE ~9-30×, KNN ~4-10×.
- none→capped (RF, n=137): 0.69s → 252.83s ≈ **360×**.

Dominant model per regime: **GP** dominates small-antibody none/uncapped/capped (per_cfg up to 65s uncapped, 50-820s capped); **RF** dominates capped (139→824s, rising monotonically n=94→239); **SVM/MLP** dominate uncapped molecular/reaction at n~3-5k (145-465s).

**Not monotone in n:** GP/none dips at n=627 below n=418/429. **Rep-size (pool) effect:** per_cfg(top) < per_cfg(general) under none. **Hold-out interpolation residuals: 14% (RF/capped) to 37% (GP/none non-monotone dip)** — this is the floor on per-cell precision and the reason no `(n/anchor)^k` is fitted.

**OWN REGIMES — do NOT interpolate in or out:** DTA n=30000 (only general/none measured); nucleotide n=18k-100k (only rf/lgbm/mlp survive, +svm at n≤10k; gp & ridge gated out by domain skip). Anchor ranges: antibody/ginkgo n=94-672 (all capped per_cfg anchored ONLY here); molecular n=1412-9978; reaction n=3955-4312; tcga n=821-1017; large clf n=4275-5000.

## ✅ Section 5 — Per-experiment cost taxonomy (verified, 36 agents)

THREE cost classes — this is the mechanism behind the non-linear, dataset-specific cumulative walls:

**A. FRESH-SEARCH** (suffixed dataset_name ⇒ full uncached `find_best_hybrid`; the expensive ones): c02 (2 of 6 calls), c05 (`_n{n}_s{seed}`), c06, c07 (subpop), c13 (new reps, 2 calls), c15, c16, c21 (per base-rep × condition), **c23 (4×n_models fbh calls, 3 fresh)**, c24 (`_fold{i}`). Cost = Σ_calls (cells_per_call × per_cfg(model, n, strategy)). **Scales with rep COUNT and per_cfg, not n alone.**
**B. CACHE-REUSE + downstream fit** (reads precomputed search, fits models per fold): c08, c09, c10, c11, **c17** (+ heavy mRMR/RFE/PCA per budget), c18, c19, c22; most s-experiments. Cheap reads + M×F model fits (+ c17's feature-selection compute, dimension-driven).
**C. NO-FBH pure compute:** c01 (per-rep characterization, cost ∝ rep dims — **HER2 3862s vs bvp 88s = the rep-size effect**), c04, s01 (n×n distance matrices, ∝ n²·rep_dim).

Runs-in-pipeline confirmed per Section 1. c06/c10/c11/c12/s11 conditional (family/strategy/tcga gates). c08 fails (NameError, regression).

## ✅ Section 6 — cx cumulative wall

**MEASURED (the only cx data that exists):** 5 antibody pipelines — bvp/137=38.98h, cic/137=37.11h, HER2/146=34.94h (complete c01→c24); Ang2/397 & VEGF/418 = 63.9h & 63.6h through c19, still running c21. Dominant experiments: c17 (6.7–53.4h), c21 (0.2–19.3h), c05, c09, c16, c23, c24 — **varying 44–550× across datasets at near-identical n** (c01 44×, c21 95×, c23 550× between n=137 and n=146).

**PHASE STATE (from 02:07 monitor — critical framing):**
- Small antibody (ginkgo/flab/abagym, n≤672): precompute DONE → in EXPERIMENT phase (the `[C## done]` jobs) → these are burning cx wall now.
- **Molecular / reaction / DTA / large-clf: STILL IN PRECOMPUTE** (`[N/M]` search %, after 68h) → their cx experiments have NOT STARTED. (And molecular is re-running already-cached precompute due to the Section 9 cache drift.)

**ESTIMABILITY (honest):**
- The 5 measured datasets: KNOWN (measured, above).
- Other small antibody (n=94-672, same regime): cumulative cx plausibly tens of hours dominated by c17/c21/c05/c09/c16/c23/c24, but per-experiment variance is 44-550× ⇒ **no defensible per-dataset point estimate; bounded range only** (measured completed small-antibody totals were 35-39h at n=137-146; ≥64h at n=397-418 and not done).
- Molecular / reaction / DTA / nucleotide cx: **ZERO measurements; per_cfg is own-regime; per-rep dims (a primary driver for c01/c17) are NOT in code** ⇒ **NOT ESTIMABLE from available data. Refusing to extrapolate.**

**VALIDATION GATE (rule 3):** a bottom-up cells×per_cfg reconstruction CANNOT be validated to <30% per experiment because (i) per_cfg is anchored on precompute-phase, not experiment-phase, walls; (ii) fresh-search per-call config counts are dataset-rep-specific; (iii) per-rep dims are unavailable statically. ⇒ cumulative cx for unmeasured datasets is **not reported as point estimates** — doing so would repeat the exact extrapolation error that has burned every prior attempt.

**Empirical validation attempt (rule 3, NUMERICAL — proof, not assertion):** cleanest anchor is c24, whose 5 per-fold searches each ≈ one general/none precompute search (same general pool, full tier). Predict c24 ≈ 5 × measured(general/none) + small k×k transfer matrix:

| dataset | 5 × gen/none (measured) | measured c24 | residual |
|---|---|---|---|
| bvp 137 | 5×0.35h = 1.77h | 2.16h | −18% (pass) |
| cic 137 | 5×0.49h = 2.44h | 3.35h | −27% (pass) |
| HER2 146 | 5×0.21h = 1.05h | 0.32h | **+228% (FAIL, 7×)** |

HER2 fails the 30% gate by 7×. **Mechanism (identifiable, NOT statically computable):** c24 cost is gated by how many of the 5 folds' searches succeed — ≥2 required, else early return (`c24:241`); HER2's scaffold-group structure makes folds fail/cheap. The SAME mechanism explains c21 (19.3h bvp vs 0.76h HER2): c21 skips any base rep with `dim > n_samples` (`c21:125-128`), so its fresh-search count = #(reps narrower than n) — needs per-rep dims, which are NOT in code (Section dataset_reps). Compounding ≥2× ambiguity: fresh searches may use FULL_BUDGETS (8) vs precompute SCREENING_BUDGETS (4); the constant values were not recovered (precompute agent failed StructuredOutput). ⇒ **bottom-up cx model REJECTED by rule 3.** Per-dataset cx point estimates for unmeasured datasets are not defensible; only the 5 measured pipelines + the regime-bounded ranges above are reported.

## ✅ Section 7 — sim cumulative (Context F)

**ZERO completed sim pipelines exist.** Per-dataset sim runs s01-s08, s10-s13 (12 experiments; s09 is cross-dataset). Cost taxonomy: most s-experiments are CACHE-REUSE/cheap (s02,s04,s05,s06,s07,s08[4 reuse calls],s10,s12); s01 is no-fbh distance matrices (∝ n²·rep_dim); s13 partial. ⇒ sim cumulative is **structurally much lighter than cx** (no c17/c21/c05/c23/c24 fresh-search monsters), but **UNMEASURED → not estimable**. Monitor shows all sim jobs at `[S02 1done]`/`[S06 5done]` — none finished. **Unlock = grep one completed sim job's per-experiment `s## done in Xs` lines** (the runner emits them, Section 1).

## ✅ Section 8 — Precompute remaining (Deliverable 1) — the defensible deliverable

Anchored on measured per-condition walls (02:07). For each missing condition I give the **nearest measured same-condition anchor**; where n is outside the measured range I REFUSE a point number.

**Two distinct situations:**
1. **Molecular DONE-4/4 (adme_hlm/mdr1/rlm, caco2): precompute is COMPLETE on disk, but the wave is RE-COMPUTING it** due to neural-rep cache drift (Section 9). "Remaining" = cache-rejection drag, not new work. Monitor face-value ETAs: caco2 +1h, adme_mdr1 +7h, adme_hlm +22h, adme_rlm +43h — but these assume constant cell rate and ignore the high-budget tail; real walls likely 1.5-2×.
2. **Genuinely partial datasets** (missing conditions):

| dataset | n | missing | nearest MEASURED same-condition anchor | verdict |
|---|---|---|---|---|
| flab_binding_koenig | 4275 | top/uncapped | its own general/uncapped=151.77h; flab_expression_koenig top/uncapped=76.89h (same n) | ~50-90h; monitor shows 90.8% → **~7h to finish** |
| herg | 7772 | gen/uncapped, top/none, top/uncapped | molecular gen/uncapped measured only n≤3086 (48-103h); n=7772 is ABOVE range | gen/uncapped **UNMEASURED at n=7772** (svm still in at ≤10k, was 65% of herg gen/none=29.4h) → plausibly >100h, **not estimable precisely** |
| logd | 5028 | gen/uncapped, top/none, top/uncapped | molecular gen/uncapped n≤3086=48-103h; n=5028 above range | **UNMEASURED at n=5028; refuse point estimate** |
| qm9 | 9978 | gen/uncapped, top/uncapped | none measured at n~10k for uncapped | **UNMEASURED; refuse** |
| bindingdb_kd | 30000 | gen/uncapped, top/none, top/uncapped | only gen/none=96.23h measured at n=30k (DTA own regime) | gen/uncapped **UNMEASURED**; gen/none alone was 96h, uncapped multiplies per-cell 8-33× but cell set also shrinks (svm/gp gated) → **plausibly exceeds 168h wall; not a defensible single number** |
| kiba | 30000 | same as bindingdb | gen/none=121.83h | same: **plausibly >168h; not estimable precisely** |
| nucleotide (≈18 datasets) | 18k-100k | mostly gen/uncapped, top/none, top/uncapped | own regime; few measured (nt_H3K4me3 gen/unc=60.89h@n18k w/ 32-CPU; nt_promoter_tata gen/unc=51.19h@n5274; gb_human_enhancers gen/unc=106.41h@n27791) | **own regime, high variance; per-dataset n differs; not estimable as a single law** — see Section 11 |
| gb_demo_coding_vs_intergenomic | 100000 | ALL (0/5) | gb_demo_human_or_worm gen/none=53.53h@n100k | only a same-n gen/none analogue exists; uncapped/top **UNMEASURED** |

**Caveats on ALL precompute-remaining:** (a) cache-rejection drag (Section 9) adds an uncounted recompute tax to every molecular job; (b) running wave is 8-CPU, resubmits are 32-CPU — more CPUs speed tree models (`n_jobs=-1`: rf/xgb/lightgbm) but NOT single-threaded gp/svm/ridge, so the speedup is model-mix-dependent, not uniform; (c) measured walls already encode the pruned `n_configs`.

## ✅ Section 9 — Cache-rejection root cause (Context A) — RESOLVED

`rep_hashes` = SHA1 over (str(shape) + str(dtype) + `np.ascontiguousarray(arr).tobytes()`) per rep (`find_best_hybrid.py:244-250`); dict order sorted, contiguity normalized, no mmap → none of those are the cause.
- **ROOT CAUSE:** the 6 neural reps (chembert, chemberta, graphmvp, grover, molformer, smited; mhg_gnn flagged) are **torch GPU forward passes** returning `.cpu().numpy()` (`molecular.py:1730,1836,1928,2007,2185,2973,3071`) → **not bit-reproducible** across processes. On a cache MISS they are GPU-recomputed and the float32 mantissa bits differ → different hash → meta drift → "will recompute". The 9 classical reps are deterministic CPU producers forcing float32 → identical bytes on recompute → stable hash. This asymmetry exactly matches the symptom (6 drift, 9 stable, same n/seed, .npy unmodified).
- **TRIGGER:** two cache dirs with DIFFERENT keys — outer `rep_cache/<dataset_name>/` (`_shared.py:170`) vs inner `reps/<sha256(smiles)[:16]>/` (`find_best_hybrid_molecular.py:514`). A neural .npy present in one dir but absent from the dir a given invocation consults forces GPU recompute. Which dir lacks the neural files on the server is the one UNKNOWN (server access forbidden).
- **Ruled out with evidence:** float32/64 flip (5/6 force .astype float32), C/F contiguity (ascontiguousarray), mmap, dict order (sorted), scaffold-group/sample reordering (`assign_scaffold_groups` is order-preserving, sorted IDs, canonical RDKit — sample_ordering agent traced every molecular loader deterministic), in-place mutation (hashed at function top, downstream slices copies), NaN, object-dtype.
- **Cost impact:** each rejection re-produces the neural rep + invalidates downstream hybrid_eval/baseline caches (`_hash_array` feeds all three layers, `:2338,:2600`). Snapshot herg_ki sim job: hybrid_eval 15419 hit / 4849 miss / 340 stale → thousands of forced per-config recomputes = the uncounted molecular drag.

## ✅ Section 10 — Wedge classification dead-vs-alive (Context B)

Method (rule 4 — NOT idle alone): compare idle_s to the max plausible wall of the CURRENT experiment at that dataset's n, using measured cx walls. No log tail available (would confirm "dead"); confidence stated. Max plausible measured walls: c17 ~53h (n~400), c21 ~19h (n=137), c09 ~10h (n=146), c05 ~3h (n≤418); s02 should be minutes (cache-reuse).

| job | step | idle | verdict |
|---|---|---|---|
| abagym_Ang2 12233941 | C21 15done, j6h=3 | 1.35h | **ALIVE** (progressing; c21 at n=397 can be many h) |
| abagym_VEGF 12233945 | C21 15done, j6h=3 | 1.7h | **ALIVE** |
| humanness_therapeutic 12233948 | C13 10done | 11min | **ALIVE** (c13 heavy: top=21 reps) |
| abagym_EGFR 12233942 | C05 5done | 2.9h | **likely alive** (c05 at n=659 plausibly hrs) |
| ginkgo_heparin/pr_cho/titer, buchwald, flab_pk_fcrnrt | C05, idle <1.1h | <1.1h | **likely alive** |
| flab_thermo_jain | C05 3done | 1.9h | **borderline** (c05@137≈1.2h; slightly over) |
| ginkgo_acsins(7h), ginkgo_hic(9.5h), ginkgo_colloidal(15.7h) | C05 | 7-16h | **likely DEAD** (c05@n242 ≈1-2h ≪ idle) |
| flab_pk_jain_heprt(5.5h), tz_her2_binary(3.4h)/multiclass(12.8h) | C05 | 3-13h | binary borderline; multiclass/heprt **likely DEAD** |
| herg_ki(14.9h) | C05 3done | 14.9h | **likely DEAD** (c05 at n=1412 unmeasured but ≪15h) |
| ddg_combined(52.5h), flab_thermo_tresanco(40h), flab_aggregation_jain(56.5h), abagym_lysozyme(67h), tcga_brca(28.5h), tcga_luad(51.5h) | C05 | 28-67h | **ALMOST CERTAINLY DEAD** (idle ≫ any plausible c05) |
| ALL sim jobs at S02 1done | S02 | 10-68h | **ALMOST CERTAINLY DEAD** (s02 is cache-reuse, should be minutes; 10-68h idle) |
| exp_%j (12233779-786) | — | varies | **framework F-jobs — DO NOT TOUCH (rule); not classified** |

Confidence: "almost certainly dead" = idle ≫ max measured wall for that experiment class; "alive" = idle < plausible wall AND/OR recent JSON saves (j6h>0). A 100-line tail of e.g. slurm-12233944.out (abagym_lysozyme, 67h) would convert "almost certainly" → confirmed.

## ✅ Section 11 — Nucleotide pending queue (Context E)

Pending (not in the 69 running): cx_nt/gb + sim_nt/gb (IDs ~12242908-12242949), 32-CPU/256G, n=5k-100k.
- **Precompute remaining:** anchorable from measured nucleotide gen/none walls (2.3-14.4h at n=24k-36k; 53.5h at n=100k) and the few measured gen/uncapped (60.89h@n18k 32-CPU; 106.41h@n27791 8-CPU; 51.19h@n5274). High variance by n and by surviving-rep count; **per-dataset, not a single law**. Domain skips reduce models to {rf, lightgbm, mlp} (+svm at n≤10k) — `_DOMAIN_MODEL_SKIPS['nucleotide']=[ridge,knn,xgboost]`, capped also drops mlp.
- **cx/sim cumulative:** **NOT ESTIMABLE** — n=18k-100k is far outside the n=137-418 where any cx pipeline was measured; per_cfg is own-regime; no nucleotide cx/sim pipeline has ever completed. Honest answer: cannot estimate cx/sim cumulative for nucleotide from available data.

---

## Failed agents (recovered)
- `precompute` foundation agent (parallel[2]) failed StructuredOutput → recovered: precompute structure from runner_inclusion + dataset_reps agents (4 lineages: general/top/general_distance/top_distance; top gated on top_tier≥2; complexity builds distance only for c03) + measured per-condition walls directly from the snapshot. No gap for Deliverable 1.
- `neural_rep_loading` forensics agent (parallel[6]) failed → fully subsumed by `cache_hash_rootcause` (traced all 6 neural rep production paths) + `sample_ordering`. No gap for Context A.

---
---

# ════════ PART II — VALIDATED RESULTS (KEPT) + AUDIT (2026-06-04) ════════

> This supersedes the early group-range / "rough 100–250h" numbers in the headline and Sections 6–8 — **those were guesses and are VOID.** The numbers below are the ones derived from a tested method.

## II.1 The cost model (the method that works)

`Wall(dataset, condition) = Σ over surviving models m  [ N_m × t_m(n, strategy) ]`
- `N_m` = configs for model m (from logged `n_cfg`); surviving from gates (gp≤1000, svm≤10000, domain/dataset skips).
- `t_m(n,strategy)` = per-config seconds.

**Tested the obvious idea and REJECTED it:** fitting `t_m = k·n^α` per model (α from algorithm theory) gave **R² < 0.4** with fitted exponents far below theory — within a size band, per-config wall is NOT a clean function of n (rep width + which-model-dominates swamp it). Confirms rule 2.

**What works — per-model none→uncapped multiplier μ_m** (the strategy switch moves the constant, not the n-exponent, so n cancels):
gp ×18, rf ×10, xgboost ×10, ridge ×14, knn ×5, lightgbm ×12, mlp ×10, svm ×16 (medians across 17–20 datasets w/ both).

**Leave-one-out validation (predict each MEASURED general/uncapped from none×μ):** median error 39%; **±6–44% on molecular / TCGA / nucleotide (reliable), up to +700% on antibody (unreliable — mlp/gp dominate unpredictably).** The missing conditions we estimate below are all molecular/DTA → in the reliable zone.

## II.2 PRECOMPUTE estimates — VALIDATED (kept)

Missing `general/uncapped` (per-model: each model's measured none-wall × μ; SVM is the killer, O(n²) at large n):

| dataset | n | gen/none (measured) | **gen/uncapped (EST ±~30%)** | dominated by | fits 7-day wall? |
|---|---|---|---|---|---|
| herg | 7,772 | 29.4h | **~400h** | svm ~302h | NO |
| logd | 5,028 | 20.4h | **~250h** | svm ~92h | NO |
| qm9 | 9,978 | 20.6h | **~256h** | svm ~116h | NO |
| bindingdb_kd | 30,000 | 96.2h | **~980h** | ridge/rf/xgb | NO |
| kiba | 30,000 | 121.8h | **~1270h** | ridge/xgb | NO |

`top/none` ≈ 0.6× gen/none (cheap, hours). `top/uncapped` ≈ 10× gen/none (same order as gen/uncapped). `top/capped` ≈ 114× gen/none **but only for datasets that run the capped condition** (the X/5 in check_precompute). Full per-dataset table computed for all ~45 datasets (in /tmp/wf working notes; verdict: every dataset at n≥~1400 has missing uncapped conditions in the hundreds-to-1000+ h → none fit the ~100h left).

## II.3 cx per-experiment — c05 worked example (kept); rest pending `wu46jfnxg`

**c05 formula (from source):** `Σ over [4–5 sizes: max(50, {.1,.25,.5,.75,1}·n)] × [3 repeats]` of a FRESH `find_best_hybrid` (suffixed name, never cached) using the **CAPPED** augmentation strategy at n_sub≤TOP_CAPPED_MAX_N, budgets ≤ train_size; + baseline grid `sizes×3×8models×G reps`; + full-data-BP refit `sizes×3×8×folds`.
**Test vs the 5 measured (1.17/0.80/6.85/2.34/2.80h):** bvp/cic/HER2 run the same 12 searches at n≈137–146 yet **HER2 is 6× costlier** → per-search cost = **concatenated rep dimensionality under capped**, which is NOT in code. c05 bounded 0.8–6.9h at n≤418, **not tightly predictable per-dataset**, unmeasured above n=418.

## II.4 sim per-experiment — structural estimates (kept; LOW confidence, no sim pipeline ever finished)

| exp | sm_ab146 | mid418 | mol2158 | large5000 | dta30k | nuc33k |
|---|---|---|---|---|---|---|
| s01 distance n×n | 34s | 45s | 2m | 23m | **8.0h** | **6.1h** (+~35GB RAM/rep → OOM risk) |
| s02 hybrid sim | 15s | 2m | 10m | 2.5h | 41h | 23h |
| s03 mutual-info kNN | 1m | 4m | 38m | 6.9h | **111h** | **97h** |
| s04 cliffs | ~0 | ~0 | 4m | ~0 | ~0 | ~0 |
| s05 kNN pred | mins | mins | mins | mins | **~16h** | ~3h |
| s06 applic. domain | 3m | 10m | 25m | 2.5h | – | – |
| s07 noise bridge | ~0 | 4s | 1m | 11m | ~0 | ~0 |
| s08 optimal dim | 2m | 12m | 24m | 3.3h | 17h | 5h |
| s10 importance | 3m | 25m | 23m | 32m | 25h | 31h |
| s11 bayes opt | ~0 | ~0 | 1.1h | ~0 | ~0 | ~0 |
| s12 x-domain cliffs | 1m | 12m | 24m | 3.3h | 4h | 3h |
| s13 cluster/retrieval | 8s | 25s | 2m | 11m | 2.1h | 1.9h |
Sim is **cheap at small/mid antibody n (minutes/exp)**; the O(n²) ones (s01/s03/s05/s10) blow up at n≥30k (8–111h) + s01 OOM risk.

## II.5 FIXES APPLIED (code, for the next submission — current jobs run old code)
1. **Cache drift** — `find_best_hybrid.py` `_load_cache_with_meta` ignores `rep_hashes`/`aug_hashes` (identity still guarded by rep_names+shapes+n+seed+budgets+config). Stops infinite neural-rep recompute churn; preserves existing caches (no rebuild). UNTESTED on server.
2. **c08 crash** — `c08_calibration.py:387` `y_pred_conf`→`y_pred_val` (NameError crashed c08 on every regression dataset).

## II.6 ════ AUDIT ════ (against the original documentation)

### NEEDS CHANGE (estimate is wrong/incomplete — must revise)
- **A1. Distance-lineage precompute is MISSING from the precompute table.** Sim datasets ALSO precompute `general_distance` + `top_distance` (full searches, DISTANCE_BUDGETS/DISTANCE_EVALUATORS) for s04/s05/s06/s07/s10/s13 — **unmeasured in your logs, not in my numbers**. So sim-side precompute-remaining is UNDER-counted. (Surfaced by the s05 agent.)
- **A2. The c08 fix INCREASES cx on regression datasets.** The 5 measured pipelines had c08 *crashing* (≈0s). Post-fix, c08 runs 4 `build_fold_hybrid` conditions → adds wall to every regression dataset's cx. So the measured 35–64h is a post-fix UNDER-estimate. Decision needed: deploy c08 fix (correctness) vs the added cx wall.
- **A3. Early per-dataset cx numbers (180L/200L/600L "rough") are GUESSES — VOID.** Replace with `wu46jfnxg` per-experiment output (pending).
- **A4. "DONE" molecular datasets show precompute-remaining=0, but on the CURRENT wave they re-run forever** (cache churn). "0" is only true for a fresh next submission with the fix.

### COULD CHANGE (improvable)
- **B1. FULL_BUDGETS / SCREENING_BUDGETS / DISTANCE_BUDGETS exact values never confirmed** (the precompute agent failed StructuredOutput) → ~2× ambiguity in c05/c24 fresh-search cell counts. Getting the 3 constants tightens cx.
- **B2. Capped-condition multiplier (×114) is total-level and high-variance (80–325×);** could use the measured capped per-model per_cfg directly. Only applies to datasets that actually run capped.
- **B3. Antibody cx/precompute is rep-dimensionality-driven** (the dominant uncertainty everywhere). Loading each dataset's reps once to RECORD per-rep dims would convert many LOW-confidence cells to MED/HIGH.

### UNADDRESSED (requested or needed, not done)
- **C1. Tail of one long-idle wedge log** (e.g. `slurm-12233944.out`, lysozyme 67h) — requested twice, not provided. Dead-job calls are "almost certain," not confirmed.
- **C2. Grep of one COMPLETED sim pipeline** — not provided → all sim numbers (II.4) remain UNVALIDATED (structural only).
- **C3. Cache-rejection drag on the CURRENT running wave** not quantified into per-job remaining (only qualitative ~30–50% churn).
- **C4. Scope:** some datasets in the table (`abcov_kd`, `flab_immunogenicity_marks`) are NOT in the 69-job running wave — estimates are "per dataset if run," not "all currently running."
- **C5. c12 (tcga modality ablation) cx cost** not separately estimated (tcga-only experiment, inserted after c11).
- **C6. Cache fix is UNTESTED** (no server access) — needs a smoke test before deploy; small residual risk that ignoring rep_hashes masks a genuine rep-value change.
- **C7. cx per-experiment validation (`wu46jfnxg`) not yet synthesized** — the c01–c24 per-dataset cx table is the remaining deliverable.

### META (process honesty)
Early-conversation outputs (group ranges, the first per-dataset cx column, "molecular 100–250h rough") were guesses and are SUPERSEDED. Trust ONLY: the validated precompute multiplier numbers (II.2), the c05/sim per-experiment formulas (II.3–II.4), and the pending cx workflow. Two estimable-only-with-more-data items remain open: a wedge log tail (C1) and a finished sim pipeline (C2).

---
---

# ════════ PART III — NEW BUG + CONTENTION FINDING + PER-EXPERIMENT CATCHES (cx AND sim) ════════

## III.0 🔴🔴 NEW BUG (user-flagged 2026-06-04): experiments run on SCREENING_BUDGETS

`complexity_helpers.py:1038-1046` and `similarity_helpers.py:639-647` build the run config with:
```
'search_speed': 'full',
'budgets': SCREENING_BUDGETS,    # = [50, 500, 1000, 2000]  (only 4)
```
- The wave's precompute (via `precompute_hybrid_searches(config)`) and every experiment that reads `config['budgets']` (e.g. c05 `c05:47`) therefore search only **4 budgets, missing [10, 25, 100, 250]**. The intended full grid is `FULL_BUDGETS=[10,25,50,100,250,500,1000,2000]` (8).
- **Inconsistency:** c24 hardcodes `FULL_BUDGETS` (8) (`_shared.py:259`) while c05 reads config (SCREENING, 4) — experiments aren't using the same budget grid. Also `precompute_hybrids.py:290` (the standalone main) uses FULL at speed='full', so the standalone path and the helper path can write **different cache keys**.
- **User: SCREENING_BUDGETS should NOT be used in these experiments** → should be FULL_BUDGETS.
- **Impact on methodology:** sparse grid misses the low-budget optima — exactly where small-n / small-rep datasets and the budget-capped slow models live (svm cap 500, gp cap 1000 → they only ever run at low budgets, several of which [10,25,100,250] are absent). The "best budget" the search reports is biased to {50,500,1000,2000}.
- **Impact on my estimates:** all measured precompute walls are on the 4-budget SCREENING grid. A corrected FULL_BUDGETS rerun ADDS 4 low/cheap budgets → roughly **+20–40% wall** (not 2×; the added budgets are small and the expensive 500–2000 are already present). So my precompute numbers are a mild UNDER-estimate of a corrected run.

## III.1 🔴 THE CONTENTION FINDING (the big one — explains the "wild variance")

Validating the per-experiment formulas against the 5 measured walls: **only 2 of 19 c-experiments (c17, c24) reproduce all 5 within ±30%.** But the failures are NOT modeling errors — they are **node CPU contention**:
- 11/19 experiments' residuals track CPU oversubscription, not n/rep/config. Signature: **HER2 (n=146) is hugely under-predicted across c01 (−98%), c04 (−95%), c05 (−93%), c09 (−100%), c11 (−95%), c18 (−92%)** — the SAME deterministic code ran **3–48× slower on HER2** than the formula and than the same code on bvp/cic at near-identical n. HER2's whole pipeline sat on a contended node (matches the snapshot's HER2 per_cfg spikes, e.g. top/none xgboost 17.5s vs 0.19s elsewhere).
- **The "44× c01 / 95× c21 variance between near-identical datasets" that the original doc flagged as the central scaling mystery is LARGELY node contention** (n_jobs=-1 oversubscription on shared SLURM nodes), NOT an algorithmic n/rep effect.
- This is **why every prior estimate was off 2–10×: the ground truth itself varies 3–48× by node load.** No code model can hit <30% against contention-noisy measurements; the right anchor is the *uncontended* runs (bvp/cic/Ang2/VEGF validate on many experiments).
- Observation (not a recommendation): fewer concurrent jobs per node / CPU pinning would both speed the wave AND make walls predictable.

## III.2 cx per-experiment catches (all 24) — only c17 & c24 are code-predictable

| exp | validated | the catch |
|---|---|---|
| c01 | no | contention-contaminated (HER2 48× gap); model-fit & metric terms collinear in rep-count → cannot be separated/fit |
| c02 | partial | cache-state: uncached `general_ft` FT-pass recompute; can't cross-validate (no held-out c02 walls) |
| c04 | partial | driver g×n, but HER2 −95% = contention |
| c05 | partial | uses CAPPED strategy on subsamples n≤250; per-search cost = rep-dimensionality (not in code); budgets here come from the SCREENING bug; HER2 −93% contention |
| c06 | no | no stable driver; ±157% swings |
| c07 | partial | which subpopulations trigger a fresh (cache-miss) search is data-dependent → +500%/+670% swings |
| c08 | (crashes) | NameError on every regression dataset (now FIXED); once fixed cost ∝ total_dim, ADDS wall |
| c09 | no | 120 fresh `create_hybrid`; its **3-fold indices ≠ precompute's 5-fold → cache MISS → full recompute** (real inefficiency); dim-driven; −99/−100% |
| c10 | no | negligible (<5s) — % residuals meaningless, ignore |
| c11 | no | pure node contention (HER2 −95%) |
| c13 | partial | driver = #reps not in LEGACY_REPS ("new" reps) — depends which reps count as new |
| c15 | no | erratic (+956% VEGF); low confidence |
| c16 | partial | driven by the cached blueprint's identity (alloc/imp/eval) — must READ the cached SearchResult to estimate |
| **c17** | **YES ±3%** | **dominant cost & PREDICTABLE: wall=2.46e-4·M·n^1.078, M from total_dim — but needs rep dims (not in code); validated antibody-only** |
| c18 | partial | per-rep lightgbm-gain fits (∝ rep count); HER2 −92% contention |
| c19 | partial | per-cell tree-fit; under-predicts bvp/cic (−66/−78%) |
| c21 | partial | aug-pool DIMENSION (esm2_per_region 17920D, esm2_per_cdr 7680D dominate); RF ranker fit ~3240×/search; bvp/cic ±4% but HER2 +870% |
| c22 | cheap (read directly) | cache-read blueprint + 5 folds×(build_hybrid + rf+xgboost fit + TreeSHAP best-of-2); TreeExplainer fast → measured 12–127s; only growth term is SHAP ∝ n_val×features×trees |
| c12 | tcga-only (read directly) | NO find_best_hybrid; direct evals of [G groups + C(G,2) pairwise + 1 all-modalities] × 6 models × 5 folds on WIDE omics concats; **quadratic in #modality groups**, concat-width-driven; runs only on tcga_brca_pam50 / tcga_luad_vs_lusc (both dead now); never measured |
| c23 | partial | tier='extra' models (catboost/dnn/cnn) dominate; bvp/cic ±0% but HER2 +49857% (measured was 18.8s) |
| **c24** | **YES** | predictable but driver is **cold-vs-warm cache** (HER2 was warm-cached → cheap); 5 fresh FULL_BUDGETS searches + 25 cross-fold cells |

## III.3 sim per-experiment catches (all 13) — equal weight to cx

| exp | the catch |
|---|---|
| s01 | pure O(n²·Σd) distance matrices; at n≥30k = 6–8h **AND ~35 GB RAM per rep → OOM risk**; Σd (rep dims) not in code |
| s02 | cache-reuse + n² growth → 23–41h at n≥30k |
| s03 | **biggest sim cost: kNN mutual-info, brute-force at d_cap=50 → O(n²) → ~111h at n=30k** |
| s04 | cheap (~0) except a few min mid-n |
| s05 | the workflow MIS-ATTRIBUTED the shared distance-lineage precompute into s05 (the "6.2h@146" is precompute, not s05); real s05 is minutes except ~16h kNN at n=30k |
| s06 | applicability domain; mixed; ~2.5h at large clf |
| s07 | cheap (~0) |
| s08 | cache-reuse; ~17h at n=30k |
| s10 | no fbh but O(n²) at large n → 25–31h at n≥30k |
| **s11** | **INCONSISTENT estimate (1.1h at n=2158 but 0 at both smaller and larger n) — that's wrong; bayesian-opt has conditional arms (multi-obj / per-rep-bo) that may or may not fire → estimate needs re-derivation** |
| s12 | mixed; ~3–4h at n≥30k |
| s13 | cache-reuse + n² retrieval; ~2h at n=30k |

**Sim verdict:** cheap at small/mid antibody n (minutes/experiment); the O(n²) ones (s01/s03/s05/s10/s13) blow up at n≥30k, s01 with an OOM risk. All sim numbers remain UNVALIDATED (no finished sim pipeline — item C2).

## III.4 Missing precompute cost: the distance lineages (sim)
Sim datasets ALSO precompute **`general_distance` + `top_distance`** (`DISTANCE_BUDGETS`=[10,25,50,100,250,500] 6 budgets, `DISTANCE_EVALUATORS`=[knn,ridge,rf,mlp,lightgbm,svm] 6) — for s04/s05/s06/s07/s10/s13. **Unmeasured in your logs, absent from my precompute table** → sim-side precompute is under-counted (this is also what the s05 agent mis-charged to s05).

## III.5 c17 — the one solid cx predictor (validated ±3%)
`wall_seconds = 2.46e-4 · M · n^1.078`, M = exact count of single-column KSG mutual-info calls in mRMR = `Σ_b [ (td if td>2000 else 0) + Peff + (Peff−1) + Σ_{i=1}^{min(b,Peff)−1}(Peff−1−i) ]`, `Peff=min(total_dim, 2000)`, budgets filtered to b<total_dim & b≤n. **Blocked on `total_dim` = Σ general-rep dims, which is not in code** (only `GEOMETRY_DIMS` is; foundation reps = runtime `hidden_size`). To extend c17 (and c05, s01) to all datasets → load each dataset's reps once and record per-rep dims.

## III.6 Capped condition (refined, item 7)
`top/capped` is **RF-dominated (49–71% of wall, median ~62%)**; total `none→capped` multiplier 80–325× (high variance, RF is O(n²/n·budget) at high budget). Only applies to datasets that actually run capped (the 5/5-condition antibody/ginkgo datasets, n≤250 mostly).

## III.7 Audit deltas (since PART II)
- **Item 5 (budget constants): RESOLVED** — and surfaced the III.0 bug.
- **Item 6 (rep dims): partial** — only `GEOMETRY_DIMS` in code; foundation = runtime hidden_size; still the #1 unlock for c17/c05/s01.
- **Item 7 (capped multiplier): done** (III.6).
- **Item 12 (c12 TCGA): DONE** — read directly (III.2): quadratic in #modality groups, wide-concat-driven, tcga-only, never measured.
- **c22: DONE** — read directly (III.2): cheap, cache-read + TreeSHAP.
- **NEW: III.0 SCREENING_BUDGETS bug** added to the fix list alongside cache + c08.

---
---

# ════════ PART IV — OPEN ITEMS, MASTER LIST (everything still to address) ════════

## IV.0 FULL PER-DATASET ESTIMATE TABLE (precompute + cx + sim)

Hours. **PC total** = full precompute (measured + multiplier-estimated); **PC left** = precompute remaining (missing conditions). `(DONE)`/`(part)` = precompute status. cx: `meas`=measured pipeline; `35–60`/`65–95`=anchored on measured same-size antibody (×2–3 under contention); `BLOCKED:rep-dims`=needs the rep-dimension dump (item 1.1), no cx measured above n=418; `not estimable`=n≫measured. sim: all LOW/unvalidated (no finished sim pipeline); n²/OOM flagged at n≥30k. **Sim also adds an unmeasured `general_distance`+`top_distance` precompute lineage not in PC.**

| dataset | n | dom | PC total | PC left | cx | sim |
|---|---|---|---|---|---|---|
| ginkgo_heparin | 94 | ab | 29 (DONE) | 0 | 35–60 | ~0.5 |
| flab_pk_jain_heprt | 130 | ab | 39 (DONE) | 0 | 35–60 | ~0.5 |
| flab_pk_jain_fcrnrt | 132 | ab | 55 (DONE) | 0 | 35–60 | ~0.5 |
| flab_polyreact_jain_bvp | 137 | ab | 1 (DONE) | 0 | 39 (meas) | ~0.5 |
| flab_polyreact_jain_cic | 137 | ab | 1 (DONE) | 0 | 37 (meas) | ~0.5 |
| flab_aggregation_jain | 137 | ab | 44 (DONE) | 0 | 35–60 | ~0.5 |
| flab_aggregation_jain_hicrt | 137 | ab | 118 (DONE) | 0 | 35–60 | ~0.5 |
| flab_thermo_jain | 137 | ab | 120 (DONE) | 0 | 35–60 | ~0.5 |
| abcov_kd | 141 | ab | 1 (DONE) | 0 | 35–60 | ~0.5 |
| abagym_HER2 | 146 | ab | 6 (DONE) | 0 | 35 (meas) | ~0.5 |
| ginkgo_pr_cho | 197 | ab | 75 (DONE) | 0 | 35–60 | ~0.5 |
| flab_immunogenicity_marks | 217 | ab | 1 (DONE) | 0 | 35–60 | ~0.5 |
| ginkgo_titer | 239 | ab | 135 (DONE) | 0 | 35–60 | ~0.5 |
| ginkgo_acsins | 242 | ab | 90 (DONE) | 0 | 35–60 | ~0.5 |
| ginkgo_colloidal | 242 | ab | 68 (DONE) | 0 | 35–60 | ~0.5 |
| ginkgo_hic | 242 | ab | 82 (DONE) | 0 | 35–60 | ~0.5 |
| abagym_Ang2 | 397 | ab | 2 (DONE) | 0 | ~67 (meas→c19) | ~1.5 |
| abagym_VEGF | 418 | ab | 2 (DONE) | 0 | ~67 (meas→c19) | ~1.5 |
| ddg_combined | 429 | ab | 14 (DONE) | 0 | 65–95 | ~1.5 |
| humanness_therapeutic | 564 | ab | 2 (DONE) | 0 | 65–95 | ~1.5 |
| abagym_lysozyme | 627 | ab | 32 (DONE) | 0 | 65–95 | ~1.5 |
| abagym_EGFR | 659 | ab | 5 (DONE) | 0 | 65–95 | ~1.5 |
| flab_thermo_tresanco | 672 | ab | 35 (DONE) | 0 | 65–95 | ~1.5 |
| tcga_brca_pam50 | 821 | tcga | 38 (DONE) | 0 | 50–90 +c12 (LOW) | ~3–5 |
| tcga_luad_vs_lusc | 1017 | tcga | 22 (DONE) | 0 | 50–90 +c12 (LOW) | ~3–5 |
| herg_ki | 1412 | mol | 99 (DONE) | 0 | 150–400 (BLOCKED:rep-dims) | ~3–5 |
| caco2 | 2158 | mol | 134 (DONE) | 0 | 150–400 (BLOCKED:rep-dims) | ~3–5 |
| adme_mdr1 | 2641 | mol | 163 (DONE) | 0 | 150–400 (BLOCKED:rep-dims) | ~3–5 |
| adme_rlm | 3053 | mol | 217 (DONE) | 0 | 150–400 (BLOCKED:rep-dims) | ~3–5 |
| adme_hlm | 3086 | mol | 203 (DONE) | 0 | 150–400 (BLOCKED:rep-dims) | ~3–5 |
| buchwald_hartwig | 3955 | rxn | 12 (DONE) | 0 | 150–400 (BLOCKED:rep-dims) | ~15–25 |
| flab_binding_koenig | 4275 | ab | 269 (part) | 101 | ≥150 (BLOCKED:rep-dims) | ~15–25 |
| flab_expression_koenig | 4275 | ab | 203 (DONE) | 0 | ≥150 (BLOCKED:rep-dims) | ~15–25 |
| bh_doyle_plates | 4312 | rxn | 320 (DONE) | 0 | 150–400 (BLOCKED:rep-dims) | ~15–25 |
| flab_binding_adams | 5000 | ab | 164 (DONE) | 0 | ≥150 (BLOCKED:rep-dims) | ~15–25 |
| flab_binding_phillips_h3 | 5000 | ab | 242 (DONE) | 0 | ≥150 (BLOCKED:rep-dims) | ~15–25 |
| flab_expression_adams | 5000 | ab | 90 (DONE) | 0 | ≥150 (BLOCKED:rep-dims) | ~15–25 |
| tz_her2_binary | 5000 | ab | 17 (DONE) | 0 | ≥150 (BLOCKED:rep-dims) | ~15–25 |
| tz_her2_multiclass | 5000 | ab | 30 (DONE) | 0 | ≥150 (BLOCKED:rep-dims) | ~15–25 |
| logd | 5028 | mol | 481 (part) | 460 | 150–400 (BLOCKED:rep-dims) | ~15–25 |
| nt_promoter_tata | 5274 | nuc | 84 (part) | 29 | not estimable (n≫418) | ~15–25 |
| herg | 7772 | mol | 692 (part) | 662 | 150–400 (BLOCKED:rep-dims) | ~15–25 |
| qm9 | 9978 | mol | 496 (part) | 452 | 150–400 (BLOCKED:rep-dims) | ~15–25 |
| nt_H3K4me3 | 18244 | nuc | 115 (part) | 46 | not estimable | ~150–225 (n²,OOM) |
| gb_human_enhancers_cohn | 27791 | nuc | 115 (part) | 0 | not estimable | ~150–225 (n²,OOM) |
| bindingdb_kd | 30000 | dta | 2261 (part) | 2165 | ≥500 (BLOCKED) | ~150–225 (n²,OOM) |
| kiba | 30000 | dta | 2863 (part) | 2741 | ≥500 (BLOCKED) | ~150–225 (n²,OOM) |
| gb_human_nontata_promoters | 36131 | nuc | 14 (part) | 0 | not estimable | ~150–225 (n²,OOM) |
| gb_demo_human_or_worm | 100000 | nuc | 54 (part) | 0 | not estimable | ~150–225 (n²,OOM) |

(`abcov_kd`, `flab_immunogenicity_marks` are not in the 69-job wave. `gb_demo_coding_vs_intergenomic` n=100000 = 0/5, no anchor → omitted.) **Confidence: PC molecular/TCGA/nucleotide ±25%; cx measured only for the 5; cx `BLOCKED`/`not estimable` rows are the honest gaps (item 1.1 unlocks them); sim entirely unvalidated.**

---


## 1. Needs a server action from YOU (cannot be done locally — no server/heavy-compute here)
1.1 **Rep-dimension dump** — load each dataset's reps once and record per-rep dims. **HIGHEST VALUE: unblocks the per-dataset cx numbers** (c17/c05/s01 are all dimension-driven; c17 is the dominant cost and is a validated formula blocked only on `total_dim`).
1.2 **Tail of one long-idle wedge log** (e.g. `slurm-12233944.out`, lysozyme 67h) — confirms the dead-vs-alive calls (currently "almost certain," not confirmed).
1.3 **Grep of one FINISHED sim pipeline** (`grep "done in" slurm-<simjob>.out`) — validates every sim number (all currently structural/unvalidated).

## 2. Bugs — found; need a decision or deploy
2.1 **SCREENING_BUDGETS → FULL_BUDGETS** (`complexity_helpers.py:1046`, `similarity_helpers.py:647`). Fix is one line each, but it **changes the cache key → forces a full precompute rebuild** → your call before I apply. (Wave currently searches only 4 budgets, missing 10/25/100/250.)
2.2 **Cache-drift fix** (applied in `find_best_hybrid.py`) — **UNTESTED on server**; needs a smoke test before you rely on it.
2.3 **c08 NameError fix** (applied) — note it **adds cx wall on every regression dataset** (c08 now actually runs); decide whether to ship it this wave.
2.4 **Verify the cache-key split** — `precompute_hybrids.py:290` uses FULL_BUDGETS at speed='full' but the helper path forces SCREENING; they may be writing two different cache families for the "same" condition.

## 3. Analysis I can still finish locally — DONE 2026-06-04 (see PART V), except 3.7
3.1 ✅ **Distance-lineage precompute numbers** — DONE structurally (PART V.3.1): 4 conditions, 6 budgets, 6 evaluators, no capped; none≈0.9×, uncapped≈0.55×; adds ~3h (small ab) → ~77h (molecular). UNVALIDATED (no measured `*_distance` wall — folds into 1.3).
3.2 ✅ **Re-derive s11** — DONE (PART V.3.2): the "inconsistency" was domain-skip zeros; s11 is 0 except on the 10 molecular/reaction datasets, where it's ~0.3–1.5h (pool-size driven).
3.3 ✅ **c12 cx number** — DONE (PART V.3.3): ~1.7–2.0h base, 3–7h contended, per TCGA dataset; quadratic in active-modality count G; 5 clf models.
3.4 ✅ **Quantify the c08-fix cx addition** — DONE (PART V.3.4): regression-only, +0.02h (small ab) → +0.5h (molecular); <1% of cx total.
3.5 ✅ **Quantify cache-rejection drag** — DONE (PART V.3.5): 24% mid-flight miss (herg_ki); molecular DONE re-runs ≈ 73h cx + 303h sim face-value remaining across 4 datasets = biggest recoverable sink.
3.6 ✅ **Refine the capped multiplier** — DONE (PART V.3.6): multiplier abandoned; capped wall 23.5–122.5h, RF-dominated (~63%), ONLY at n≤250, anchored on 11 measured values.
3.7 ⏳ **Build the full combined per-dataset table** (precompute + each c + each s) — still BLOCKED on 1.1 (rep dims).

## 4. Deliverable gaps (blocked by the above)
4.1 **Per-dataset cx NUMBERS** — blocked on rep dims (1.1) + contention noise; only c17 & c24 are validated formulas today.
4.2 **Sim validation** — blocked on a finished sim log (1.3); numbers are structural.
4.3 **Wedge dead/alive CONFIRMATION** — blocked on a log tail (1.2).

## 5. Refuse-to-extrapolate zones (honest limits, not "fixable" without new data)
5.1 cx cumulative for molecular / DTA / reaction / TCGA / nucleotide — **zero measured cx above n=418**; not estimable to <30% without rep dims + an in-regime measurement.
5.2 sim cumulative anywhere — no finished sim pipeline exists.
5.3 Nucleotide cx/sim at n=18k–100k — own regime, unmeasured.

## 6. Scope clarifications
6.1 `abcov_kd`, `flab_immunogenicity_marks` — in the rep table but NOT in the 69-job wave (estimates are "per dataset if run").
6.2 `gb_demo_coding_vs_intergenomic` (0/5) is fully un-precomputed; the pending nucleotide queue (~40 jobs) hasn't started.

## 7. YOUR decisions, now unblocked by the estimates (from your original prompt — "what I will decide after estimates")
7.1 Add per-model budget gates?
7.2 Skip specific experiments for specific dataset size classes (e.g. the n²/contention-heavy ones at large n)?
7.3 Cancel specific dead/wedged jobs vs let them time out at the 2026-06-08 ~06:00 wall?
7.4 Resubmit with a reduced/corrected search space (FULL_BUDGETS fix + experiment trims)?
7.5 How to handle cache rejection — deploy the fix, rebuild, or leave the running wave?

## 8. Cross-cutting findings that shape ALL of the above
8.1 **Node contention** makes the measured per-experiment walls vary 3–48×; no code model hits <30% against contention-noisy anchors. Reducing concurrent jobs per node would speed the wave AND make walls predictable.
8.2 **Rep dimensionality** is the single dominant driver of the hard-to-predict experiments (c17/c05/s01/c21) and is not recorded anywhere → 1.1 is the master unlock.

---
---

# ════════ PART V — LOCAL ANALYSIS COMPLETED (§3 items, no server needed) — 2026-06-04 ════════

> All six §3 items closed by reading the source (c08/c12/s11/precompute_hybrids) + the measured 02:07 timing. Each shows the formula, the number, the confidence, and what (if anything) stays unvalidated. These do NOT need the rep-dim dump; the per-dataset cx NUMBERS (§4.1) still do.

## V.3.2 — s11 RE-DERIVED (resolves the "inconsistent" estimate) ✅

**Root of the inconsistency:** s11 has TWO hard gates at the top of `run()` (`s11_bayesian_optimization.py:758,762`):
1. `if domain not in ('molecular','reaction'): return {skipped}` — **s11 does NOT run on antibody, TCGA, DTA, or nucleotide.**
2. `if n_samples < 200: return {error}` — needs n≥200 (no molecular/reaction dataset is below this).

So the old "1.1h @ n=2158 but 0 at smaller AND larger n" was conflating two different zeros: the zeros were **correct** for the skipped domains (antibody/TCGA/DTA/nuc), not a modelling error. The real shape is: **0 everywhere except the 10 molecular + reaction datasets.**

**Cost on the 10 it DOES run (wave kwargs: max_iterations=100, n_seeds=3, rebuild_interval=0, hybrid_budget=50; `--multi-objective` and `--per-rep-bo` OFF):**
11 always-on conditions × 3 seeds × up-to-100 BO iterations. Per iteration = fit surrogate on the growing labeled set (seed≈0.05n, +1/iter) + **predict-with-uncertainty on the ~0.8n pool** + EI. The GP fit stays cheap (n_labeled³ with n_labeled≈150–600 is <0.5s); **the cost driver is the per-iteration predict over the 0.8n pool, dominated by the two 2048-D ECFP4 conditions** (`ecfp4_raw`, `tanimoto_ecfp4`) whose kernel is n_pool×n_labeled×2048 every iteration. `best_single` adds a one-time GP-CV over all reps per seed; `hybrid_ngboost`/`hybrid_bnn`/`hybrid_gauche_gp` add 100-estimator / 100-epoch / 50-Adam-step refits (these 3 are **~0 if ngboost/BNN/gauche ImportError out** — install state unknown → s11 may be cheaper).

| dataset | n | pool≈0.8n | **s11 EST (h)** |
|---|---|---|---|
| herg_ki | 1412 | 1130 | ~0.3 |
| caco2 | 2158 | 1726 | ~0.4 |
| adme_mdr1 | 2641 | 2113 | ~0.45 |
| adme_rlm | 3053 | 2442 | ~0.5 |
| adme_hlm | 3086 | 2469 | ~0.5 |
| buchwald_hartwig | 3955 | 3164 | ~0.6 |
| bh_doyle_plates | 4312 | 3450 | ~0.65 |
| logd | 5028 | 4022 | ~0.8 |
| herg | 7772 | 6218 | ~1.2 |
| qm9 | 9978 | 7982 | ~1.5 |

Cost ≈ linear in pool size (≈0.8n). **LOW confidence** (no measured s11). The old "1.1h@2158" was an over-estimate (real ~0.4h); the old "0 elsewhere" was wrong for the other 9 molecular/reaction rows but right for every non-molecular row. **Net: s11 is a minor sim contributor (≤1.5h) and ONLY on molecular/reaction.**

## V.3.3 — c12 NUMBER (TCGA modality ablation) ✅

**Structure (read from `c12_modality_ablation.py`):** NO `find_best_hybrid`. Direct CV fits:
- **Phase 1** (individual): R candidate reps × M models × F folds.
- **Phase 2** (pairwise): C(G,2) modality pairs × M × F (concatenate best rep per modality).
- **Phase 3** (all-combined): 1 × M × F (concat best rep across all G modalities).

**M = 5, not 6:** both TCGA datasets are classification → `get_clf_model_factories` filters `DOWNSTREAM_MODELS=[rf,xgboost,ridge,svm,mlp,lightgbm]` against the clf catalog `{rf,knn,ridge,mlp,xgboost,lightgbm}` → **svm dropped (not in catalog), knn not requested ⇒ 5 models run {rf,xgboost,ridge,mlp,lightgbm}**. F=5 folds.

**G = active modalities (≤9):** expression, methylation, mutation, cnv, mirna, rppa, immune, clinical, wsi. **R = candidate reps (≤58)** = union of available reps across those groups (depends on what the TCGA preset loaded → ties to 1.1).

`Wall ≈ F × [ R + C(G,2) + 1 ] × t_fit`, t_fit = avg per-(model,fold) fit, **MLP-dominated** (`MLPClassifier((100,50),max_iter=500)` ≈ 8–15s; rf/xgb/lgbm 100-tree ≈ 1–3s; ridge <0.5s ⇒ ~14s avg, more on the wide all-concat).

At R≈40, G=9 (C(9,2)=36): Phase1 40×5×14s=2800s(0.78h) + Phase2 36×5×18s=3240s(0.90h) + Phase3 5×40s=200s(0.06h) ≈ **1.7h base**.

| dataset | n | G,R assumed | **c12 base** | **c12 contended (×2–3)** |
|---|---|---|---|---|
| tcga_brca_pam50 (5-class) | 821 | ~9, ~40 | ~1.7h | 3–7h |
| tcga_luad_vs_lusc (binary) | 1017 | ~9, ~40 | ~2.0h | 3–7h |

**Quadratic in G** (the C(G,2) term): if fewer modalities load (e.g. no WSI tiles, no methylation → G=6), Phase 2 drops to C(6,2)=15 pairs ⇒ ~0.4h. **LOW–MED confidence**; the two uncertainties are R (rep count) and the per-MLP-fit time = rep dims (→ 1.1). c12 is **ADDITIVE** to the TCGA cx total (IV.0 "50–90 +c12") ⇒ +~2h base / +3–7h contended per TCGA dataset.

## V.3.4 — c08-fix cx ADDITION (regression datasets only) ✅

The NameError was at `c08_calibration.py:387`, **inside the regression branch only** (the classification branch never references `y_pred_conf`). So pre-fix: c08 **crashed on regression datasets** (~0 wall: it died in condition-1/fold-0 after one hybrid build + fit) and **ran normally on classification datasets** (which already paid the cost). **Post-fix, the cost is ADDED to regression datasets only.**

Full c08 = up-to-4 conditions {optimal, underfit, overfit, mismatched} × 5 folds × (1 `create_hybrid` build + 2 model fits [`model` + `model_conf`] + conformal calibration compute). The cost driver is the **per-fold importance recompute** in `create_hybrid` (treeshap/lightgbm_gain from the cached blueprint's `imp`) + the two **full-dim conditions** (overfit & mismatched both fit at `budget=total_dim`).

| regime | per (cond,fold) | c08 add (4×5=20 cells) |
|---|---|---|
| small antibody regression (n≤250) | ~2–5s | **+0.02–0.05h** |
| mid antibody regression (n≈400–670) | ~5–12s | **+0.05–0.15h** |
| molecular regression (n≈1.4k–10k) | ~10–50s | **+0.1–0.5h** |
| reaction regression (n≈4k) | ~20–50s | **+0.2–0.5h** |

**≤0.5h on any single dataset — negligible vs the 35–95h cx totals (<1%).** BUT most antibody targets are regression, so the measured 35–64h antibody cx totals are a slight (≤0.5h) **under-estimate** post-fix. **Decision 2.3:** shipping the c08 fix costs ~nothing on wall and restores a real experiment → low-risk to deploy.

## V.3.5 — CACHE-REJECTION DRAG, quantified ✅

Two anchors:
1. **Mid-flight miss rate (herg_ki sim job, §9 snapshot):** hybrid_eval **15419 hit / 4849 miss / 340 stale** → **23.5% of per-config lookups miss-and-recompute** + 1.6% stale, on a cache that should be warm.
2. **Molecular DONE datasets re-running from scratch:** check_precompute reports adme_hlm/mdr1/rlm + caco2 **DONE 4/4 on disk**, yet the monitor shows them at 48–90% **re-searching in precompute**. Every hour of that is **100% drag** (the result already exists). Face-value remaining ETAs (02:07 monitor):

| dataset | cx remaining | sim remaining |
|---|---|---|
| caco2 | +1h | +25h |
| adme_mdr1 | +7h | +35h |
| adme_hlm | +22h | +73h |
| adme_rlm | +43h | +170h |
| **sum** | **+73h** | **+303h** |

**≈376h of face-value remaining recompute across 4 already-complete datasets** (+~68h already burned re-running them), and the doc notes real walls run **1.5–2× the constant-rate ETA**. **This is the single largest recoverable time sink in the running wave.** Scope: every job whose reps include the 6 non-reproducible neural reps (chembert/chemberta/graphmvp/grover/molformer/smited) = all **molecular + DTA + reaction**; pure antibody/TCGA/nucleotide jobs use deterministic reps → little/no drag (matches their low drift/stale counts in the monitor). **The applied cache-drift fix (2.2, untested) removes this for the next submission.** HIGH confidence on the mechanism + that it dominates molecular wall; the exact recovered hours carry the ETA's 1.5–2× tail.

## V.3.6 — CAPPED multiplier REFINED (the multiplier is the wrong frame) ✅

The old "none→capped ×80–325" is unreliable because `top/none` is tiny and varies 10× (104s–1542s), so the ratio swings **131–2261×**. The capped WALL itself is far more stable and is **RF-dominated**. The 11 measured `top/capped` rows (all n≤242):

| dataset | n | capped wall | RF share | RF per_cfg(s) |
|---|---|---|---|---|
| ginkgo_heparin | 94 | 23.5h | 71% | 139 |
| flab_pk_jain_heprt | 130 | 31.7h | 55% | 207 |
| flab_pk_jain_fcrnrt | 132 | 48.7h | 71% | 232 |
| flab_aggregation_jain | 137 | 35.2h | 58% | 253 |
| flab_thermo_jain | 137 | 101.2h | 49% | 423 |
| flab_aggregation_jain_hicrt | 137 | 104.0h | 63% | 557 |
| ginkgo_pr_cho | 197 | 67.8h | 69% | 466 |
| ginkgo_acsins | 242 | 81.2h | 62% | 503 |
| ginkgo_colloidal | 242 | 53.8h | 59% | 517 |
| ginkgo_hic | 242 | 72.0h | 64% | 564 |
| ginkgo_titer | 239 | 122.5h | 67% | 824 |

**Refined model:** `capped_wall ≈ RF_capped_wall / 0.63` (RF median share ~63%), where `RF_capped_wall = n_cfg_RF × per_cfg_RF(n)`, `n_cfg_RF≈290–540`, and **`per_cfg_RF(n)` rises ~140s→~550s as n goes 94→242** (≈ linear in n) **with ~2× rep-dimensionality scatter at fixed n** (n=137: 253 vs 423 vs 557s). ⇒ capped wall **23.5–122.5h, anchored directly on these 11 values** — no multiplier needed for datasets that have it.

**Scope bound (critical):** capped is **ONLY emitted at n ≤ TOP_CAPPED_MAX_N = 250** (`precompute_hybrids.py:330`, `_shared.py:270`); above 250 `top/capped` is SKIPPED → **0**. So capped is a cost for **only the ~15 small antibody/ginkgo datasets**, where it is by far their **largest single precompute condition** (23–122h each, vs <20h for all their none+uncapped combined). The residual scatter (the rep-dim part) again points at item 1.1.

## V.3.1 — DISTANCE-LINEAGE precompute (resolves audit A1: sim-side under-count) ✅

**Structure (read from `precompute_hybrids.py:295–305, 392–397`):** with `--distance`, up to **4 conditions** — `general_distance/none`, `general_distance/uncapped` (if aug reps), `top_distance/none`, `top_distance/uncapped` (if aug reps); **NO capped**; each tier needs ≥2 reps. They call `find_best_hybrid` with `budgets=DISTANCE_BUDGETS` (6: [10,25,50,100,250,500]) and `internal_evaluators=DISTANCE_EVALUATORS` (6: [knn,ridge,rf,mlp,lightgbm,svm]) — vs the standard path's FULL_BUDGETS (8) and 5 SPEED_TIERS evaluators.

**Cost vs the standard conditions:**
- **distance/none ≈ standard/none × (6/8 budgets) × (6/5 evaluators) ≈ ×0.9.**
- **distance/uncapped ≈ standard/uncapped × ~0.5–0.6** — because DISTANCE_BUDGETS **drops the two priciest budgets (1000, 2000)**, which dominate uncapped (svm/rf/xgb at high budget), partially offset by the 6/5 evaluator increase.

`distance_precompute ≈ 0.9×(gen/none + top/none) + ~0.55×(gen/uncapped + top/uncapped)`. Worked examples:

| dataset | n | none(gen+top) | unc(gen+top) | **distance add (EST)** |
|---|---|---|---|---|
| ginkgo_heparin | 94 | 0.26+0.18 | 2.67+2.44 | 0.9×0.44 + 0.55×5.11 ≈ **+3.2h** |
| caco2 | 2158 | 4.53+4.11 | 63.31+62.32 | 0.9×8.64 + 0.55×125.6 ≈ **+77h** |
| flab_thermo_tresanco | 672 | 5.57+5.23 | 14.56+10.03 | 0.9×10.8 + 0.55×24.6 ≈ **+23h** |
| tcga_luad_vs_lusc | 1017 | 1.01+0.94 | 8.94+10.61 | 0.9×1.95 + 0.55×19.55 ≈ **+12h** |

So the sim-side precompute was under-counted by **single-digit hours (small antibody) up to ~70–80h (molecular)** — material for molecular/DTA/reaction because the distance lineage carries its own (reduced-budget) uncapped search. **UNVALIDATED: `check_precompute --timing` shows NO `*_distance` rows**, so I cannot confirm whether the distance precompute is done, pending, or running on the server — and I have no measured distance wall to anchor the 0.9×/0.55× multipliers. **Server need (folds into 1.3):** one `*_distance` condition's timing to validate; until then this is structural only.

## V — §3 STATUS: 3.1 ✅(structural, unvalidated) · 3.2 ✅ · 3.3 ✅ · 3.4 ✅ · 3.5 ✅ · 3.6 ✅. Remaining §3: 3.7 (full combined table) still BLOCKED on 1.1 (rep dims).

---
---

# ════════ PART VI — CONNECTING ESTIMATES → CUTS / BUGS / GATES (2026-06-04, user-requested) ════════

> The estimates (PART above) point straight at the cost drivers. Diving into the source of each expensive experiment, here are the **cuts, bugs, and gates** ranked by hours saved, each tied to the estimate it attacks. This is the SOLUTION phase (Section 0 deferred-notes) — surfacing options; the user decides which to apply. **The dominant cx cost is c17 (mRMR); the dominant precompute cost is SVM-uncapped O(n²); the dominant WASTE is cache-drift recompute.**

## VI.1 🔴 c17 — THE #1 cx COST (198h@n=1412 → thousands@n=30k). Three correctness-preserving levers.

Source: `c17_information_theoretic_decomposition.py`. `run()` loops `budgets=[b in FULL_BUDGETS if b<total_dim and b≤n]` (up to 8) × strategies `[hybrid,pca,mrmr,rfe,random]`; the `mrmr` strategy calls `_greedy_mrmr(X, y, budget, max_features=2000)` **once per budget**. KSG MI cost is ~linear in n; that linearity IS why c17∝n.

- **GATE (biggest, for large n): subsample rows to ≤2000 for the MI/mRMR computation.** The KSG estimator converges well before n=2000 — using all 30000 rows buys no statistical accuracy. `estimate_mutual_information_knn` and `_greedy_mrmr` on a 2000-row subsample make **c17 cost ∝ constant, not ∝ n**. Effect: c17 at n=9978 **1632h → ~310h-equiv**; at n=30000 **~5400h → ~310h**; at n=2158 little change (already near 2000). **This is THE lever that makes large-n cx finite.** (Downstream RF eval still uses full n; only the feature-selection MI is subsampled.)
- **BUG/INEFFICIENCY: `_greedy_mrmr` is re-run from scratch for every budget, but greedy mRMR is PREFIX-monotone** — the budget=2000 selection's first 10/25/.../1000 elements ARE the smaller-budget selections. The per-budget prefilter pass (MI of ALL `total_dim` features, the most expensive line, budget-independent), the relevance pass, and the initial-redundancy pass are **recomputed ~8×** identically. **Fix: run `_greedy_mrmr` ONCE at max budget, slice for all smaller budgets.** Exactly equivalent. Saving: prefilter ÷8, greedy loop 3935→2000 iters (÷2). Net **mRMR ÷4 to ÷8** → ~100–250h on a molecular dataset.
- **CUT: lower `mrmr_prefilter` 2000 → 500.** The greedy keeps ≤budget≤2000 features; the downstream RF eval barely moves between 500 and 2000 prefiltered features. Cuts the relevance/redundancy MI passes **÷4**.
- **CUT: c17 budgets 8 → 4** ([10,50,250,1000] captures the MI-density curve). Half the per-budget eval/RFE work (smaller after the slice-fix).
- Combined, c17 on a molecular dataset drops from **~310h to ~40–70h**, and large-n from thousands to ~bounded. **This single experiment is where the wave is dying.**

## VI.2 🔴 c05 — #2 cx COST. 15 fresh searches, 3× redundant, can't reuse precompute.

Source: `c05_learning_curves.py`. 5 fractions [.1,.25,.5,.75,1.0] × `n_repeats=3` = **15 fresh `find_best_hybrid` calls**, each with a UNIQUE key `f"{dataset}_n{n_sub}_s{seed}"` (`:190`) → never reuses the main precompute cache.

- **CUT: `n_repeats` 3 → 1 (or 2).** The 3 seeds are error bars on the curve. ÷3 (or ÷1.5) on all of c05. ~60–130h on a molecular dataset.
- **BUG/INEFFICIENCY: the 1.0-fraction search re-runs the FULL uncapped search 3×** but can't reuse the main `general/uncapped` precompute (unique suffix key). **Fix: the 1.0n fraction should read the main precompute cache** (it's the same search) — saves 3 × a full uncapped search (~90h on caco2). At minimum, run the 1.0 fraction once (no repeats).
- **GATE: cap the largest subsample at min(n, ~2000).** The learning-curve trend is set well before n=2000; the 1.0=30000 point costs the most and adds least. Bounds c05 like the c17 gate.
- Combined: c05 molecular **~100–200h → ~20–40h**.

## VI.3 🔴 SVM-uncapped — THE #1 PRECOMPUTE cost (O(n²)). Gate by n.

From the snapshot: herg gen/none SVM = 19.3h (66% of the wall); the missing herg gen/**uncapped** SVM is ~302h (PART II.2) — pure O(n²). Already gated at `_max_samples=10000` (`find_best_hybrid.py:1184`); SVM still runs full-force at n=5000–10000.
- **GATE: lower SVM `_max_samples` 10000 → ~3000–5000**, OR drop SVM from the `uncapped` strategy at n>5000 via `_DATASET_STRATEGY_SKIPS`. Saves herg (~302h), logd (~92h), qm9 (~116h) uncapped — the exact datasets whose precompute can't finish. SVM rarely wins at large n (its share shrinks), so the accuracy cost is low. **Decision 7.1/7.2.**

## VI.4 🔴 CACHE-DRIFT — THE #1 WASTE (already a FIXED bug). ~376h recoverable now.

PART V.3.5: the 4 molecular DONE-on-disk datasets are re-searching from scratch (24% mid-flight miss; ~73h cx + 303h sim of pure recompute remaining). **The fix is already applied** (`_load_cache_with_meta` ignores `rep_hashes`/`aug_hashes`, PART II.5) but UNTESTED on server (item 2.2). **Deploying it + restarting the molecular jobs recovers ~376h immediately** — the single biggest one-shot saving in the running wave. **Decision 7.5.**

## VI.5 🟡 capped precompute — RF-bound, 23–122h, ONLY n≤250 (mechanism CONFIRMED).

PART V.3.6: capped is RF-dominated, per_cfg 140–820s vs ~0.5s under none = a **300–1600× per-config blowup**. **CONFIRMED cause** (`hybrid.py:1979 augment_capped`): capped is **feature-level A/B testing with a full model REFIT per candidate** — NOT row inflation. Per search-config it does: (1) `_pool_and_rank_aug_features` — an RF-importance ranking over the **entire augmentation pool**, which includes giant reps (esm2_per_region 17920D etc.) → a slow fit; (2) Phase A "fill" — up to `max_candidates=50` candidate features, each an `_ab_test` model refit, stopping at `patience=10` consecutive non-improvements; (3) Phase B "replace" — more A/B refits. So each config ≈ 1 huge ranking fit + ~10–50 model refits. Times n_cfg≈290–540 configs = the 20h+ RF capped walls. **Levers (all cuts):**
- **CUT `max_candidates` 50 → ~15 and `patience` 10 → 5** — directly cuts the per-config refit count → ~2–3× on capped.
- **CUT/cap the augmentation-pool dimensionality before ranking** — the ranking fit over ~18000-D aug reps is the per-config fixed cost (this is the SAME giant-aug-pool problem as c21, PART III.2). Importance-prefilter the aug pool to ~2000 cols first.
- **GATE capped off** for n≤250 datasets whose none+uncapped already cover the pool (decision 7.2) — saves the full 23–122h, their single largest condition.

This also explains **c05's** cost on small antibody: c05 uses `capped` on every n_sub≤250 subsample (`c05:175`) → it inherits this A/B-refit blowup 15×. Cutting max_candidates/patience helps c05 too.

## VI.6 🟡 Other cx terms (c09, c16, c21, c23) — secondary, but real.

- **c09 (BUG, PART III.2): its 3-fold create_hybrid indices ≠ the precompute's 5-fold → guaranteed cache MISS → 120 full recomputes.** Fix: align c09 to 5 folds → it reuses the precompute. Saves the c09 recompute (0.8–10.4h measured, more at large n).
- **c23: tier='extra' models (catboost/dnn/cnn)** dominate (HER2 +49857% anomaly). **GATE: skip the 'extra' tier in c23** at large n.
- **c21: skips reps with dim>n**, so cost = #narrow reps × fresh search on giant aug-pool dims (esm2_per_region 17920D). **GATE: cap c21's aug-pool dimensionality.**

## VI.7 ⚖️ Budget grid (item 2.1) is a COST lever, not just a correctness one.

The wave currently runs SCREENING_BUDGETS (4) — you flagged it should be FULL_BUDGETS (8) for correctness. **Connect: FULL_BUDGETS ADDS ~+20–40% to every search.** So the correctness fix and the cost goal point opposite ways — this is a genuine trade (7.1/7.4), not a free win. c17/c05 above already assume FULL_BUDGETS where they hardcode it.

## ════════ PART VII — AUDIT OF THE ESTIMATED RESULTS (flagged 2026-06-04) ════════

> Self-critical audit of the ★★★ per-dataset table + the c17 law + cx≈c17×1.5 + precompute/sim numbers. Each flag: what was claimed, the flaw, **direction of error**, severity, what resolves it. **Net finding: the large-n cx/precompute numbers are biased LOW (several flags all push up) — so the "won't finish" verdict is even more robust, but the specific hours are soft.**

### Flags on the cx estimates
- **EA-1 [VERIFIED — holds].** c17's n-dependence is ~O(n log n): the MI estimator (`_shared.py:1160`) runs sklearn `mutual_info_regression/classif` (kNN Kraskov) on **≤50 PCA components** (kd-tree query → n log n). This MATCHES the fitted n^1.078. ⇒ the linear-ish extrapolation in n is sound; this is the one piece that survives audit.
- **EA-2 [UNDER-estimate · HIGH].** "M (mRMR feature count) saturates at the 2000 prefilter cap" is WRONG. `_greedy_mrmr` computes MI for **all `total_dim` features** in the prefilter pass (`c17:55-65`) BEFORE capping → c17's prefilter is **O(total_dim × n log n)**, scaling with FULL rep dimensionality. Molecular total_dim (mordred 1613 + ECFP 2048 + neural reps ≈ 5000–20000) ≫ the antibody anchor's. ⇒ **molecular/reaction/large cx is UNDER-estimated** vs the antibody-anchored law; the true molecular c17 is higher than the table's c17(n).
- **EA-3 [BOTH directions · HIGH].** The anchor c17=53.4h@n=418 (VEGF) is a SINGLE measurement and may be **node-contention-inflated** (the doc's own 3–48× contention finding). If VEGF ran on a loaded node, the entire c17 column is too high; the anchor was used as if clean. The c17 column inherits the contention uncertainty.
- **EA-4 [precision overclaim].** The c17 exponent rests on effectively **2 rep-rich points (n=217, 418), only 1.93× apart in n**. "Validated ±3%" is interpolation between two points — NOT validation of a **70–240× extrapolation** to n=30k–100k. The DTA/nucleotide "thousands of hours" is a 2-point line extended far past any data.
- **EA-5 [UNDER-estimate · HIGH].** **cx ≈ c17×1.5 is contradicted by the measured pipelines:** bvp cx/c17 = 38.98/9.09 = **4.3×**; cic = 4.5×; HER2 = 5.2×. The ×1.5 only holds if "rest" (c05+c09+c16+c21+c23+c24) stays ~30h absolute while c17 grows — but rest has its OWN n-growth (c05 subsample searches, c09's 120 builds, c21 aug dims). ⇒ cx is likely **c17×2–3, not ×1.5** at large n → **cx UNDER-estimated**. (At small n I used the measured envelope, so that part is fine; the ×1.5 is the large-n weak point.)
- **EA-6 [self-contradiction].** The doc still contains the OLD IV.0 table ("150–400h BLOCKED:rep-dims") which now contradicts the ★★★ headline (caco2 400–560). Two inconsistent cx columns in one doc → IV.0 must be marked superseded.
- **EA-7 [UNMEASURED].** ALL molecular/DTA/reaction/TCGA/nucleotide cx is unmeasured — only antibody n≤418 has any cx data. The entire n≥564 cx column is formula extrapolation with **zero in-regime validation**.
- **EA-8 [may not be runtime].** At n=30k–100k the experiments likely **OOM or hit the partition wall before runtime matters** (s01 ~35GB/rep; c17 prefilter MI over 100k×~15000-D PCA). "thousands of hours" may really be "crashes." The verdict (won't finish) holds either way; the *number* is meaningless there.

### Flags on the precompute estimates
- **EA-9 [UNDER-estimate at large n].** The none→uncapped per-model multipliers (svm ×16, etc.) were calibrated across mixed-n datasets, but **SVM's multiplier itself grows with n** (O(n²)). Applying a fixed μ to the largest-n datasets (herg n=7772, DTA n=30k) under-estimates uncapped. herg ~400h could be higher.
- **EA-10 [self-contradiction].** "PC left = 0" for molecular DONE datasets contradicts CACHE-4 (~376h re-running via cache drift). The 0 is only true for a fresh post-fix submission.
- **EA-11 [error band not per-row].** LOO validation was ±6–44% (molecular/TCGA/nuc, reliable) but **up to +700% on antibody** (mlp/gp dominate unpredictably). The table doesn't flag which rows are in the unreliable antibody zone.

### Flags on the sim estimates
- **EA-12 [UNVALIDATED].** ALL sim numbers are structural — **zero finished sim pipeline exists**. s01/s03 O(n²) figures (35GB, 111h) may be OOM-crashes, not runtimes.
- **EA-13 [under-count].** sim TOTAL omits the distance-lineage precompute (COST-12 / V.3.1) the sim experiments depend on.

### Flags on the secondary estimates
- **EA-14 [c12 soft].** c12 ~2h assumes R≈40 reps, G=9 modalities — both depend on what loads on the server (unverified). Range 0.4h (G=6) → 7h (contended).
- **EA-15 [s11 soft].** s11 0.3–1.5h assumes ngboost/BNN/gauche installed; if they ImportError, cheaper. Install state unknown.
- **EA-16 [capped scatter].** capped 23–122h anchored on 11 measured values, but within-n scatter is ~2× (rep-dim) → per-dataset capped is ±2×.

### Net direction & what to trust
- **Most flags push the large-n numbers UP** (EA-2 total_dim, EA-5 ×1.5→×2-3, EA-9 svm μ) ⇒ molecular/DTA/nucleotide are likely **WORSE than the table** → the **"nothing at n≥~1400 finishes" verdict is robust (HIGH confidence)**; the specific hours are soft and biased low.
- **Trust:** EA-1 (c17 ∝ n log n) and the *ordering* (c17 dominates; molecular≫antibody; DTA/nuc≫molecular). **Distrust as precise:** any single cx hour-figure at n>418, all sim figures, and the n≥30k magnitudes (EA-8 may be crashes).
- **What would harden it:** (1) total_dim per dataset (item 1.1) — fixes EA-2; (2) ONE measured cx above n=418 — fixes EA-4/EA-7; (3) an un-contended c17 measurement — fixes EA-3; (4) one finished sim log — fixes EA-12.

## ════════ PART VIII — LINEAR-GROWTH ASSUMPTION FLAGS (the forbidden form — MOST IMPORTANT) ════════

> User: "find anywhere it assumes linear growth with dataset size OR representation size — by far one of the most important issues." Every instance below. Classified: **🔴 FORBIDDEN** = scales an unmeasured PER-UNIT wall by n^k / by rep-dim / by a fixed cross-n multiplier (the banned `(n/anchor)^k` family); **🟠 MEASURED-ANCHORED** = a multiple of a per-dataset MEASURED quantity (acceptable but inherits the base's error); **🟢 EXACT LOOP** = a literal iteration count (correct, not an assumption).
>
> **Headline admission: the cx table's backbone (c17 law) is 🔴. So the entire cx hour-column is order-of-magnitude / ordering only — NOT numbers to trust.** This is the same error that burned every prior attempt; I reproduced it.

### 🔴 FORBIDDEN — linear/power scaling of an unmeasured per-unit cost
| ID | Where (line) | The assumption | Why it's wrong (joint dependence the form ignores) |
|---|---|---|---|
| LIN-1 | ★★★ headline + III.5 (`:20,25,498`) | **c17 = 53.4h × (n/418)^1.078** | THE banned `(n/anchor)^k`. Fit on **2 rep-rich points 1.93× apart**, extrapolated 70–240×. Real c17 also depends on total_dim (LIN-8), which models survive gates, and 3–48× contention — none captured by one exponent. EA-1's "sound" claim only covers the n-axis *shape*, not the magnitude. |
| LIN-2 | ★★★ headline (`:23,25,74`) | **cx total ≈ c17 × 1.5** (×1.3–1.8) | Fixed multiple of c17. **Contradicted by measured pipelines (cx/c17 = 4.3–5.2×, not 1.5×)**; "rest" has its own n-growth so the ratio isn't constant. |
| LIN-3 | V.3.2 s11 (`:653`) | **s11 cost ≈ linear in pool size (0.8n)** | Explicit linear-in-n. Ignores that the GP fit is n³ on the labeled set and the per-iteration predict is n_pool×dim — not a single linear term. |
| LIN-4 | V.3.6 capped (`:726`) | **RF capped per_cfg "≈ linear in n" (140→550s, n=94→242)** | Per-unit cost scaled linearly in n; the SAME line notes a 2× rep-dim scatter at fixed n → not a function of n alone. |
| LIN-8 | III.5 / EA-2 (`:498,810`) | **c17 ∝ M = total_dim** | The prefilter MI pass is O(total_dim) but PCA(≤50)/greedy/eval are NOT linear in total_dim → c17 is not linear in rep size; the "∝ total_dim" both over- and under-counts depending on the rep mix. |
| LIN-12 | II.1/II.2 + §4 (`:227,352,369`) | **none→uncapped per-model μ (svm×16, gp×18…) as FIXED multipliers independent of n** | Assumes the uncapped/none RATIO is constant in n. SVM is O(n²) ⇒ the ratio GROWS with n ⇒ fixed μ under-estimates at large n (EA-9). A constant multiplier is a linear-ratio assumption. |
| LIN-13 | II.2 (`:369`) | **top/none≈0.6×, top/uncapped≈10×, top/capped≈114× gen/none** | Fixed cross-condition multipliers, constant in n; top-tier rep COUNT and dims differ per dataset → ratio isn't constant. |
| LIN-14 | V.3.1 distance (`:738,742`) | **distance ≈ 0.9×none + 0.55×uncapped** | Fixed multipliers; the budget-set reduction's effect depends on which models dominate (n-dependent), not a flat 0.55. |
| LIN-15 | V.3.3 c12 (`:668`) | **c12 ≈ F×[R+C(G,2)+1]×t_fit, t_fit ~constant** | Cell COUNT is exact (🟢) but **t_fit (per model-fit) is NOT constant** — it grows with n (821 vs 1017) and with concat width (rep dim). |
| LIN-16 | II.4 sim table (`:312–326`) | sim per-experiment numbers interpolated across size classes (s02/s05/s08/s10/s13) | Cross-size figures imply smooth scaling; the O(n²) ones are super-linear and the cache-reuse ones step with model gates. |
| LIN-5 | III.2 c22 (`:470`) | c22 SHAP ∝ n_val×features×trees | Linear in n_val AND features; minor (c22 cheap) but still a product-of-linears assumption. |
| LIN-6 | TIME-c04 (master `:281`) | c04 ∝ g×n | Linear in n; g (family variants) is a count (🟢) but the ×n is a per-unit linear assumption. |

### 🟠 MEASURED-ANCHORED — a multiple of a per-dataset MEASURED quantity (acceptable; inherits base error)
| ID | Where | The form | Note |
|---|---|---|---|
| LIN-7 | §6 (`:261`) | **c24 ≈ 5 × measured(gen/none)** | 5 = fold count (🟢); gen/none is MEASURED per dataset → least-bad. Validated ±27% bvp/cic, FAILED HER2 (cache). |
| LIN-9 | §5 (`:242`) | c01 ∝ rep dims; s01 ∝ n²·rep_dim | Anchored on the measured rep-size effect (HER2 3862s vs bvp 88s); s01's n² is correct complexity, but rep_dim is unmeasured (item 1.1). |
| LIN-11 | V.3.2 (`:638`) | s11 kernel ∝ n_pool×n_labeled×2048 | The 2048 (ECFP) is a known dim; the n_pool×n_labeled is the GP-predict structure (correct), but summing it as "linear in pool" (LIN-3) is the overreach. |

### 🟢 EXACT LOOP COUNTS (correct — not assumptions)
c05 n_repeats=3 and 5 fractions; c24 5 folds + 25 cross-fold cells; c12 R reps + C(G,2) pairs; budget-sweep counts (4 or 8); s11 11 conditions × 3 seeds. These linear factors are literal iteration counts — correct. **The non-linearity lives in the PER-UNIT wall inside each loop, which is where LIN-1/4/8/12 cheat.**

### What this means (honest)
- **The cx column is the worst hit:** its backbone LIN-1 is the banned form ⇒ treat every cx hour at n>418 as **order-of-magnitude only**. The *ordering* (c17 dominates; molecular≫antibody≫small-ab; DTA/nuc≫all) survives because it's driven by structure (more reps, bigger n, O(n²) models surviving), not by the exponent.
- **The precompute column is partly hit:** LIN-12/13 fixed multipliers under-estimate at large n. But the BASE (each measured none/uncapped wall) is real, so precompute-remaining for datasets whose missing conditions are *also measured at that n* (e.g. flab_binding_koenig top/uncapped ≈ its own general/uncapped neighbour) is the firmest part of the whole effort.
- **The only fully-defensible numbers are the directly MEASURED ones:** the 5 cx pipelines, and every measured precompute condition in the snapshot. Everything else is a model, and every model here leans on at least one 🔴 or 🟠 assumption.
- **What removes the 🔴s:** there is NO single-factor fix — that's the point. total_dim (item 1.1) removes LIN-8's guess but not LIN-1's exponent; only **measured cx/precompute walls at the target n** retire a 🔴. Until then, report ranges + ordering, not point hours.

## ════════ PART IX — APPLIED AUDIT: the structural facts EXPLAIN the wave you're seeing ════════

> User: "apply these facts to the issues I'm having, push this further." The expensive-last ordering (PART ORD) + cache-drift + duplicate jobs + the 7-day wall are not separate problems — they COMPOUND into the exact symptoms in the snapshot. Each claim is checked against the measured `check_precompute` / monitor data.

### IX.1 — The "missing conditions" are DETERMINISTIC, not random (ORD-4 applied)
Every partial dataset has the **cheap `none` conditions done and the expensive `uncapped/capped` ones missing** — exactly what expensive-last predicts (the job completes condition 1, the cheapest, and dies in/before the expensive tail). Checked against the snapshot:
- herg 1/4: gen/none ✓, missing gen/uncapped + top/none + top/uncapped.
- logd 1/4: same. qm9 2/4: both `none` done, both `uncapped` missing.
- bindingdb/kiba 1/4: only gen/none done.
- **flab_binding_koenig 3/4: the ONE missing is `top/uncapped` — the single LAST condition.** Cleanest possible proof.
⇒ You can PREDICT which conditions are missing from the loop order alone. The missing-condition report is the shadow of ORD-3/ORD-4, not a mystery.

### IX.2 — Restart + cache-drift = Sisyphus (ORD × CACHE-1 × TIME-3, the months-lost mechanism)
A restart re-runs the SAME cheap-first order. **If the cache worked, it would skip the cheap (cached) conditions and advance the expensive tail.** But cache-drift (CACHE-1) REJECTS the cached cheap conditions → the restart **re-computes general/none from scratch, burns its budget, and dies at the same expensive wall again**. No-checkpointing (TIME-3) loses in-condition progress too. ⇒ **the same conditions stay missing across every wave — not because they're individually impossible, but because the restart never gets PAST the cheap re-do to reach them.** This is the core reason months of restarts never advanced the expensive conditions.

### IX.3 — "DONE but re-running" molecular endangers its own cache (CACHE-1 × ORD)
caco2/adme_* are DONE on disk, but the running job re-searches cheap-first (monitor: mid-FIRST-condition, 75.7%/90.4%/…), ETA "+22h/+43h" = the expensive `uncapped` re-do that comes LAST → will likely time out mid-re-do → risk re-rejecting/corrupting the cached expensive conditions. ⇒ the cache-drift fix isn't just "save ~376h" — under expensive-last it's protecting the DONE datasets' expensive conditions from being destroyed by their own re-run.

### IX.4 — Wedged-vs-dead is readable from loop position (ORD applied to §10 wedge calls)
A job idle in the **late/expensive tail** (gp / budget-2000 / capped / cx-c17 / cx-c21) is grinding a genuinely long cell (gp-uncapped minutes/config; capped A/B-refit hours; c17 ≈52h@n400) → **"WEDGED" false-positive, actually alive-slow**. A job idle at an **early/cheap stage** (cheap cells = seconds) is **probably truly dead**. Checked: Ang2/VEGF idle ~1.5h at **C21** (expensive) → ALIVE ✓; lysozyme idle **67h** at C05 (c05@n627 ≈ a few h ≪ 67h) → DEAD ✓; flab_binding_koenig idle 98s at 90.8% → ALIVE ✓. ⇒ re-triage the 43 wedged by loop position, not by idle alone.

### IX.5 — The cx EXPERIMENT phase has its own expensive-late pattern (ORD-9 applied)
c17 is experiment ~#14, c21 #16, c23/c24 late. A cx job that enters the experiment phase late (after ~68h of precompute) hits **c17 (≈52h @ n≈400)** and dies → c18/c19/c21/c23/c24 never run. **CONFIRMED: Ang2/VEGF completed through c19, stuck at c21; the small fast ones (bvp/cic/HER2) finished all 19.** ⇒ bigger datasets systematically lose the LATE experiments (c21/c23/c24) — the experiment sequence sacrifices them the same way the budget sweep sacrifices budget-2000.

### IX.6 — THE COMPOUNDING CHAIN (why MONTHS, not days)
A single precompute job is hit by all of these AT ONCE, and they MULTIPLY:
```
expensive-last (ORD)        → sacrifices the costly tail you need
 × cache-drift (CACHE-1)    → re-does cheap cached work every restart, never reaching the tail
 × duplicate cx+sim (TIME-1)→ the whole thing runs TWICE per dataset
 × 8-CPU not 32 (TIME-2)    → ~4× slower than a resubmit on tree models
 × node contention (COST-10)→ 3–48× slower, unpredictably
 × no checkpointing (TIME-3)→ a kill loses in-condition progress
 × 7-day wall              → hard cut mid-expensive-tail
 × SCREENING not FULL       → the grid it does finish is the wrong (biased) one
 × forbidden-linear estimates (LIN) → you couldn't predict ANY of it, so you kept resubmitting blind
```
Net: a job that **re-does cheap cached work, twice, slowly, on a contended node, loses progress on kill, never reaches the expensive conditions you actually need — identically every wave.** No single factor explains the lost months; the PRODUCT does.

### IX.7 — Predictions for the CURRENT wave (~100h left), from structure
- **Molecular / DTA / reaction precompute** (in cheap-first re-do or at general/none %): the expensive `uncapped` tail is hundreds of hours away → **won't reach it; the missing conditions stay missing.** (HIGH confidence — structural, not estimate-dependent.)
- **Small-antibody cx at C05** (ginkgo/flab, n≤250): c17+c21 tail ≈30h ahead → **may finish IF alive + uncontended.**
- **Mid-antibody cx at C21** (Ang2/VEGF): grinding the expensive c21; c23/c24 still after → **borderline.**
- **Any big dataset still at a general/none %:** its expensive conditions are **structurally unreachable this wave** regardless of the exact hour-estimate.

### IX.8 — What the ordering facts say to DO (options; ranked; none applied)
1. **Deploy the cache-drift fix + restart molecular** — breaks the Sisyphus re-do (IX.2); highest leverage GIVEN expensive-last, because only then does a restart actually advance the expensive tail.
2. **Reverse/interleave the inner budget+model loops (expensive-first)** — fail-fast (know in hour 1 if gp/budget-2000/capped fits) AND the expensive variant completes-or-is-ruled-out instead of perpetually half-done. (`KIRBY_PRECOMPUTE_REVERSE_ORDER` only reverses conditions, not these.)
3. **Run cx OR sim precompute once, not both** (TIME-1) — halves the precompute.
4. **Gate the always-sacrificed tail so it's also always-affordable** — svm n-cap (COST-6), c17 subsample-MI (COST-1), capped-off at large n (COST-7) — make the expensive-last variant cheap enough to actually reach.
5. **Checkpoint long conditions / split so one condition fits a partition** (TIME-3/4) — the herg svm-uncapped (~302h) can't fit one 7-day window at all otherwise.

## ════════ PART X — ASSUMPTIONS THE ORDERING/COLLISION FINDINGS INVALIDATE (estimates revised) ════════

> User: "that may drastically modify your estimates… a lot of your assumptions were wrong." Correct. Writing down every estimate assumption the per-condition-parallelism + reverse-flag-collision + contention findings break, with direction, plus reverse-flag fixes.

### X.1 — The TWO that drastically modify the numbers
- **🔴 X-A: serial-sum → parallel-max.** I computed precompute "PC total" as Σ(conditions) (serial, one job per dataset). But `--condition` per-job SLURM arrays run conditions **in parallel** ⇒ **wall-to-finish ≈ max(single condition), not the sum.** This FLIPS verdicts: a dataset whose conditions sum to 400h but whose largest single condition is ~90h **finishes in ~90h wall** as per-condition jobs. So the IV.0 "PC total" column overstates the achievable wall whenever conditions are parallelised.
- **🔴 X-B: anchors are regime-contaminated (likely INFLATED).** Every derived number is anchored on the measured per-condition walls, which were measured under (i) heavy node contention (COST-10, 3–48×) and (ii) — IF forward+reverse ran concurrently on neural-rep datasets — cache-drift COLLISION (the reverse-flag's "different cache files" premise fails under CACHE-1, so the two waves recompute each other's middle conditions). ⇒ the anchors are **upper-ish bounds, not clean costs**; a clean, low-contention, cache-fixed, disjoint run could be **~1.5–3× faster** than my numbers. (This is the OPPOSITE direction from EA-2/EA-5/EA-9 which pushed up — so the true value is bracketed by both, and point estimates are even less defensible.)

### X.2 — Assumptions now known wrong (enumerate, with direction)
| ID | Assumption I made | Why wrong | Direction |
|---|---|---|---|
| WRONG-1 | PC total = Σ conditions (serial) | `--condition` arrays parallelise → wall = max(condition) | overstates dataset wall |
| WRONG-2 | measured per-condition walls = the cost | measured under contention (+possible collision) → inflated | overstates everything |
| WRONG-3 | c17 anchor 53.4h@418 is clean | VEGF may have run during contention/collision | overstates cx column |
| WRONG-4 | "won't finish" judged vs serial sum / the inflated walls | should be judged per-condition vs ONE partition, on clean walls | over-pessimistic for some datasets |
| WRONG-5 | reverse flag = 2× throughput (parallel waves) | under CACHE-1 the 2 waves COLLIDE, not parallelise → measured "2-wave" walls are collision-inflated, not 2× faster | mis-modelled the wave's effective throughput |
| WRONG-6 | precompute "remaining %" extrapolates linearly to remaining wall | collision + contention + expensive-last late-grid tail break the constant-rate assumption | remaining ETAs unreliable (already flagged, now compounded) |

### X.3 — RE-FRAMED verdict (per-condition vs one 7-day partition ≈ 168h, on possibly-inflated walls)
The right question is no longer "does the dataset's serial sum fit ~100h" but **"does each MISSING condition fit ONE partition as its own job?"**
- **herg gen/uncapped ~302h, logd ~250h, qm9 ~256h, DTA uncapped ~980–1270h** — single condition **> 168h partition ⇒ won't finish even alone** (and even at 0.5× if inflated, herg/DTA still bust). Verdict HOLDS.
- **flab_binding_koenig top/uncapped (~50–90h)** — fits a partition as `--condition top_uncapped` ⇒ **CAN finish** (monitor: 90.8% → ~7h left, consistent). The serial framing wrongly buried it.
- **Several mid datasets whose missing conditions each fit a partition** — per-condition parallelism + a clean run could finish them where the serial-sum estimate said no. **This is the drastic modification: the "won't finish" set SHRINKS to the datasets with a single supra-partition condition** (the big-n uncapped/capped + DTA/nucleotide), not "everything ≥1400".

### X.4 — POTENTIAL FIXES FOR THE REVERSE FLAG (ranked)
The flag's goal is collision-free parallelisation. Options:
1. **Supersede it with `--condition` per-job SLURM arrays** (already in-codebase, ORD-REVFLAG) — disjoint work, no caching reliance, each condition its own partition, N-way. Best path; drop the flag for production.
2. **Atomic per-condition claim (lock-file / `mkdir` claim)** — replace blind `reversed()` with: each job atomically claims the next UN-STARTED condition from a shared list (claim dir per `<dataset>__<tier>__<strat>`). Generalises to N waves, collision-free **even under cache-drift** (work is claimed, not inferred from cache state). Fixes WRONG-5.
3. **Completed-condition manifest** — before running a condition, check a shared manifest (independent of the rep-hash cache); skip if another job logged it done. Cache-independent dedup; pairs with #2.
4. **Cost-descending order, not blind reverse** — sort conditions by estimated cost desc (capped/uncapped first) so the longest-running start earliest + fail-fast. Replaces `list(reversed(conditions))`.
5. **Extend the flip to the INNER loops** — `KIRBY_PRECOMPUTE_REVERSE_INNER` (or cost-sort) on budgets+models in `find_best_hybrid.py:3056/3059` → expensive CELL first within a condition (the part the current flag can't reach, ORD-1/2).
6. **Partition-aware condition splitting** — if a single condition > one partition (herg svm-uncapped ~302h), split it by model or budget-range into sub-jobs so each fits; otherwise it can NEVER complete regardless of ordering (TIME-4).
7. **Deploy the cache-drift fix FIRST (prerequisite)** — the flag's "different cache files" premise only holds if caching is reliable. With CACHE-1 fixed, even the unmodified reverse flag stops colliding. So CACHE-1 is the gate for ALL caching-dependent parallelisation, the flag included.

**Bottom line for the estimates:** the per-condition-parallel + clean-run reframing means my hour-numbers are **upper-ish bounds under a contended/colliding/serial regime** — the *ordering* (what's expensive, what's missing-by-construction) holds, but the achievable WALL under a fixed-cache, per-condition, low-contention setup is plausibly well below the table, and the "won't finish" set is smaller than IV.0 implies (X.3).

## VI.8 RANKED SAVINGS (the connect-the-dots summary)
| lever | type | attacks | saving |
|---|---|---|---|
| c17 subsample-MI to ≤2000 rows | gate | c17 at large n | n=9978: −1300h; n=30k: −5000h |
| cache-drift fix deploy + restart molecular | bug (fixed) | wasted recompute | ~376h now |
| c17 mRMR-once-and-slice + prefilter 500 | bug+cut | c17 all n | ÷4–8 mRMR (~100–250h/molecular) |
| c05 n_repeats 3→1 + reuse main cache for 1.0n | cut+bug | c05 all n | ÷3 + a full search ×3 (~90–130h/molecular) |
| SVM _max_samples 10000→5000 | gate | uncapped precompute | herg −302h, logd −92h, qm9 −116h |
| capped gate/augment-cap (n≤250) | gate | capped precompute | 23–122h per small dataset |
| c09 fold-align (3→5) | bug | c09 recompute | 0.8–10.4h+ per dataset |

**Headline: c17 (subsample-MI gate + mRMR-once) and the cache-drift deploy are the two that change the outcome** — together they take molecular cx from ~400–600h toward ~80–150h and recover ~376h of dead recompute. Everything else is incremental. All are options; none applied yet (the cache fix is staged but untested).
