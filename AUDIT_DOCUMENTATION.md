# ANNITIA Documentation Forensic Audit
**Date:** 2026-05-08 (deadline day)
**Auditor:** Background agent
**Scope:** All ANNITIA planning + analysis docs in `tests/experiments/annitia/`,
`tests/results/annitia/full/`, and `ANNITIA_TRANSFER_ROADMAP.md`.

Adelaide has ~10 submissions left and is frustrated because previous summaries
"missed context" and "ignored her suggestions". This audit is forensic — it
flags contradictions, silently-false `[x]` checkboxes, punted critical work,
and miscalibrated analysis claims.

---

## TL;DR — five highest-impact findings

1. **Scoring formula contradiction (the leaderboard_history `_comment` is wrong).**
   `tests/experiments/annitia/leaderboard_history.json:2` says the LB is "**average**
   of hep + death concordance". The actual formula is `0.7 * Cindex_hepatic +
   0.3 * Cindex_death` (`run.py:210`, `README.md:13`, `ANNITIA_TRANSFER_ROADMAP.md:13`,
   `build_submission_zip.py:91`). **All MORNING_BRIEFING.md "expected macro" estimates
   for the subgroup variants use the wrong arithmetic** — the directional verdicts
   may still be correct, but the numbers Adelaide is being shown
   (`subgroup_blend ~0.89-0.90`, `subgroup_hep_only ~0.88`) are computed from a
   false formula. Real LB-equivalent numbers are ~0.86-0.88.
   *Why it matters today:* if Adelaide is making include/exclude decisions on
   numerical thresholds (e.g. "0.9302 floor"), the numbers in the briefing aren't
   on the same scale as the LB.

2. **`oof_to_leaderboard.md`/`json` is stale — the markdown report shows
   the OLD (averaging) calibration even though the script was fixed.**
   The script `oof_to_leaderboard.py` was edited (mtime 03:09) so that line 152
   correctly computes `oof_lb = 0.7 * h_c + 0.3 * d_c`. But the artifacts at
   `tests/results/annitia/full/oof_to_leaderboard.{md,json}` are from 02:34 — they
   still report `OOF_avg = 0.8786` and a slope of 2.34 derived from the wrong
   formula. The MORNING_BRIEFING never re-ran the calibration script.
   *Why it matters today:* the calibration line `LB = -1.1365 + 2.3403 · OOF_avg`
   in the briefing is meaningless. The "projected LB" table is bogus.

3. **Round 5 hparam tuning never ran. `tests/results/annitia/hparam/` is empty.**
   `MORNING_BRIEFING.md:172` says "experiment_hparam.py crashed in 3s with `top
   view set = []`. Bug in CV summary read. Not worth debugging today." But
   `SUBMISSION_PLAN_2026_05_08.md:17` (slot 3) and the auto-chain `[x] auto_chain.sh
   waits for Round 4, runs Round 5...` checkbox in `USER_SUGGESTIONS.md:60`
   both reference Round-5-tuned recipes as if they're real. The plan still lists
   "Best Round-5-tuned single-model recipe" as a Tier 2 submission slot.
   *Why it matters today:* slot #3 in `SUBMISSION_PLAN_2026_05_08.md` is empty.
   Adelaide needs to know that submission can't ship and replan.

4. **T2 NHANES pretraining is one recipe, not the 60-cell ablation.**
   Roadmap T2.3 promises 5 subsets × 6 recipes × 2 endpoints = 60 conditions.
   `external/zeroshot.json` shows ALL 5 subsets crashed with
   `'NoneType' object is not callable`. `external/cv_summary.json` shows
   `t23_subsets_run: []`. Only `pretrained-mlp__nhanes_proj` (single recipe)
   landed in `branches/`. Hep CV 0.706 LOW, death CV 0.756 LOW per
   `oof_diagnostics.md:37,61`. `MORNING_BRIEFING.md:164` admits "Round 4
   cross-endpoint pretraining added basically nothing" — but
   `USER_SUGGESTIONS.md:51` still has the entire T2 plan as `[ ] pending`,
   and `SUBMISSION_PLAN_2026_05_08.md:33` describes Round 4 as if NHANES T2.x
   ran multi-recipe. **Three docs disagree about what actually ran.**

5. **All foundation models the roadmap promised (MITRA, TabTune, SurvTRACE,
   TabDDPM, TabSyn) are absent. None landed.** No `oof_*mitra*` or `oof_*tabtune*`
   or `oof_*survtrace*` in `branches/`. Only TabPFN-v1 landmark, TabPFN-2.5
   discretised, TabSurv (D2H/H2D/K10/K15), and DSM-K4 made it. The ANNITIA
   roadmap (T1.1-T1.5) is largely on paper. `USER_SUGGESTIONS.md:52` flags
   this honestly: torch <2.4 blocked them, conda env attempt "in progress"
   — but `LOCAL_TESTING.md:39-46` lists them all as runnable conditions
   (just "skipped if absent") which understates how much work is missing.

---

## Section 1 — Contradictions between documents

### 1.1 Scoring formula
- `tests/experiments/annitia/leaderboard_history.json:2`
  > `"_comment": "Trustii leaderboard scores keyed by submission CSV stem.
  > Concordance index = **average of hep + death** concordance on the
  > val_test holdout."`
- `tests/experiments/annitia/run.py:210`
  > `score = 0.7 * c_h + 0.3 * c_d`
- `tests/experiments/annitia/README.md:13`
  > "The score is `0.7 * Cindex_hepatic + 0.3 * Cindex_death`"
- `ANNITIA_TRANSFER_ROADMAP.md:13`
  > "Score weight = 0.7 hep + 0.3 death."
- `tests/experiments/annitia/build_submission_zip.py:91`
  > `0.7 * hepatic_search['best']['mean_cindex'] + 0.3 * death_search['best']['mean_cindex']`
- `tests/experiments/annitia/oof_to_leaderboard.py:8` (docstring, recently fixed)
  > "leaderboard score is LB = 0.7 * Cindex_hepatic + 0.3 * Cindex_death"

Conclusion: 5 sources agree on 0.7/0.3, ONE (the `_comment`) is wrong. Adelaide
flagged this in the audit prompt. The problem: the wrong number propagated into
the morning briefing's macro-estimate language even though the *code* is right.

### 1.2 Hepatic event count
- `README.md:20`: "Hepatic events are rare (47 / 1253)"
- `oof_diagnostics.md:7`: "hepatic: 47 events"
- `failure_analysis.md:13`: "**n_valid=984, n_events=39**, event_rate=0.040"
- `ANNITIA_TRANSFER_ROADMAP.md:12`: "hepatic (47 events / 1253)"
- `MORNING_BRIEFING.md:157`: "Hep (47 events)"

`failure_analysis.py` strictly drops invalid rows + applies endpoint-specific
filters → 984 valid, 39 events. The other counts are pre-filter. Both can be
correct, but no doc reconciles them. *Why this matters:* if the briefing's
"47 events / SE_C ≈ 0.025" calculation uses 47 (line 159) but the failure
analysis uses 39, the noise floors disagree.

### 1.3 Death event count and dataset size
- `ANNITIA_TRANSFER_ROADMAP.md:12-13`: "death (~750 events / 990 after 21% blank
  exclusion)"
- `oof_diagnostics.md:8`: "death: **76 events**"
- `failure_analysis.md:31`: "n_valid=695, **n_events=76**"
- `README.md:124`: "Dropping those patients shrinks the death-endpoint training
  set to 966"
- `MORNING_BRIEFING.md:157`: "Death (76 events)"
- `project_annitia_2026_05_08.md:33` (memory): "47 events / 1106 patients"

The roadmap's "~750 events" is wildly out of line with everything else. If the
roadmap was used to set expectations (e.g. "hepatic is the bottleneck because
death already has plenty"), and reality is 76 deaths not 750, the bottleneck
hierarchy is different.

### 1.4 Round 4 verdict
- `USER_SUGGESTIONS.md:51`: "[ ] Round 4 EXTERNAL stage adds T2.0 (NHANES
  pretrain × 5 subsets) + T2.1 (zero-shot) + T2.2 (6 transfer recipes) + T2.3
  (5×6×2 ablation) — ~70 more models"  *(unchecked, treated as future work)*
- `MORNING_BRIEFING.md:164`: "Round 4 cross-endpoint pretraining added basically
  nothing — variants are informationally redundant with the existing pool"
  *(treated as completed and evaluated)*
- `SUBMISSION_PLAN_2026_05_08.md:15-16`: "`meta_avg_aware_weighted_v2`
  (Round 4 expanded pool, same recipe)" + "`meta_avg_grand_all_v2`
  (Round 4 expanded, equal weight)" *(treated as available, with new pool)*
- `ANNITIA_TRANSFER_ROADMAP.md` describes T2 as the upcoming work for tonight.

The truth: only `pretrained-mlp__nhanes_proj` (single recipe, single subset)
plus 4 cross-endpoint transfer branches (catboost-D2H/H2D, mlp-D2H/H2D,
tabsurv-D2H/H2D) actually ran. All 5 zero-shot subsets crashed
(`external/zeroshot.json`). T2.3 ablation: 0 cells. So:
"Round 4 added basically nothing" is fair *for what ran*, but the original
T2 roadmap is essentially undelivered, which the morning briefing buries.

### 1.5 Tier1 fallback semantics
- `MORNING_BRIEFING.md:99`: "tier1_fallback — **identical** to
  meta_avg_aware_weighted (already used as anchor for the 0.9302 LB)"
- `SUBMISSION_PLAN_2026_05_08.md:112`: "`tier1_fallback.sh` rebuilds the
  original 0.9302 recipe from the existing pool — that's our floor."

Reading `tier1_fallback.sh:23-32`: it's a deterministic rebuild from the
same 15 variants with the same weights. So "identical" is correct in
intent, but the rebuilt CSV `sub_master_tier1_fallback.csv` is a NEW file
written by the script — Adelaide should know whether re-running this gives
byte-identical output to the 0.9302 anchor or whether floating-point order
of operations could nudge it. Not verified anywhere.

### 1.6 OOF C anchor numbers
- `auto_chain.log` summary tail (last block):
  > "hepatic: best=rank_mean OOF C=0.8980; death: best=specialist_override OOF C=0.9736"
- `MORNING_BRIEFING.md:55`: "Existing ensemble pool best | 0.8980 | **0.9736**"
- `oof_diagnostics.md:51`: best single death model = `tabpfn-landmark` 0.933
  HIGH-trust (not 0.9736 — that's an ensemble fusion variant, not a base)

The 0.9736 is a fusion-level OOF C, but in the `MORNING_BRIEFING.md:62-63` it
sits in the same column as the global Cox 0.7439 as if comparable single
models. Not labelled. Confusing.

### 1.7 OOF→LB calibration projection
- `oof_to_leaderboard.md` (stale, lines 7-9, 14-21):
  > Linear: LB = -1.1365 + 2.3403 · OOF_avg
  > Projected LB at OOF=0.94 → 1.063 (impossible — bounded by 1.0)
- `oof_to_leaderboard.py:152` (current code) computes the LB-formula OOF, not
  the average → would generate a different fit, not yet run.

The projection table in the .md is non-physical (>1.0). This alone should have
flagged the bug. No one re-ran the calibration after fixing the script.

---

## Section 2 — User suggestions NOT implemented

### 2.1 Bullets in `USER_SUGGESTIONS.md` marked `[ ]` (unchecked) that are
deadline-day-critical

- `:32` **Bootstrap C-index confidence intervals — not just point estimate.**
  Partially done in `oof_diagnostics.py` (bootstrap on per-model OOF). NOT done
  for variant-level OOF C — which is what the submission plan is using to
  pick recipes. The MORNING_BRIEFING/SUBMISSION_PLAN both quote point
  estimates without CIs.
- `:33` **Repeated K-fold across multiple seeds.** Listed in
  `GENERALIZATION_IDEAS.md:68` as "highest-lift autonomous tonight" item.
  No file in `tests/experiments/annitia/` does multi-seed K-fold. No
  `seed=43|44|45` artifacts in `branches/`. **Top recommended item, never
  shipped.**
- `:34` **Hold-out fusion-test set.** `refuse_holdout.py` exists, but reading
  it shows it's a *post-hoc selector* (picks variants by similarity to mean
  on a 20% slice), not the holdout-test that GENERALIZATION_IDEAS.md called
  for (10% pure holdout that fusion CANNOT see). The base models already
  saw all rows via 5-fold OOF, which the docstring lines 5-13 acknowledges:
  "this is NOT a perfect imitation of the test-time conditions". So the
  `[x] holdout_pick` recipe in the briefing is an approximation labelled as
  the suggestion fulfilment.
- `:35` **Per-fold C distribution (not just mean).** Done at base-model
  level in `oof_diagnostics.md` (fold C ± σ column). NOT done at variant
  level — the briefing/plan pick variants by point OOF C only.
- `:36` **Group-K-fold by patient cluster.** `patient_representation.py`
  produces clusters, but no group-K-fold using them. Briefing line 102 says
  "5 val patients are beyond train Mahalanobis p99, and cluster 2 (97 train
  / 1 val) is a real distribution-shift signal" — but no submission accounts
  for that.
- `:37` **Better metrics + validation metrics to plan submission strategy
  when limited.** Direct quote from Adelaide. The submission plan's
  decision-rule logic depends on point OOF C. No Brier, no IBS, no per-fold
  histogram, no OOF-LB calibration with CIs. This is the suggestion she asked
  about given submissions are scarce, and it's still open.

### 2.2 `[x]` checkboxes in USER_SUGGESTIONS.md whose artefact is missing
or weaker than claimed

- `:24` `[x] Use uncertainty for per-patient model routing (specialist
  override on top-X% anomalous)` — `refuse_variants.py` has
  `specialist_override`, OK.
- `:46` `[x] Pretrain on death, fine-tune on hep (TabSurv-D2H, CatBoost-D2H,
  MLP-D2H)` — all 3 OOF files exist in `branches/`. Verified.
- `:47` `[x] Pretrain on hep, fine-tune on death (TabSurv-H2D, CatBoost-H2D,
  MLP-H2D)` — same. Verified.
- `:50` `[x] NHANES external dataset pretraining for pretrained-mlp branch` —
  ONE recipe, ONE subset. The roadmap T2 is 60 cells; the morning briefing
  treats this single branch as fulfilling "NHANES Round 4". Misleading but
  technically `[x]` for the literal text.
- `:60` `[x] auto_chain.sh waits for Round 4, runs Round 5, re-fuses, builds
  journals` — auto_chain.sh exists at `/tmp/annitia_runs/auto_chain.sh`,
  but Round 5 (hparam) crashed and never produced output. The chain ran
  through to FUSION but Round 5 step is silently empty. Checkbox is true for
  "the script exists and ran", false for "Round 5 actually tuned hparams".

### 2.3 `GENERALIZATION_IDEAS.md` — items NOT implemented despite being top picks

The "Top picks worth implementing tonight (autonomous)" block at lines
67-71 lists 4 items:
- `(E.31)` **Multi-seed K-fold** — **NOT IMPLEMENTED**. No `seed=43|44|45`
  evidence in branches/. Adelaide explicitly called this "improves stability
  *for free*".
- `(D.26)` **Isotonic calibration** — Implemented as `refuse_isotonic.py`.
  Done.
- `(A.6)` **Random-subset dropout fusion** — Implemented as
  `refuse_bootstrap.py` → `sub_master_bootstrap_dropout.csv`. Done.
- `(F.36)` **Hold-out fusion-test set** — Partial; see §2.1.

So 1 of 4 top picks is missing entirely (multi-seed), 1 is half-done (holdout).

The "Actions to run autonomously tonight" list (lines 79-83):
- `[ ] Add refuse_hierarchical.py to auto_chain` — done
- `[ ] Add bootstrap-fusion variant to auto_chain` — done
- `[ ] Add isotonic-calibration variant to auto_chain` — done
- `[ ] Add hold-out-fusion-test methodology to auto_chain` — partial

All four are still `[ ]` in `GENERALIZATION_IDEAS.md` despite being done in
practice — minor doc-hygiene issue but signals the document hasn't been
updated post-implementation. (USER_SUGGESTIONS.md `:62-64` echoes the same
todo as `[ ]`.)

### 2.4 ANNITIA_TRANSFER_ROADMAP — almost entirely undelivered

- T1.1 MITRA — NOT delivered (no `oof_*mitra*` files; `LOCAL_TESTING.md:39`
  shows it would need an autogluon install). TabPFN-2.5-disc IS delivered.
- T1.2 TabTune PEFT — NOT delivered (no `oof_*tabtune*`).
- T1.3 TabSurv — DELIVERED (TabSurv-K10/K15/D2H/H2D OOFs in branches/).
- T1.4 Synthetic augmentation — DELIVERED for GMM + TabEBM-fallback-to-GMM.
  Diffusion variants (TabDDPM, TabSyn) NOT delivered (no synthcity install).
  *Note:* `oof_hepatic_synaug-tabebm-rsf-500-n100` and `oof_hepatic_synaug-gmm-rsf-500-n100`
  have IDENTICAL pooled C 0.854 in `oof_diagnostics.md:27` — strongly
  suggests TabEBM silently fell back to GMM (same numbers). Memory file
  acknowledges this.
- T1.5 SurvTRACE — NOT delivered (no `oof_*survtrace*`).
- T2.0-T2.3 NHANES pipeline — only T2.2f-style stacking with 1 recipe / 1
  subset delivered. T2.1 zero-shot crashed for ALL subsets. T2.3 ablation:
  0 cells.
- T3 (NIDDK / SteatoSITE / LITMUS / UKBB / TARGET-NASH) — manual user actions,
  out of scope for tonight.

This is a roadmap problem more than an audit problem, but Adelaide has been
told repeatedly that "T1 + T2 are the path past 0.9302". In reality only
T1.3, T1.4-partial, plus a single T2 recipe shipped. The morning briefing
buries this under "Round 4 added basically nothing" without acknowledging
the roadmap shortfall.

---

## Section 3 — Defensive claims that may be punts

### 3.1 "Hierarchical fusion underperforms — don't pursue"
Memory `project_annitia_2026_05_08.md:34` and tacitly the briefing.
Looking at evidence: `MORNING_BRIEFING.md:75-76` actually treats HIER
(`refuse_hierarchical.py` output) as one of the **Tier 2** submissions
(slot #2), with note "Most-different recipe in the quality-controlled pool
(ρ=0.87 hep / 0.88 death to anchor). Real recipe shift, NOT a downgraded
prediction." So the memory's claim that HIER underperforms FLAT contradicts
the briefing recommending HIER for submission. Either:
- The memory is stale (HIER tested earlier, briefing tested newer expanded
  pool variant), or
- The briefing is over-recommending an under-performing variant.

`refuse_hierarchical.py` was authored 2026-05-08 01:20 (docstring claims
"with ~70 NHANES T2.x branches" expected). Since those 70 branches don't
exist, the hierarchical fusion is operating on a tiny pool — making the
"per-source pre-fusion" pattern weaker than designed.

*Why it matters today:* if Adelaide submits HIER (slot #2) expecting the
"real recipe shift" claim and it's a downgraded prediction, that's a
wasted submission. Cross-check the OOF numbers in
`refuse_*_summary.json` files against the briefing's claims before
shipping.

### 3.2 "0.9302 is near the noise ceiling"
`MORNING_BRIEFING.md:155-156`. Justified by: "SE_C ≈ 0.025 with 47 hep
events". But: (a) val_test has different N events than train; SE on the
LB measurement is unspecified; (b) the briefing's own numbers show 5
submissions clustered in [0.9269, 0.9302] — that 0.0033 spread IS within
the 0.025 SE, so "near ceiling" is a one-sigma observation, not a hard
floor. Adelaide could still see 0.95 with the right recipe; the briefing's
"don't bother probing" tone is over-defeatist.

### 3.3 "Per-patient routing UNDERPERFORMS aggregate averaging"
Memory `:30`. Empirical: the patient-aware variants ARE in the
`meta_avg_aware_weighted` 0.9302 anchor (10 base × 1 + 5 patient-aware × 2,
weighted *up*). So they're not "underperforming" — they're *part of* the
best result with 2x weight. The claim is true for *raw* per-patient
routing (sub_master_anomaly_router_meta etc) at submission level, but
useful as components inside the meta-learner. The briefing line 81
distinguishes these correctly; the memory line is too blunt.

### 3.4 "Subgroup-routed Cox is worse, not different"
`MORNING_BRIEFING.md:166`. Evidence is the OOF table in lines 50-55:
subgroup_routed hep 0.7638 vs Global Cox 0.8401 vs ensemble pool 0.8980.
True for OOF C, but the *information* it carries (cluster 2 = 97 train / 1
val distribution shift signal) is acknowledged as diagnostic value at line
102. The "DO NOT SUBMIT" verdict is correct given negative EV; calling it
"worse, not different" is true *as a submission* but the diagnostic insight
isn't captured anywhere actionable for tomorrow. Could be a feature — e.g.
downweight cluster-2 val predictions — but no code does this.

### 3.5 "Round 4 cross-endpoint pretraining added basically nothing"
`MORNING_BRIEFING.md:164`. Looking at `oof_diagnostics.md:30,52`:
`catboost-D2H` hep CV 0.838 LOW; `catboost-H2D` death CV 0.901 MEDIUM.
These sit in the middle of the rankings — not catastrophic, not
breakout. The "added basically nothing" claim refers to delta on the
fusion. But did anyone actually *measure* the fusion-with-vs-without
delta? `meta_avg_aware_weighted_v2` adds the new pool, but its CV
isn't compared against `meta_avg_aware_weighted` on the OOF in a
single doc. Adelaide submission #1 in the briefing IS this comparison,
which is fine — but the verdict is being prejudged before the LB
returns.

---

## Section 4 — MORNING_BRIEFING vs reality

### 4.1 "expected macro" estimates use wrong formula
- `MORNING_BRIEFING.md:18`: "subgroup_hep_only expected macro ~0.88"
- `MORNING_BRIEFING.md:19`: "subgroup_blend expected macro ~0.89-0.90"
- `MORNING_BRIEFING.md:94`: "drops macro ~0.05 below 0.9302" (so ~0.88)

If "macro" means simple average:
  - subgroup_hep_only = (0.83 hep_LB + 0.93 anchor death_LB) / 2 = 0.88. **Matches briefing.**
  - subgroup_blend ≈ same arithmetic → 0.89-0.90. **Matches briefing.**

If "macro" means actual LB:
  - subgroup_hep_only = 0.7 × 0.83 + 0.3 × 0.93 = 0.581 + 0.279 = **0.860**. ~0.07 below anchor, not 0.05.
  - subgroup_blend ≈ 0.7 × 0.86 + 0.3 × 0.93 = 0.602 + 0.279 = **0.881**.

So the briefing is computing simple averages. Directional verdicts are
correct (don't submit, would tank LB), but specific numbers are off by
0.02-0.04. Critically, this same "macro" framing might be used in
decision rules — `MORNING_BRIEFING.md:117` "0.85-0.92 → moderate signal
loss. Don't escalate hier" thresholds were calibrated against the wrong
formula.

### 4.2 hep CV 0.85 / death CV 0.92 "matched 0.9302's profile"
`SUBMISSION_PLAN_2026_05_08.md:86`: "OOF C of the recipe must be ≥ 0.85
hep / 0.92 death (matched 0.9302's profile)"

Cross-check: `oof_diagnostics.md:21` hepatic best is 0.868. Anchor's hep
profile is fusion-level OOF, which could be much higher. The 0.85/0.92
threshold is unsourced. From `MORNING_BRIEFING.md:55`: "Existing ensemble
pool best | 0.8980 | 0.9736" — i.e., **fusion** OOF is 0.898 hep, 0.974
death. So setting 0.85 hep as floor is loose; 0.92 death is much *below*
the 0.974 fusion ceiling. Decision rule effectively excludes nothing on
death side. Probably fine but inconsistent with the "ceiling" framing.

### 4.3 GUID bug
`MORNING_BRIEFING.md:30-43`: "7 broken submissions caught and fixed".
True — `pre_submit_check.py` exists, GUID bug acknowledged on line 287
of `patient_representation.py`. The CSVs `sub_master_subgroup_*.csv`
all have integer trustii_id now. Verified.

But: `tests/results/annitia/full/patient_representation_val.csv` (the
diagnostic CSV, not the submission) still has GUID `trustii_id` column —
that's a separate file, intentional, but Adelaide should be told
explicitly that the diagnostic CSVs and the submission CSVs use
different conventions.

### 4.4 OOF/calibration claims
The briefing references `oof_diagnostics` and `oof_to_leaderboard`
findings but as noted in §2 of TL;DR, the calibration .md is stale. Do
not trust the briefing's numerical projections derived from the .md.

### 4.5 Subgroup numbers attributed to journals
`MORNING_BRIEFING.md:64-65`: "The journal files for `subgroup_blend`
and `subgroup_hep_only` claim OOF 0.8980/0.9736 — those are the anchor's
numbers, auto-generated boilerplate. Do not trust journal text on these
three submissions." This is a HONEST flag. Good. It means the journals
themselves have miscalibrated claims that propagate to anyone who reads
them blind. Adelaide should know that `build_journal.py` doesn't
parameterise OOF per recipe.

---

## Section 5 — Relationship of USER_SUGGESTIONS vs GENERALIZATION_IDEAS

`USER_SUGGESTIONS.md` is a tracking checklist; `GENERALIZATION_IDEAS.md` is a
brainstorm. Cross-mapped:

| GENERALIZATION_IDEAS line | Description | Code? | USER_SUGGESTIONS counterpart |
|---|---|---|---|
| A.1-4 | rank-mean/median/trimmed/rrf | `refuse_variants.py` | `[x] :11` |
| A.5 | bootstrap aggregation of fusion | `refuse_bootstrap.py` | `[ ]` (mentioned :64) |
| A.6 | Random-subset dropout fusion | `refuse_bootstrap.py` | `[ ]` |
| A.7 | Bayesian model averaging | NOT IMPLEMENTED | absent |
| A.8 | NNLS meta-learner | `refuse_nnls.py` | `[ ]` (mentioned briefing) |
| B.10 | source-stratified fusion | `refuse_hierarchical.py` | `[x] :17` |
| B.11 | quality-stratified weighting | refuse_hierarchical | `[x] :17` |
| B.12 | subjective gating on pretraining | refuse_hierarchical | `[x] :19` (`_gate_pretraining_source`) |
| B.13 | Disagreement-based dropout | NOT IMPLEMENTED | absent |
| C.15-19 | anomaly/cluster routing | `refuse_meta_aware.py` | `[x] :15` |
| C.20 | Soft routing via gating network | NOT IMPLEMENTED | "low-hanging" :77 |
| C.21 | Per-time-bin routing | NOT IMPLEMENTED | absent |
| C.22 | Cohort-specific specialists | partial via subgroup_routed | not in suggestions |
| D.23 | Mid-alpha Coxnet | `refuse_reselection.py` | `[x] :30` |
| D.24 | Strong L2 stack | NOT IMPLEMENTED | absent |
| D.25 | Conformal prediction | NOT IMPLEMENTED | absent |
| D.26 | Isotonic calibration | `refuse_isotonic.py` | top pick :69 |
| D.27 | Bin-wise discordance correction | NOT IMPLEMENTED | "low-hanging" :77 |
| E.28 | Synthetic positive aug | `synthetic_aug.py` | `[x] :73` |
| E.29 | Mixup | NOT IMPLEMENTED | absent |
| E.30 | Pseudo-labelling | NOT IMPLEMENTED | absent |
| E.31 | Multi-seed K-fold | **NOT IMPLEMENTED** | top pick :68 |
| F.33 | repeated K-fold | NOT IMPLEMENTED | `[ ] :33` |
| F.34 | nested CV | refuse_reselection | `[x] :29` |
| F.35 | Group-K-fold | NOT IMPLEMENTED | `[ ] :36` |
| F.36 | Hold-out fusion-test | partial (refuse_holdout) | `[ ] :34` |
| G.39 | Distillation | NOT IMPLEMENTED | "low-hanging" :77 |
| H.41-44 | bagging variants | NOT IMPLEMENTED | mentioned :74 |

Net: **about a third of GENERALIZATION_IDEAS items have code, two-thirds don't.**
Important: the *top picks* list (lines 67-71) is supposed to be the curated
"do these tonight" — and 1 of 4 (multi-seed) is silently undone.

---

## Section 6 — Has Adelaide's earlier directions been addressed?

(Cross-referenced against CLAUDE memory `project_annitia_2026_05_08.md`.)

### 6.1 Uncertainty
- ✅ IsolationForest anomaly + ensemble disagreement implemented
  (`analyse_residuals.py`, `refuse_meta_aware.py`).
- ✅ Used as routing signals (specialist_override, anomaly_blend).
- ❌ Bootstrap C-index CIs at variant level — done at base-model level only.
- ❌ Conformal prediction — never started.
- ❌ Brier / IBS calibration metrics — never computed despite roadmap T1.4
  explicitly demanding them.

### 6.2 Noise mitigation
- The roadmap (T1.4 line 162) requires "Calibration check (Goorbergh JAMIA
  2022): augmentation can wreck calibration. Report Brier score and
  integrated Brier score alongside C-index. Do not submit augmented model
  if calibration degrades > 10%."
- **NOT IMPLEMENTED.** `synthetic_aug.py` exists; no calibration check
  exists; no Brier/IBS reporting in any artifact. The synthaug variants
  are in the pool without the gate Adelaide's roadmap demanded.

### 6.3 Reverse engineering
- `cross_reference_failures.py` + `valtest_contention.py` do reverse-
  engineer val_test failures by linking contention to training failure
  patterns. ✅
- But the result (`cross_reference.md`: 38% hep / 36% death contested
  patients map to FP-like training failures) is *reported* but not *acted
  on*. No submission downweights or specialises on those neighbours.

### 6.4 Failure-based scoring
- `failure_analysis.py` produces FN/FP/hardest-30 sets and feature
  comparisons. ✅ Output exists at `failure_analysis.md`.
- No fusion variant uses the failure flags as features or weights. The
  Section 3.4 / 4.4 finding is consistent: the analysis exists, the action
  doesn't.
- Adelaide's directive (paraphrased from memory) asked for failure-based
  decisions *in submission*. Currently failure analysis is purely
  diagnostic.

### 6.5 Methodology validation
- `[x] HONEST OOF C (nested 5-fold for any learned weights)` — refuse_reselection.py
  done.
- `[ ] Bootstrap CI on variant OOF` — undone (top of §6.1).
- `[ ] Hold-out fusion-test set (true holdout)` — partial.

Adelaide's broader directive: "Better metrics + validation metrics to plan
submission strategy when limited" (`USER_SUGGESTIONS.md:35`). Still open.
This is the meta-suggestion that justifies all of §6.1-6.4. None of the
deadline-day docs (briefing, plan) actually use richer metrics; they all
fall back to point OOF C and Spearman to anchor.

---

## Section 7 — Critical pending items for deadline-day decisions

In priority order:

1. **Re-run `oof_to_leaderboard.py` so the .md is on the corrected formula.**
   The morning briefing's projections are bogus until this is regenerated.
   Trivial: `python tests/experiments/annitia/oof_to_leaderboard.py`.

2. **Recompute the "expected macro" numbers in MORNING_BRIEFING.md
   §"Reality check on subgroup variants" (lines 17-19, 91-96) using the
   correct 0.7/0.3 weighting.** Directional verdicts likely hold but
   thresholds and decisions on "below 0.9302 floor" depend on it.

3. **Verify the `meta_avg_aware_weighted_v2` and `meta_avg_grand_all_v2`
   bring net new information vs the original.** Check
   `refuse_variants_summary.json` for OOF deltas. The briefing assumes
   they do; the v2 zips exist; but the ρ=0.99 to anchor (briefing line 70)
   is suspiciously high for a "Round 4 expanded pool" and might mean the
   new branches washed out.

4. **Resolve the empty Round 5 hparam slot in submission plan slot #3.**
   Either fix `experiment_hparam.py` (briefing line 172 says "not worth
   debugging") and run, or replace with a different experiment. As-is,
   slot #3 is empty.

5. **Multi-seed K-fold for top variants** (GENERALIZATION_IDEAS top pick
   E.31) — the only top pick truly missing. Even just rebuilding
   `meta_avg_aware_weighted` with seeds 43/44/45 base models, averaged,
   would be a real submission slot.

6. **Pre-submission audit of HIER and isotonic_avg recipes' OOF.**
   `MORNING_BRIEFING.md:74-78` recommends them as Tier 2 but doesn't quote
   their OOF C numbers — just Spearman correlation to anchor. Run a one-shot
   check that `refuse_hierarchical_summary.json` and
   `refuse_isotonic_summary.json` exist and report C ≥ briefing's claimed
   threshold before shipping.

7. **Calibration check on synthaug variants** (roadmap T1.4 explicit
   requirement, ignored). Either drop synaug variants from the ensemble or
   compute IBS. Cheap to do; high methodology-points value (Trustii's 30%
   qualitative scoring).

---

## Section 8 — Documentation hygiene

- `MORNING_BRIEFING.md` is dated "REVISED 03:55 after sanity sweep" but
  references an `auto_chain.log` whose tail is timestamped 03:14 — i.e.
  the briefing references the log's tail block but isn't itself reflecting
  the absence of Round 5 / hparam outputs.
- `SUBMISSION_PLAN_2026_05_08.md` is dated 01:30 and was not updated after
  the 03:14 auto_chain finish. It's been overtaken by MORNING_BRIEFING but
  not deprecated.
- `USER_SUGGESTIONS.md` and `GENERALIZATION_IDEAS.md` were last edited 01:21-22
  and don't reflect the 03:09 oof-script fix or the 03:55 briefing.
- `leaderboard_history.json` `_comment` is materially wrong (§1.1) and was
  the trigger for this audit.
- `ANNITIA_TRANSFER_ROADMAP.md` is dated 2026-05-07 and largely undelivered;
  no document acknowledges this gap honestly. The morning briefing's
  "ceiling" framing implicitly papers over it.
- `tests/experiments/annitia/MORNING_BRIEFING.md` and the submissions plan
  use slightly different recipe orderings (Tier 1/2/3/4 vs the matrix
  table). Reconciling is for Adelaide; no doc claims they're identical.

---

## Appendix A — File mtimes (for staleness)

| file | mtime |
|---|---|
| `oof_to_leaderboard.py` | 2026-05-08 03:09 |
| `oof_to_leaderboard.md` | 2026-05-08 02:34 (**stale**) |
| `oof_to_leaderboard.json` | 2026-05-08 02:34 (**stale**) |
| `oof_diagnostics.md` | 2026-05-08 03:11 |
| `failure_analysis.md` | 2026-05-08 03:07 |
| `cross_reference.md` | 2026-05-08 03:09 |
| `MORNING_BRIEFING.md` | 2026-05-08 03:55 |
| `SUBMISSION_PLAN_2026_05_08.md` | 2026-05-08 01:30 (**superseded**) |
| `USER_SUGGESTIONS.md` | 2026-05-08 01:21 (**not updated since fixes**) |
| `GENERALIZATION_IDEAS.md` | 2026-05-08 01:22 (**not updated since fixes**) |
| `leaderboard_history.json` | 2026-05-08 04:08 |
| `tier1_fallback.sh` | 2026-05-08 01:32 |

---

## Appendix B — Verified artefacts

OOF .npy files present in `tests/results/annitia/branches/` at audit time
(spot-check vs claims):
- ✅ tabpfn-landmark (death only — used as HIGH-trust per oof_diagnostics)
- ✅ tabpfn25-disc-K10 (hep), K20 (death)
- ✅ tabsurv-D2H (hep), H2D (death), K10/K15 single-endpoint
- ✅ catboost-D2H (hep), catboost-H2D (death)
- ✅ mlp-D2H (hep), mlp-H2D (death)
- ✅ pretrained-mlp__nhanes_proj (both endpoints) — single NHANES recipe
- ✅ dsm-K4 (3 view sets each endpoint)
- ✅ tsfm-{cox, coxnet, gbsa, lgbm_aft, ngboost, rsf} (both endpoints)
- ✅ synaug-{none, gmm, tabebm}-{cox, rsf} for hep only
- ❌ NO mitra files
- ❌ NO tabtune files
- ❌ NO survtrace files
- ❌ NO tabddpm/tabsyn synaug variants (only GMM + TabEBM-fallback)
- ❌ NO multi-seed (seed=43|44|45) variants

Submission CSVs and zips for the briefing's 8-recipe plan all present in
`tests/results/annitia/submissions/`. Verified.

---

## Appendix C — One-line "why this matters today" recap

| Finding | Submission impact |
|---|---|
| Wrong _comment formula in leaderboard_history | Decisions on "below 0.9302 floor" use simple-avg arithmetic, not LB arithmetic |
| Stale oof_to_leaderboard.md | The briefing's projection table (lines pointing to OOF=0.94→1.06) is bogus |
| Round 5 empty | Submission plan slot #3 has no recipe to ship |
| T2 mostly undelivered | "Round 4 added basically nothing" verdict is comparing to a no-show, not a tested baseline |
| Foundation models absent | Roadmap-class lift (TabPFN/MITRA on rare-class hep) was never tested |
| Multi-seed K-fold missing | Top "for free" stability boost is the cheapest unused submission slot |
| Synaug calibration not checked | Methodology-points risk for the qualitative 30% if reviewer asks |
| HIER recipe targets missing pool | "Quality-stratified weighting" works on a thinner pool than designed |
| Briefing macro arithmetic wrong | Specific numbers (0.88, 0.89-0.90) are off by 0.02-0.04 vs LB scale |
| Variant-level CIs absent | Submission ranking decisions are point estimates only; can't tell if 0.9285 vs 0.9302 is signal or noise |
