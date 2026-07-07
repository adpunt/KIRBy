# Presentation Analysis Protocol (2026-07-07)

Written analysis protocol for the presentation datasets. Purpose: fix **how**
every number is read *before* the numbers are final, so that when fresher
results land overnight you re-run one command and the analysis is identical —
no re-deciding methodology under time pressure.

All results live on the server. This document is the read recipe; the executable
companion is `tests/analysis/quick_hybrid_summary.py` (read-only, stdlib-only).
On the server: `git pull && python tests/analysis/quick_hybrid_summary.py tests/results`
and paste the output.

---

## 0. Hard constraints (do NOT relitigate mid-analysis)

1. **No averages as a headline.** Never a cross-dataset or cross-domain mean.
   Report per-dataset, then within a domain report **counts** (`+` better / `=`
   tie / `−` worse) and the **range/median** of Δ. Domain segmentation is allowed
   (antibody / molecular / reaction / tcga / nucleotide); pooling across domains
   is not.
2. **Direction-correct every Δ.** Direction is authoritative from
   `kirby.registry.DATASET_REGISTRY[ds].lower_is_better` (the analyzer loads it;
   verified 2026-07-07). `lower_is_better = True` (RMSE/MAE regression): caco2,
   herg_ki, logd, qm9, adme_hlm, adme_rlm, adme_mdr1, buchwald_hartwig,
   bh_doyle_plates. **`herg` is CLASSIFICATION (AUROC) → higher-better** — do not
   put it in the lower-better bucket. Everything else higher-better. Δ is signed so
   **positive always = hybrid better**. `improvement()` in the analyzer already
   does this; the JSON does **not** persist the direction, so it is re-derived from
   the registry every run (known recurring sign bug — the one thing that silently
   corrupts a slide).
3. **Same model on both sides.** Compare the hybrid against the best single rep
   *for the same model you'd deploy* — i.e. the model with the best single-rep
   score. Never best-model-hybrid vs worst-model-baseline.
4. **Tie band.** Treat |Δ| ≤ 0.01 (in ρ / R² / AUROC units) as a tie, not a win.
   State the band on the slide.
5. **Read-only.** Claude has no server access. Every number on a slide traces to
   one pasted analyzer run + this recipe.

---

## 1. What is actually analysis-ready (server state, 2026-07-07)

Coverage columns: `cx /20` = complexity experiment cells done, `sim /12`,
`noise /12`. "Ready for the core claim" needs the c02 cell done (it is one of the
first cx cells to finish; a dataset at cx ≥ 3/20 generally has c02).

### Ready NOW — lead the talk with these
| Domain | Datasets | State |
|---|---|---|
| **Antibody (core claim)** | abagym_Ang2, abagym_EGFR, abagym_HER2, abagym_VEGF, abcov_kd, flab_immunogenicity_marks, flab_polyreact_jain_bvp, flab_polyreact_jain_cic, humanness_therapeutic | cx 16–20/20, sim 12/12 — full suite |
| **Antibody (developability panel)** | ginkgo_hic, ginkgo_acsins, ginkgo_colloidal, ginkgo_heparin, ginkgo_pr_cho, ginkgo_titer, ginkgo_tm2, ginkgo_aggregation, ginkgo_purity | sim 12/12; cx 3–10/20 (similarity-ready, complexity partial) |
| **Reaction (strongest non-Ab)** | buchwald_hartwig | cx 13/20, sim 11/12, noise 12/12 |

### Thin now — will hopefully fill overnight; protocol is pre-written so they drop in
| Domain | Datasets | Blocker |
|---|---|---|
| Antibody headlines | flab_thermo_tresanco (VHH Tm), flab_expression_koenig | cx 3/20, sim 1/12 — c02 present but similarity not; treat as single-point until sim lands |
| **Molecular (your primary domain)** | caco2, herg, herg_ki, adme_mdr1 (precompute DONE); adme_hlm/rlm, logd, qm9 (precompute partial) | cx 0–3/20, sim 0–1/12 — **not core-claim-ready yet** |
| Reaction | bh_doyle_plates | cx 0/20 (just started) |
| TCGA | tcga_brca_pam50, tcga_luad_vs_lusc | cx 3/20, sim 1/12 |
| Nucleotide | nt_H3K27me3, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_promoter_tata, gb_human_enhancers_cohn | cx 0–3/20; only gb_human_enhancers_cohn has sim 12/12 |

**Honest framing for the talk:** the mature, multi-condition evidence is
**antibody** (25+ datasets, full grid) plus **buchwald_hartwig**. Molecular — the
domain you care most about — is precomputed but the experiments have barely run
(cx 1–3/20). Present molecular as "in flight, early read" or hold it, rather than
over-claiming from 1–3 conditions. Do not average molecular into an antibody
headline to make it look covered.

---

## 2. Core-claim protocol — c02 (hybrid vs best single rep)

**Question:** does a mixed-representation hybrid beat the single best representation
on the model you'd deploy?

**Files:** `tests/results/complexity/c02_complexity_matching_grid/<dataset>.json`
(canonical path under `complexity/`; there is a stale duplicate at
`tests/results/c02_complexity_matching_grid/` — ignore it).

**Read per dataset:**
1. `baselines[model]` → `best_single_rep`, `best_single_score`. The **deployed
   model** = the model with the best `best_single_score` (direction-aware). Take
   the **top-2** such models.
2. `best_by_model[model]` → the best hybrid config for that same model
   (`score_mean`, `score_std`, `budget`, `allocation`, `n_features`).
3. `Δ = improvement(best_single_score, hybrid.score_mean, dataset)` — same model
   both sides, direction-corrected.
4. `conditions.<name>[model]` gives the hybrid under each search condition
   (`no_aug`, `uncapped`, `top_tier_only`, `frozen_plus_ft`, `per_model_top_rep_dim`).
   Report which condition produced the winning hybrid (`hybrid_used`).

**Report (per domain, no averages):**
- Table: dataset | deployed model | best rep + score | best hybrid + score | Δ | condition.
- Domain roll-up: `+N better / =N tie / −N worse` at the deployed model, and again
  at "both top-2 models". Median Δ and [min, max] range. **Counts and ranges only.**
- Hybrid **size** vs single-rep dim (from §"HYBRID SIZE" reader): flag cases where
  the hybrid *wins with fewer features* — that's a strong slide ("smaller and
  better"), and it holds in the current antibody batch.

The analyzer already emits all three tables (`analyze_c02_top2`,
`analyze_c02_by_model`, `analyze_c02_hybrid_size`). Nothing new to write for c02.

---

## 3. Confidence intervals & significance

**What IS stored:** every config carries `fold_scores` (5 CV folds). The winning
hybrid therefore has 5 fold scores → report **hybrid mean ± 95% CI**
(t-interval, df = 4): `CI = mean ± 2.776 · std / sqrt(5)`.

**What is NOT stored:** `baselines` holds only the scalar `best_single_score` —
no per-fold baseline scores. So a *paired* hybrid-vs-base fold test is **not
computable from c02 alone.** Two honest options (decision for you, §8):
- **(A) Pragmatic / today:** plot the hybrid mean with its 95% CI and the base as
  a reference line; call it "beats base" only when the base falls **below the
  lower CI bound**. Understates significance (ignores fold pairing) but is
  defensible and needs no new runs.
- **(B) Proper paired test:** recover per-fold single-rep scores from
  `c01_rep_characterization/<dataset>.json` or the baseline survey (same 5 folds),
  then paired t / Wilcoxon across folds on (hybrid_fold − base_fold). Needs a
  reader confirming the fold alignment matches; do only if a dataset is a
  headline slide and time allows.

For the ginkgo panel and abcov_kd (flagged "CIs" in your notes): use (A) now,
upgrade the 2–3 headline ones to (B) if the fold files line up.

---

## 4. Per-domain protocols

### 4.1 Antibody — the lead story
- **Headline claim:** "Across N antibody property tasks, the importance-guided
  hybrid beats the best single representation on the deployed model in X/N cases,
  ties in Y, loses in Z; median lift +M ρ, range […]." Fill from the c02 roll-up.
  In the current batch this is strongly positive (rf/xgboost/mlp near-uniformly
  positive; ridge/svm mostly positive with a few losses).
- **"Smaller and better" slide:** the hybrid frequently wins at a feature budget
  far below the single rep's native dim — pull the smaller? column from the size
  reader.
- **abcov_kd (top card, n≈1780, identity-clustered):** headline single antibody.
  Add the **germline control** — show the hybrid lift survives against a
  germline/naive-distance null model, not just the best learned rep. If the
  germline baseline hasn't been computed on the server, mark the control
  "pending" on the slide rather than dropping it.
- **abagym_Ang2 / VEGF / EGFR:** frame explicitly as **single-antibody DMS** with
  a valid split — the lift is within one antibody's mutational landscape, a
  different (and legitimate) claim from the cross-antibody abcov_kd.
- **humanness_therapeutic:** saturated (AUROC ≈ 0.97, Δ ≈ +0.004). Use as a
  **sanity check only** — "the method doesn't break a solved task" — never a
  headline.
- **flab_thermo_tresanco (VHH/nanobody Tm):** intended headline, but currently
  cx 3/20 & sim 1/12 → single-condition point. Present as "early, promising" or
  hold until more cx lands.

### 4.2 Developability panel (ginkgo_*) — the honest-negative panel
- Frame as: "developability endpoints the field considers hard." Show the panel
  **as a panel** — per-endpoint Δ with the tie band drawn — including the flat/
  negative ones (ginkgo_colloidal ≈ 0, ginkgo_pr_cho slightly negative). The
  credibility of the positive cases (heparin, hic, acsins) rests on showing you
  didn't cherry-pick.
- These have sim 12/12 → the **similarity** story (s01/s02, §5) is the ready one
  here; complexity is partial, so lead with similarity for ginkgo.

### 4.3 Reaction — buchwald_hartwig (strongest non-Ab)
- Leave-additive-out split, direction = **lower_is_better** (RMSE). cx 13/20,
  sim 11/12, noise 12/12 — the most complete non-antibody dataset. Give it its
  own slide as the "generalizes beyond proteins" evidence. Double-check the sign:
  a *negative* raw RMSE delta is an *improvement* → `improvement()` flips it.
- bh_doyle_plates (cx 0/20) is not ready; mention as replication-in-progress.

### 4.4 Molecular — primary domain, currently early
- caco2, herg, herg_ki, adme_mdr1 have precompute DONE but cx 1–3/20. If overnight
  runs push these to ≥ c02-done, the §2 protocol applies unchanged (they're
  already in the analyzer's `_LOWER_BETTER_FALLBACK`). Until then: either omit, or
  a single "early read, N conditions only" caveat slide. Do not roll molecular
  into an antibody count.

### 4.5 TCGA & Nucleotide — supporting, thin
- tcga_luad_vs_lusc / tcga_brca_pam50: TSS-grouped classification, AUROC. c02 is
  present (cx 3/20); report as single supporting points ("holds on genomics-scale
  clf too"), CI via §3(A). Note the tcga ridge case is a rare loss — keep it in.
- Nucleotide: only gb_human_enhancers_cohn has sim 12/12; the nt_* set is cx 0–3.
  Treat as "domain-breadth in progress," not a claim.

---

## 5. Supporting experiments (already have readers)

- **s01 similarity_baseline** (`similarity/s01_similarity_baseline/`): best-rep
  self-augmentation-coefficient (SAC) per dataset — the single-rep reference.
- **s02 hybrid_similarity** (`similarity/s02_hybrid_similarity/`): hybrid distance
  space vs best single rep. This is the **ready** story for the ginkgo panel
  (sim 12/12). Same +/=/− count framing, direction-aware, no averages.
- **noise (n01–n04)**: only where noise ≥ 4/12. buchwald_hartwig (12/12),
  flab_pk_jain_fcrnrt (12/12), flab_aggregation_jain (12/12) can support a "hybrid
  degrades gracefully under label noise" slide. n05–n13 too thin — skip.

---

## 6. Re-run recipe for tomorrow (post-overnight)

1. On the server: `git pull` (get any new result JSONs synced) —
   `python tests/analysis/quick_hybrid_summary.py tests/results > /tmp/summary.txt`
   and paste `summary.txt`.
2. Re-read against §2/§4. Any dataset that crossed into "c02 done" moves from the
   thin table (§1) to the ready table — the framing is already written, so it is a
   copy-paste, not a re-analysis.
3. Re-confirm direction on every newly-arrived molecular/reaction dataset (the sign
   bug is the one thing that silently corrupts a slide).
4. Regenerate CIs (§3) only for datasets that changed.

---

## 7. Slide-count sanity (what the evidence currently supports)

- 1 headline: antibody core claim (counts + range, "smaller & better").
- 1 single-dataset deep-dive: abcov_kd (+ germline control if ready).
- 1 honest-negative panel: ginkgo developability (via similarity s02).
- 1 cross-domain generalization: buchwald_hartwig.
- 1 caveat/roadmap: molecular + tcga + nucleotide "in flight."
- 1 method/robustness: noise + calibration where ≥4/12.

---

## 8. Open decisions for you (methodology is yours)

1. **CI method:** §3(A) pragmatic for all, or upgrade 2–3 headlines to §3(B)
   paired? (Default: A now, B for abcov_kd only if fold files align.)
2. **Molecular:** omit entirely, or one honest "early read" slide from 1–3
   conditions?
3. **Tie band:** 0.01 the right threshold, or tighter for AUROC (saturated tasks
   like humanness)?
4. **flab_thermo_tresanco:** keep as an early headline, or demote to "pending"
   until cx fills?
5. **abcov_kd germline control:** is the germline/naive baseline computed on the
   server, or does it need a run before it can go on the slide?
