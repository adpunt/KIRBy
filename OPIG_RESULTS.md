# OPIG results — the story, dataset by dataset

I've selected the hero result for each point so you don't have to. One dataset per
experiment (or the one contrast pair where the contrast *is* the point). Numbers read
straight from `~/Downloads/prelimResults`. CIs shown for small-molecule only.

---

## Slide — Molecular

**Point:** on small molecules the hybrid beats the best single descriptor using the same model.

**Hero: adme_hlm** (metabolic stability). svm on pdv: RMSE **0.454 → 0.421** (±0.005) — the
tightest win in the whole run; the baseline is clearly outside the interval.
**Confirms: adme_mdr1** (permeability). svm on smited: **0.467 → 0.431** (±0.021).

*Say:* "Same model, same folds — swap the single descriptor for the hybrid blueprint and the
error drops, tightly on adme_hlm, again on adme_mdr1." (qm9 wasn't in this run; caco2/herg_ki
are weaker and I left them off.)

---

## Slide — Antibody (the core of the talk; split across 2–3 slides)

### 1. It works, cleanly
**Hero: flab_expression_koenig** (antibody expression). lightgbm on esm3: **0.681 → 0.772**
(Spearman ρ). The strongest clean antibody win — a big jump from an already-strong baseline.

### 2. When and why — abagym_EGFR **vs** abagym_HER2 (the centerpiece)
Same task (antibody binding), two targets, opposite outcomes. This one pair carries five
experiments — it's the "when does KIRBy help" slide.

| | abagym_EGFR (wins) | abagym_HER2 (stalls) |
|---|---|---|
| reps available to draw from (c01) | **26** | 13 |
| best rep / its effective rank (c01) | saprot, rank **259** | mutation_features, rank **24** |
| core claim (c02) | ridge/prot_t5 0.251 → 0.372 (**+0.121**) | 0.453 → 0.361 (**−0.092**) |
| with more data (c05) | edge appears: −0.040 (n=329) → **+0.035** (n=659) | frozen at n=146, −0.092 |
| dump all reps together (c11 PCA-concat) | 0.317 (hybrid 0.338 beats it) | collapses **0.453 → 0.151** |
| calibration error (c08) | 0.041 | 0.147 |

*Say:* "Identical task. EGFR has 26 rich representations to mix and enough data — the hybrid
finds signal none of them hold alone, and its edge *grows* as data arrives. HER2's signal is
already in one compact 48-dimensional mutation vector, n is stuck at 146, and there's nothing
to combine — so the hybrid can't help. That's the rule for the whole method in one pair."
(Figure: `egfr_her2_why.png` — the three ingredients side by side.)

### 3. The blueprint's importance scores are real signal
**Hero: ddg_combined** (binding ΔΔG). Weighting retrieval by the hybrid's per-feature
importance vs not: **Δ +0.282** — the biggest such gain in the set. The importances the
blueprint is built on carry information; they're not decoration.

### 4. Beyond accuracy — the hybrid as a *retrieval* space (which Ginkgo to use)
Of the nine Ginkgo developability assays, the two worth showing tell different stories:
- **ginkgo_tm2** (thermostability) is the accuracy win: svm/igt5 **0.107 → 0.288**.
- **ginkgo_hic** (the recognisable HIC assay) is *not* an accuracy win (0.582 → 0.589, a tie),
  but it's where the hybrid **distance space retrieves better** — largest similarity lift of
  the nine (s02 +0.053) and it wins retrieval (s13).

*Say:* "Feature thermostability for the accuracy story and HIC for the retrieval/similarity
story — HIC's regression number is flat, so don't sell it as accuracy."

---

## Slide — Other domains

### Genomic
**Hero: nt_promoter_tata** (DNA promoter). lightgbm on kmer5: AUROC **0.661 → 0.710**, and the
hybrid does it at **lower dimension** (500 vs 1024) — wins *while* compressing.
**Confirms: gb_human_enhancers_cohn** (DNA enhancer): 0.480 → 0.506.

### Reaction — the most striking single result
**Hero: buchwald_hartwig** (reaction yield). Hybrid advantage vs the best single rep across
training size:

| training size n | ~400 | ~4000 |
|---|---|---|
| hybrid advantage (RMSE, +=better) | **+1.176** | +0.077 |

*Say:* "With only ~400 reactions the hybrid slashes error; by ~4000 the single rep has caught
up. KIRBy is a small-data method — it buys you the most exactly when labels are expensive."

---

## Slide — Negative (one honest example, no table)

> **abcov_kd** (antibody binding Kd) is the clean failure, and it's failure *for a reason*. The
> best single representation, abroberta, already scores 0.488. Build the hybrid on it and the
> score drops to 0.428. Concatenate everything (PCA-concat) and it's 0.387 — also worse. Give
> it more data and it gets worse, not better (0.488 at the largest n → −0.060). When one
> representation already holds the signal, selecting and blending can only dilute it. That's
> the same mechanism as HER2 — and we show it rather than hide it.

---

### Notes for honesty
Forward-compatibility (does a *new* rep improve the hybrid) and noise-robustness produced no
usable results in this run — don't claim them. Everything above is one dataset per point, read
from the report; nothing is aggregated across datasets.
