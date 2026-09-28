# Handoff: censoring deep-dive noise experiment

Written 2026-09-28. Paste everything below the line into a new session.

---

Read `CLAUDE.md` at the repo root first and follow it in every reply. The
owner is the engineer and decides the design. Your first deliverable is a
design proposal with options and numbers, not code. Build nothing until she
signs it off.

## What censoring is here

Some assays cannot measure past a limit. The lab records the limit instead
of the true value, so a recorded label at the limit means "at the limit or
beyond it", not "exactly this". NoiseInject calls this censoring. It clips
every label beyond a cut-off to the cut-off
(`~/repos/NoiseInject/noiseInject/core.py:736-779`). It supports
`side='upper'` (labels above the limit set to it) and `side='lower'`
(labels below it set to it). The level is the fraction of labels clipped.

## The real example: adme_hlm

adme_hlm is human liver microsome clearance from the Polaris ADME-Fang set,
read from `~/.cache/kirby/polaris_adme_fang.csv`, column `LOG_HLM_CLint`.
Counted 2026-09-28:

- adme_hlm: 958 of 3,087 molecules have exactly the lowest value, 0.6757
  (log10), which is 4.74 on the original scale.
- adme_rlm (rat, column `LOG_RLM_CLint`): 346 of 3,054 molecules have
  exactly its lowest value, 1.0279, which is 10.66.
- The highest value is shared by 1 molecule (adme_hlm) and 2 (adme_rlm).

The reading is that the assay cannot resolve clearance slower than that
value. So these labels are censored at the LOWER end. This is inferred from
the counts. It has not been checked against the source paper (Fang et al.
2023). Check the paper for the assay's lower limit first, and say what you
found.

Local state, not re-checked: `~/.cache/kirby/rep_cache/adme_hlm` holds 9
representations (atom_pair, avalon, coulomb_matrix, ecfp4, maccs, mordred,
pdv, rdkfp, top_torsion) and no `_identity.json`. No stored hybrid search
for adme_hlm or adme_rlm was found in either `result_db`.

## What exists already: n30

`tests/experiments/noise/n30_censored_likelihood.py` is the one experiment
about censoring. Read its docstring in full. In short, it clips training
labels at the upper end, then compares ways of handling the clipped ones. Each
is scored against the fit on the clipped labels, on the same held-out
molecules:

- raise every clipped label by one estimated amount (`constant_lift`)
- raise each clipped label by its own estimated amount (`per_molecule_lift`,
  with a shuffled control)
- replace each clipped label with a truncated-normal expectation
- drop the clipped molecules
- pull every label toward an out-of-fold prediction (winsorising)
- keep the labels and fit a model that treats them as bounds (xgboost AFT, a
  censored likelihood)

Its limits, read from the code on 2026-09-27:

- It only clips artificially, and only at the upper end. The lift formulas
  assume upper censoring; lower censoring needs them mirrored.
- It has no path for real censored labels. `n30:124` says whether adme_hlm's
  floor is real is unsettled. The owner now treats it as real censoring.
- At level 0.0 only the winsorising arms run, because no limit exists.
- `xgboost_aft` is compared with plain xgboost on lifted labels. It differs in
  the model and the labels at once.
- The random-flag control exists only with `KIRBY_NOISE_CONTROLS=1`.

An earlier measurement outside the repo (herg_ki and caco2, artificial upper
clip, one representation, 3 folds) is in
`/Volumes/seagate/local_noise_run/loop_headroom/FINDINGS.md` sections 6 and
11-14. Treat it as a dated claim, not a result.

## The uncertainty finding to re-test

From `~/repos/qsar_qm_models/paper.tex`, lines 770 and 778. Read them
yourself.

- Under censoring, NGBoost, QRF and the GPs shrank their predicted
  uncertainty as clipping narrowed the label spread.
- Per molecule, predicted uncertainty was LOWEST on the clipped labels.
  Spearman between uncertainty and injected error: -0.84 to -0.80 across
  folds for NGBoost, and -0.47 to -0.41 for the GP. Both were measured on
  QM9 with half the labels clipped, on PDV. On logD every model except the
  two GPs was at -0.29 or lower in every fold.
- The paper suggests using this in reverse: the molecules a model is most
  certain about are the likeliest to be clipped.

This was measured on single representations in that paper's own pipeline.
The owner wants it re-tested with the same models on the same single
representation, and on KIRBy hybrid representations.

## What the owner asked for

1. **Artificial censoring**, swept over the fraction clipped.
2. **Methods tailored to censoring**: learning-with-noisy-labels methods and
   cleaning methods designed for clipped labels.
3. **Uncertainty as a detector of clipped labels, in reverse**: lowest
   predicted uncertainty means likeliest clipped. Use it both on its own,
   graded against which labels were clipped, and inside existing mitigation
   methods, for example choosing which labels to lift or down-weight.
4. **The real case**: adme_hlm (and adme_rlm) run through the same methods.

## Rules that apply (from the owner)

- **The main measurement is D against D'.** D is the training labels as given
  (recorded, or clipped). D' is the labels after a method. The same model is
  fitted on both and scored on the same held-out molecules. On adme_hlm, D is
  the recorded labels. Helpers are in `d_vs_dprime.py`.
- **No knowledge that only exists because noise was injected.** No method may
  use how many labels were clipped, where the limit is, or which molecules
  moved. It can use only what a user holding the labels could see. Grading
  against the true clipped set is evaluation and is allowed. Where a method
  needs a count, sweep a fixed share (0.05, 0.10, 0.20 of training molecules).
- **Single representations are each model's own top 2**, from its stored
  search (`_shared.noise_base_reps_by_model`). Never a fixed list.
- **Every cleanup has a random control** of the same size.
- **Run the whole sweep. Never collapse an axis.** No significance tests;
  report per-fold scores and their spread. No pooling across datasets.
- **Name every way two compared arms differ.** If more than one, say so.

## Questions to put to the owner, with numbers, before building

1. Extend n30, or write a new experiment file? Say what each costs in code
   and in re-runs.
2. **Is a detector needed when the limit is visible?** On adme_hlm all 958
   clipped labels share one exact value, so they can be found without any
   model. Where does detection add something? For example: labels recorded
   with rounding or noise near the limit, several sources with different
   limits, or a hidden limit. Propose which of these to simulate.
3. Which models give per-molecule uncertainty in KIRBy today, and which from
   the paper need porting? Check `n23_uncertainty_cleaning.py` (`SOURCES`:
   hetero_gp, ngboost, heteroscedastic_nn, gaussian_process) and n32's
   detectors (ngboost_rescaled, quantile_forest_rescaled). Say which of the
   paper's models (NGBoost, QRF, GP variants, networks with a variance head)
   are missing.
4. Upper, lower, or both sides for the artificial clipping? adme_hlm is lower.
5. Which datasets? Say which have stored hybrid searches locally. A hybrid
   for adme_hlm needs a search to be built first.

---
