# Budget-split normalization — handoff

# ⛔ DONE. This brief has been carried out. Nothing here needs running again.

**f32 ran locally on 2026-09-10** — `tests/experiments/framework/f32_budget_split_normalisation.py`,
nine datasets, 45 folds. The decision is **Shape B**, and the `f1_weighted` null is the f1 of
always predicting the majority training class. Both answers are on the `>>` lines of
`FRAMEWORK_FINALIZATION.md` §3.9, and the full write-up with every number is in §5 under
*f32 — the budget split needs one normalisation rule per metric*. §3.9 is closed; the code
change itself is not made.

**Everything below is the brief as it was written on 2026-09-09**, kept because it holds the
original measurements. Two of its instructions were not followed and the reasons are in the
§5 write-up: the experiment goes through `create_hybrid` on its own outer folds rather than
`nested_cv_evaluate`, and two of the nine datasets have no representation pool on this
machine.

**Next free f-number: f33.**

---


---

## The problem, plainly

`create_hybrid` decides how many features each representation gets by making the share
proportional to that representation's score. Making a share proportional to a score only
means something if zero on that score means "this representation contributes nothing".

Spearman has that property. Zero correlation is no signal.

Negative RMSE does not. Its zero is a *perfect* prediction, so distance from zero measures
the wrong thing. AUROC does not either — chance is 0.5, not 0, and 0.5 is a positive
number, so nothing corrects it.

The result is that the same five representations, in the same order, with the same gaps,
split a budget differently depending on which metric the dataset happens to report.

---

## What was measured

Measured on 2026-09-09 by calling `_sanitize_baseline_scores` and
`_compute_allocation_weights` directly with made-up scores. These are the real allocation
functions, not a stand-in. The scores are illustrative; the behaviour is the code's.

**Budget of 100 across five representations.**

| representation scores | resulting split |
|---|---|
| Spearman 0.700 0.697 0.695 0.694 0.693 | 20 · 20 · 20 · 20 · 20 |
| −RMSE −0.300 −0.303 −0.305 −0.306 −0.307 | 50 · 29 · 14 · 7 · 0 |
| AUROC 0.90 0.70 0.55 0.52 **0.50** | 28 · 22 · 17 · 17 · **16** |
| the same AUROC, re-centred to 2·(auc−0.5) | 60 · 30 · 7 · 3 · 0 |

A representation at chance gets 16 of 100 features.

**The weighting schemes respond to different halves of a normalisation.** Same five scores,
transformed three ways:

| | as measured | ×10 (scale only) | +5 (offset only) | min-max |
|---|---|---|---|---|
| `performance_weighted` (linear) | 31·24·20·16·9 | **31·24·20·16·9** | 21·20·20·20·19 | 40·28·20·12·0 |
| `power_2` | 42·26·18·11·3 | **42·26·18·11·3** | 22·21·20·19·18 | 54·27·14·5·0 |
| `softmax_T1` | 25·22·20·18·15 | 75·17·6·2·0 | **25·22·20·18·15** | 31·23·19·16·11 |
| `rank_exp_d0.3` | 33·25·18·14·10 | **33·25·18·14·10** | **33·25·18·14·10** | **33·25·18·14·10** |

Read that as: `linear` and `power_2` ignore the scale and respond only to where zero sits.
`softmax` is the mirror image — it ignores the offset and responds only to the scale.
`rank_exp` ignores both, because it reads the order.

**Two consequences that matter for the grid.**

- **min-max and subtract-the-minimum are the same transform** for `linear` and `power_2` —
  verified, identical splits — because dividing by the range cancels in a ratio. They differ
  only under `softmax`. Do not spend cells on both for the ratio schemes.
- **`softmax_T1` is an equal split on realistic score spreads.** Five representations within
  0.007 of each other give 20·20·20·20·20 whether the metric is Spearman or −RMSE. Widen the
  spread to 0.40 and it is still 26·21·18·18·17. This is a documented softmax failure mode —
  narrow-range inputs give a flat distribution — and `softmax_T1` is one of the five default
  allocation methods. **Confirm this against a real search's scores before acting on it;**
  the spreads above are illustrative.

---

## The two shapes this can take

The user of `find_best_hybrid` supplies data, a model and a budget. They cannot be asked
which normalisation they want. So there are two options and no third.

### Shape A — the search picks it

Normalisation becomes another axis, chosen inside the search the way the allocation method
already is. Nothing is hardcoded and it can differ per dataset.

**Cost.** Counting only the cells that are not duplicates of each other: `rank_exp` needs no
normalisation axis at all (1 cell). `performance_weighted`, `power_2` and `wglobal_power2`
need the offset choices only (3 each). `softmax_T1` needs the scale choices only (3). That
is 13 where there are now 5 — **2.6× the configuration count** of every default search.

It also adds selection variance on a knob whose measured effect is small: the allocation
axis moves results by 0.009–0.018 while the model choice moves them by 0.27
(`project_knob_sensitivity_2026_07_16`).

### Shape B — a hardcoded rule, one line per metric

The metric is already a known input — the caller passes it, or the registry supplies it.
So a rule keyed on the metric needs nothing from the user and costs nothing at search time.

**The rule is nearly forced by what each metric means.** Across the 70 registry entries:

| metric | datasets | what zero means today | rule |
|---|---|---|---|
| `spearman` | 25 | no signal | **already correct, no change** |
| `mcc` | 21 | chance | **already correct, no change** |
| `rmse` | 9 | perfect prediction | skill score against predicting the mean |
| `auc` | 8 | worse than chance | 2·(auc − 0.5) |
| `ci` | 4 | worse than chance | 2·(ci − 0.5) |
| `mae` | 1 | perfect prediction | skill score against predicting the median |
| `f1_weighted` | 2 | nothing clean | **open — see below** |

46 of 70 datasets already behave correctly and do not move. 22 are corrected by a definition
nobody argues about. Two need a decision.

**The one genuinely open sub-question: `f1_weighted` on `tcga_brca_pam50` and one other.**
There is no agreed null. The candidates are the f1 of always predicting the majority class,
or a permutation null. Someone has to pick, and the experiment can measure both.

**Precedent.** This is the standard *skill score* construction from forecasting:
`1 − (your error / a reference model's error)`, where perfect is 1 and no-better-than-the-
reference is 0. It is the established way to make metrics on different scales comparable
before weighting by them.

---

## What f32 has to answer

Not "which normalisation is best". The rule above is already determined for 68 of 70
datasets by definition. The experiment is a **check on Shape B**, and it has two questions:

1. **Does the corrected split beat the raw one?** Compare the per-metric rule against what
   the code does today. If it is a wash, Shape B still wins on being explainable, but you
   need to know.
2. **Does the best normalisation track the metric, or the dataset?** This is the question
   that decides B versus A. If two datasets reporting the same metric want different
   normalisations, a per-metric rule cannot express that and the axis has to go in the
   search. If they agree, Shape B is enough and the search never grows.

Question 2 is the one that matters. Design for it.

---

## Suggested design

Answering question 2 needs **more than one dataset per metric family** — one each cannot
tell "the metric decides" from "the dataset decides".

- **Datasets: two per metric family, four families.** Two `rmse` (e.g. herg_ki, caco2), two
  `spearman` (e.g. flab_thermo_tresanco and one ginkgo set), two `auc`
  (tcga_luad_vs_lusc, humanness_therapeutic), two `mcc` (any two nucleotide sets). The
  `mcc` and `spearman` pairs are the control — the rule says they do not move, so if they
  move, the rule is wrong.
- **Normalisation arms: raw (today), subtract-min-always, min-max, per-metric skill score,
  rank.** Five.
- **Weighting schemes: `performance_weighted`, `power_2`, `softmax_T1`.** They respond to
  different halves of the normalisation, so collapsing them would hide the effect. Drop the
  duplicate cells named above rather than running them.
- **Hold fixed:** one model, one internal evaluator, one importance method, 5 folds. The
  evaluator and importance axes are measured noise; the model axis dominates everything and
  would swamp the effect.
- **Budgets:** at least two. Budget is the dominant lever and the split's shape interacts
  with it.

**Cost is not yet measured.** Time one cell before committing to the grid.

---

## Rules this experiment must follow

- Report `delta_mean_of_folds` from `nested_cv_evaluate`. Never `.all_results` or
  `search_mean`.
- It is a **framework** experiment. It goes in `tests/experiments/framework/` as f32, never
  in the noise or similarity suites.
- Do not cut an arm, a dataset or a budget for compute. A cut needs a stated reason —
  degenerate, broken, or undefined.
- 5 folds. Do not drop to 3 for speed.
- The result goes into `FRAMEWORK_FINALIZATION.md` §4, and closes §3.9.

---

## Where the code is

- `_sanitize_baseline_scores` — `src/kirby/hybrid.py:821`. The shift. Subtracts the minimum
  only when the minimum is negative. This is the line the whole item is about.
- `_compute_allocation_weights` — `src/kirby/hybrid.py:1019`. Turns scores into weights.
  The `linear`, `power_law`, `softmax` and `rank_exp` branches are the ones measured above.
- `_score_rep_for_budget` — `src/kirby/hybrid.py:2521`. Produces the score. Ends
  `return -raw if lower_is_better else raw`, which is where an RMSE becomes negative.
- `_score_component` — `src/kirby/hybrid.py:224`. **Prior art in the same file.** The blend
  path already converts every component to a correlation bounded in [−1, 1] with zero
  meaning no signal, because an RMSE cannot be averaged with a Spearman. The plain
  budget-split path never got the same treatment.
