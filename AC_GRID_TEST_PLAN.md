# AC grid test plan

Started 2026-11-06. One section per TBD. Each says what has to be measured before
Adelaide can decide it, and nothing about what the answer should be.

Why this file exists, her words on 2026-11-06: "days of back and forth server sessions led
me to get the answer to exactly one TBD, which is NOT ENOUGH".

What is already decided is in `HANDOFF_AC_STUDY.md` section 9a, not here.

---

## TBD2, which representations

### Her decision so far

Single representations: the top base plus ecfp4, or the next best if the top is ecfp4.

Hybrids: the first hybrid of the general tier's predictive search. Seven candidates left:

1. predictive search, model 1, first hybrid — **taken**
2. predictive search, model 1, second hybrid
3. predictive search, model 2, first hybrid
4. predictive search, model 2, second hybrid
5. similarity-objective search — one per dataset, fits no model
6. distance tier, knn only
7. predictive search, binary input only
8. similarity-objective search, binary input only, all representations

### What has to run

All eight hybrids on several small-molecule datasets, fast tier, so each one has a score
on the same molecules and folds.

### ⛔ THE TRAP IN SCORING A REPRESENTATION, which decides which metrics are usable

Every activity-landscape score in the literature is computed INSIDE a representation.
SALI divides the label difference by one minus the similarity. iCliff and ROGI are
functions of the pairwise distances. So ranking representations by any of them asks each
representation to grade itself, and the answer moves with the representation rather than
with the chemistry.

The ROGI authors say this themselves: ROGI "strongly depends on the molecular
representation and the distance metric used", and they suggest it "might be possible to
use ROGI to compare variants of the same representation" — variants of one, not one
against another. The 2026 Geometry preprint puts the same point as a fact about cliffs:
"depending on the representation, the pair may be classified as an activity cliff, or not
a cliff at all".

So the metrics split in two, and only the first kind can choose between representations.

**Usable: scored against something outside the representation.**

- the labels — a prediction or retrieval task where the label is the answer key
- an external cliff flag — MoleculeACE's `cliff_mol`, which no representation here produced
- an independent structural truth — maximum common substructure overlap
- an injected noise mask — the corruption is known, so finding it is a real score

**Not usable for this choice: scored inside the representation.** SALI, SARI, iCliff,
ROGI, local roughness, the rivality index. They describe a landscape; they cannot rank the
spaces the landscape was drawn in. They stay useful for describing one dataset under one
fixed representation.

### The metrics, and whether the code already has them

Her Metric 1, prediction from nearest neighbours, is `eval_knn_readacross` in
`tests/experiments/similarity_metrics/simeval.py`: scaffold-grouped folds,
distance-weighted kNN on the precomputed distance matrix, reported as Spearman and
negative RMSE. Cited to Golbraikh, Muratov, Fourches & Tropsha, JCIM 2014.

Her Metric 2, whether it helps remove noise, is already written twice in that same file
and is the strongest answer to her question:

- `eval_noise_detect` — inject noise, flag each molecule by how far its label sits from
  the median of its k nearest neighbours in this representation, and score the AUROC
  against the known injected mask. Named there as Wilson / edited-nearest-neighbour
  editing quality.
- `eval_noise_recovery` — estimate each label from its neighbours, and correlate the
  correction it implies against the injected error on the corrupted molecules. A good
  space gives a correction equal and opposite to the error.

Both are non-circular: the injected mask is outside every representation.

Four more in the same file, all with the label or an external flag as the answer key:

- `eval_modi_binary` — the class-balanced rate at which a molecule's nearest neighbour
  shares its activity class, median split. Golbraikh et al. JCIM 2014.
- `eval_patterson` — among the closest few per cent of pairs, the 95th percentile of the
  label difference. A valid space bounds the label difference from above at small
  distances. Patterson, Cramer, Ferguson, Clark & Weinberger, J Med Chem 1996, 39:3049.
- `eval_vs_enrichment` — take the most active quarter as actives, rank the rest from each
  active by distance, and report BEDROC. Truchon & Bayly, JCIM 2007, 47:488.
- `eval_cliff_auroc` — does local label disagreement among the k nearest neighbours in
  this space flag an externally defined cliff set. Guha & Van Drie 2008, JCIM 48:646.
- `eval_violation_rate` — the share of nearest-neighbour pairs whose label difference
  exceeds the global median.

⛔ NONE OF THESE IS REACHABLE FROM THE CLIFF EXPERIMENTS. `simeval.py` is imported only
by the eight standalone scripts in `tests/experiments/similarity_metrics/`, checked
2026-11-06. Nothing in `tests/experiments/cliffs/` or the noise suite calls it.

### Not in the code

Maximum common substructure overlap as an independent structural answer key. Huber &
Pollmann, J Cheminform 2026, benchmark fingerprints against it precisely to avoid the
circularity above, and `SIMILARITY_STUDY_HANDOFF.md` section 3 records that nobody has
graded metrics on both structural fidelity and activity coherence on the same targets.
MCS is NP-hard and slow, and a methyl-to-nitro swap barely moves it while moving activity
a lot, so it is necessary and not sufficient.

### What her own prior literature report already settled

`SIMILARITY_STUDY_HANDOFF.md` section 3, from a deep-research report read in full
earlier:

- ECFP with Tanimoto is hard to beat, and pretrained embeddings and graph networks do not
  reliably beat it (Riniker & Landrum 2013; O'Boyle & Sayle 2016; van Tilborg 2022;
  Praski 2025, 25 models by 25 datasets).
- The coefficient choice is low-stakes: Tanimoto, Dice, Cosine and Soergel are close
  (Bajusz 2015; Todeschini 2012).
- Counts against bits matters more than the coefficient (Huber & Pollmann 2026).
- The open gap it names: nobody has closed "a better metric gives measurably cleaner
  labels", benchmarked against the roughly 0.5-log noise floor (Kramer 2012, 0.54 pKi
  single-measurement SD; Kalliokoski 2013, 0.69 pKi and 0.87 pIC50 paired-difference
  sigmas).

That last point is why her Metric 2 is the one worth building: it is the gap, not a
convenience.

### Open, for her

Which of the eight metrics above the hybrid comparison is scored on, and on which
small-molecule datasets.

---

## TBD3, the similarity cutoff and whether it is global, per domain or per dataset

### What has run, and why it did not answer it

`moleculeace_agreement` over MoleculeACE's 30 targets, 2026-11-06, 3.54 hours. Every
cutoff of her five coefficients scored against their `cliff_mol` flag.

It cannot settle TBD3. Their flag is itself a fixed 0.90 cutoff on fingerprint overlap, so
definitions shaped like theirs win by construction: every row above her own definitions is
their any-of-three similarity at 0.85 to 0.95. Her best definition reached a pooled
Matthews correlation of 0.4316 and 0.1309 on its worst target.

### What has to run

`cliff_sets` and `definition_agreement` on her six small-molecule datasets, where the
problem she is claiming to solve shows up: one fixed cutoff gives 0.2 per cent of QM9's
molecules and 74 per cent of Caco-2's.

Both are built and are ledger units. Measured cost of the definition sweep, 2026-11-06:
2.52 microseconds times the square of the molecule count per similarity matrix, so about
4 minutes a fold at 1,412 molecules and 201 minutes at 9,978.

### Open, for her

Nothing until those run.

---

## TBD4, MoleculeACE's cutoff

Decided 2026-11-06. Their published 0.90, which beat every other cutoff of their rule at
every gap swept: pooled Matthews 0.7647 against 0.5876 at 0.95 and 0.5387 at 0.85. One
sentence is in `PAPER_3_OUTLINE.md`, marked as not a published result.

---

## TBD5, the cutoff on one hybrid overlap in place of their three scores

### What blocks it

No hybrid builds on MoleculeACE's 30 targets. The hybrid search asks `DECISIONS.md` for
two downstream models per dataset and none of the 30 has a row, so the search refuses.

### What has to happen first

Two models recorded for each of the 30 targets, or the cutoff swept on her own datasets
instead and scored by something other than their flag.

---

## TBD6, the share of molecules the moved gap aims at

### What has run

The same 30-target run swept 5, 10, 15, 20, 25 and 30 per cent. Agreement was still
climbing at 30, the top of the grid, and their flag marks 18,831 of 48,714 molecules,
which is 38.7 per cent.

### What has to run

The same sweep with shares past 30 — 35, 40, 45 — so the turnover is inside the grid. The
share axis is nearly free: every share is read off one sorted vector per similarity
criterion, so this costs about the same 3.5 hours.

⛔ It needs `--force-rerun`. The resume file is keyed on the target and the scopes, not on
the settings, so a plain re-run would skip all 30 targets and return the old shares. That
key should include the settings.

### What it still will not settle

A larger share scores better partly because it moves her base rate toward their 38.7 per
cent. The extended grid says whether there is a peak at all or whether agreement tracks
the base rate all the way up; inside the current grid those two look identical.

---

## TBD7, training a model on MoleculeACE's cliff flag

### What has run

One target as a crash check, 2026-11-04, local, so debug. CHEMBL4203_Ki, 731 molecules,
five scaffold-grouped folds, 151 seconds. Her re-implementation of their rule reproduces
their flag exactly. A trained classifier reached a median Matthews correlation of 0.2902
across folds against 0.1663 for ECFP4 Tanimoto at 0.55.

### What has to run

The same over all 30 targets, if she wants it at all.

---

## Her M and her L

Neither comes from any run in this file. Both are "the best" of the noise experiments'
own results, which have not run.

---

## Sources read on 2026-11-06

Code, opened this session: `tests/experiments/similarity_metrics/simeval.py`,
`tests/experiments/noise/cliff_rules.py`, `figures_ch5_7/tab_distance_tracks_activity.tex`,
`SIMILARITY_STUDY_HANDOFF.md` section 3.

Literature, searched this session rather than recalled:

- ROGI, roughness of molecular property landscapes, and its dependence on the
  representation — https://arxiv.org/pdf/2207.09250
- ROGI-XD used to compare representations, and the warning that roughness ignores
  train-test shift — https://arxiv.org/html/2611.08420
- iCliff, a global roughness metric, correlated with mean pairwise SALI over 30 AC
  databases; bioRxiv preprint, not verified peer-reviewed —
  https://www.biorxiv.org/content/10.1101/2025.03.09.642269v1.full
- SALI's definition and its undefined value at similarity one; SARI for SAR continuity —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3159128/
- The Geometry of Activity Cliffs, representation dependence of the cliff label —
  https://arxiv.org/html/2605.30831v1
- A ROGI implementation shipping SARI, MODI and RMODI — https://pypi.org/project/rogi/

No chemistry-specific benchmark of label-noise cleaning turned up; the label-quality
benchmarks found are image and tabular (AQuA, https://arxiv.org/pdf/2306.09467v2;
CleanPatrick, https://arxiv.org/html/2505.11034v2). That absence is consistent with what
`SIMILARITY_STUDY_HANDOFF.md` section 3 already records as the open gap.
