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

### The five datasets (hers, 2026-11-06)

`qm9`, `herg_ki`, `caco2`, `herg`, `logd`. `adme_mdr1` is out, her words: "remove
MDR1-MDCK. I don't need THAT much data and I have more experience with logD".

`qm9` is already the HOMO-LUMO gap at 5,000 molecules:
`tests/find_best_hybrid_molecular.py:532` calls `load_qm9(n_samples=5000,
property_idx=qm9_property_index('gap'))`.

Molecule counts, and where each number came from:

one row is one dataset, the count is molecules

| dataset | molecules | source of the count |
|---|---|---|
| herg_ki | 1,412 | `tests/experiments/dataset_policy.py:232` |
| caco2 | 2,158 | `PRECOMPUTE_RUN_PLAN_2026_09_14.md:621`, dated |
| qm9 | 5,000 | `tests/find_best_hybrid_molecular.py:532` |
| logd | 5,028 | `REP_CACHE_REBUILD_2026_09_16.md:29`, dated |
| herg | 7,772 | `tests/experiments/_shared.py:4291`, dated |

⛔ `herg` carries a binary label, held as `'classification', 'auc'` in
`tests/experiments/dataset_policy.py:249`. Three of the eight scorers are
degenerate on it, not merely noisier. The label difference between two molecules
is 0 or 1, so `eval_patterson`'s 95th percentile is 1 almost always,
`eval_violation_rate` compares against a median of 0 or 1, and
`eval_vs_enrichment`'s most-active-quarter cut lands on a class boundary rather
than a quantile. `eval_noise_detect` and `eval_noise_recovery` inject noise on a
label scale a binary label does not have.

### ⛔ ONE OF THE EIGHT HAS A REPRESENTATION AS ITS ANSWER KEY

`eval_cliff_auroc` scores against `cliff_ground_truth`, which is ECFP4 Tanimoto
at or above 0.55 with a label jump above twice the median absolute deviation.
`simeval.py:462` already marks it `'cliff_auroc': False` in
`EVAL_IS_ACTIVITY_GROUNDED`.

Its docstring argues this is not circular because "a metric never scores
itself". That holds for a single representation competing against ECFP4. It does
not hold for a hybrid, because a hybrid may CONTAIN ecfp4, and then it is partly
being scored against itself. The restricted binary hybrids are the ones most
likely to contain it.

### Measured cost of the eight scorers, 2026-11-06

Timed on random distances and random labels, because the cost of every one of
these is set by the molecule count and by k, not by the values. Script at
`/Volumes/seagate/kirby_scratch/simeval_timing/time_scorers.py`, local machine.

one row is one dataset, the number is seconds to run all eight scorers once on
one distance matrix, plus the cost of building that matrix

| dataset | all eight scorers | one Tanimoto distance matrix | total, one hybrid |
|---|---|---|---|
| herg_ki, 1,412 | 0.8 s | 0.2 to 0.4 s | ~1.2 s |
| caco2, 2,158 | 2.2 s | not timed | ~3 s |
| qm9, 5,000 | 21.4 s | 2.9 s | ~24 s |
| logd, 5,028 | 16.1 s | 2.9 s | ~19 s |
| herg, 7,772 | 97.1 s | 6.2 to 13.3 s | ~105 s |

The 5,000 and 5,028 rows are 21.4 s and 16.1 s for the same code at almost the
same molecule count. That 32 per cent spread is this machine's run-to-run noise,
not a property of the datasets, so no number here should be read tighter than
about a third.

The slowest three at 7,772 molecules: `eval_violation_rate` 37.3 s,
`eval_vs_enrichment` 12.9 s, `cliff_ground_truth` 11.1 s.
`eval_violation_rate` builds the whole molecule-by-molecule label difference
matrix and then `triu_indices` over it, which is 483 MB plus 483 MB at that
count; peak resident memory reached 2,076 MB.

**All five datasets, one hybrid, all eight scorers: about 155 seconds. Eight
hybrids: about 21 minutes.** The scorers are not the cost of this comparison.
The cost is the eight hybrid searches, which this timing says nothing about.

### The metric-against-metric comparison, her idea on 2026-11-06

Her words: "It would be really interesting to compare, see what's produces the
same rankings vs what's unique vs what tracks with other results (what's
useful)". Three separate outputs, written down here and not yet built.

**Which metrics give the same ranking.** Rank the eight hybrids by each metric on
each dataset, then correlate every metric's ranking against every other
metric's. One eight-by-eight matrix of rank correlations per dataset, five
datasets. A pair that agrees on all five is redundant and one of the two can
carry the comparison.

Two pairs are near-duplicates before any number is computed, by construction and
not by finding. `eval_knn_readacross` returns a Spearman and a negative RMSE from
the SAME predictions. `eval_modi_binary` and `eval_violation_rate` are both
nearest-neighbour label agreement, at one neighbour and at five.

**Which metric is unique.** A metric whose ranking correlates with nothing else
is either carrying its own signal or is noise. Those are told apart by whether
its own ranking is stable across the five datasets, so each metric also needs the
spread of its hybrid ranking across datasets, reported as the raw ranks.

⛔ The whole basis is eight hybrids. A rank correlation over eight items is read
as a direction and nothing finer, and no significance test is run on it.

**Which metric tracks a downstream result.** Correlate each metric's ranking of
the eight hybrids against the score a fitted model gets on that same hybrid, in
the dataset's own metric. This is the only one of the three that answers "useful".

⛔ It needs an input that does not exist: a downstream model score per hybrid per
dataset. That is a model fit per hybrid per fold, not a scorer call, and its cost
is not measured.

### Open, for her

Which of the eight metrics the hybrid comparison is scored on, and whether the
downstream-score arm above is in scope.

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
