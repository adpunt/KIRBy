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

### The metrics, decided 2026-11-06: ALL EIGHT

Her words: "great yes please add all 8". Every hybrid is scored on all eight
scorers in `tests/experiments/similarity_metrics/simeval.py`:
`eval_knn_readacross`, `eval_modi_binary`, `eval_patterson`,
`eval_cliff_auroc`, `eval_violation_rate`, `eval_vs_enrichment`,
`eval_noise_detect`, `eval_noise_recovery`.

The grid is 8 hybrids times 5 datasets times 8 metrics = 320 numbers.

### ⛔ THE METRICS AGAINST EACH OTHER IS AN ANALYSIS TO DO ONCE THE RESULTS ARE BACK

Not a setting to decide before the run, and not something the run itself
answers. Her words, 2026-11-06: "It would be really interesting to compare, see
what's produces the same rankings vs what's unique vs what tracks with other
results (what's useful) / This is an interesting question" — and "indicate that
this is something we have to analyse once the results are back".

It costs nothing to run. Once the 320 numbers exist, the first two questions
below are arithmetic on them, in seconds. Nothing has to be decided now, and no
metric is dropped in advance on the strength of any of it.

Three separate outputs, written down here and not yet built.

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

**Which metric tracks other results.** Her words: "what tracks with other
results (what's useful)". Which results those are is not recorded here, because
she has not said, and the earlier attempt to fill it in with a model's score on
each hybrid was Claude's invention and is cut above.

One of the eight is already a cliff test and the other seven are not.
`eval_cliff_auroc` scores whether label disagreement among a molecule's five
nearest neighbours flags an externally defined cliff set. The other seven score
whether close molecules have close labels, and say nothing about cliffs.

### ⛔ TWO THINGS CLAUDE INVENTED AND PUT IN THIS FILE AS HER DECISIONS. BOTH CUT 2026-11-07.

**"The downstream-score arm".** Correlating each metric's ranking of the eight
hybrids against a model's cross-validated score on that same hybrid, 40 fits.
No sentence in her outline asks for this. Her own downstream test for a cliff
detection method is section 0's D, D' and D'': "the difference between D'' and D
and D'' and D' should indicate the relative performance of different AC
detection methods." That is `cliff_aware_cleanup.py`, already written. A model's
RMSE on a hybrid's features is a different question and was not asked.

**"Which similarity-to-distance conversion".** The eight scorers want a distance
and `cliff_rules.py:1093` stores `1/(1+d)`, so this was written up as a setting
for her. It is not one: the distance comes straight from the hybrid's feature
table with the same `scipy_cdist` call `cliff_rules.py:1092` already makes, and
nothing is converted. The stored similarity exists for the cliff definitions,
which are a separate job.

### Open, for her

Nothing. Her third comparison question, "what tracks with other results", is
answered against results she already has, and the sentence naming which ones is
hers to write.

### Not measured

The 40 hybrid searches, 8 hybrids on 5 datasets. That is the cost of this whole
comparison; the eight scorers are about 21 minutes for the full grid and the
metric-against-metric analysis is seconds on top.

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

The same sweep with shares past 30, so a turnover is inside the grid rather than at its
edge.

THE GRID, decided by her on 2026-11-07, her word: "45". `TARGET_SHARES` at
`tests/experiments/cliffs/settings.py` now holds 5, 10, 15, 20, 25, 30, 35, 40 and 45 per
cent. Their own flag marks 38.7 per cent of molecules, so that rate is inside the grid with
two points above it.

Cost: the 30 targets over 6 shares took 3.54 hours on 2026-11-06. The per-share cost was
not measured on its own. Every share is read off one sorted vector per similarity
criterion rather than a new n-by-n matrix, so added shares add reads and not matrices.

The resume key was fixed on 2026-11-07, so a re-run no longer needs `--force-rerun`: the
partial file records which settings each target's row was computed under, a row from other
settings is recomputed, and the run prints how many rows it refused to reuse.

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

## ⛔ WHAT THIS FILE LEFT OUT UNTIL 2026-11-07, AND WHY

It was written with one section per TBD number. Everything in her outline
without a TBD number had nowhere to go and fell out without being noticed. Her
words: "they are mentioned repeatedly, and i have pasted the same plan at least
4 times in chats". The sections below are those items. None of them is a new
idea; each is a line of her own outline.

---

## mmpdb, one named chemical change

Her outline, under which pairs count as similar: "mmpdb (one named chemical
change)".

### What exists

`matched_molecular_pairs` is one of the 44 similarity criteria in
`cliff_rules.SIMILARITY_CRITERIA`. The named rule pairing it with a gap is
`dablander`: matched pairs on the similarity side,
`absolute_2_half_cliffs_dropped` on the gap side.

It reads a pair list the dataset's loader ships. The three ChEMBL targets of
Dablander et al. 2023 ship one; no other dataset does, so on every other dataset
this criterion records that the loader shipped no matched pairs rather than
running.

### Open, recorded in `HANDOFF_AC_STUDY.md` 9b and not decided

On `chembl_qsar_d2`, 6,333 molecules, all 35,484 pairs arrive. The loader prints
460 cliffs and the rule here counts 456. Four pairs have a potency difference of
exactly 2.000: `cliff_rules._far` is a strict `gap > cutoff` and the loader's
count is at or above. Which of the two Paper 3 uses is hers.

---

## GraphCliff, SemiMol and the 2025 foundation-model paper

Her outline, under direct AC prediction, items 1 to 3. Item 4 is TBD7 and has
its own section above.

### What exists in the code

`definition_agreement.py:78` names all three:
`('graphcliff', 'semimol', 'foundation_model_2025')`. For each one it reads a
per-molecule cliff flag from

    /Volumes/seagate/external_cliff_flags/<source>__<dataset>.json

holding `source`, `what_it_is`, `reference`, `smiles` and `cliff_molecule`. The
flags are matched BY SMILES, never by row order, because an outside model's
molecule order is its own. A molecule the dataset has and the file does not is
left out of the comparison and counted, rather than treated as a non-cliff: a
missing prediction is not a prediction of "no".

So every one of her definitions is already scored against all three the moment a
file appears, and nothing has to be written to make that happen.

### ⛔ WHAT BLOCKS IT

No file exists for any of the three. Checked 2026-11-07 by running
`cliffs/_smoke_held_out_and_agreement.py`, which prints: "the three published
models: /Volumes/seagate/external_cliff_flags holds no flag file, so the run
records it as missing and writes no row".

Obtaining those three models' per-molecule predictions on her five datasets is
the prerequisite. It is not a run of her own code and no part of this repo does
it.

### Open, for her

Whether these three are obtained, and for which datasets. Nothing in the code
waits on a setting; it waits on the files.

---

## Her two confirmation tests

Her outline: "Actually two tests I can run (and should run)".

### Test one, her words: "My methods on MoleculeACE with identified ACs - see if it actually identifies them"

`cliffs/moleculeace_agreement.py`. Ran 2026-11-06 over all 30 targets in 3.54
hours. This is the one test in the whole study that has produced numbers.

### Test two, her words: "Compare the results of my AC detection methods on my datasets vs the published AC prediction models on those same datasets ... as well as the standard ECFP4 Tanimoto with standard cutoffs (I should already be producing those given my setup, but I need to highlight them)"

`cliffs/definition_agreement.py`, a ledger unit on every dataset. It scores each
definition against three kinds of comparison set: the standard ECFP4 Tanimoto
cutoffs, MoleculeACE's published rule, and a published model's flag where one is
on disk.

The ECFP4 Tanimoto half needs no outside file and runs now. The published-model
half is blocked by the section above.

### Test three is the D, D' and D'' comparison

That is `cliff_aware_cleanup.py` and it waits on her M and her L. Its section is
below.

---

## S04 and S09

Her outline: "Then how they map into experiments ... S04 - do models predict
better/worse for ACs compared to non-ACs? (held-out prediction error on cliff
molecules against non-cliff molecules)" and "S09 (the relevant part) - how do
hybrids impact AC detection on the above tests?"

### S04's question has a new unit

`cliffs/error_by_cliff_status.py` computes held-out prediction error on cliff
molecules against non-cliff molecules, under the new definitions. Never run.

### The old files are already out of both papers' run sets

Checked 2026-11-07. `tests/experiments/papers/plan.py:110` sets Paper 3's
similarity experiments to `('s01', 's02', 's05', 's08', 's10')` and Paper 2's to
`('s03',)`. The old `s04_activity_cliffs.py` (2026-02-18) and
`s12_cross_domain_cliffs.py` (2026-03-25) are in neither list, so neither runs
for either paper. The new `cliffs` category is already the only thing that does.

Her sentence, 2026-11-07: "I thought I was replacing all the AC experiments with
the new setup." That is what the ledger already does.

⛔ `s09_cross_dataset_summary.py` reads s04's stored output and fits nothing
("Dependencies: s02, s04, s05, s06, s08"), so it has no cliff numbers to
aggregate while s04 does not run. It is also not in either paper's list.

⛔ `thesis.tex:3341`, table `tab:similarity:cliffs`, was built from the old s04
and s12 runs, and the prose at `:3311` and `:3314` cites it. Those numbers exist
and the experiments behind them are no longer in the run set.

### S09's question is answered by an axis, not a unit

"How do hybrids impact AC detection" is answered by running the new units with
the hybrids in the representation list. 10 of the 44 criteria in
`cliff_rules.SIMILARITY_CRITERIA` are hybrid ones (`hybrid_bits_*` and
`hybrid_counts_*`), and each fold's two restricted hybrids are searched on that
fold's training molecules only. No separate unit computes it.

---

## The per-dataset count her outline asks for

Her outline: "Random stats from experiments: How many molecules set in at least
one cliff pair".

`cliff_rules.cliff_sets` returns `n_cliff_molecules` and `pct_cliff_molecules`
beside the pair counts, for every definition, so `cliff_sets` writes it per
definition per fold. Nothing further is needed.

---

## The two things she assigned as work, not questions

Her outline, verbatim: "Help me identify published methods that go into
determining whether or not a pair is in fact an AC" and "Help me brainstorm
anything else".

Neither has been done. `HANDOFF_AC_STUDY.md` section 8 holds the scope of the
first one. Both are searches and write-ups, not runs, and neither blocks
anything above.

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
