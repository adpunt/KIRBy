# AC grid test plan

Started 2026-11-06. One section per TBD. Each says what has to be measured before
Adelaide can decide it, and nothing about what the answer should be.

Why this file exists, her words on 2026-11-06: "days of back and forth server sessions led
me to get the answer to exactly one TBD, which is NOT ENOUGH".

What is already decided is in `HANDOFF_AC_STUDY.md` section 9a, not here.

This file has two parts. Everything down to "Her M and her L" is Part 1, the cliff tests. Part 2,
added 2026-11-07, is the noise grid and how the cliff sweep changes it, and ends with the prompts
for the chats that finish both.

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
(Corrected 2026-11-07: `eval_knn_readacross` gives two numbers, so it is 360. Part 2, section N4.)

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
(Corrected 2026-11-07: true of s04 and s12, not of n47. Part 2, section N3, item 7.)

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

# PART 2: THE NOISE GRID, ADDED 2026-11-07

Why this is in this file. Her words: "I have a noise grid chat that needs a handoff, and
I'm going to combine it with AC_GRID_TEST_PLAN.md", and on 2026-11-07: "I also have an AC
cliff sweep, which will not only impact this but will expand the noise grid. So I need you
to add on to the other plan to account for everything that was supposed to be in this
handoff."

`PAPER_3_NOISE_EXPERIMENTS.md` stays the record of her noise decisions. This part says how
those decisions meet the cliff sweep, what is still wrong in the code, what is open, and
which chats finish it. The chat prompts are at the end.

Words used here. A piece is one noise kind, level and draw on one fold, one model and one
model input. A result is one cleaned training set scored for that piece. The first run has
no put-back and no twins. The second run adds them for results she names.

Everything marked "Checked 2026-11-07" was read or run in the code that day. Nothing in
this part is a scientific result, and nothing here recommends a value.

---

## N1. What she has decided about noise

One line is one decision, with her words.

- Results first, checks after: "First the results, then the checks for what I need".
- Model inputs in n44: "Lets do this: base_1 and hybrid only".
- Second base representation, all five experiments: "okay then the cut to all".
- Three noise kinds cut, all five: "Yes cut the noise from all". Those are `dense_oofmse`,
  `hetero_hidden` and `sparse_dprime_up_p10`.
- `hetero_seen` stays: "Keep hetero_seen". It keeps levels 0, 0.1, 0.3 and 1.0.
- Draws stay at 3, and 1 under censoring: "Don't modify the draws thats legit".
- Larger datasets run fewer levels: "LARGE DATASETS EXCLUDING QM9 RUN LESS". QM9: "that's
  a really important baseline to have".
- Censoring on every dataset: "nah keep censoring in for everything".
- No repeated work: "just make sure the same thing doesn't get run multiple times".
- Time recording, no limit: "yea just record the times of everything".
- No more timing sessions now: "No I'm good for now. But keep track of timing. They'll be
  tests in the future and this needs to be included in the handoff".
- The grid still gets cut: "Beyond just implementation I do still need to cut down on the
  noise grid".
- The grid review waits for the cliff work: "hold off on the grid review, I'm waiting for
  ACs".
- Extra hybrids in n43 and n47: "this needs to be pushed off to the handoff". Still open,
  section N7.
- The second batch of noise code stays: "yes that code was supposed to be implemented that
  is correct".
- Five folds and the dataset's two decided models are unchanged (Adelaide, 2026-10-30).
- The cutoff is chosen from a first portion of the noise grid, 2026-11-09: "let's do a select
  portion of the noise grid, and that after a certain set of experiments have run, determine
  which cutoff I want to stick with ... in the intial set I'll sweep the cutoffs, but I'll
  still be collecting data for the main grid, two birds, one stone." Which portion is open.
- That first portion, 2026-11-09. Datasets: "qm9, herg_ki, caco2, logd, herg". Folds: "Let's
  just do 1 fold" (which fold she has not said). Model input: "Let's just look at hybrid".
  Noise kinds, levels and which cleaning methods get the cutoff test are still open.
- Every cliff number moves out of n47 into the cliff experiments, under every definition,
  2026-11-09. Her words: "It should never use the old rule, it sohuld be swep aross all ways of
  determinign ACs like everything"; of moving the check of whether cliffs come in groups or as
  lone pairs into the cliff experiments, "that seems like a much better option"; and to "every
  cliff number lives in one place, with every definition", "yes".

Her hybrid decision from the cliff chat, which this part depends on: "the main set of
hybrids don't need to run over those 6, the distance and two binary options can be
restricted to noise/cliff experiments".

---

## N2. The state of the noise code

**Built and on the shared branch.** All of it reached the branch through other chats'
commits (031f4535, 11eddcb1, b497ba42).

- The kind cuts and the second-base cut in n43 to n47.
- Time recording: `tests/experiments/noise/noise_timing.py`.
- First run in n44: no put-back and no twins.
- Level 0 runs once, under `gaussian`, in n43, n44 and n45. n46 keeps its own because its
  censoring family differs by kind. n47 reads level 0 inside itself.
- `reliability_weight` runs once per neighbour setting instead of once per detector.
- Second run in n44: `KIRBY_N44_SECOND_RUN` names a file listing the results. It refuses to
  start with no results or no cliff definition. Its rows go in their own files.

**Tests.**

- n44 passed on the current code, 2026-11-07, on made-up data. That includes the second run:
  it wrote only the named result and its twins, with plain, cliff put-back and random put-back rows.

- n43 and n45 passed on the current code.
- The second run's many cliff definitions, 2026-11-09: `_smoke_n44.py` passed, exit 0, on
  made-up data. `"all"` built 1,184 definitions, the reference rules among them, each
  put-back row names its definition, and a name that is not a definition is reported.
- n46 and n47 passed on the first batch and were not re-run after the second.
- `tests/test_noise_alignment.py::test_decided_grid` fails. It expects
  `sparse_dprime_up_p10` without the promote flag. Whether it failed before this work is
  not checked.
- The script that looks for further repeated training sets was stopped and has no result.
  The one number so far: 1,030 of 4,032 results in a test piece read a saved fit.

**Time recording covers less than she asked.** Checked 2026-11-07: n43 to n47 record, and
n15, n20, n36 and every file in `tests/experiments/cliffs/` do not.

---

## N3. Where the cliff sweep reaches the noise grid

Seven places. Each says what the noise code does today and what the sweep asks of it.

**1. Which molecules the second run puts back.** Fixed 2026-11-09. Until then the second
run reached one definition only, ECFP4 Tanimoto at 0.55 with twice the median absolute
deviation. Her words: "This should NEVER be the only cliff defnition". Now its JSON file
names `cliff_definitions`: a list of definition keys from `cliffs/definitions.py`, or
`"all"`. There is no default. The masks are built per fold on the training molecules and
per noise version from the labels as they look. Every put-back row carries
`|cliff=<definition key>|`. The old rule is still reachable as `n25_existing`, but only
when it is named. On 48 made-up molecules with two representations, `"all"` built 1,184
definitions in 1.1 seconds.

**2. Definitions multiply the second run, and also its masks.** Every chosen result gains
2 fits per definition per piece (cliff put-back and random put-back), or 8 with twins. A
noise version also changes the labels, and the cliff mask is computed from "the labels as
they look" (`n42._cliff_masks`). So each definition needs a new mask per noise version per
fold. The cost of those masks is not measured.

**3. Whether random put-back can run at all depends on the definition.** Random put-back
needs as many non-cliff molecules as cliff molecules. `thesis.tex:4148` records it as
impossible on Caco-2, hERG blocking and LogD under twice the median absolute deviation.
Under a moved gap, her share sets the cliff count. So TBD6 affects which datasets get the
comparison. Each row now records when the random draw came up short.

**4. The test of D, D' and D'' is a second route to the same claim.** `cliff_aware_cleanup`
injects no noise, because her sentence is "Take dataset D". The second run in n44 works
under injected noise, where the corrupted molecules are known. They differ in two ways at
once: which labels they start from, and which detectors and cleanups they can run.

**5. Which hybrids the neighbours are found in.** Today a noise experiment searches 26
neighbour settings on the larger noise grid and 13 on the smaller one
(`similarity_space.GROUP_SPACES`). Her distance hybrid is already among them. Neither binary
hybrid is. Tanimoto runs only on ECFP4 and only on the larger grid, and Dice runs nowhere.
TBD1 has five coefficients and TBD2 has eight hybrids. So her "restricted to noise/cliff
experiments" adds settings the code does not hold yet.

**6. Which hybrids a model is fitted on.** n44 fits `base_1` and `hybrid`. Her
"restricted to noise/cliff experiments" does not say whether a binary hybrid is a model
input, a neighbour space or both. As a neighbour space it adds settings. As a model input it
adds whole pieces. Section N5 has the numbers for each.

**7. The noise measurements still compute old cliff numbers.** `n47.n01_part` and n47's n14
part compute cliff counts and held-out error on cliff molecules under the old fixed rule.
Her outline says "I want to get rid of these numbers" and "Make this a separate set of AC
experiments". On 2026-11-07 she said: "I thought I was replacing all the AC experiments with
the new setup." So two experiments will report held-out error on cliff molecules, n47 under
one old rule and `error_by_cliff_status` under the new definitions. The line in this file
saying the new category "is already the only thing that does" holds for s04 and s12 and not
for n47.

---

## N4. Where the code and her decisions disagree

Checked 2026-11-07, by running the ledger's own plan functions and reading the settings.
One entry is one disagreement. No fix is written here.

**The ledger would not run her noise grid.** She decided five folds, and four levels on
datasets of 5,000 molecules or fewer. `runner.common_env` removes `KIRBY_NOISE_PROMOTE`, and
`papers/worker.py` starts from it. Run that way, n44 on Caco-2 gets fold 0 only and levels 0
and 1.0, and n47 gets fold 0 only. So that is two ways it differs: one fold rather than five, and two levels rather
than four. The timing sessions avoided this by setting the flag by hand.

**`hetero_seen` has no ledger unit.** She wrote "Keep hetero_seen". `plan.noise_units` reads
`conditions_for_task`, whose list does not hold it. On Caco-2 the n44 units are `gaussian`,
`outlier_p05`, `grouped_shifted`, `censoring` and `shift_only_p10`, which is five kinds and
not her six. Run by hand, n44 does include it.

**Three more noise experiments run in the ledger and are in no noise decision.** n15, n20
and n36 are in `SINGLE_DATASET_EXPERIMENTS`. None of the three reads the kind cuts, records
time or appears in `PAPER_3_NOISE_EXPERIMENTS.md`'s counts. Through the ledger n15 runs folds
0 to 2, not five. n36 runs only where a loader keeps repeat measurements.

**Classification was never part of the noise decisions.** Five Paper 3 datasets are
classification: `herg`, `cdr_flexibility_h3`, `cdr_flexibility_l3`, `tz_her2_binary` and
`tz_her2_multiclass`. Each runs six flip kinds at six levels with 3 draws. Her decisions
name regression kinds. "LARGE DATASETS EXCLUDING QM9 RUN LESS" is not applied to `herg`,
which has 7,772 molecules and keeps all six levels.

**Cliff units are planned on 28 datasets, and her scope is small molecules.**
`PAPER_3_OUTLINE.md` records "Scope of the activity cliff work (Adelaide, 2026-11-04): small
molecules only". `papers/plan.py` gives cliff units to molecular and antibody datasets, so 14
antibody datasets carry them.

**The model inputs differ between the two suites.** `cliff_aware_cleanup` and
`error_by_cliff_status` fit `base_1`, `base_2` and `hybrid`, and one comment there says these
are "the same three the noise suite fits". The noise suite has cut `base_2`. Her TBD2 line,
"top base + ecfp4", is about which representations define a cliff, and neither file
fits ecfp4 by name.

**Her M and L can name something the unit refuses.** `cliff_aware_cleanup` accepts detectors
from `ESTIMATOR_REGISTRY` and cleanups from `n25.cleanups_for`. n44 runs 26 cleanups, and 16
of them are not in that list, such as `rewrite` and `exp_downweight`. 10 of n44's detector
names are outside the registry, such as `oof_residual` and `knn_shapley`. Her M and L are
"the best" of n44's results, so a winner could be refused.

**The fit count printed by `cliff_aware_cleanup` is low.** The printed product has no term
for L's own settings, and the row loop runs one per setting.

**The distance search is not what she said.** Her words: "this should literally just search
over knn and nothing else". On hERG Ki `tests/experiments/decided_grid.json` still gives
that tier the models svm, dnn and knn, and the evaluators knn, rf, dnn, lightgbm and svm.

**The two binary searches are not what she said.** Her list: "general tier, prediction
objective, binary input only" and "general tier, similarity objective, binary input only
(n_reps=all, not swept over for both)". `definitions.restricted_hybrids` builds `bits` and
`bits+counts`, both by the predictive search. Nothing in `tests/experiments/cliffs/` runs the
similarity objective on binary input or fixes `n_reps`. Neither search is a ledger search step.

**The settings file does not hold her TBD2 line.** `CLIFF_SETTINGS.json` holds the TBD values
the cliff units read, and was last changed in git on 2026-11-05. Its `representations` is
`"all"`, and she said "top base + ecfp4". Its `_candidates`
block still lists target shares to 30. The code reads shares from `settings.py`, so the 45
per cent runs regardless.

**The eight-hybrid comparison has no script.** Nothing joins the hybrids to `simeval.py`. Also
`simeval.run_battery` returns nine numbers from eight scorers, because `eval_knn_readacross`
gives two. So the grid is 360 numbers rather than the 320 written above.

---

## N5. How big the noise grid is now

Recounted 2026-11-07 from `n44.run_unit` with fitting stubbed out
(`scratchpad/paper3_grid_counts/count_all_regression.py`). This is n44's first run, before
the review. These are counts of results and not times. Sizes the registry does not hold were
read from `AC_GRID_TEST_PLAN.md`, the thesis and the handoffs. Whether `qm9_mu` and `qm9_alpha`
are also at 5,000 molecules is not checked.

One row is one group of datasets.

| group | datasets | versions of the labels | pieces per dataset | results per dataset | results, group total |
|---|---|---|---|---|---|
| Regression, four levels | 17 | 49 | 980 | 4,175,010 | 70,975,170 |
| Regression, four levels, one model takes no weight | 3 | 49 | 980 | 3,661,175 | 10,983,525 |
| Regression, two levels | 2 | 23 | 460 | 1,932,270 | 3,864,540 |
| `herg_ic50`, two levels, svm skipped above 10,000 rows | 1 | 23 | 230 | 966,135 | 966,135 |
| Classification, six levels | 5 | 91 | 1,820 | 2,568,020 | 12,840,100 |

Regression comes to 86,789,370 results. With classification it is 99,629,470. n43, n45, n46,
n47, n15, n20 and n36 are not in these numbers. `PAPER_3_NOISE_EXPERIMENTS.md` sections 4 and
5 predate the `reliability_weight` change. A base-input piece now has 5,468 results and a
hybrid-input piece 3,601, or 4,988 and 3,121 where no oracle flags exist.

**What each cliff decision adds to it.** One row is one candidate, on one dataset, computed
from those per-piece numbers.

| added to n44's first run | results on a four-level dataset | results on a two-level dataset |
|---|---|---|
| Nothing | 4,175,010 | 1,932,270 |
| One more model input, same size as a hybrid | +1,630,090 | +751,430 |

One row is one change to the neighbour settings. The number is how many settings a piece is
screened over.

| added to the neighbour settings | settings, larger noise grid (26 today) | settings, smaller noise grid (13 today) |
|---|---|---|
| Second base and its augmented version removed | 20 | 13, it holds no second base |
| One new space under today's distances | 29 | 15 |
| One new space under today's distances plus Tanimoto and Dice | 31 | 17 |

n44 keeps 3 neighbour settings per piece after a screen, so more settings do not change its
result count. They do add screen time. n45 grades every setting with no screen, so its count
grows with them. It has not been recounted since the decisions.

**What a definition costs elsewhere.** One row is one candidate set of definitions on one
dataset, counted with `definitions.expand`, her five coefficients, 9 target shares, and the
number of fits `cliff_aware_cleanup` prints for one M, one L and 3 outlier shares. That is
270 fits per definition, plus L's own settings, which it does not print.

| definitions | what the set holds | count | model fits in `cliff_aware_cleanup` |
|---|---|---|---|
| References only | published rules plus MoleculeACE's rule at 6 cutoffs | 64 | 17,310 |
| Continuous top base, ecfp4, first hybrid | her TBD2 so far | 1,674 | 452,010 |
| Binary top base, ecfp4, first hybrid | her TBD2 so far | 2,094 | 565,410 |
| Those with all eight hybrids, two of them binary | every TBD2 candidate | 5,944 | 1,604,910 |

Each extra continuous hybrid adds 490 definitions and each extra binary hybrid 910. Nothing
has been cut from these numbers.

---

## N6. Timing, which goes with every later chat

Her words: "keep track of timing. They'll be tests in the future and this needs to be
included in the handoff". She has also said a Claude estimate of 8.8 hours took about 3, so a
new estimate carries the word estimate and any measured number beside it.

Read recorded times with `python tests/experiments/noise/noise_timing.py <folder>`. Logs from
the two timed sessions are in `/data/stat-ecr/scat9264/paper3_timing`. Those sessions ran with
8 cores, on `outlier_p05`, with dnn only.

One row is one measurement, in seconds.

| what | Caco-2 | LogD |
|---|---|---|
| one dnn fit, median | 1.438 | 2.570 |
| `hetero_gp` | 915 | not measured, read from a saved score |
| `residual_low_disagreement` | at least 723 | 3,864 |
| `quantile_regression` | 475 | 4,017 |
| `flag_detectors` | 278 | 2,083 |
| search on one fold, as logged inside that session | 458 | 1,273 over 2 runs |

Claude's estimate, not a measurement: a Caco-2 dnn `base_1` piece takes about 3 hours, and a
LogD one at least 7.5 hours.

Not measured: any svm fit, a whole piece, a hybrid-input piece, the 40 hybrid searches of
TBD2, the cliff masks per noise version, and anything inside the cliff units beyond the sweep
timing in `HANDOFF_AC_STUDY.md` 9b.

---

## N7. Open, for her

One line is one setting. None has a value until she writes it.

**Before the grid review.**

- TBD2: how many of the other seven hybrids. Waits on the eight-hybrid comparison.
- TBD3: the cutoff and its scope. Chosen from the first portion of the noise grid (section
  N1, 2026-11-09). `cliff_sets` and `definition_agreement` alone cannot choose it.
- Which portion of the noise grid runs first, and which cutoffs it sweeps.
- How the cliff runs reach the server. Every cliff unit sits in the ledger's `paper3` step
  with the noise units (`papers/plan.py`, `paper_units`), so opening it for the cliffs also
  starts the noise run. Whether a run by hand does its own per-fold hybrid searches is not
  checked.
- hERG in the eight-hybrid comparison. Five of the eight scorers do not work on a 0/1
  label (Part 1, "The five datasets"), and Prompt 1 runs all eight on it.
- TBD5. No hybrid builds on MoleculeACE's 30 targets, because none has two models recorded.
- TBD7. `train_cliff_model_on_moleculeace` is empty, so `moleculeace_cliff_model` does not run.
- mmpdb. Not installed, so it runs only on the three targets that ship their own pairs. And
  whether a gap of exactly 2.000 counts: 460 cliffs or 456.

Not a decision, work for the next chat: a `cliff_sets` fold is estimated, not measured, at
about 8 hours on hERG and 3.3 on LogD and QM9, scaled from 15.8 minutes at 1,412 molecules by
the square of the molecule count. Sessions last at most 4 hours. Whether a fold can resume
partway is not checked.
- TBD6: the share. Waits on the sweep to 45 per cent.
- Which datasets the TBD3 runs use. This file says six and the TBD2 comparison uses her five.
- Which rows the eight scorers read for a hybrid searched on one fold's training molecules:
  all molecules, or training molecules only. It is not stated. `simeval` scores a whole matrix
  with its own folds, and a hybrid's features are chosen using labels.
- Which noise the two noise scorers inject. `simeval.inject_label_noise` moves 15 per cent of
  labels with sigma 0.6, and that is none of her six kinds.
- Cliff units on antibody datasets: small molecules only, as the outline says, or the 28 the
  ledger plans.

**In the grid review.**

- Everything in `PAPER_3_NOISE_EXPERIMENTS.md` section 5: detectors, cleanups, their settings,
  detector feature spaces, neighbour settings, pairings and `gp_smooth`.
- Whether n43 keeps `hybrid__general2` and `weighted_hybrid`, and n47 keeps `hybrid__general2`.
- Whether the second-base cut also covers neighbour lists.
- Whether the distance and binary hybrids are neighbour spaces, model inputs or both.
- Whether Tanimoto and Dice join the neighbour distances.
- The classification grid: flip kinds, levels, and whether larger datasets run fewer.
- What n15, n20 and n36 run, and on what grid.

**After the first run.**

- Her M and her L.
- Which results get the second run: "First the results, then the checks for what I need".
- Which cliff definitions the second run and `cliff_aware_cleanup` read.
- Which model inputs `cliff_aware_cleanup` fits.
- Whether a definition built from one model's hybrid applies to pieces of the other model.
- Whether Paper 3 splits, from her outline: "If I can prove 1 and 2 well enough, then paper 3
  gets split into two papers".

---

## N8. The order

Claude's proposal, not reviewed by her. An arrow is "needs the result of".

- Eight-hybrid comparison, TBD3 runs, 45 per cent sweep: need nothing from noise.
- Fixing the ledger and connecting the noise code to the definitions: needs nothing from them.
- Grid review → the comparison and TBD3 results, because of "I'm waiting for ACs", and the
  corrected counts.
- First run → the grid review, and `paper3` opened by her.
- M, L and the chosen results → the first run.
- Second run and `cliff_aware_cleanup` → M, L, chosen results and the chosen definitions.

⛔ `paper3` has to stay closed until the first two entries of section N4 are fixed, or the
ledger runs fold 0 at two levels and without `hetero_seen`. Her ask, 2026-11-05: "the paper
2/3 experiment ledger should have paper 3 experiments on hold - confirm this is the case".
The code starts with every step closed. The server's own file has never been read back here.
She can read it with `python3 tests/slurm_scripts/queue_ledgers.py check`, which changes
nothing and lists any Papers steps still closed.

Four prompts follow. Prompts 1 and 2 can run side by side. Prompt 3 waits for both. Prompt 4
waits for the first run. The items under "The two things she assigned as work" and the three
published models above run when she says so and are not in these prompts.

---

## Prompt 1: finish the cliff tests and get them onto the server

Paste everything inside the block into a new Claude Code chat in `~/repos/KIRBy`.

```
Read ~/repos/KIRBy/CLAUDE.md in full and follow it in every reply. Then read MEMORY.md in the
memory folder.

Read AC_GRID_TEST_PLAN.md in full, Part 1 and Part 2. It is the plan. HANDOFF_AC_STUDY.md
section 0 is Adelaide's outline and overrides everything where they differ. Section 9a holds
what she has decided.

Her words: "the AC grid plan is the final touches on implementation", and "AC grid test is an
extension of the existing code". Do not redesign anything. Read the code before any claim.

THE JOB. Get the measurements that TBD2, TBD3 and TBD6 wait on, with her running the server.

BUILD, without asking. Check each against the code first, because the plan's section N4 was
true on 2026-11-07 and code moves.
1. The distance search as she said it: "this should literally just search over knn and
   nothing else".
2. Her two binary searches: "general tier, prediction objective, binary input only" and
   "general tier, similarity objective, binary input only (n_reps=all, not swept over for
   both)".
3. A script that scores all eight hybrids, fast tier, on qm9, herg_ki, caco2, herg and logd,
   with all eight scorers in tests/experiments/similarity_metrics/simeval.py, imported
   read-only. It keeps every number, saves after each hybrid so a 4-hour session loses little,
   and resumes.
4. Time recording in that script, cliff_sets, definition_agreement and
   moleculeace_agreement, through tests/experiments/noise/noise_timing.py. Her words: "keep
   track of timing".
5. The metric-against-metric comparison stays an analysis for after the results.

TWO OPEN SETTINGS STOP THE COMPARISON. Bring them first, one per line, with what changes either
way: which rows the scorers read for a hybrid searched on one fold's training molecules, and
which noise the two noise scorers inject. Enter each as a named input with no default.

ALSO OPEN: which datasets the TBD3 runs use, and whether cliff units stay on antibody datasets.
Her outline has "small molecules only" and the ledger plans 28 datasets.

RUNS. She runs every server command. Before you ask her to run anything, say what it is for,
what decision it feeds, and how long it should take, with the unit, marked estimate or
measured. Her words about the last round: "oh thats what those tests were doing, you should
have said earlier". Three runs:
- The eight-hybrid comparison.
- cliff_sets and definition_agreement.
- moleculeace_agreement with target shares to 45 per cent. It resumes without --force-rerun.

Do not run the noise suite. Do not open paper3 in the ledger. Do not choose a value for her.
A discussion about options is not permission to build. She has had to stop a chat for doing
that.

WHEN A RESULT COMES BACK: say which TBD it feeds, give the raw numbers in the dataset's own
metric, and stop. She decides. Write her decision into AC_GRID_TEST_PLAN.md with her words as
soon as she makes it. Never write a verdict of yours into her documents.

Rules. Never ssh to the server. One `srun ... --pty /bin/bash` line, then plain commands
inside. No sbatch. Sessions last at most 4 hours. No `rm` in anything she pastes. No patient
data anywhere. Do not touch Connectome. Local logs and temp files go on /Volumes/seagate, and
the laptop may be busy with other jobs, so one heavy local job at a time. Commit or push only
when she asks, and run git status before saying what is or is not committed, because other
chats commit from this same folder.

Done means: every TBD2, TBD3 and TBD6 number exists and the plan says where.
```

---

## Prompt 2: make the noise code match her decisions

Paste everything inside the block into a new Claude Code chat in `~/repos/KIRBy`.

```
Read ~/repos/KIRBy/CLAUDE.md in full and follow it in every reply. Then read MEMORY.md in the
memory folder.

Read PAPER_3_NOISE_EXPERIMENTS.md in full, then AC_GRID_TEST_PLAN.md Part 2. Sections N3 and N4
are the work. Where this prompt and PAPER_3_NOISE_EXPERIMENTS.md section 1 differ, section 1 is
right. Re-read the code before any claim; these were true on 2026-11-07.

THE JOB. The noise code does what she decided, and can read the cliff definitions when she has
chosen them. This chat does not cut the grid and does not review detectors or cleanups.

BUILD, without asking.
1. Run the tests first and report each one: _smoke_n46 and _smoke_n47 in tests/experiments/noise/,
   which have not run since the second batch, and tests/test_noise_alignment.py. Say whether test_decided_grid failed before
   this work; git history can show it.
2. The Papers ledger runs her decided grid for n43 to n47: five folds, four levels on datasets
   of 5,000 molecules or fewer plus QM9, two on larger ones, and hetero_seen as a unit. Show
   the unit list and levels for caco2, logd and qm9 before and after.
3. Finish the check for repeated training sets. Compare what each piece asks for by training
   set, not by name, and report what you find. Her words: "the same thing doesn't get run
   multiple times".
4. Done 2026-11-09: the second run reads every cliff definition in
   tests/experiments/cliffs/definitions.py by name, or "all". Check it still holds. No noise
   experiment may fix cliffs to one rule; n47 is the open case, section N7.
5a. Take the cliff parts out of n47: which molecules are cliffs, held-out error on cliff
   against non-cliff molecules, and n14's check of whether cliffs come in groups or as lone
   pairs. The first two are already in cliff_sets and error_by_cliff_status. The grouping check
   becomes part of the cliff experiments, under every definition, reading which molecules are
   paired; cliff_sets does not save the pairs yet, so add that and record what it costs. n47's
   non-cliff parts stay. Her decision is in AC_GRID_TEST_PLAN.md section N1, 2026-11-09.
5. cliff_aware_cleanup accepts every detector and cleanup n44 can run, through the same code n44
   uses. Its printed fit count includes L's own settings. Its model inputs become a named input
   with no default, because the noise suite cut base_2 and she has not said what this unit
   fits. Do the same in error_by_cliff_status.
6. Time recording in n15, n20, n36, cliff_aware_cleanup and error_by_cliff_status. Her words:
   "keep track of timing. They'll be tests in the future".
7. Recount every experiment in the Paper 3 noise list with fitting stubbed out, classification
   included, per dataset: scratchpad/paper3_grid_counts/. Update PAPER_3_NOISE_EXPERIMENTS.md
   sections 4, 5, 8, 11 and 15 and AC_GRID_TEST_PLAN.md section N5. Counts are not times.

DO NOT DECIDE. Leave these for the grid review, and do not change them: n15, n20 and n36's grids,
the classification grid, n47's old cliff numbers, which hybrids join the neighbour settings, and
everything in PAPER_3_NOISE_EXPERIMENTS.md section 5.

DO NOT RUN the noise suite on the server, and do not open paper3. If you need a server check, say
what it is for and how long it takes, marked estimate or measured, before she spends a session.

Another chat may be working in tests/experiments/cliffs/ at the same time, on the eight-hybrid
comparison, the distance search and the binary searches. Do not edit those parts.

Rules. Never ssh to the server. One `srun ... --pty /bin/bash` line, then plain commands inside.
No sbatch. Sessions last at most 4 hours. No `rm` in anything she pastes. No patient data anywhere.
Do not touch Connectome. Local logs and temp files go on /Volumes/seagate, and the laptop may be
busy, so one heavy local job at a time. Commit or push only when she asks, and run git status
before saying what is or is not committed, because other chats commit from this same folder.

Done means: every test result is reported with its output, section N4's noise entries are fixed
or named as waiting on her, and the counts in both documents match the code.
```

---

## Prompt 3: cut the noise grid, with the cliff results in front of her

Paste everything inside the block into a new Claude Code chat in `~/repos/KIRBy`, after Prompts
1 and 2 have finished.

```
Read ~/repos/KIRBy/CLAUDE.md in full and follow it in every reply. Then read MEMORY.md in the
memory folder.

Read PAPER_3_NOISE_EXPERIMENTS.md in full, then AC_GRID_TEST_PLAN.md in full, Part 1 and Part 2.

THE JOB. Adelaide reviews everything inside the noise mitigation grid and decides what goes. Her
words: "Beyond just implementation I do still need to cut down on the noise grid", "those should
be massive cuts. Right now I'm throwing everything at the kitchen sink for noise", and it should
look "at more than just detectors, but everything in that noise mitigation grid". She held this
back until the cliff results existed: "hold off on the grid review, I'm waiting for ACs".

Every cut is hers. Do not cut an axis because it is expensive. Offer options with numbers. This
chat changes documents only until she says to build.

THE AXES. Map all of them against the results before going deep on any one.
- Score detectors, neighbour detectors, flag detectors and oracle flags.
- Cleanups and every cleanup setting.
- Detector feature spaces, and which detectors pair with which cleanups.
- Neighbour settings, including the second-base question and gp_smooth.
- Everything in AC_GRID_TEST_PLAN.md section N7 under "In the grid review". Those include what the
  cliff sweep adds: the distance and binary hybrids as neighbour spaces or model inputs, and
  Tanimoto and Dice as neighbour distances.

WHAT TO READ.
- ~/Documents/ch7_noise_methods_audit.html section #s6, her two lists of methods that raised the
  top score, and HANDOFF_WHAT_WORKED.md for its outcome.
- thesis.tex Chapter 7, "Detecting and handling label noise", Results section.
- /Volumes/seagate/kirby_results_2026-10-10b/noise, the result files behind it.
- thesis_provenance/METHOD_PROVENANCE.md and NOISE_CODE_FIX_DECISIONS_2026-10-26.md, for where
  each method came from and which code changed since.
- tests/experiments/noise/ROBUSTNESS_CHECKS.md.
- The timing in AC_GRID_TEST_PLAN.md section N6, plus any new logs she gives you.
- The cliff results Prompt 1 produced, and what she decided from them.

HOW.
- "Worked" means raised the top score. The random comparison and QM9 support it and do not sort
  methods. Do not change that definition without telling her.
- A gain on measured data with no noise added is a success when QM9 is unchanged or slightly worse.
- Give score pairs in the dataset's own metric, not tallies, and no significance tests.
- Those results come from older code, mostly on fold 0, with settings cut by hand for the thesis.
  Say so where it matters. A method that never ran is not a method that failed.
- Say which entries have no results, and which are the same training set under another name.
- Offer each cut with what it saves in results, from the recounted grid, and what is lost. Give
  time only where it is measured, and say "not measured" otherwise.
- At most three decisions at once, one per line, each saying what it is before asking.
- Write her cuts into PAPER_3_NOISE_EXPERIMENTS.md as soon as she makes them, with her words. Mark
  anything of yours as unreviewed. Never write a verdict as a finding.
- cliff_aware_cleanup's M and L come from these methods, so tell her if a cut removes a candidate.

Rules. Never ssh to the server. One `srun ... --pty /bin/bash` line, then plain commands inside. No
sbatch. Sessions last at most 4 hours. No `rm` in anything she pastes. Before asking for a server
session, say what it is for and how long it takes. No patient data anywhere. Do not touch Connectome.
Commit or push only when she asks.

Done means: every axis has her decision, the code that carries it is built and tested, the grid is
recounted, and she decides whether paper3 opens.
```

---

## Prompt 4: after the first run, the second run and D, D' and D''

Paste everything inside the block into a new Claude Code chat in `~/repos/KIRBy`, only once n44's
first-run scores exist.

```
Read ~/repos/KIRBy/CLAUDE.md in full and follow it in every reply. Then read MEMORY.md in the
memory folder.

Read HANDOFF_AC_STUDY.md section 0, PAPER_3_NOISE_EXPERIMENTS.md sections 1 and 9, and
AC_GRID_TEST_PLAN.md Part 2.

THE JOB. Her two claims, from her outline: "1) prove that I am accurately able to detect ACs" and
"2) that detecting them helps me mitigate noise more effectively".

FIRST, bring her the first-run scores so she can decide, one per line:
- Her M and her L, from her words: "Look at the best outlier detection methods, call one M, and then
  subsequent best label handling methods, call one L".
- Which results get the second run: "First the results, then the checks for what I need".
- Which cliff definitions those two experiments read, from the TBD results.
- What cliff_aware_cleanup fits, and whether a definition from one model's hybrid applies to the other
  model's pieces.

Show each with its fit count, computed from her values, and its time if measured. Do not choose.

THEN, once she has decided, give her the commands: the second run through KIRBY_N44_SECOND_RUN, and
cliff_aware_cleanup through CLIFF_SETTINGS.json. Say what each run is for and how long it should take
before she starts it. Time recording stays on.

READING THE RESULT. D'' against D' and D'' against D, beside random put-back, plus how many flagged
molecules were cliff molecules. Where M flagged no cliff molecule, D'' equals D' by construction, so
say so. Whether Paper 3 splits is hers: "If I can prove 1 and 2 well enough, then paper 3 gets split
into two papers".

Rules. Never ssh to the server. One `srun ... --pty /bin/bash` line, then plain commands inside. No
sbatch. Sessions last at most 4 hours. No `rm` in anything she pastes. No patient data anywhere. Do not
touch Connectome. Commit or push only when she asks. Never write a verdict of yours into her documents.
```

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
