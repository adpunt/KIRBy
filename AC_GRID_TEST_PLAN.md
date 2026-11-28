# AC grid test plan

Started 2026-11-06. One section per TBD. Each says what has to be measured before
Adelaide can decide it, and nothing about what the answer should be.

Why this file exists, her words on 2026-11-06: "days of back and forth server sessions led
me to get the answer to exactly one TBD, which is NOT ENOUGH".

What is already decided is in the section "What she has decided" below, not in the TBD
sections.

This file has two parts. Everything down to "Her M and her L" is Part 1, the cliff tests. Part 2,
added 2026-11-07, is the noise grid and how the cliff sweep changes it, and ends with the prompts
for the chats that finish both. The appendix at the end holds the measurements the TBD sections
were written from.

---

## Her outline, verbatim

Her instruction, 2026-11-04: "my most recent instructions should obviously take
precedent, I've sent you my version of the outline with the TBDs, that's the best source
of truth if you have doubt for AC-specific things". It overrides every other line in this
file. Where anything below disagrees with it, it is right.

> This is how the AC-related experiments should look
> Which pairs count as similar
> 1. Fixed similarity score with cutoff
>     - Similarity coefficient (TBD1) with representation (TBD2) at TBD3 cutoff
>     - MoleculeACE (original with 0.9 cutoff (TBD4) and hybrid with TBD5 cutoff)
>     - mmpdb (one named chemical change)
> 2. Top percent of fixed similarity scores from (1)
>
> How far apart labels have to be
> 1. Twice the median absolute deviation of that dataset's labels.
> 2. Move the gap until some percentage of molecules are cliffs (percent TBD6)
>
> Direct AC prediction
> 1. GraphCliff
> 2. SemiMol
> 3. A 2025 foundation-model paper
> 4. Train my own model on MoleculeACE (TBD7)
>
> How do I confirm if I actually got cliffs? Well, I can't conclusively but I have some
> options:
> I can test my methods against the direct AC prediction models that exist already, that
> will be a useful data point, but otherwise I don't know. Actually two tests I can run
> (and should run):
> - My methods on MoleculeACE with identified ACs - see if it actually identifies them
> - Compare the results of my AC detection methods on my datasets vs the published AC
>   prediction models on those same datasets (so how much do my methods agree with
>   GraphCliff, SemiMol, and the published method) as well as the standard ECFP4 Tanimoto
>   with standard cutoffs (I should already be producing those given my setup, but I need
>   to highlight them)
> - Performance in downstream tasks like AC-aware noise mitigation. Look at the best
>   outlier detection methods, call one M, and then subsequent best label handling methods,
>   call one L. Take dataset D. Run M on D, giving you a list of outliers. Now we have to
>   determine if the outliers are noise or ACs. Take the AC detection method in question
>   and run it, cross-check with the list of outliers. Now first, apply L without
>   AC-awareness to get D'. Then apply L with AC-awareness to get D''. Compare the results
>   of training on D, D', and D''. Assuming M and L work, for a higher-is-better metric, we
>   should see D'' > D' > D. Potentially D > D' if the removal of ACs was actually
>   damaging. But the important thing being D'' should be the best. And then the difference
>   between D'' and D and D'' and D' should indicate the relative performance of different
>   AC detection methods.
> Help me identify published methods that go into determining whether or not a pair is in
> fact an AC
> Help me brainstorm anything else
>
> TBD1 - look at the coefficients used in my similarity research, which show the most
> promise? And how can I justify that in the paper 3 outline? It should be a set.
> TBD2 - which representations work best for similarity? Again I have the results. And for
> the love of god HYBRIDS this was WHY I DESIGNED THE HYBRIDS. It should be a set.
> TBD3/4 - I don't know what cutoff works best, and if that cutoff is per-dataset or global
> (or maybe something like per-domain). See above for how can I confirm it's a cliff, that
> will help me determine cut-offs
> TBD 5 - will definitely need a separate cutoff if I'm only trying to use one instead of 3
> (hybrid overlap instead of a check of 3 overlaps)
> TBD 6 - same with the cutoff question
> TBD 7 - is this a reasonable task? Is this straightforward?
>
> Then how they map into experiments
>
> Use the output ACs from the methods above for the following experiments:
> S04 - do models predict better/worse for ACs compared to non-ACs? ( held-out prediction
> error on cliff molecules against non-cliff molecules)
> S09 (the relevant part) - how do hybrids impact AC detection on the above tests?
> Random stats from experiments:
>     - How many molecules set in at least one cliff pair
>
> I want to get rid of these numbers (see the re-done noise experiments)
> Make this a separate set of AC experiments
> The main goal should be to connect back to AC-noise awareness - it would be AMAZING if I
> could 1) prove that I am accurately able to detect ACs (maybe with a criteria a bit more
> flexible than ECFP4 with a set cutoff which doesn't work for every dataset) and 2) that
> detecting them helps me mitigate noise more effectively
> If I can prove 1 and 2 well enough, then paper 3 gets split into two papers: noise
> methods + hybrid features and cliff detection + AC-aware noise mitigation
> If 1 and 2 are not proven, or lightly proven, then I report a smaller subset of this AC
> study within paper 3 focusing on how it connects to noise (if at all, or I can "identify"
> ACs but not in a way that's useful for me)

The goal was stated back to her on 2026-11-04 and she replied: "yes all of that is right,
and the handoff should reflect that".

> The goal is two claims. That you can detect activity cliffs with something more flexible
> than ECFP4 at a fixed cutoff, because a fixed cutoff doesn't work across your datasets.
> And that detecting them makes your noise mitigation work better. Both proven, paper 3
> splits in two. Not proven, a smaller AC piece inside paper 3.
>
> The TBDs are the settings you have to fix before any of that can run. You want them
> decided one at a time, with your own numbers in front of you, by talking to me. Not by me
> building something and showing you counts.

---

## What she has decided

Her values, written here as she gave them, from the evening of 2026-11-04. Nothing in this
section was proposed by Claude and accepted; each line is what she typed.

### TBD1, the set of coefficients

manhattan, cosine, minkowski p=0.5, tanimoto, dice.

Recorded in `tests/experiments/cliffs/CLIFF_SETTINGS.json` under `coefficients`.

### The hybrid searches, and which experiments read them

Her list, corrected by her from the six the code produces today.

Main set, every experiment.

- general search, predictive objective, model 1, first hybrid
- general search, predictive objective, model 1, second hybrid
- general search, predictive objective, model 2, first hybrid
- general search, predictive objective, model 2, second hybrid
- similarity search

Noise and cliff experiments only.

- kNN search (`general_distance` in the code)
- general search, predictive objective, binary input only
- similarity search, binary input only, n_reps=all

Her words on the split: "the main set of hybrids don't need to run over those 6, the
distance and two binary options can be restricted to noise/cliff experiments".

### TBD2, the set of representations: partly decided

In the set, her words.

- the two base representations ("I have the two bases")
- the first hybrid of the general search ("Obviously I'm using this: 1.
  predictive objective, model 1, first hybrid")

### The kNN search searches kNN and nothing else

Her correction: "this should literally just search over knn and nothing else".

Asked whether it was a problem that the `general_distance` searches already saved could no longer be read, she said:
"everything has to be rerun there are no stored distance searches that i care about".

### n_reps is fixed to all, on the binary searches only

Her words: "n_reps=all - for binary only. Because not many reps are binary only."

---

## TBD2, which representations

### Her decision so far

Single representations: the top base plus ecfp4, or the next best if the top is ecfp4.

Hybrids: the first hybrid of the general search. Seven candidates left:

1. general search, model 1, first hybrid — **taken**
2. general search, model 1, second hybrid
3. general search, model 2, first hybrid
4. general search, model 2, second hybrid
5. similarity search — one per dataset, fits no model
6. kNN search
7. general search, binary input only
8. similarity search, binary input only, all representations

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
- an injected noise mask — which labels are damaged is known, so finding them is a real score

**Not usable for this choice: scored inside the representation.** SALI, SARI, iCliff,
ROGI, local roughness, the rivality index. They describe a landscape; they cannot rank the
spaces the landscape was drawn in. They stay useful for describing one dataset under one
fixed representation.

### The metrics, and whether the code already has them

Her Metric 1, prediction from nearest neighbors, is `eval_knn_readacross` in
`tests/experiments/similarity_metrics/simeval.py`: scaffold-grouped folds,
distance-weighted kNN on the precomputed distance matrix, reported as Spearman and
negative RMSE. Cited to Golbraikh, Muratov, Fourches & Tropsha, JCIM 2014.

Her Metric 2, whether it helps remove noise, is already written twice in that same file
and is the strongest answer to her question:

- `eval_noise_detect` — inject noise, flag each molecule by how far its label sits from
  the median of its k nearest neighbors in this representation, and score the AUROC
  against the known injected mask. Named there as Wilson / edited-nearest-neighbor
  editing quality.
- `eval_noise_recovery` — estimate each label from its neighbors, and correlate the
  correction it implies against the injected error on the damaged molecules. A good
  space gives a correction equal and opposite to the error.

Both are non-circular: the injected mask is outside every representation.

Five more in the same file, all with the label or an external flag as the answer key:

- `eval_modi_binary` — the class-balanced rate at which a molecule's nearest neighbor
  shares its activity class, median split. Golbraikh et al. JCIM 2014.
- `eval_patterson` — among the closest few per cent of pairs, the 95th percentile of the
  label difference. A valid space bounds the label difference from above at small
  distances. Patterson, Cramer, Ferguson, Clark & Weinberger, J Med Chem 1996, 39:3049.
- `eval_vs_enrichment` — take the most active quarter as actives, rank the rest from each
  active by distance, and report BEDROC. Truchon & Bayly, JCIM 2007, 47:488.
- `eval_cliff_auroc` — does local label disagreement among the k nearest neighbors in
  this space flag an externally defined cliff set. Guha & Van Drie 2008, JCIM 48:646.
- `eval_violation_rate` — the share of nearest-neighbor pairs whose label difference
  exceeds the global median.

⛔ NONE OF THESE IS CALLED BY THE CLIFF EXPERIMENTS. `simeval.py` is imported by 18 scripts in `tests/experiments/similarity_metrics/`
and by `f18_distance_metric_sweep.py` and `f23_noise_cleaning_metric.py` in
`tests/experiments/framework/`, checked 2026-11-18. `tests/experiments/cliffs/binary_hybrid.py`
imports f18, which imports simeval for its coefficient lists. Nothing in
`tests/experiments/cliffs/` or the noise suite calls one of these metrics.

### Not in `simeval.py`

Maximum common substructure overlap as an independent structural answer key. It exists
elsewhere: `tests/experiments/similarity_metrics/mcs_cache.py` computes MCS distance on
sampled molecule pairs, `structural_yardstick.py` scores a representation's distances
against it by Spearman, and `_structural_fidelity_score` in `src/kirby/find_best_hybrid.py`
does the same for a hybrid. Nothing in the cliff code calls them. Huber &
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
`tests/experiments/dataset_policy.py:249`. Three of the eight metrics are
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

### Measured cost of the eight metrics, 2026-11-06

Timed on random distances and random labels, because the cost of every one of
these is set by the molecule count and by k, not by the values. Script at
`/Volumes/seagate/kirby_scratch/simeval_timing/time_scorers.py`, local machine.

one row is one dataset, the number is seconds to run all eight metrics once on
one distance matrix, plus the cost of building that matrix

| dataset | all eight metrics | one Tanimoto distance matrix | total, one hybrid |
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

**All five datasets, one hybrid, all eight metrics: about 155 seconds. Eight
hybrids: about 21 minutes.** The metrics are not the cost of this comparison.
The cost is the eight hybrid searches, which this timing says nothing about.

### The metrics, decided 2026-11-06: ALL EIGHT

Her words: "great yes please add all 8". Every hybrid is scored on all eight
metrics in `tests/experiments/similarity_metrics/simeval.py`:
`eval_knn_readacross`, `eval_modi_binary`, `eval_patterson`,
`eval_cliff_auroc`, `eval_violation_rate`, `eval_vs_enrichment`,
`eval_noise_detect`, `eval_noise_recovery`.

The grid is 8 hybrids times 5 datasets times 8 metrics = 320 numbers.
(Corrected 2026-11-07: `eval_knn_readacross` gives two numbers, so it is 360. Part 2, section N4.)

### Her settings for the eight-hybrid comparison, decided 2026-11-18

One line is one setting, in her words.

- Which molecules the eight metrics read: that fold's training molecules only. "Over that 80".
- Budgets: the 2000-feature budget is cut for this comparison. "Let's cut the 2K runs for this
  set of experiments". So 10, 25, 50, 100, 250, 500 and 1000 features.
- `herg` is out. "we're obviously not running herg or herg_blocking". Four datasets: herg_ki,
  caco2, qm9, logd.
- Noise conditions, and the level the two noise metrics run at: outliers and gaussian, at level
  1, which is a fraction of the label SD. "The noise conditions and levels you laid out are fine
  with me". The evidence put to her: `~/Documents/ch7_noise_methods_audit.html`, where outliers
  at level 1 reach average precision 0.31 to 0.42 against chance 0.05 to 0.06 on every
  regression dataset, and cleaning raised the top score on herg_ki under gaussian, grouped and
  outlier at level 1 and above. Shifts, censoring and sparse shifts are at or near chance at the
  smaller levels.
- It runs in interactive server sessions and must not depend on the Seagate drive. "This will be
  done in interactive sessions on the server - I'll need a complete set of commands to run it.
  No dependencies on seagate".

Open: how many repetitions, and how many folds.

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
nearest-neighbor label agreement, at one neighbor and at five.

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
nearest neighbors flags an externally defined cliff set. The other seven score
whether close molecules have close labels, and say nothing about cliffs.

### ⛔ TWO THINGS CLAUDE INVENTED AND PUT IN THIS FILE AS HER DECISIONS. BOTH CUT 2026-11-07.

**The downstream-score comparison.** Correlating each metric's ranking of the eight
hybrids against a model's cross-validated score on that same hybrid, 40 fits.
No sentence in her outline asks for this. Her own downstream test for a cliff
detection method is her outline's D, D' and D'': "the difference between D'' and D
and D'' and D' should indicate the relative performance of different AC
detection methods." That is `cliff_aware_cleanup.py`, already written. A model's
RMSE on a hybrid's features is a different question and was not asked.

**"Which similarity-to-distance conversion".** The eight metrics want a distance
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
comparison; the eight metrics are about 21 minutes for the full grid and the
metric-against-metric analysis is seconds on top.

---

## TBD3, the similarity cutoff and whether it is global, per domain or per dataset

### What has run, and why it did not answer it

`moleculeace_agreement` over MoleculeACE's 30 targets, 2026-11-06, 3.54 hours. Every
cutoff of her five coefficients scored against their `cliff_mol` flag.

It cannot settle TBD3. Their flag is itself a fixed 0.90 cutoff on any of ECFP Tanimoto, scaffold-ECFP Tanimoto or scaled Levenshtein on the SMILES, so
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

Decided 2026-11-06. Their published 0.90, which beat every other cutoff of their definition of an AC at
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

DECIDED by her on 2026-11-27, her words: "I said yes to training a model on moleculeace".
`train_cliff_model_on_moleculeace` is now `true` in `tests/experiments/cliffs/CLIFF_SETTINGS.json`,
so the unit runs without `--run-anyway` and its output is not stamped a crash check.

### What has run

One target as a crash check, 2026-11-04, local, so debug. CHEMBL4203_Ki, 731 molecules,
five scaffold-grouped folds, 151 seconds. Her re-implementation of their definition of an AC reproduces
their flag exactly. A trained classifier reached a median Matthews correlation of 0.2902
across folds against 0.1663 for ECFP4 Tanimoto at 0.55.

### What has run, all 30 targets

Server, 2026-11-27, 583.4 seconds, output
`/data/stat-ecr/scat9264/cliff_tbd7/moleculeace_cliff_model.json`. One row is one arm. The
number is the Matthews correlation against MoleculeACE's own per-molecule flag. Every target
and every fold goes into one list per arm, 30 targets times 5 folds, and the three numbers are
that list's median, its lowest single fold and its highest single fold. No per-target best is
reported, and her instruction of 2026-11-28 is that the per-target best is what she would use:
"I've never cared about averages. I would only in practice use the BEST one ... This is a TOOL."
The literature for this task is in `thesis_provenance/LITERATURE_SEARCH_TBD7_2026-11-28.md`:
no published work trains a classifier on their per-molecule flag, so there is no published
number to compare these to.

| arm | model | representation | median | lowest fold | highest fold |
|---|---|---|---|---|---|
| their rule re-implemented | - | ecfp4 at 2048 bits | 1.0000 | 0.9501 | 1.0000 |
| trained model | rf | ecfp4_count | 0.4464 | 0.0000 | 0.7511 |
| trained model | rf | atompair_count | 0.4451 | -0.0311 | 0.6840 |
| trained model | lightgbm | ecfp4_count | 0.4436 | -0.0311 | 0.7146 |
| trained model | rf | rdkit | 0.3716 | -0.0572 | 0.6729 |
| ECFP4 Tanimoto 0.55, gap twice the median absolute deviation | - | ecfp4 at 2048 bits | 0.3107 | -0.1217 | 0.6560 |
| every molecule called not-a-cliff | - | - | 0.0000 | 0.0000 | 0.0000 |

The 20 trained arms run from 0.3716 to 0.4464; the two rows between `lightgbm` on
`ecfp4_count` and `rf` on `rdkit` are the rest of that range and are in the output file. The
first row is their definition computed twice, not a baseline.

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
`cliff_rules.SIMILARITY_CRITERIA`. The named definition of an AC pairing it with a gap is
`dablander`: matched pairs on the similarity side,
`absolute_2_half_cliffs_dropped` on the gap side.

It reads a pair list the dataset's loader ships. The three ChEMBL targets of
Dablander et al. 2023 ship one; no other dataset does, so on every other dataset
this criterion records that the loader shipped no matched pairs rather than
running.

### Open, and not decided

On `chembl_qsar_d2`, 6,333 molecules, all 35,484 pairs arrive. The loader prints
460 cliffs and the definition of an AC here counts 456. Four pairs have a potency difference of
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
flags are matched BY SMILES, never by molecule order, because an outside model's
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
cutoffs, MoleculeACE's published definition of an AC, and a published model's flag where one is
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

### S09's question is answered by the hybrids in the representation list, not by a separate unit

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

Neither has been done. Both are searches and write-ups, not runs, and neither
blocks anything above.

The first one's scope: what the published literature uses to decide that a pair
is a real activity cliff rather than a measurement error. That covers the
similarity measures and cutoffs, the potency gaps, the repeat-measurement and
curation rules, and any prospective confirmation. On 2026-11-04 a Claude session
reduced this to a two-option question in a chat reply, and then, after she
objected, started a literature search without being told to. Neither was right.
It runs when she says so.

What already sits in the code as a published source, from the module records in
`cliff_rules.py`. One line is one source.

- Hu, Furtmann, Stumpfe and Bajorath 2015, for ECFP4 Tanimoto at 0.55.
- van Tilborg, Alenicheva and Grisoni 2022, for the any-of-three definition of an AC at 0.90 with a
  1-log-unit gap.
- Dablander et al. 2023, for a matched pair with a 2.0-unit gap.
- Kramer, Kalliokoski, Gedeck and Vulpetti 2012, for a 0.54 pKi repeat spread.
- A 2605 arXiv preprint, recorded second-hand through a research session with no paper read.
- One gap rule attributed to the Bajorath group by a research session, marked unverified in
  the code.

The second one is a brainstorm of further ways to decide that a pair is a cliff.
Material has accumulated, none of it hers and none of it decided. One line is one
candidate.

- Assay type as a filter before detection.
- Whether any one dataset mixes IC50, Ki and Kd internally. Unchecked.
- Censored values, and whether any dataset ships them into the cliff calculation. Unchecked.
- Restricting cliff pairs to molecules measured in the same document, which hERG IC50
  records.
- Stereoisomer pairs, a criterion in the code that finds 0 pairs on all 30 MoleculeACE
  targets.
- Crossing a cliff mask with the molecules whose labels the noise experiments damaged on purpose.
- Dablander's band of 1 to 2 log units dropped from both sides, a third gap option not in
  her list of two.
- The ChEMBL curator flag, already run, which covers 215 of the 9,802 molecules in hERG IC50 (6 of them under "Potential transcription error").

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

Words used here. A configuration, in the noise chapter's sense, is one model on one input. A result is one cleaned training set scored for one configuration, with its fold, noise condition, noise level and repetition. A repetition is one repeat of the noise injection, with new random noise. The standard cleanup run has
no put-back, no shuffled detector scores (`shuffled_scores_twin`) and no shuffled cleanup results (`shuffled_twin`). The variant cleanups run adds them for results she names.

Everything marked "Checked 2026-11-07" was read or run in the code that day. Nothing in
this part is a scientific result, and nothing here recommends a value.

---

## N1. What she has decided about noise

One line is one decision, with her words.

- Results first, checks after: "First the results, then the checks for what I need".
- Inputs of the configurations in n44: "Lets do this: base_1 and hybrid only".
- Second base representation, all five experiments: "okay then the cut to all".
- Three noise conditions cut, all five: "Yes cut the noise from all". Those are `dense_oofmse`,
  `hetero_hidden` and `sparse_dprime_up_p10`.
- `hetero_seen` stays: "Keep hetero_seen". It keeps levels 0, 0.1, 0.3 and 1.0.
- Repetitions stay at 3, and 1 under censoring: "Don't modify the draws thats legit".
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
  just do 1 fold" (which fold she has not said). Input of the configurations: "Let's just look at hybrid".
  Noise conditions, levels and which cleaning methods get the cutoff test are still open.
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
commits (031f4535, b497ba42).

- The noise condition cuts and the second-base cut in n43 to n47.
- Time recording: `tests/experiments/noise/noise_timing.py`.
- Standard cleanup run in n44: no put-back, no shuffled detector scores and no shuffled cleanup results.
- Level 0 runs once, under `gaussian` (under `uniform` on a classification dataset), in n43, n44 and n45. n46 keeps its own because its
  censoring family differs by noise condition. n47 reads level 0 inside itself.
- `reliability_weight` runs once per neighbor setting instead of once per detector.
- Variant cleanups run in n44: `KIRBY_N44_SECOND_RUN` names a file listing the results. It refuses to
  start with no results or no cliff definition. Its rows go in their own files.

**Tests.**

- n44 passed on the code of 2026-11-07, on made-up data. That includes the variant cleanups run:
  it wrote only the named result, its `random_twin` and its `shuffled_twin`. Across all three, the
  versions written were plain, Cliff put-back and Random put-back, and no others. Commit 7d07b139
  of 2026-11-18 removed `random_twin` from n44.

- n43 and n45 passed on the current code.
- The variant cleanups run's many cliff definitions, 2026-11-09: `_smoke_n44.py` passed, exit 0, on
  made-up data. `"all"` built 1,184 definitions, the reference definitions among them, each
  put-back row names its definition, and a name that is not a definition is reported.
- n46 and n47 passed on the first batch and were not re-run after the second. Run 2026-11-16 on
  the code before that day's changes: `_smoke_n46.py` exit 0, 57 checks ok, 57 s; `_smoke_n47.py`
  exit 0, all passed, 28 s. Logs in `/Volumes/seagate/kirby_smoke_logs/2026-11-16/`.
- `tests/test_noise_alignment.py::test_decided_grid` failed before this work, since commit
  58794630 of 2026-10-07. That commit made the screen (two levels, no `sparse_dprime_up_p10`)
  the default without the promote flag, and the test was not changed. Checked 2026-11-16 by
  running the test at that commit's parent (passed) and at the commit (failed). The test now
  checks the screen without the flag and, with it, the noise conditions and levels the flag sets: 6 passed.
- The repeated training sets check ran 2026-11-16 without neighbor settings and 2026-11-18
  with them, 2,225 s on the laptop (section N4, "The same training set is scored under several
  names").
- `tests/experiments/cliffs/_smoke_cleanup_and_error_split.py`, 2026-11-18, after the n44
  commit 7d07b139 of that day: exit 0, all passed, 2,140 rows, 610 s.
- `_smoke_n36.py`, 2026-11-18: exit 0, 43 passed. The test had not followed n36: it replaced
  `get_best_search_for_downstream`, which n36 no longer has. It now stands in for
  `fold_configurations` and `decided_models` on its made-up data, and reads the neighbor
  rules under their setting names. The timing log has search, building inputs, pass and run lines.
- `_smoke_n20.py`, 2026-11-18: exit 0, 38 passed. Its made-up datasets had no row in
  DECISIONS.md, so `decided_grid` refused them. The test now adds a row for them in its own process
  (rf, budget 10, rf as internal evaluator); DECISIONS.md is unchanged. The timing log has inject,
  search, building inputs, fold and run lines. n15's per-step timing is not checked by a test.
- `tests/test_repeated_training_set_fitted_once.py`, 2026-11-18: 3 passed. A training set two
  methods both produce is fitted once.

**Time recording.** n43 to n47 recorded already. Since 2026-11-16 n15, n20, n36,
`cliff_aware_cleanup` and `error_by_cliff_status` record too (`PAPER_3_NOISE_EXPERIMENTS.md`
section 8). The other files in `tests/experiments/cliffs/` were not changed here.

---

## N3. Where the cliff sweep reaches the noise grid

Seven places. Each says what the noise code does today and what the sweep asks of it.

**1. Which molecules the variant cleanups run puts back.** Fixed 2026-11-09. Until then the variant
cleanups run reached one definition only, ECFP4 Tanimoto at 0.55 with twice the median absolute
deviation. Her words: "This should NEVER be the only cliff defnition". Now its JSON file
names `cliff_definitions`: a list of definition keys from `cliffs/definitions.py`, or
`"all"`. There is no default. The masks are built per fold on the training molecules and
per noise condition, level and repetition from the labels as they look. Every put-back row carries
`|cliff=<definition key>|`. The old definition of an AC is still reachable as `n25_existing`, but only
when it is named. On 48 made-up molecules with two representations, `"all"` built 1,184
definitions in 1.1 seconds.

**2. Definitions multiply the variant cleanups run, and also its masks.** Every chosen result gains
2 fits per definition (Cliff put-back and Random put-back), or up to 6 counting the shuffled detector scores and shuffled cleanup results of each. Each chosen result itself also gains its own shuffled detector scores and shuffled cleanup results, once, not once per definition. Results with detector `none` get no shuffled detector scores. A cleanup that changes labels on a classification dataset gets no shuffled cleanup results. Shuffled cleanup results are also skipped when the cleanup's result holds no list of kept molecules, no weights and no new labels (`controls.py` `shuffled_twin`, `_shape`). Shuffled detector scores are also skipped when the cleanup, run again on the shuffled scores, gives no result. Each
noise condition, level and repetition also changes the labels, and the cliff mask is computed from "the labels as
they look" (`n42._cliff_masks`). So each definition needs a new mask per noise condition, level and repetition, per
fold. The cost of those masks is not measured.

**3. Whether Random put-back can run at all depends on the definition.** Random put-back
needs as many non-cliff molecules as cliff molecules. `thesis.tex:4148` records it as
impossible on Caco-2, hERG blocking and LogD under twice the median absolute deviation.
Under a moved gap, her share sets the cliff count. So TBD6 affects which datasets get the
comparison. n44 now records an error, `random_put_back_short`, when Random put-back finds fewer non-cliff molecules than cliff molecules.

**4. The test of D, D' and D'' is a second route to the same claim.** `cliff_aware_cleanup`
injects no noise, because her sentence is "Take dataset D". The variant cleanups run in n44 works
under injected noise, where the molecules with damaged labels are known. They differ in two ways at
once: which labels they start from, and which detectors and cleanups they can run.

**5. Which hybrids the neighbors are found in.** Today a noise experiment searches 20
neighbor settings on the larger noise grid and 11 on the smaller one (26 and 13 only where the augmented spaces are kept)
(`similarity_space.GROUP_SPACES`). The top hybrid of the `general_distance` tier, `hybrid__distance1`, is already among them. Neither binary
hybrid is. Tanimoto runs only on ECFP4 and only on the larger grid, and Dice runs nowhere.
TBD1 has five coefficients and TBD2 has eight hybrids. So her "restricted to noise/cliff
experiments" adds settings the code does not hold yet.

**6. Which hybrids a model is fitted on.** n44 fits `base_1` and `hybrid`. Her
"restricted to noise/cliff experiments" does not say whether a binary hybrid is the input of a configuration,
a neighbor space or both. As a neighbor space it adds settings. As the input of a configuration it
adds whole configurations, each at every fold, noise condition, noise level and repetition. Section N5 has the numbers for each.

Checked again 2026-11-16: item 1 still holds in `n44_detect_and_clean.py`
(`second_run_spec`, `pass_cliff_sets`). No other noise experiment in n43 to n47 reads a definition of an AC.

**7. Fixed 2026-11-16.** n47 no longer computes any cliff number. `n01_part`, `cliff_reference`
and the cliff parts of n14's record are gone from n47; what n14 recorded without cliffs stays
(label spread, repeat measurements, the noise diagnostics from `stage0_diagnostics`). The check of whether cliffs
come in groups or as lone pairs is now on every `cliff_sets` row, as `grouping`, under every
definition. It reads the pairs `cliff_sets` now saves in `<dataset>__cliff_pairs.npz`. On the
smoke's made-up data (60 training molecules, 20 definitions) the grouping took 0.01 s of 1.7 s
and the pair file was 5,007 bytes. The cost on a real dataset is not measured; each run records
`seconds_grouping_check` per fold and `cliff_pair_file_bytes`.

What it said before: `n47.n01_part` and n47's n14
part compute cliff counts and held-out error on cliff molecules under the old fixed definition of an AC.
Her outline says "I want to get rid of these numbers" and "Make this a separate set of AC
experiments". On 2026-11-07 she said: "I thought I was replacing all the AC experiments with
the new setup." So two experiments will report held-out error on cliff molecules, n47 under
one old definition of an AC and `error_by_cliff_status` under the new definitions. The line in this file
saying the new category "is already the only thing that does" holds for s04 and s12 and not
for n47.

---

## N4. Where the code and her decisions disagree

Checked 2026-11-07, by running the ledger's own plan functions and reading the settings.
One entry is one disagreement. No fix is written here. Updated 2026-11-16: each noise entry
now says whether it was fixed that day or waits on her.

**The ledger would not run her noise grid.** She decided five folds, and four levels on
datasets of 5,000 molecules or fewer. `runner.common_env` removes `KIRBY_NOISE_PROMOTE`, and
`papers/worker.py` starts from it. Run that way, n44 on Caco-2 gets fold 0 only and levels 0
and 1.0, and n47 gets fold 0 only. So that is two ways it differs: one fold rather than five, and two levels rather
than four. The timing sessions avoided this by setting the flag by hand.
FIXED 2026-11-16. The Papers worker sets `KIRBY_NOISE_PROMOTE` to the unit's own experiment for
n43 to n47 in the `paper3` step. Caco-2 and QM9 units now get folds 0 to 4 and levels 0, 0.5, 1.0
and 1.5 (censoring 0, 0.1, 0.25, 0.4); LogD (5,028 molecules) gets folds 0 to 4 and levels 0
and 1.0 (censoring 0 and 0.25). QM9 loads 4,995 molecules, so the 5,000 rule gives it four
levels; there is no separate QM9 exception in the code. A run made before `paper3` opens needs
`python tests/experiments/papers/ledger.py extend --datasets <datasets>` (or `--all`) to add the new units; `replan`
keeps old steps as they were.

**`hetero_seen` has no ledger unit.** She wrote "Keep hetero_seen". `plan.noise_units` reads
`conditions_for_task`, whose list does not hold it. On Caco-2 the n44 units are `gaussian`,
`outlier_p05`, `grouped_shifted`, `censoring` and `shift_only_p10`, which is five noise conditions and
not her six. Run by hand, n44 does include it.
FIXED 2026-11-16. `n43__hetero_seen` and `n44__hetero_seen` are ledger units, at 0, 0.1, 0.3 and
1.0 on every dataset size. n45, n46 and n47 do not run `hetero_seen` by hand either; you have not
said whether they should. n45 by hand also runs `outlier_only_p05`, which is in no Paper 3 noise
decision; the ledger gives it no unit.

**Three more noise experiments run in the ledger and are in no noise decision.** n15, n20
and n36 are in `SINGLE_DATASET_EXPERIMENTS`. None of the three reads the noise-condition cuts, records
time or appears in `PAPER_3_NOISE_EXPERIMENTS.md`'s counts. Through the ledger n15 runs folds
0 to 2, not five. n36 runs only where a loader keeps repeat measurements.
Time recording added 2026-11-16. Their grids are unchanged and wait for the grid review.

**Classification was never part of the noise decisions.** Five Paper 3 datasets are
classification: `herg`, `cdr_flexibility_h3`, `cdr_flexibility_l3`, `tz_her2_binary` and
`tz_her2_multiclass`. Each runs six label-flip noise conditions at six levels with 3 repetitions. Her decisions
name regression noise conditions. "LARGE DATASETS EXCLUDING QM9 RUN LESS" is not applied to `herg`,
which has 7,772 molecules and keeps all six levels.
Waits on her, in the grid review. Unchanged.

**Cliff units are planned on 28 datasets, and her scope is small molecules.**
`PAPER_3_OUTLINE.md` records "Scope of the activity cliff work (Adelaide, 2026-11-04): small
molecules only". `papers/plan.py` gives cliff units to molecular and antibody datasets, so 14
antibody datasets carry them.
FIXED 2026-11-18, from her outline sentence. `papers/plan.py` gives cliff units to
small-molecule datasets only: 14 datasets, none antibody. The ledger tests pass, 31 of 31.

**The inputs of the configurations differ between the two suites.** `cliff_aware_cleanup` and
`error_by_cliff_status` fit `base_1`, `base_2` and `hybrid`, and one comment there says these
are "the same three the noise suite fits". The noise suite has cut `base_2`. Her TBD2 line,
"top base + ecfp4", is about which representations define a cliff, and neither file
fits ecfp4 by name.
FIXED 2026-11-16. Which inputs each unit's configurations use is a setting with no value,
`cliff_aware_cleanup_model_inputs` and `error_by_cliff_status_model_inputs`, from the first and second
base representations and the first and second hybrids. Until she writes them, the ledger holds both units
(`cliffs/required.py`) and a run by hand stops and names the setting.

**Her M and L can name something the unit refuses.** `cliff_aware_cleanup` accepts detectors
from `ESTIMATOR_REGISTRY` and cleanups from `n25.cleanups_for`. n44 runs 26 cleanups, and 16
of them are not in that list, such as `rewrite` and `exp_downweight`. 10 of n44's detector
names are outside the registry, such as `oof_residual` and `knn_shapley`. Her M and L are
"the best" of n44's results, so a winner could be refused.
FIXED 2026-11-16. `cliff_aware_cleanup` accepts every detector and cleanup n44 can run, by the
name n44 writes before '@', and builds them with n44's own functions and settings: score and
neighbor detectors, flagging methods, every cleanup at n44's settings, paired as n44 pairs them. n44's
flags of known content are not accepted, because this experiment injects no noise. D' reads no
definition, so it is built once and every definition puts its cliff molecules back into it.

**The fit count printed by `cliff_aware_cleanup` is low.** The printed product has no term
for L's own settings, and the row loop runs one per setting.
FIXED 2026-11-16. It now prints, per fold, model and input, one fit on D, the number of cleaned
training sets including L's settings and up to 3 kept neighbor settings, and two fits per
definition per cleaned set. On the smoke (4 detectors and 5 cleanups named) it allowed 125
cleaned sets per unit and built 107.

**The same training set is scored under several names.** Her words, 2026-11-04: "just make
sure the same thing doesn't get run multiple times". Checked 2026-11-18 on two n44 configurations, each at one fold, noise condition, level and repetition, on
made-up data (128 training molecules, rf, `outlier_p05` level 1), neighbor settings included.
Every result's training set was hashed and compared within its configuration. In the base-input configuration,
1,228 of 5,188 results repeat a training set already scored, leaving 3,960 different training
sets. In the hybrid-input configuration, 746 of 3,461 repeat one, leaving 2,715.
Causes, read in the code:
- `shrink_flagged_one_weight` and `shrink_flagged_per_row` re-flag by out-of-fold error and read
  only how many the detector flagged, so every detector at one share of training molecules
  flagged gives one set.
- `huber_weight` gives every molecule weight 1 when the detector's median score is 0.
- `bayes_shrink` leaves the labels unchanged when the flagged molecules fit no worse; 162
  results in the base-input configuration were D itself.
Seen on this data, cause not read:
- `js_shrink` and `move_confidence` gave the same set on flags of known content at the same strength.
- `neighbor_residual` and `distance_weighted` flagged the same molecules at k = 10.
FIXED, no code change needed, checked 2026-11-18. The model fit is keyed on the content of the
training set, the held-out molecules and the model, never on the method's name
(`noise_cache.cached_fit_predict`). So a repeated training set is fitted once and read from the
fit cache under every other name. `tests/test_repeated_training_set_fitted_once.py` checks it: 3
passed. Each name keeps its own result row. The fit cache is on unless `KIRBY_NOISE_CACHE=0`,
which only the test scripts set.
CHANGED 2026-11-20, her words: "Shouldn't I just used the fix number and not run the detector?"
and "okay continue doing that". `shrink_flagged_one_weight` and `shrink_flagged_per_row` pick
their own molecules and read from a detector only how many it flagged. They now read no detector.
Each runs once per share of training molecules flagged (5%, 10% and 20%), in the model's own
space, under detector `none`. Every group is in
`/Volumes/seagate/kirby_smoke_logs/2026-11-16/n44_groups.json`; details in
`PAPER_3_NOISE_EXPERIMENTS.md` section 6.

**As committed, the search in the `general_distance` tier is not what she said.** Her words: "this should literally just search
over knn and nothing else". On hERG Ki `tests/experiments/decided_grid.json` still gives
that tier the models svm, dnn and knn, and the evaluators knn, rf, dnn, lightgbm and svm. In the working tree (uncommitted, 2026-11-18) `decided_grid.decided_search` narrows that tier to models knn and evaluators knn (`_knn_only_distance`); the JSON file is unchanged.

**The two binary searches are not what she said.** Her list: "general tier, prediction
objective, binary input only" and "general tier, similarity objective, binary input only
(n_reps=all, not swept over for both)". `definitions.restricted_hybrids` builds `bits` and
`bits+counts`, both by the general search. Nothing in `tests/experiments/cliffs/` runs the
similarity objective on binary input or fixes `n_reps`. Neither search is a ledger search step.

**The settings file does not hold her TBD2 line.** `CLIFF_SETTINGS.json` holds the TBD values
the cliff units read, and was last changed in git on 2026-11-05. Its `representations` is
`"all"`, and she said "top base + ecfp4". Its `_candidates`
block still lists cliff shares to 30. The code reads shares from `settings.py`, so the 45
per cent runs regardless.

**The eight-hybrid comparison has no script.** Nothing joins the hybrids to `simeval.py`. Also
`simeval.run_battery` returns nine numbers from eight metrics, because `eval_knn_readacross`
gives two. So the grid is 360 numbers rather than the 320 written above.

---

## N5. How big the noise grid is now

Recounted 2026-11-18 from the code, with fitting stubbed out
(`scratchpad/paper3_grid_counts/count_paper3.py`, output `paper3_counts.csv`). Every count is
as the Papers ledger now runs it. n43 to n47 run five folds with the noise conditions and levels
that `KIRBY_NOISE_PROMOTE` sets; n15, n20 and n36 run the
screen. These are counts, not times. Molecule counts come from loading each of the 28
datasets (`dataset_sizes.py`); `qm9`, `qm9_mu` and `qm9_alpha` each load 4,995 molecules.

**n44's standard cleanup run.** One row is one group of datasets.

| group | datasets | noise condition, level and repetition combinations, level 0 counted once | configurations, each at one fold, noise condition, noise level and repetition, per dataset | results per dataset | results, group total |
|---|---|---|---|---|---|
| Regression, four levels | 17 | 49 | 980 | 3,789,030 | 64,413,510 |
| Regression, four levels, one model takes no weight | 3 | 49 | 980 | 3,275,195 | 9,825,585 |
| Regression, two levels | 2 | 23 | 460 | 1,752,810 | 3,505,620 |
| `herg_ic50`, two levels, dnn only (svm skipped above 10,000 molecules) | 1 | 23 | 230 | 845,355 | 845,355 |
| Classification, as the code stands | 5 | 91 | 1,820 | 2,199,470 | 10,997,350 |

Regression comes to 78,590,070 results. With classification it is 89,587,420. A base-input
configuration at one fold, noise condition, level and repetition has 4,952 results and a
hybrid-input one 3,295, or 4,502 and 2,845 where no flags of known content exist. Recounted
2026-11-20, after `shrink_flagged_one_weight` and `shrink_flagged_per_row` stopped reading a
detector; on 2026-11-18 regression was 86,756,250 and the total 97,753,600. Classification is
unchanged, because those two cleanups are regression only. The shuffled detector scores rerun added on 2026-11-18 is not in these numbers.

Two numbers changed from the 2026-11-07 count, and neither is a change in the grid:
- `herg_ic50`: 966,135 before, 933,015 on 2026-11-18 (845,355 after 2026-11-20). The earlier script counted the two-learner
  detector, but the code runs it only when a dataset has a second model, and `herg_ic50` has dnn alone.
- Classification: 2,568,020 per dataset before, 2,199,470 now. No script in
  `scratchpad/paper3_grid_counts/` gives the earlier number. The code before and after the
  2026-11-18 n44 commit gives 1,444 results per base-input configuration and 973 per hybrid-input configuration.
  The classification grid is waiting on you.

`chembl_qsar_d2` has 6,333 molecules, so the code gives it two levels. The older list of eleven
datasets in `similarity_space.py` puts it with the four-level datasets; the code prints that
disagreement and uses the molecule count.

**The other experiments.** One row is one group of datasets. Each number is a count per
dataset. What one counts differs by experiment:

- n43: one combination of noise condition, level and repetition, on one fold. No model or input is counted.
- n45 and n46: one configuration, a model on the first base representation or on the first
  hybrid, at one fold, noise condition, level and repetition.
- n47: one configuration, a model on one of three inputs, at one fold, noise condition, level and repetition.
- n15: one combination of noise condition, level and repetition, on one fold, with one search on the damaged labels.
- n20: one combination of noise condition, level and repetition on fold 0, plus one run per noise condition on the labels as they come.
- n36: fold 0 with one of 2 seeds, where repeat measurements exist. The seed picks which measurement each molecule measured more than once starts with.

| group | datasets | n43 | n45 | n46 | n47 | n15 | n20 | n36 |
|---|---|---|---|---|---|---|---|---|
| Regression, four levels | 20 | 245 | 800 | 880 | 1,320 | 9 or 15 | 23 | 0 |
| Regression, two levels | 2 | 115 | 280 | 360 | 540 | 9 | 23 | 0 |
| `herg_ic50` | 1 | 115 | 140 | 180 | 270 | 9 | 23 | 2 |
| Classification | 5 | 455 | 1,820 | 0 | 2,880 | 9 or 15 | 30 | 0 |

n15 runs one noise condition (`DEFAULT_STRATEGY_REG`, or `DEFAULT_STRATEGY_CLF` for
classification), with 3 combinations of level and repetition. These run on 3 folds (9 in the
table), or on 5 folds (15) for the datasets in `ALL_FOLDS_DATASETS`.
n36 runs only where repeat measurements exist, which is `herg_ic50` alone among these datasets.
n46 and n36 are regression only. n43's 9 network methods, n45's fits and n46's three structure
passes run inside what the table counts and are not counted in it.

**What each cliff decision adds to it.** One row is one candidate, on one dataset, computed
from those numbers per configuration at one fold, noise condition, noise level and repetition.

| added to n44's standard cleanup run | results on a four-level dataset | results on a two-level dataset |
|---|---|---|
| Nothing | 3,789,030 | 1,752,810 |
| One more input for each model, same size as a hybrid | +1,488,550 | +685,850 |

One row is one change to the neighbor settings. The number is how many settings each configuration, at one fold, noise condition, noise level and repetition, is
screened over.

| added to the neighbor settings | settings, larger noise grid (26 where the augmented spaces are kept) | settings, smaller noise grid (13 where the augmented spaces are kept) |
|---|---|---|
| Second base and its augmented version removed | 20 | 13, it holds no second base |
| One new space under today's distances | 29 | 15 |
| One new space under today's distances plus Tanimoto and Dice | 31 | 17 |

n44 keeps 3 neighbor settings per configuration at one fold, noise condition, noise level and repetition after a screen, so more settings do not change its
result count. They do add screen time. n45 grades every setting with no screen, so its count
grows with them. It has not been recounted since the decisions.

**What a definition costs elsewhere.** One row is one candidate set of definitions on one
dataset, counted with `definitions.expand`, her five coefficients, 9 cliff shares, and the
number of fits `cliff_aware_cleanup` prints for one M, one L and 3 outlier shares. That is
270 fits per definition, plus L's own settings, which it does not print.

| definitions | what the set holds | count | model fits in `cliff_aware_cleanup` |
|---|---|---|---|
| References only | published definitions of an AC plus MoleculeACE's definition at 6 cutoffs | 64 | 17,310 |
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

Claude's estimate, not a measurement: a Caco-2 dnn configuration on `base_1`, at one fold, noise condition, noise level and repetition, takes about 3 hours. A
LogD one takes at least 7.5 hours.

Not measured: any svm fit, and a whole configuration on either input, at one fold, noise condition, noise level and repetition.
Also not measured: the 40 hybrid searches of TBD2, the cliff masks per noise condition, level and
repetition, and anything inside the cliff units beyond the `cliff_sets` timing in the appendix.

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
- hERG in the eight-hybrid comparison. Five of the eight metrics do not work on a 0/1
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
- Which molecules the eight metrics read for a hybrid searched on one fold's training molecules.
  DECIDED by her, 2026-11-16: that fold's training molecules only. Her words: "Over that 80".
- Which noise the two noise metrics inject. `simeval.inject_label_noise` moves 15 per cent of
  labels with sigma 0.6, and that is none of her six noise conditions.
- What "the two bases" means. Her TBD2 line is "I have the two bases". It is not stated
  whether that is ecfp4 and atom_pair by name, which are the two she listed earlier, or
  whichever two single representations each fold's own search picks; the code's `base_1` and
  `base_2` are the second. Her other TBD2 words are "top base + ecfp4".
- Whether the three published models' flag files are obtained, and for which datasets.
  Nothing in the code waits on a setting; it waits on the files.
- Whether `ActivityCliffDetector` (`noise_mitigation_methods.py:770`) becomes a detector in
  the noise suite's detector list or is dropped as a duplicate. It has two free thresholds.
  It is reached today by `representations/molecular.py` and `tests/similarity_experiments.py`, and by the name `activity_cliffs` through
  `get_mitigation_method` in `tests/experiments/similarity_optimized_hybrid.py`,
  `tests/noise_mitigation_advanced.py` and `tests/noise_mitigation_extended.py`, checked 2026-11-18.
- Where s01, s02, s04, s09, s12, `simeval` and `nary_similarity` land in the restructured
  layout.
- What the continuous landscape scores are for. The scores named in `cliff_rules.CONTINUOUS_SCORES` are computed
  by `cliff_census.continuous_scores` and nothing reads its output, checked 2026-11-17. Either one of them defines a cliff and
  joins the sweep, or they are reported as landscape descriptions only.
- Whether ACNet and the Bajorath group's published cliff sets are downloaded as a second
  external cliff set after MoleculeACE. `cliff_census.py` records `'acnet': {'downloaded':
  False}`.
- Whether `opig_activity_cliff_qsar` builds Dablander's other two pair settings, or the paper
  states it runs on two of his four. Checked 2026-11-17: its docstring names M_inter and
  M_test, and the string `M_cores` is in no file.
- Whether the half-cliff band, 1 to 2 log units dropped from both sides, is carried into the
  noise experiments or stays in the Dablander replication only.
- Whether dropping a named molecule and refitting is run, and on which dataset. The honest
  form costs one fit per molecule per fold, so it is a dataset-size decision. Nothing in the
  repository does it per molecule.

**Two things that disagree in the cliff settings file.**

`tests/experiments/cliffs/CLIFF_SETTINGS.json` holds `definition_scope` as `["per_fold"]` and
`held_out_cliff_rule` as `["same_as_training"]`. Both were narrowed from `"all"` in commit
dd020ed7 of 2026-11-05, whose subject is about `moleculeace_agreement` taking `--data-dir`,
and no quote of hers is recorded for either value. The record of what was built on 2026-11-04
says of the same two settings: "Both values run". So the file runs one value per setting and the
record says two, and no sentence of hers picks either side.

**Named in her outline and not built, carried here so they are not lost.**

- The kNN search, which searches knn and nothing else (section N4).
- The similarity search on binary input with `n_reps` fixed to all (section N4).
- The cliff-call Matthews correlations `opig_activity_cliff_qsar` already computes. Its
  `ac_mcc` field exists, checked 2026-11-17, and the thesis reports its prediction errors
  and none of them.

**In the grid review.**

- Everything in `PAPER_3_NOISE_EXPERIMENTS.md` section 5: detectors, cleanups, their settings,
  detector feature spaces, neighbor settings, pairings and `gp_smooth`.
- Whether n43 keeps `hybrid__general2` and `weighted_hybrid`, and n47 keeps `hybrid__general2`.
- Whether the second-base cut also covers the spaces where neighbors are found.
- Whether the kNN and binary hybrids are spaces where neighbors are found, inputs of configurations, or both.
- Whether Tanimoto and Dice join the neighbor distances.
- The classification grid: label-flip noise conditions, levels, and whether larger datasets run fewer.
- What n15, n20 and n36 run, and on what grid.

**After the standard cleanup run.**

- Her M and her L.
- Which results get the variant cleanups run: "First the results, then the checks for what I need".
- Which cliff definitions the variant cleanups run and `cliff_aware_cleanup` read.
- Which inputs the configurations in `cliff_aware_cleanup` use.
- Whether a definition built from one model's hybrid applies to the other model's configurations, with their folds, noise conditions, noise levels and repetitions.
- Whether Paper 3 splits, from her outline: "If I can prove 1 and 2 well enough, then paper 3
  gets split into two papers".

---

## N8. The order

Claude's proposal, not reviewed by her. An arrow is "needs the result of".

- Eight-hybrid comparison, TBD3 runs, 45 per cent sweep: need nothing from noise.
- Fixing the ledger and connecting the noise code to the definitions: needs nothing from them.
- Grid review → the comparison and TBD3 results, because of "I'm waiting for ACs", and the
  corrected counts.
- Standard cleanup run → the grid review, and `paper3` opened by her.
- M, L and the chosen results → the standard cleanup run.
- Variant cleanups run and `cliff_aware_cleanup` → M, L, chosen results and the chosen definitions.

⛔ `paper3` has to stay closed until the first two entries of section N4 are fixed, or the
ledger runs fold 0 at two levels and without `hetero_seen`. Her ask, 2026-11-05: "the paper
2/3 experiment ledger should have paper 3 experiments on hold - confirm this is the case".
The code starts with every step after `reps` closed. The server's own file has never been read back here.
She can read it with `python3 tests/slurm_scripts/queue_ledgers.py check`, which changes
nothing and lists any Papers steps still closed.

Four prompts follow. Prompts 1 and 2 can run side by side. Prompt 3 waits for both. Prompt 4
waits for the standard cleanup run. The items under "The two things she assigned as work" and the three
published models above run when she says so and are not in these prompts.

---

## Prompt 1: finish the cliff tests and get them onto the server

Paste everything inside the block into a new Claude Code chat in `~/repos/KIRBy`.

```
Read ~/repos/KIRBy/CLAUDE.md in full and follow it in every reply. Then read MEMORY.md in the
memory folder.

Read AC_GRID_TEST_PLAN.md in full, Part 1 and Part 2. It is the plan. Its section "Her
outline, verbatim" is Adelaide's outline and overrides everything where they differ. Its
section "What she has decided" holds her values.

Her words: "the AC grid plan is the final touches on implementation", and "AC grid test is an
extension of the existing code". Do not redesign anything. Read the code before any claim.

THE JOB. Get the measurements that TBD2, TBD3 and TBD6 wait on, with her running the server.

BUILD, without asking. Check each against the code first, because the plan's section N4 was
true on 2026-11-07 and code moves.
1. The kNN search as she said it: "this should literally just search over knn and
   nothing else".
2. Her two binary searches: "general tier, prediction objective, binary input only" and
   "general tier, similarity objective, binary input only (n_reps=all, not swept over for
   both)".
3. A script that scores all eight hybrids, fast tier, on qm9, herg_ki, caco2, herg and logd,
   with all eight metrics in tests/experiments/similarity_metrics/simeval.py, imported
   read-only. It keeps every number, saves after each hybrid so a 4-hour session loses little,
   and resumes.
4. Time recording in that script, cliff_sets, definition_agreement and
   moleculeace_agreement, through tests/experiments/noise/noise_timing.py. Her words: "keep
   track of timing".
5. The metric-against-metric comparison stays an analysis for after the results.

TWO OPEN SETTINGS STOP THE COMPARISON. Bring them first, one per line, with what changes either
way: which molecules the metrics read for a hybrid searched on one fold's training molecules, and
which noise the two noise metrics inject. Enter each as a named input with no default.

ALSO OPEN: which datasets the TBD3 runs use. Cliff units run on the 14 small-molecule datasets
only, from her outline's "small molecules only" (fixed 2026-11-18).

RUNS. She runs every server command. Before you ask her to run anything, say what it is for,
what decision it feeds, and how long it should take, with the unit, marked estimate or
measured. Her words about the last runs: "oh thats what those tests were doing, you should
have said earlier". Three runs:
- The eight-hybrid comparison.
- cliff_sets and definition_agreement.
- moleculeace_agreement with cliff shares to 45 per cent. It resumes without --force-rerun.

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
2. The Papers ledger runs the folds, levels and noise conditions she decided for n43 to n47: five folds, four levels on datasets
   of 5,000 molecules or fewer plus QM9, two on larger ones, and hetero_seen as a unit. Show
   the unit list and levels for caco2, logd and qm9 before and after.
3. Finish the check for repeated training sets. Compare what each configuration, with its fold, noise condition, noise level and repetition, asks for by training
   set, not by name, and report what you find. Her words: "the same thing doesn't get run
   multiple times".
4. Done 2026-11-09: the variant cleanups run reads every cliff definition in
   tests/experiments/cliffs/definitions.py by name, or "all". Check it still holds. No noise
   experiment may fix cliffs to one definition of an AC; n47's cliff numbers moved to the cliff experiments on 2026-11-16 (not committed), section N3 item 7.
5a. Take the cliff parts out of n47: which molecules are cliffs, held-out error on cliff
   against non-cliff molecules, and n14's check of whether cliffs come in groups or as lone
   pairs. The first two are already in cliff_sets and error_by_cliff_status. The grouping check
   becomes part of the cliff experiments, under every definition, reading which molecules are
   paired; cliff_sets saves the pairs in `<dataset>__cliff_pairs.npz` and records what that costs, since 2026-11-16 (section N3 item 7). n47's
   non-cliff parts stay. Her decision is in AC_GRID_TEST_PLAN.md section N1, 2026-11-09.
5. cliff_aware_cleanup accepts every detector and cleanup n44 can run, through the same code n44
   uses. Its printed fit count includes L's own settings. Which inputs its configurations use becomes a named input
   with no default, because the noise suite cut the second base representation and she has not said what this unit
   fits. Do the same in error_by_cliff_status.
6. Time recording in n15, n20, n36, cliff_aware_cleanup and error_by_cliff_status. Her words:
   "keep track of timing. They'll be tests in the future".
7. Recount every experiment in the Paper 3 noise list with fitting stubbed out, classification
   included, per dataset: scratchpad/paper3_grid_counts/. Update PAPER_3_NOISE_EXPERIMENTS.md
   sections 4, 5, 8, 11 and 15 and AC_GRID_TEST_PLAN.md section N5. Counts are not times.

DO NOT DECIDE. Leave these for the grid review, and do not change them: n15, n20 and n36's grids,
the classification grid, n47's old cliff numbers, which hybrids join the neighbor settings, and
everything in PAPER_3_NOISE_EXPERIMENTS.md section 5.

DO NOT RUN the noise suite on the server, and do not open paper3. If you need a server check, say
what it is for and how long it takes, marked estimate or measured, before she spends a session.

Another chat may be working in tests/experiments/cliffs/ at the same time, on the eight-hybrid
comparison, the kNN search and the binary searches. Do not edit those parts.

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

Every cut is hers. Do not cut a setting because it is expensive. Offer options with numbers. This
chat changes documents only until she says to build.

THE SETTINGS. Map all of them against the results before going deep on any one.
- Score detectors, neighbor detectors and flagging methods, including flags of known content.
- Cleanups and every cleanup setting.
- Detector feature spaces, and which detectors pair with which cleanups.
- Neighbor settings, including the second-base question and gp_smooth.
- Everything in AC_GRID_TEST_PLAN.md section N7 under "In the grid review". Those include what the
  cliff sweep adds: the kNN and binary hybrids as neighbor spaces or as inputs of configurations, and
  Tanimoto and Dice as neighbor distances.

WHAT TO READ.
- ~/Documents/ch7_noise_methods_audit.html section #s6, her two lists of methods that raised the
  top score. Checked over the result files on 2026-10-29: all 35 entries of list 1 stay; list 2
  keeps Constant lift, Truncated-normal mean, Censored EM and Own lift, gains Delete clipped,
  and loses the XGBoost accelerated-failure-time interval model.
- thesis.tex Chapter 7, "Detecting and handling label noise", Results section.
- /Volumes/seagate/kirby_results_2026-10-10b/noise, the result files behind it.
- thesis_provenance/METHOD_PROVENANCE.md and NOISE_CODE_FIX_DECISIONS_2026-10-26.md, for where
  each method came from and which code changed since.
- PAPER_3_OUTLINE.md, Methods, "What counts as a method having raised the score", for her
  definition and the two checks that support it.
- The timing in AC_GRID_TEST_PLAN.md section N6, plus any new logs she gives you.
- The cliff results Prompt 1 produced, and what she decided from them.

HOW.
- "Worked" means raised the top score. Shuffled detector scores and QM9 support it and do not sort
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

Done means: every setting has her decision, the code that carries it is built and tested, the grid is
recounted, and she decides whether paper3 opens.
```

---

## Prompt 4: after the standard cleanup run, the variant cleanups run and D, D' and D''

Paste everything inside the block into a new Claude Code chat in `~/repos/KIRBy`, only once n44's
scores from the standard cleanup run exist.

```
Read ~/repos/KIRBy/CLAUDE.md in full and follow it in every reply. Then read MEMORY.md in the
memory folder.

Read AC_GRID_TEST_PLAN.md's section "Her outline, verbatim", PAPER_3_NOISE_EXPERIMENTS.md
sections 1 and 9, and AC_GRID_TEST_PLAN.md Part 2.

THE JOB. Her two claims, from her outline: "1) prove that I am accurately able to detect ACs" and
"2) that detecting them helps me mitigate noise more effectively".

FIRST, bring her the scores from the standard cleanup run so she can decide, one per line:
- Her M and her L, from her words: "Look at the best outlier detection methods, call one M, and then
  subsequent best label handling methods, call one L".
- Which results get the variant cleanups run: "First the results, then the checks for what I need".
- Which cliff definitions those two experiments read, from the TBD results.
- What cliff_aware_cleanup fits, and whether a definition from one model's hybrid applies to the other
  model's configurations, with their folds, noise conditions, noise levels and repetitions.

Show each with its fit count, computed from her values, and its time if measured. Do not choose.

THEN, once she has decided, give her the commands: the variant cleanups run through KIRBY_N44_SECOND_RUN, and
cliff_aware_cleanup through CLIFF_SETTINGS.json. Say what each run is for and how long it should take
before she starts it. Time recording stays on.

READING THE RESULT. D'' against D' and D'' against D, beside Random put-back, plus how many flagged
molecules were cliff molecules. Where M flagged no cliff molecule, D'' equals D' by construction, so
say so. Whether Paper 3 splits is hers: "If I can prove 1 and 2 well enough, then paper 3 gets split
into two papers".

Rules. Never ssh to the server. One `srun ... --pty /bin/bash` line, then plain commands inside. No
sbatch. Sessions last at most 4 hours. No `rm` in anything she pastes. No patient data anywhere. Do not
touch Connectome. Commit or push only when she asks. Never write a verdict of yours into her documents.
```

---

# APPENDIX: the measurements the TBD sections were written from

Every number names the file it came from. Nothing here is a recommendation.

---

## A1. TBD1, the coefficients

Her own server sweep is stored at `tests/results/framework/f34_full_sweep/`. It covers four
datasets with 12 representations each, 16 representation names across the four, 14
coefficients and five folds. It holds 2,020 rows. In 290 of them the coefficient was skipped
as "undefined: requires non-negative input, this representation has negative values", and in 20 as "undefined on
this block".

One row is one coefficient. The number is the best score that coefficient reaches on that
dataset, searching over every representation, neighbor weighting and neighbor count,
averaged over her five folds. The first three numbers are RMSE, lower better, in pKi on hERG
Ki and in log units on Caco-2 and MDR1-MDCK efflux. The last is AUC on hERG blocking, higher
better. The bracket is how many of that dataset's 12 representations the coefficient is
defined on. Recomputed from the stored rows on 2026-11-04.

| coefficient | hERG Ki | Caco-2 | MDR1-MDCK efflux | hERG blocking |
|---|---|---|---|---|
| manhattan | 0.5604 (12) | 0.3173 (12) | 0.4862 (12) | 0.8475 (12) |
| minkowski p=0.5 | 0.5653 (8) | 0.3190 (7) | 0.4886 (7) | 0.8487 (7) |
| minkowski p=0.3 | 0.5698 (8) | 0.3202 (7) | 0.4890 (7) | 0.8490 (7) |
| euclidean | 0.5700 (12) | 0.3161 (12) | 0.4886 (12) | 0.8423 (12) |
| correlation | 0.5813 (8) | 0.3143 (7) | 0.4877 (7) | 0.8438 (7) |
| cosine | 0.5818 (12) | 0.3137 (12) | 0.4856 (12) | 0.8450 (12) |
| tversky | 0.5981 (3) | 0.3295 (4) | 0.5409 (4) | 0.8246 (4) |
| dice | 0.5997 (4) | 0.3296 (5) | 0.5401 (5) | 0.8247 (5) |
| tanimoto | 0.5997 (4) | 0.3308 (5) | 0.5408 (5) | 0.8247 (5) |
| soergel | 0.6068 (1) | 0.3319 (1) | 0.5560 (1) | 0.8270 (1) |
| chebyshev | 0.6347 (8) | 0.3323 (7) | 0.5308 (7) | 0.7977 (7) |
| mahalanobis | 0.6651 (6) | 0.3306 (6) | 0.6028 (6) | 0.8187 (7) |

The same quantity restricted to ECFP4's 2,048 bits, from the same stored rows. One row is one
coefficient. Six coefficients are defined on 0/1 features.

| coefficient | hERG Ki | Caco-2 | MDR1-MDCK efflux | hERG blocking |
|---|---|---|---|---|
| dice | 0.6095 | 0.3296 | 0.5823 | 0.8154 |
| tanimoto | 0.6095 | 0.3308 | 0.5823 | 0.8146 |
| tversky | 0.6100 | 0.3295 | 0.5851 | 0.8150 |
| cosine | 0.6109 | 0.3294 | 0.5937 | 0.8147 |
| euclidean | 0.6215 | 0.3328 | 0.6685 | 0.8153 |
| manhattan | 0.6215 | 0.3321 | 0.6685 | 0.8153 |

How far apart the best two settings are, against how far apart her folds are. On hERG Ki
under Manhattan the best setting averages 0.5604 pKi RMSE over five folds, and the second
averages 0.5611 pKi. That is a difference of 0.0007 pKi. The best setting's own five folds
run from 0.5078 to 0.6334 pKi, a range of 0.1255 pKi.

Tanimoto and Dice rank pairs identically, so only the cutoff differs. On hERG Ki, ECFP4 Dice
at 0.75 and ECFP4 Tanimoto at 0.60 both call 7,254 pairs of molecules similar. Both put 362
of the 1,412 molecules in a cliff pair. Source: `/Volumes/seagate/cliff_census_sweep/herg_ki__sweep.json`,
a local run, which that file's own `what_it_is` field describes as "A local run of this is not
a result; it picks which definitions to carry."

Her published justification already in her thesis, `thesis.tex` line 973: "\citet{Bajusz2015}
compared eight coefficients and found Tanimoto, Dice, Cosine, and Soergel the best and
largely equivalent, while similarities derived from Euclidean and Manhattan distances were
not recommended on their own."

Two numbers from her earlier runs, counted by a Claude session on 2026-11-04 and not
re-counted since. `s01_similarity_baseline` scores Manhattan, cosine and Minkowski p=0.5
only. Its score is the Spearman correlation between a pair's distance and the difference of
its two labels. Manhattan is the best of those three on 7 of 10 representations on hERG Ki, 8
of 10 on Caco-2 and 8 of 10 on LogD. Cosine is best on 8 of 10 on QM9. In
`/Volumes/seagate/kirby_s13_coef/`, on hERG Ki under Manhattan the hybrid's neighbors have
closer labels than atom pairs' (0.5575 against 0.5746 pKi) and under cosine they do not
(0.5753 against 0.5739 pKi).

**Not measured.** No coefficient has been scored against a cliff set, which is the use TBD1
is for. Every number above scores a coefficient by neighbor prediction, by distance against
label difference, or by clustering and retrieval. Tanimoto, Dice, Soergel and Tversky have
never been scored by the distance-against-label-difference Spearman that `s01` and `s02` use.
`f34` covers four of her six small-molecule datasets; LogD and QM9 are not in it.

Each coefficient she names needs a path to a cliff set. Seven scores in
`tests/experiments/noise/cliff_rules.py` already have one: Tanimoto and Dice each in a 0/1
and a count form, Tversky on 0/1 features only, Soergel on count features only, and cosine.
Eight more reach a cliff set by one route only: 1 divided by (1 plus Euclidean distance) over
every representation z-scored and concatenated. Those eight are euclidean, manhattan,
minkowski, chebyshev, correlation, canberra, braycurtis and mahalanobis. That module marks
the route `reference_independent: False`.

---

## A2. TBD2, the representations

From the same stored `f34_full_sweep`. One row is one representation. The number is the best
score it reaches on that dataset, searching over every coefficient, neighbor weighting and
neighbor count, averaged over five folds. Units as in the table above: RMSE in pKi, log
units, log units, then AUC. A dash means that representation was not run on that dataset.
Recomputed from the stored rows on 2026-11-04.

| representation | hERG Ki | Caco-2 | MDR1-MDCK efflux | hERG blocking |
|---|---|---|---|---|
| hybrid, kNN internal evaluator | 0.5604 | 0.3171 | 0.4856 | 0.8456 |
| hybrid, general search, RF internal evaluator | 0.5786 | 0.3137 | 0.4862 | 0.8490 |
| atom_pair | 0.5948 | 0.3347 | 0.5401 | 0.8273 |
| pdv | 0.6009 | 0.3213 | 0.5165 | -- |
| graph_kernel | 0.6068 | 0.3317 | 0.5560 | 0.8282 |
| ecfp4 | 0.6095 | 0.3294 | 0.5823 | 0.8154 |
| mordred | 0.6101 | -- | 0.5437 | 0.8249 |
| mhg_gnn | 0.6148 | 0.3308 | 0.5585 | 0.8334 |
| grover | 0.6218 | 0.3298 | 0.6027 | 0.8113 |
| smited | 0.6299 | 0.3370 | 0.5266 | -- |
| chembert | 0.6300 | -- | -- | -- |
| avalon | 0.6367 | 0.3381 | 0.5871 | 0.8028 |
| maccs | -- | -- | 0.5638 | -- |
| mol2vec | -- | -- | -- | 0.8112 |
| molformer | -- | 0.3345 | -- | 0.8185 |
| top_torsion | -- | 0.3331 | -- | 0.8071 |

A hybrid comes first on all four of those datasets. The best single representation is
atom_pair on hERG Ki at 0.5948 pKi and pdv on Caco-2 at 0.3213 log units. It is pdv on
MDR1-MDCK efflux at 0.5165 log units and graph_kernel on hERG blocking at 0.8282 AUC.

Which single representations her own kNN search ranked first and second per fold, read from
`/Volumes/seagate/kirby_results_2026-10-10b/similarity/s04_activity_cliffs/<dataset>.json`,
field `folds[*].selection.knn_base`, which holds two names per fold. On hERG Ki atom_pair is
first on 4 of 5 folds and ecfp4 on the fifth. On MDR1-MDCK efflux pdv is first and smited
second on all five. On LogD mordred is first on 4 of 5. On QM9 smited is first on 4 of 5 and
mol2vec on the fifth. On Caco-2 pdv is first on 2 folds, grover on 2 and ecfp4 on 1. On hERG
blocking atom_pair is first on 3 of 5 and is in the top two on all five.

Her thesis tables. `figures_ch5_7/tab_distance_tracks_activity.tex` scores each dataset by
the Spearman correlation between pair distance and absolute label difference, mean over outer
folds. The kNN hybrid is higher than the best single representation on all 14 datasets. On
the six small-molecule datasets it goes hERG Ki 0.0729 to 0.2625, LogD 0.0765 to 0.2687 and
QM9 0.1298 to 0.3779. The other three go MDR1-MDCK efflux 0.2286 to 0.2883, Caco-2 0.1517 to
0.1969 and hERG blocking 0.0257 to 0.0552.

`figures_ch5_7/tab_nearest_neighbour_prediction.tex` predicts a held-out molecule from its
five nearest training molecules. The hybrid is more accurate on all six small-molecule
datasets. LogD goes 0.6766 to 0.5831 RMSE in log units and hERG Ki 0.6280 to 0.5822 pKi.
Caco-2 goes 0.3386 to 0.3246 log units and MDR1-MDCK efflux 0.5543 to 0.5202 log units. QM9
goes 0.6692 to 0.4205 eV and hERG blocking 0.8143 to 0.8335 AUROC.

What weighting a single representation's features by importance does to that gain, from
`thesis.tex` line 3298: "Weighting both the single representation and the hybrid shrinks the
gain wherever the hybrid wins, on LogD from 0.089 to 0.025 RMSE."

On cliff molecules the comparison reverses against a weighted single representation.
`/Volumes/seagate/kirby_s09_out/s09_cross_dataset_summary/results.json` was written
2026-09-23 on three datasets, Manhattan only. Its field
`cliff_rmse_reduction_pct_vs_weighted_rep` compares the hybrid with the best
importance-weighted single representation on cliff molecules. The hybrid is worse by 0.23 per
cent of RMSE on Caco-2, 4.11 per cent on hERG Ki and 0.12 per cent on MDR1-MDCK efflux. The
same file's `cliff_rmse_reduction_pct` compares against an unweighted single representation.
There the hybrid is better by 1.14 per cent on Caco-2 and 4.29 per cent on MDR1-MDCK efflux.
It is worse by 0.89 per cent on hERG Ki.

**Not measured.** A hybrid has never been used to decide which pairs are similar. Twelve of
the 44 similarity criteria in `cliff_rules.py` read a hybrid, and none has produced a number.
In `/Volumes/seagate/ac1_smoke/herg_ki.json` all twelve sit in `similarity_criteria_excluded`
with the reason "hybrid_bits was not loaded" (seven) or "hybrid_counts was not loaded" (five). In `/Volumes/seagate/cliff_census/herg_ki.json`
the field `representations_loaded` is the empty list, and none of the twelve appears in that file: it lists 13 available and 9 excluded similarity criteria, none of them a hybrid.

In `s04_activity_cliffs` and `s12_cross_domain_cliffs` the cliff set itself came from ECFP4:
the field `cliff_definition.fingerprint_note` in s04 and `cliff_info.structural_source` in s12 reads "ECFP4 radius 2 over 2048 bits ...
Tanimoto at or above 0.55". So the hybrids were scored on cliffs ECFP4 chose.

The method her thesis used to compare representations at a fixed number of similar pairs has
its ingredient in the code and no file runs it. `thesis.tex` line 1123 states the result: "On
hERG $K_i$ the number of cliff molecules then ranges from 138 (atom pairs) to 267 (WL graph
kernel). Of the pairs that ECFP4 calls similar, only 57\% to 74\% are also among another
representation's closest pairs, and on QM9 only 1\% to 24\%."

One dataset in Paper 3's small-molecule list has no per-dataset representation list, so no
hybrid can be built for it as the code stands. `DATASET_REPS` in
`tests/experiments/complexity/complexity_helpers.py` holds entries for 59 datasets. It has no
entry for `herg_ic50`. The three ChEMBL targets, `chembl_qsar_d2`, `chembl_qsar_fxa` and `chembl_qsar_mpro`, each have one.

What each hybrid adds to the grid, measured 2026-11-04 on hERG Ki's 1,129 fold-0 training
molecules under her five coefficients. One row is one thing added on top of the two base representations
plus the first hybrid, which is 1,467 definitions by itself.

| added | definitions |
|---|---|
| another continuous hybrid | +343 |
| a binary-input hybrid | +637 |
| one more single binary representation | +539 |
| one more single continuous representation | +245 |

All eight hybrids plus the two base representations is 4,456 definitions. Binary and count cost more because tanimoto
and dice run only on 0/1 and count features, so a binary or count representation gets five of her coefficients and a
continuous one gets three.

Fit time does not scale with this, except in one experiment. Measured 2026-11-04 on the same
shape with a 900-feature hybrid: one svm fit and predict is 2.06 s and one dnn is 9.18 s. The
held-out error split fits once per fold and per configuration (one model on one input) -- 30 fits, about three
minutes for the dataset -- and a definition only splits the predictions afterwards, so
representations add no fits there. The downstream D, D', D'' test costs 540 fits per
definition, about 50 minutes.

On the binary searches, measured 2026-11-04: hERG Ki has 3 binary representations of its 10
(atom_pair, avalon, ecfp4). The binary count on the other datasets is not checked.

---

## A3. TBD3 and TBD4, the cutoff and its scope

The definition of an AC her thesis uses, `thesis.tex` line 1120: "Two molecules are similar if their ECFP4
Tanimoto similarity is at least 0.55 \citep{Hu2015} ... We chose the gap on each dataset so
that 15\% of the molecules are cliffs, so the gap follows the spread of the labels."

One row is one dataset. Molecules is the whole dataset. Cliff molecules is how many sit in at
least one cliff pair at a 15 per cent cliff share. Gap is the label difference that share
forces, in that dataset's own label units. From `thesis.tex` table
`tab:flaws:distance_summary`, lines 1165 to 1172.

| dataset | molecules | cliff molecules | gap |
|---|---|---|---|
| hERG Ki | 1,412 | 211 | 1.76 pKi |
| LogD | 5,028 | 731 | 2.90 log units |
| Caco-2 | 2,158 | 324 | 1.21 log units |
| RLM clearance | 3,053 | 458 | 0.34 log units |
| MDR1-MDCK efflux | 2,641 | 397 | 0.20 log units |
| HLM clearance | 3,086 | 462 | 0.27 log units |
| QM9 | 9,978 | 1,495 | 0.05 eV |

What one fixed pair of settings does to the share of molecules called cliffs. One row is one
dataset. The number is the share of a fold's training molecules in at least one cliff pair,
across her five folds. ECFP4 Tanimoto is held at 0.55 and the gap at twice the labels' median
absolute deviation. Source: `s04_activity_cliffs/<dataset>.json`, field
`cliff_definition.by_fold[*].uncalibrated_pct_cliff_compounds`. The hERG Ki row was re-read
on 2026-11-04, over five folds of 1,129 or 1,130 training molecules. The other four were read
by a Claude session on 2026-11-04.

| dataset | share of training molecules in a cliff pair |
|---|---|
| hERG Ki | 27.4 to 32.3 per cent |
| Caco-2 | 71.3 to 73.6 per cent |
| MDR1-MDCK efflux | 3.9 to 4.4 per cent |
| LogD | 65.2 to 67.3 per cent |
| QM9 | 0.2 to 0.4 per cent |

One definition of an AC, five datasets, 0.2 per cent to 74 per cent of molecules.

Turning it round. One row is one dataset. Both numbers are the gap that puts 15 per cent of a
fold's training molecules in a cliff pair, across her five folds. Source: the same files,
field `cliff_definition.by_fold[*].activity_threshold`.

| dataset | gap, in label units | gap, as a multiple of that dataset's median absolute deviation |
|---|---|---|
| hERG Ki | 1.595 to 1.750 pKi | 2.56 to 2.97 |
| Caco-2 | 1.157 to 1.196 log units | 5.35 to 5.74 |
| MDR1-MDCK efflux | 0.131 to 0.162 log units | 0.34 to 0.41 |
| LogD | 2.750 to 2.900 log units | 3.44 to 3.63 |
| QM9 | 0.030 to 0.038 eV | 0.03 to 0.04 |

The spread between folds of one dataset is small next to the spread between datasets.

What moving the similarity cutoff alone does. One row is one dataset. Every number is the
share of that dataset's molecules in at least one cliff pair, as a percentage, with the gap
held at twice the median absolute deviation. Source: the three local sweep files in
`/Volumes/seagate/cliff_census_sweep/`.

| dataset | share at 0.40 | share at 0.55 | share at 0.70 | share at 0.90 |
|---|---|---|---|---|
| hERG Ki | 50.1 | 31.9 | 12.1 | 0.6 |
| Caco-2 | 94.6 | 75.3 | 38.9 | 0.6 |
| MDR1-MDCK efflux | 19.7 | 4.7 | 0.9 | 0.0 |

On hERG Ki those four cutoffs rest on 23,611, 10,809, 2,829 and 103 similar pairs.

Where a published gap in absolute label units produces nothing. On Caco-2, with ECFP4
Tanimoto at 0.55 and a 2.0-unit gap, 0 of the 2,158 molecules are in a cliff pair, on 24,858
similar pairs. Three more definitions of an AC give 0 cliff molecules on that dataset. One is the same
similarity at a 2.0-unit gap under a second name. The others are Morgan Tanimoto at 0.90 on
143 similar pairs and the RDKit fingerprint under Dice at 0.90 on 13,469 similar pairs.
Source: `/Volumes/seagate/cliff_census/caco2.json`, field `named_rules`.

Where two definitions both give a cliff set, they disagree. One row is one pair of named
definitions of an AC on hERG Ki's 1,412 molecules. "Both" counts molecules each definition calls a cliff molecule.
Source: `/Volumes/seagate/cliff_census/herg_ki.json`, field `named_rule_agreement`.

| first definition | second definition | both | only first | only second | Jaccard |
|---|---|---|---|---|---|
| hu2015 | n01 | 119 | 0 | 332 | 0.2639 |
| moleculeace | n01 | 227 | 89 | 224 | 0.4204 |
| geometry_morgan | n01 | 5 | 0 | 446 | 0.0111 |
| geometry_rdkit_dice | hu2015 | 43 | 3 | 76 | 0.3525 |
| hu2015 | moleculeace | 58 | 61 | 258 | 0.1538 |
| hu2015 | n14 | 119 | 0 | 0 | 1.0000 |

The last row is two names for the same two criteria.

How lopsided the similarity scale itself is, `thesis.tex` line 1133: "Of the 996,166 hERG
$K_i$ pairs, 93\% have an ECFP4 Tanimoto similarity below 0.20, and only 1.1\% reach 0.55.
Over all pairs the median label difference is 0.81 p$K_i$, but above a similarity of about
0.4 it is 0.20 to 0.30 p$K_i$, although 8.3\% of the 774 pairs at 0.8 or above still differ
by more than 1 p$K_i$."

The one external cliff set that has been scored. One row is one cliff definition, pooled over
MoleculeACE's 30 targets and 48,714 molecules, against their own per-molecule `cliff_mol`
flag, which is set on 18,831 of those molecules. Source:
`tests/results/comparisons/opig_moleculeace_cliff_check.json`, written 2026-11-02, 30
targets, 0 failed, 8,035.8 seconds.

| similarity | gap | our cliff molecules | their cliff molecules | both | pooled Matthews correlation |
|---|---|---|---|---|---|
| moleculeace_any_090 | absolute_1 | 18,844 | 18,831 | 18,823 | 0.99875 |
| ecfp4_tanimoto_055 | mad_2x | 19,064 | 18,831 | 11,042 | 0.31722 |
| ecfp4_tanimoto_055 | solved_target_15pct | 7,313 | 18,831 | 4,633 | 0.21315 |
| ecfp4_tanimoto_090 | absolute_1 | 1,133 | 18,831 | 1,130 | 0.19355 |

The first row is her own re-implementation of MoleculeACE's published definition of an AC against
MoleculeACE's own flag. Of the 30 targets, 18 match their flag exactly; the worst is
CHEMBL231_Ki, where her re-implementation adds 7 molecules to their 237, Matthews correlation 0.98085.

**Not measured.** No swept cutoff has been scored against anything external.
`score_against_labels` in `cliff_rules.py` is called from one file only,
`tests/experiments/opig_moleculeace_cliff_check.py`, and that file iterates the named
criteria rather than the sweep's points. Only 13 of the 44 similarity criteria reached its
pooled table, and no hybrid criterion did, because it builds a context from SMILES and a
label and loads no representations.

The sweep has never been run as a sweep for all six datasets.
`/Volumes/seagate/cliff_census_sweep/` holds three sweep files, for hERG Ki, Caco-2 and MDR1-MDCK
efflux, all local runs.

Nothing measures a cutoff per domain. Caco-2 and MDR1-MDCK efflux are both ADME assays in log
units. The gap reaching 15 per cent of molecules is 5.35 to 5.74 times the median absolute
deviation on one and 0.34 to 0.41 times on the other. Her thesis records a related point at
line 2922: "Only the two hERG datasets measure an activity, but we use the term on all six
small-molecule datasets."

---

## A4. TBD5, one hybrid overlap in place of their three scores

What MoleculeACE's definition of an AC is, from `tests/results/comparisons/opig_moleculeace_cliff_check.json`,
field `what_cliff_mol_is`: "similarity at or above 0.90 on any of ECFP Tanimoto,
scaffold-ECFP Tanimoto or scaled Levenshtein on the SMILES, with a 1-log-unit potency
difference. It is one published rule, not a reading of nature."

Her own implementation reproduces their flag exactly on one target. Source:
`/Volumes/seagate/moleculeace_benchmark/_smoke.json`, CHEMBL214_Ki, 3,317 molecules, 625.1
seconds. Her `moleculeace_any_090` with a 1-log-unit gap and their shipped `cliff_mol` flag
both mark the same 1,222 molecules. 0 molecules are only in hers and 0 only in theirs.

Her other definitions of an AC on that same target, same file. One row is one definition against their flag
of 1,222 cliff molecules out of 3,317.

| similarity | gap | her cliff molecules | both | only hers | only theirs |
|---|---|---|---|---|---|
| ecfp4_tanimoto_055 | mad_2x | 1,151 | 707 | 444 | 515 |
| ecfp4_tanimoto_055 | absolute_2 | 707 | 479 | 228 | 743 |
| ecfp4_tanimoto_055 | solved_target_15pct | 498 | 355 | 143 | 867 |
| rdkit_dice_090 | absolute_2 | 389 | 270 | 119 | 952 |
| morgan_tanimoto_090 | absolute_2 | 31 | 31 | 0 | 1,191 |

What dropping two of the three scores costs in strictness, with the gap at twice the median
absolute deviation. MoleculeACE's any-of-three rule at 0.90 calls 2,594 pairs similar on hERG
Ki, 8,377 on Caco-2 and 282 on MDR1-MDCK efflux. One ECFP4 Tanimoto score at the same 0.90
calls 103, 143 and 6 pairs similar. The single ECFP4 Tanimoto cutoff reaching the same number
of similar pairs as the any-of-three rule is about 0.70 on hERG Ki, 0.65 on Caco-2 and 0.60
on MDR1-MDCK efflux. The any-of-three rule puts 12.6 per cent of hERG Ki's molecules in a
cliff pair, 48.9 per cent on Caco-2 and 2.2 per cent on MDR1-MDCK efflux. Read from the three
local sweep files in `/Volumes/seagate/cliff_census_sweep/` by a Claude session on
2026-11-04.

Which of their three scores makes their cliffs, from the docstring of
`sim_moleculeace_one_score` in `cliff_rules.py` line 1008. Of MoleculeACE's 1,222 cliff
molecules on CHEMBL214_Ki, 1,055 came from the string measure and 148 from the fingerprint.
That is a docstring claim; the arithmetic does not close to 1,222 and the file does not say
why.

**Not measured.** No hybrid overlap score has been computed on any dataset, for the reason
under A2. It cannot be scored against MoleculeACE's flag as things stand: that needs a hybrid
built on their molecules, and `/Volumes/seagate/kirby_cache/similarity_metrics/sim_data/` is
empty. `/Volumes/seagate/moleculeace_reps/` holds representation files for 15 of the 30 targets, written 2026-11-04 by `build_moleculeace.py`, and none of them holds a hybrid. The census sweep grid in `cliff_census.py` moves no hybrid overlap cutoff. `tests/experiments/cliffs/definitions.py` moves it over the 14 values of `hybrid_overlap_cutoffs`, 0.30 to 0.95, for each hybrid representation it is given.

---

## A5. TBD6, the share a moved gap aims at

Her thesis and her `s04` run both use 15 per cent of a fold's training molecules.

What the other cliff shares cost. One row is one cliff share on hERG Ki's 1,412 molecules,
at ECFP4 Tanimoto 0.55. Gap is in pKi. Source:
`/Volumes/seagate/cliff_census_sweep/herg_ki__sweep.json`, a local run.

| cliff share aimed at | gap | cliff molecules | achieved share |
|---|---|---|---|
| 5 per cent | 2.274 | 71 | 5.03 per cent |
| 10 per cent | 1.970 | 143 | 10.13 per cent |
| 15 per cent | 1.760 | 211 | 14.94 per cent |
| 20 per cent | 1.554 | 283 | 20.04 per cent |
| 25 per cent | 1.408 | 352 | 24.93 per cent |
| 30 per cent | 1.260 | 423 | 29.96 per cent |

The same six cliff shares on her other two swept datasets. One row is one dataset. Every
number is a gap in that dataset's own label units, and each column heading is a cliff share
of that dataset's molecules. Read by a Claude session on 2026-11-04.

| dataset | 5 per cent | 10 per cent | 15 per cent | 20 per cent | 25 per cent | 30 per cent |
|---|---|---|---|---|---|---|
| Caco-2, log units | 1.54 | 1.34 | 1.21 | 1.11 | 1.04 | 0.98 |
| MDR1-MDCK efflux, log units | 0.76 | 0.41 | 0.20 | 0.08 | 0.00 | 0.00 |

On MDR1-MDCK efflux the gap runs out at about 25 per cent. Only 653 of its 2,641 molecules,
24.7 per cent, sit in a similar pair at all.

The share a held-out molecule is judged by is not the share the gap was fitted to. Her own
record, `THESIS_OUTLINE.md` lines 2269 to 2273, marked CHECKED 2026-10-09: "the cliff rule
aims at 15% of training molecules. Held-out molecules come out at about 40%: logd folds 39.1%
to 44.5%, herg 38% to 46%, caco2 28% to 33%, herg_ki 19% to 27%. The held-out rule is 'at
least two similar training molecules differ by more than the gap'. It is looser than the
training rule. So 'cliff molecule' means a different thing in training and held-out
molecules."

Her open decision on that, `THESIS_OUTLINE.md` lines 2274 to 2275, verbatim: "[DECIDE: the
held-out cliff rule. Keep it and state the two shares, which changes text only. Or apply the
training rule to held-out molecules, which needs s04 and s12 re-run.]"

Where a low share runs out of molecules to score. `thesis.tex` line 3314: "QM9 and
MDR1--MDCK efflux have too few cliff molecules to judge, 48 and 49 in all, and only two folds
on each hold the 10 needed for a score." And line 3309: "The antibody datasets have too few
cliff antibodies to score, for example 11 of the 429 held-out antibodies on $\Delta\Delta G$."

Whether a chosen gap clears the assay's own noise, on the one dataset of hers with repeat
measurements. `/Volumes/seagate/cliff_gap_vs_repeats/herg_ic50.json`: hERG IC50 holds 12,008
measurements over 9,802 molecules, 1,218 molecules measured more than once, and 652 of those
have repeats that disagree at all. Over those 652, the spread of one molecule's own repeats in
pIC50 has a median of 0.166, quartiles 0.050 and 0.450, and a 95th percentile of 1.608. Over
all 1,218 repeated molecules the median spread is 0.0014 pIC50, because 566 agree exactly.

One row is one definition of an AC in that file. The multiple is the gap divided by the median repeat spread
of the 652 disagreeing molecules. The last column counts cliff pairs where both molecules were
measured more than once. It then says how many of those have a gap smaller than the two
spreads added together.

| definition of an AC | gap | multiple of median repeat spread | pairs with both repeated, of which gap below their own spreads |
|---|---|---|---|
| hu2015 | 2.000 pIC50 | 12.0 | 115, of which 1 |
| calibrated to 15 per cent | 1.868 pIC50 | 11.2 | 142, of which 3 |
| moleculeace | 1.000 pIC50 | 6.0 | 136, of which 16 |
| n01 | 0.980 pIC50 | 5.9 | 596, of which 70 |

The same question on datasets with no repeats, using Kramer's 0.54 pKi as the floor. Hold
ECFP4 Tanimoto at 0.55 and the gap at twice the median absolute deviation. The share of cliff
pairs whose gap falls inside that floor is then 0.0 per cent on hERG Ki and 0.0 per cent on
MDR1-MDCK efflux. On Caco-2 it is 31.1 per cent. At one times the median absolute deviation
the three shares are 0.0, 64.0 and 30.7 per cent. Read from the three local sweep files by a
Claude session on 2026-11-04. Her thesis states the floor at line 1047: "Independent p$K_i$
measurements of the same compound against the same target differ by $0.44$ log units on
average, with a standard deviation of $0.54$ \citep{Kramer2012}".

**Not measured.** Nothing measures how the chosen share changes the noise-mitigation result,
which is the second of her two claims. Five of her six small-molecule datasets have no repeat
measurements, so the noise-floor check falls back to one published number on those.

---

## A6. TBD7, training her own cliff model on MoleculeACE

The data is on the drive. `/Volumes/seagate/moleculeace_benchmark/` holds 30 CSV files, one
per target, fetched 2026-11-02. Each file carries the fields `smiles`, `exp_mean [nM]`, `y`,
`cliff_mol`, `split` and `y [pEC50/pKi]`.

Counted from `tests/results/comparisons/opig_moleculeace_cliff_check.json`, field
`per_target`: 48,714 molecules over the 30 targets, 615 to 3,657 molecules per target. The
per-molecule cliff flag is set on 18,831 of the 48,714. Per target the share of flagged
molecules runs from 8.8 per cent (64 of 731 molecules, CHEMBL4203_Ki) to 54.0 per cent (794 of
1,471 molecules, CHEMBL4792_Ki). The median over the 30 targets is 39.2 per cent.

The shipped `split` column marks 38,912 molecules train and 9,802 test. The run read it and
discarded it, with its reason recorded in the same file's `split_column_ignored` field: "no
split is needed to compare two labellings of the same molecules, and this lab does not use
published or random splits."

**Measured on 2026-11-27, superseding what this paragraph said.** The TBD7 section above
holds the numbers from all 30 targets. `moleculeace_cliff_model.py` takes `cliff_mol` as its
target, and the representations and Bemis-Murcko scaffold groups now exist as 30 `sim_*.npz`
files on `/Volumes/seagate/moleculeace_reps` and on the server.

The labels are loadable now without the MoleculeACE package and without the network.
`cliff_rules.moleculeace_cliff_labels(csv_path)` returns SMILES, the `y [pEC50/pKi]` column
and the `cliff_mol` flag, and `opig_moleculeace_cliff_check.py --no-fetch` reads the Seagate
folder directly. `CHEMBL214_Ki.csv` holds 3,317 molecules, of which 1,222 carry `cliff_mol` set.

---

## A7. Two measured costs of the cliff units

`cliff_sets` on 1,412 molecules with 13 representations: 12,740 definitions in 15.8 minutes
for one fold, 0.33 GB of memory. Five folds is 1.3 hours. The cost is in the square of the
molecule count, so a dataset of 9,978 molecules is about 50 times that per fold unless the
settings narrow the definitions.

Two things got it there from 37 minutes a fold and 72 GB. The n-by-n similarity matrices are
dropped when the group that reads them finishes: one matrix is 8 MB at 1,412 molecules and
398 MB at 9,978, and 13 coefficients on 14 representations ask for 182 of them. And the gap
that puts a chosen share of molecules in a cliff pair is read off the sorted per-molecule
largest gap to a similar partner, which is exact, instead of bisecting a continuous interval
80 times.

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
