# Handoff: the activity cliff study

Rebuilt 2026-11-04.

Section 0 is Adelaide's own outline, pasted by her. It overrides every other line in
this file. Where anything below disagrees with section 0, section 0 is right.

Three conditions this file was written under.

Every number names the file it came from. No number here is cited from another handoff.

Nothing here is a recommendation. A verdict written by Claude into her document cannot
later be told apart from a measured result.

The new activity cliff experiments have no names. She has not named them. The names AC1
to AC5 were invented by Claude sessions and are gone from this file. S04 and S09 are her
existing similarity experiments. Section 0 calls the new work "a separate set of AC
experiments". So they are not used as names for it here.

---

## 0. Adelaide's outline, verbatim

Her instruction, 2026-11-04: "my most recent instructions should obviously take
precedent, I've sent you my version of the outline with the TBDs, that's the best source
of truth if you have doubt for AC-specific things".

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

---

## 1. The goal

This was stated back to her on 2026-11-04 and she replied: "yes all of that is right,
and the handoff should reflect that".

> The goal is two claims. That you can detect activity cliffs with something more flexible
> than ECFP4 at a fixed cutoff, because a fixed cutoff doesn't work across your datasets.
> And that detecting them makes your noise mitigation work better. Both proven, paper 3
> splits in two. Not proven, a smaller AC piece inside paper 3.
>
> The TBDs are the settings you have to fix before any of that can run. You want them
> decided one at a time, with your own numbers in front of you, by talking to me. Not by me
> building something and showing you counts.

Two claims. Nothing in this file adds a third.

Her seven TBDs are the settings. Each is hers to fix, one at a time. Sections 2 to 7 take
them in her numbering, one section each.

---

## 2. TBD1, the set of similarity coefficients

### What she asked

"look at the coefficients used in my similarity research, which show the most promise?
And how can I justify that in the paper 3 outline? It should be a set."

### Her numbers

Her own server sweep is stored at
`/Users/apunt/repos/KIRBy/tests/results/framework/f34_full_sweep/`. It covers four
datasets with 12 representations each, 16 representation names across the four, 14
coefficients and five folds. It holds 2,020 rows. In 290 of them the coefficient was
skipped as "undefined: requires non-negative input, this representation", and in 20 as
"undefined on this block".

One row below is one coefficient. The number is the best score that coefficient reaches
on that dataset, searching over every representation, neighbour weighting and neighbour
count, averaged over her five folds. The first three numbers are RMSE, lower better, in
pKi on hERG Ki and in log units on Caco-2 and MDR1-MDCK efflux. The last is AUC on hERG
blocking, higher better. The bracket is how many of that dataset's 12 representations
the coefficient is defined on. Recomputed from the stored rows on 2026-11-04.

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

The same quantity restricted to ECFP4's 2,048 bits, from the same stored rows. One row
is one coefficient. Six coefficients are defined on 0/1 features.

| coefficient | hERG Ki | Caco-2 | MDR1-MDCK efflux | hERG blocking |
|---|---|---|---|---|
| dice | 0.6095 | 0.3296 | 0.5823 | 0.8154 |
| tanimoto | 0.6095 | 0.3308 | 0.5823 | 0.8146 |
| tversky | 0.6100 | 0.3295 | 0.5851 | 0.8150 |
| cosine | 0.6109 | 0.3294 | 0.5937 | 0.8147 |
| euclidean | 0.6215 | 0.3328 | 0.6685 | 0.8153 |
| manhattan | 0.6215 | 0.3321 | 0.6685 | 0.8153 |

How far apart the best two settings are, against how far apart her folds are. On hERG Ki
under Manhattan the best setting averages 0.5604 pKi RMSE over five folds, and the
second averages 0.5611 pKi. That is a difference of 0.0007 pKi. The best setting's own
five folds run from 0.5078 to 0.6334 pKi, a range of 0.1255 pKi.

Tanimoto and Dice rank pairs identically, so only the cutoff differs. On hERG Ki, ECFP4
Dice at 0.75 and ECFP4 Tanimoto at 0.60 both call 7,254 pairs of molecules similar. Both
put 362 of the 1,412 molecules in a cliff pair. Source:
`/Volumes/seagate/cliff_census_sweep/herg_ki__sweep.json`, a local run, which that
file's own `what_it_is` field describes as "A local run of this is not a result; it
picks which definitions to carry."

Her published justification already in her thesis, `thesis.tex` line 973:
"\citet{Bajusz2015} compared eight coefficients and found Tanimoto, Dice, Cosine, and
Soergel the best and largely equivalent, while similarities derived from Euclidean and
Manhattan distances were not recommended on their own."

Two numbers from her earlier runs, counted by a Claude session on 2026-11-04 and not
re-counted here. `s01_similarity_baseline` scores Manhattan, cosine and Minkowski p=0.5
only. Its score is the Spearman correlation between a pair's distance and the difference
of its two labels. Manhattan is the best of those three on 7 of 10 representations on
hERG Ki, 8 of 10 on Caco-2 and 8 of 10 on LogD. Cosine is best on 8 of 10 on QM9. In
`/Volumes/seagate/kirby_s13_coef/`, on hERG Ki under Manhattan the hybrid's neighbours
have closer labels than atom pairs' (0.5575 against 0.5746 pKi) and under cosine they do
not (0.5753 against 0.5739 pKi).

### Not measured

No coefficient has been scored against a cliff set, which is the use TBD1 is for. Every
number above scores a coefficient by neighbour prediction, by distance against label
difference, or by clustering and retrieval.

Tanimoto, Dice, Soergel and Tversky have never been scored by the
distance-against-label- difference Spearman that `s01` and `s02` use. `f34` covers four
of her six small-molecule datasets; LogD and QM9 are not in it.

### After she fixes the set

Each coefficient she names needs a path to a cliff set. Seven scores in
`tests/experiments/noise/cliff_rules.py` already have one: Tanimoto and Dice each in a
0/1 and a count form, Tversky on 0/1 features only, Soergel on count features only, and
cosine. Eight more reach a cliff set by one route only: 1 divided by (1 plus Euclidean
distance) over every representation z-scored and concatenated. Those eight are
euclidean, manhattan, minkowski, chebyshev, correlation, canberra, braycurtis and
mahalanobis. That module marks the route `reference_independent: False`.

---

## 3. TBD2, the set of representations

### What she asked

"which representations work best for similarity? Again I have the results. And for the
love of god HYBRIDS this was WHY I DESIGNED THE HYBRIDS. It should be a set."

### Her numbers

From the same stored `f34_full_sweep`. One row is one representation. The number is the
best score it reaches on that dataset, searching over every coefficient, neighbour
weighting and neighbour count, averaged over five folds. Units as in the table above:
RMSE in pKi, log units, log units, then AUC. A dash means that representation was not
run on that dataset. Recomputed from the stored rows on 2026-11-04.

| representation | hERG Ki | Caco-2 | MDR1-MDCK efflux | hERG blocking |
|---|---|---|---|---|
| hybrid, distance objective | 0.5604 | 0.3171 | 0.4856 | 0.8456 |
| hybrid, general objective | 0.5786 | 0.3137 | 0.4862 | 0.8490 |
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

Which single representations her own kNN search ranked first and second per fold, read
from
`/Volumes/seagate/kirby_results_2026-10-10b/similarity/s04_activity_cliffs/<dataset>.json`,
field `folds[*].selection.knn_base`, which holds two names per fold. On hERG Ki
atom_pair is first on 4 of 5 folds and ecfp4 on the fifth. On MDR1-MDCK efflux pdv is
first and smited second on all five. On LogD mordred is first on 4 of 5. On QM9 smited
is first on 4 of 5 and mol2vec on the fifth. On Caco-2 pdv is first on 2 folds, grover
on 2 and ecfp4 on 1. On hERG blocking atom_pair is first on 3 of 5 and is in the top two
on all five.

Her thesis tables. `figures_ch5_7/tab_distance_tracks_activity.tex` scores each dataset
by the Spearman correlation between pair distance and absolute label difference, mean
over outer folds. The kNN hybrid is higher than the best single representation on all 14
datasets. On the six small-molecule datasets it goes hERG Ki 0.0729 to 0.2625, LogD
0.0765 to 0.2687 and QM9 0.1298 to 0.3779. The other three go MDR1-MDCK efflux 0.2286 to
0.2883, Caco-2 0.1517 to 0.1969 and hERG blocking 0.0257 to 0.0552.

`figures_ch5_7/tab_nearest_neighbour_prediction.tex` predicts a held-out molecule from
its five nearest training molecules. The hybrid is more accurate on all six
small-molecule datasets. LogD goes 0.6766 to 0.5831 RMSE in log units and hERG Ki 0.6280
to 0.5822 pKi. Caco-2 goes 0.3386 to 0.3246 log units and MDR1-MDCK efflux 0.5543 to
0.5202 log units. QM9 goes 0.6692 to 0.4205 eV and hERG blocking 0.8143 to 0.8335 AUROC.

What weighting a single representation's features by importance does to that gain, from
`thesis.tex` line 3298: "Weighting both the single representation and the hybrid shrinks
the gain wherever the hybrid wins, on LogD from 0.089 to 0.025 RMSE."

On cliff molecules the comparison reverses against a weighted single representation.
`/Volumes/seagate/kirby_s09_out/s09_cross_dataset_summary/results.json` was written
2026-09-23 on three datasets, Manhattan only. Its field
`cliff_rmse_reduction_pct_vs_weighted_rep` compares the hybrid with the best
importance-weighted single representation on cliff molecules. The hybrid is worse by
0.23 per cent of RMSE on Caco-2, 4.11 per cent on hERG Ki and 0.12 per cent on MDR1-MDCK
efflux. The same file's `cliff_rmse_reduction_pct` compares against an unweighted single
representation. There the hybrid is better by 1.14 per cent on Caco-2 and 4.29 per cent
on MDR1-MDCK efflux. It is worse by 0.89 per cent on hERG Ki.

### Not measured

A hybrid has never been used to decide which pairs are similar. Twelve of the 44
similarity criteria in `tests/experiments/noise/cliff_rules.py` read a hybrid, and none
has produced a number. In `/Volumes/seagate/ac1_smoke/herg_ki.json` all twelve sit in
`similarity_criteria_excluded` with the reason "hybrid_bits was not loaded". In
`/Volumes/seagate/cliff_census/herg_ki.json` the field `representations_loaded` is the
empty list, so all twelve were excluded again.

Every number in this section is about distance, neighbour prediction, retrieval or
clustering. In `s04_activity_cliffs` and `s12_cross_domain_cliffs` the cliff set itself
came from ECFP4: the field `cliff_definition.structural_source` reads "ECFP4 radius 2
over 2048 bits ... Tanimoto at or above 0.55". So the hybrids were scored on cliffs
ECFP4 chose.

The method her thesis used to compare representations at a fixed number of similar pairs
has its ingredient in the code and no file runs it. `thesis.tex` line 1123 states the
result: "On hERG $K_i$ the number of cliff molecules then ranges from 138 (atom pairs)
to 267 (WL graph kernel). Of the pairs that ECFP4 calls similar, only 57\% to 74\% are
also among another representation's closest pairs, and on QM9 only 1\% to 24\%."

### After she fixes the set

Two datasets in Paper 3's small-molecule list have no per-dataset representation list,
so no hybrid can be built for them as the code stands. `DATASET_REPS` in
`tests/experiments/complexity/complexity_helpers.py` holds entries for 59 datasets. It
has no entry for `herg_ic50`, `chembl_dopamine_d2`, `chembl_factor_xa` or
`postera_sars_cov_2_mpro`.

Adding a representation to a cliff criterion needs a registry entry and no new
mechanics: `sim_embedding_cosine` and `sim_rep_overlap` in `cliff_rules.py` both take a
representation name.

---

## 4. TBD3 and TBD4, the cutoff and whether it is global, per domain or per dataset

### What she asked

"I don't know what cutoff works best, and if that cutoff is per-dataset or global (or
maybe something like per-domain). See above for how can I confirm it's a cliff, that
will help me determine cut-offs"

### Her numbers

The rule her thesis uses, `thesis.tex` line 1120: "Two molecules are similar if their
ECFP4 Tanimoto similarity is at least 0.55 \citep{Hu2015} ... We chose the gap on each
dataset so that 15\% of the molecules are cliffs, so the gap follows the spread of the
labels."

One row below is one dataset. Molecules is the whole dataset. Cliff molecules is how
many sit in at least one cliff pair at a 15 per cent target share. Gap is the label
difference that share forces, in that dataset's own label units. From `thesis.tex` table
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

What one fixed pair of settings does to the share of molecules called cliffs. One row
below is one dataset. The number is the share of a fold's training molecules in at least
one cliff pair, across her five folds. ECFP4 Tanimoto is held at 0.55 and the gap at
twice the labels' median absolute deviation. Source:
`s04_activity_cliffs/<dataset>.json`, field
`cliff_definition.by_fold[*].uncalibrated_pct_cliff_compounds`. The hERG Ki row was
re-read here, over five folds of 1,129 or 1,130 training molecules. The other four were
read by a Claude session on 2026-11-04.

| dataset | share of training molecules in a cliff pair |
|---|---|
| hERG Ki | 27.4 to 32.3 per cent |
| Caco-2 | 71.3 to 73.6 per cent |
| MDR1-MDCK efflux | 3.9 to 4.4 per cent |
| LogD | 65.2 to 67.3 per cent |
| QM9 | 0.2 to 0.4 per cent |

One rule, five datasets, 0.2 per cent to 74 per cent of molecules.

Turning it round. One row below is one dataset. Both numbers are the gap that puts 15
per cent of a fold's training molecules in a cliff pair, across her five folds. Source:
the same files, field `cliff_definition.by_fold[*].activity_threshold`. The hERG Ki row
was re-read here. The other four were read by a Claude session on 2026-11-04.

| dataset | gap, in label units | gap, as a multiple of that dataset's median absolute deviation |
|---|---|---|
| hERG Ki | 1.595 to 1.750 pKi | 2.56 to 2.97 |
| Caco-2 | 1.157 to 1.196 log units | 5.35 to 5.74 |
| MDR1-MDCK efflux | 0.131 to 0.162 log units | 0.34 to 0.41 |
| LogD | 2.750 to 2.900 log units | 3.44 to 3.63 |
| QM9 | 0.030 to 0.038 eV | 0.03 to 0.04 |

The spread between folds of one dataset is small next to the spread between datasets.

What moving the similarity cutoff alone does. One row below is one dataset. Every number
is the share of that dataset's molecules in at least one cliff pair, as a percentage,
with the gap held at twice the median absolute deviation. Source: the three local sweep
files in `/Volumes/seagate/cliff_census_sweep/`. The hERG Ki row was re-read here; the
other two were read by a Claude session on 2026-11-04.

| dataset | share at 0.40 | share at 0.55 | share at 0.70 | share at 0.90 |
|---|---|---|---|---|
| hERG Ki | 50.1 | 31.9 | 12.1 | 0.6 |
| Caco-2 | 94.6 | 75.3 | 38.9 | 0.6 |
| MDR1-MDCK efflux | 19.7 | 4.7 | 0.9 | 0.0 |

On hERG Ki those four cutoffs rest on 23,611, 10,809, 2,829 and 103 similar pairs.

Where a published gap in absolute label units produces nothing. On Caco-2, with ECFP4
Tanimoto at 0.55 and a 2.0-unit gap, 0 of the 2,158 molecules are in a cliff pair, on
24,858 similar pairs. Three more rules give 0 cliff molecules on that dataset. One is the
same similarity at a 2.0-unit gap under a second name. The others are Morgan Tanimoto at
0.90 on 143 similar pairs and the RDKit fingerprint under Dice at 0.90 on 13,469 similar
pairs.
Source: `/Volumes/seagate/cliff_census/caco2.json`, field `named_rules`.

Where two definitions both give a cliff set, they disagree. One row below is one pair of
named rules on hERG Ki's 1,412 molecules. "Both" counts molecules each rule calls a
cliff molecule. Source: `/Volumes/seagate/cliff_census/herg_ki.json`, field
`named_rule_agreement`.

| first rule | second rule | both | only first | only second | Jaccard |
|---|---|---|---|---|---|
| hu2015 | n01 | 119 | 0 | 332 | 0.2639 |
| moleculeace | n01 | 227 | 89 | 224 | 0.4204 |
| geometry_morgan | n01 | 5 | 0 | 446 | 0.0111 |
| geometry_rdkit_dice | hu2015 | 43 | 3 | 76 | 0.3525 |
| hu2015 | moleculeace | 58 | 61 | 258 | 0.1538 |
| hu2015 | n14 | 119 | 0 | 0 | 1.0000 |

The last row is two names for the same two criteria.

How lopsided the similarity scale itself is, `thesis.tex` line 1133: "Of the 996,166
hERG $K_i$ pairs, 93\% have an ECFP4 Tanimoto similarity below 0.20, and only 1.1\%
reach 0.55. Over all pairs the median label difference is 0.81 p$K_i$, but above a
similarity of about 0.4 it is 0.20 to 0.30 p$K_i$, although 8.3\% of the 774 pairs at
0.8 or above still differ by more than 1 p$K_i$."

The one external cliff set that has been scored. One row below is one cliff definition,
pooled over MoleculeACE's 30 targets and 48,714 molecules, against their own
per-molecule `cliff_mol` flag, which is set on 18,831 of those molecules. Source:
`/Users/apunt/repos/KIRBy/tests/results/comparisons/opig_moleculeace_cliff_check.json`,
written 2026-11-02, 30 targets, 0 failed, 8,035.8 seconds.

| similarity | gap | our cliff molecules | their cliff molecules | both | pooled Matthews correlation |
|---|---|---|---|---|---|
| moleculeace_any_090 | absolute_1 | 18,844 | 18,831 | 18,823 | 0.99875 |
| ecfp4_tanimoto_055 | mad_2x | 19,064 | 18,831 | 11,042 | 0.31722 |
| ecfp4_tanimoto_055 | solved_target_15pct | 7,313 | 18,831 | 4,633 | 0.21315 |
| ecfp4_tanimoto_090 | absolute_1 | 1,133 | 18,831 | 1,130 | 0.19355 |

The first row is her own re-implementation of MoleculeACE's published rule against
MoleculeACE's own flag. Of the 30 targets, 18 match their flag exactly; the worst is
CHEMBL231_Ki, where her rule adds 7 molecules to their 237, Matthews correlation
0.98085.

### Not measured

No swept cutoff has been scored against anything external. `score_against_labels` in
`cliff_rules.py` is called from one file only,
`tests/experiments/opig_moleculeace_cliff_check.py`, and that file iterates the named
criteria rather than the sweep's points. Only 13 of the 44 similarity criteria reached
its pooled table, and no hybrid criterion did, because it builds a context from SMILES
and a label and loads no representations.

The sweep has never been run as a sweep for all six datasets.
`/Volumes/seagate/cliff_census_sweep/` holds three files, for hERG Ki, Caco-2 and
MDR1-MDCK efflux, all local runs.

Nothing measures a cutoff per domain. Caco-2 and MDR1-MDCK efflux are both ADME assays
in log units. The gap reaching 15 per cent of molecules is 5.35 to 5.74 times the median
absolute deviation on one and 0.34 to 0.41 times on the other. Her thesis records a
related point at line 2922: "Only the two hERG datasets measure an activity, but we use
the term on all six small-molecule datasets."

None of the three confirmation tests she names in section 0 has been run. No agreement
number against GraphCliff, SemiMol or the 2025 foundation-model comparison exists
anywhere in the repo.

### After she fixes the cutoff and its scope

A function that takes one swept point and scores it against an external cliff flag does
not exist. The sweep writes counts; the external scorer reads named criteria. Both
halves exist and nothing joins them.

---

## 5. TBD5, a separate cutoff for one hybrid overlap instead of three overlaps

### What she asked

"will definitely need a separate cutoff if I'm only trying to use one instead of 3
(hybrid overlap instead of a check of 3 overlaps)"

### Her numbers

What MoleculeACE's rule is, from
`tests/results/comparisons/opig_moleculeace_cliff_check.json`, field
`what_cliff_mol_is`: "similarity at or above 0.90 on any of ECFP Tanimoto, scaffold-ECFP
Tanimoto or scaled Levenshtein on the SMILES, with a 1-log-unit potency difference. It
is one published rule, not a reading of nature."

Her own implementation reproduces their flag exactly on one target. Source:
`/Volumes/seagate/moleculeace_benchmark/_smoke.json`, CHEMBL214_Ki, 3,317 molecules,
625.1 seconds. Her `moleculeace_any_090` with a 1-log-unit gap and their shipped
`cliff_mol` flag both mark the same 1,222 molecules. 0 molecules are only in hers and 0
only in theirs.

Her other rules on that same target, same file. One row is one definition against their
flag of 1,222 cliff molecules out of 3,317.

| similarity | gap | her cliff molecules | both | only hers | only theirs |
|---|---|---|---|---|---|
| ecfp4_tanimoto_055 | mad_2x | 1,151 | 707 | 444 | 515 |
| ecfp4_tanimoto_055 | absolute_2 | 707 | 479 | 228 | 743 |
| ecfp4_tanimoto_055 | solved_target_15pct | 498 | 355 | 143 | 867 |
| rdkit_dice_090 | absolute_2 | 389 | 270 | 119 | 952 |
| morgan_tanimoto_090 | absolute_2 | 31 | 31 | 0 | 1,191 |

What dropping two of the three scores costs in strictness, with the gap at twice the
median absolute deviation. MoleculeACE's any-of-three rule at 0.90 calls 2,594 pairs
similar on hERG Ki, 8,377 on Caco-2 and 282 on MDR1-MDCK efflux. One ECFP4 Tanimoto
score at the same 0.90 calls 103, 143 and 6 pairs similar. The single ECFP4 Tanimoto
cutoff reaching the same number of similar pairs as the any-of-three rule is about 0.70
on hERG Ki, 0.65 on Caco-2 and 0.60 on MDR1-MDCK efflux. The any-of-three rule puts 12.6
per cent of hERG Ki's molecules in a cliff pair, 48.9 per cent on Caco-2 and 2.2 per
cent on MDR1-MDCK efflux. Read from the three local sweep files in
`/Volumes/seagate/cliff_census_sweep/` by a Claude session on 2026-11-04. The hERG Ki
figure of 103 similar pairs at ECFP4 Tanimoto 0.90 was re-read here and matches.

Which of their three scores makes their cliffs, from the docstring of
`sim_moleculeace_one_score` in `cliff_rules.py` line 994. Of MoleculeACE's 1,222 cliff
molecules on CHEMBL214_Ki, 1,055 came from the string measure and 148 from the
fingerprint. That is a docstring claim; the arithmetic does not close to 1,222 and the
file does not say why.

### Not measured

Every number above is for one ECFP4 Tanimoto score, not for a hybrid overlap. No hybrid
overlap score has been computed on any dataset, for the reason under TBD2.

It cannot be scored against MoleculeACE's flag as things stand. That needs a hybrid
built on their molecules. `/Volumes/seagate/kirby_cache/similarity_metrics/sim_data/` is
empty, so no representation exists for any of their 48,714 molecules.

No sweep grid entry moves a hybrid overlap cutoff. Her agreement check ran on 1 of the
30 MoleculeACE targets.

### After she fixes the cutoff

The definition is already expressible. `sim_moleculeace_one_score` and `sim_rep_overlap`
in `cliff_rules.py` both take any cutoff, and `moleculeace_hybrid_bits_tanimoto_090` is
one of the 44 named criteria. Its docstring records that the 0.90 is not theirs and is
swept separately.

---

## 6. TBD6, the share of molecules a moved gap aims at

### What she asked

"same with the cutoff question"

### Her numbers

Her thesis and her `s04` run both use 15 per cent of a fold's training molecules. The
gaps that reaches are in the table under TBD3 and TBD4.

What the other target shares cost. One row is one target share on hERG Ki's 1,412
molecules, at ECFP4 Tanimoto 0.55. Gap is in pKi. Source:
`/Volumes/seagate/cliff_census_sweep/herg_ki__sweep.json`, a local run.

| target share | gap | cliff molecules | achieved share |
|---|---|---|---|
| 5 per cent | 2.274 | 71 | 5.03 per cent |
| 10 per cent | 1.970 | 143 | 10.13 per cent |
| 15 per cent | 1.760 | 211 | 14.94 per cent |
| 20 per cent | 1.554 | 283 | 20.04 per cent |
| 25 per cent | 1.408 | 352 | 24.93 per cent |
| 30 per cent | 1.260 | 423 | 29.96 per cent |

The same six target shares on her other two swept datasets. One row below is one
dataset. Every number is a gap in that dataset's own label units, and each column heading
is a target share of that dataset's molecules. Read by a Claude session on 2026-11-04 and
not re-read here.

| dataset | 5 per cent | 10 per cent | 15 per cent | 20 per cent | 25 per cent | 30 per cent |
|---|---|---|---|---|---|---|
| Caco-2, log units | 1.54 | 1.34 | 1.21 | 1.11 | 1.04 | 0.98 |
| MDR1-MDCK efflux, log units | 0.76 | 0.41 | 0.20 | 0.08 | 0.00 | 0.00 |

On MDR1-MDCK efflux the gap runs out at about 25 per cent. Only 653 of its 2,641
molecules, 24.7 per cent, sit in a similar pair at all.

The share a held-out molecule is judged by is not the share the gap was fitted to. Her
own record, `THESIS_OUTLINE.md` lines 2269 to 2273, marked CHECKED 2026-10-09: "the
cliff rule aims at 15% of training molecules. Held-out molecules come out at about 40%:
logd folds 39.1% to 44.5%, herg 38% to 46%, caco2 28% to 33%, herg_ki 19% to 27%. The
held-out rule is 'at least two similar training molecules differ by more than the gap'.
It is looser than the training rule. So 'cliff molecule' means a different thing in
training and held-out molecules."

Her open decision on that, `THESIS_OUTLINE.md` lines 2274 to 2275, verbatim: "[DECIDE:
the held-out cliff rule. Keep it and state the two shares, which changes text only. Or
apply the training rule to held-out molecules, which needs s04 and s12 re-run.]"

Where a low share runs out of molecules to score. `thesis.tex` line 3314: "QM9 and
MDR1--MDCK efflux have too few cliff molecules to judge, 48 and 49 in all, and only two
folds on each hold the 10 needed for a score." And line 3309: "The antibody datasets
have too few cliff antibodies to score, for example 11 of the 429 held-out antibodies on
$\Delta\Delta G$."

Whether a chosen gap clears the assay's own noise, on the one dataset of hers with
repeat measurements. `/Volumes/seagate/cliff_gap_vs_repeats/herg_ic50.json`: hERG IC50
holds 12,008 measurements over 9,802 molecules, 1,218 molecules measured more than once,
and 652 of those have repeats that disagree at all. Over those 652, the spread of one
molecule's own repeats in pIC50 has a median of 0.166, quartiles 0.050 and 0.450, and a
95th percentile of 1.608. Over all 1,218 repeated molecules the median spread is 0.0014
pIC50, because 566 agree exactly.

One row below is one rule in that file. The multiple is the gap divided by the median
repeat spread of the 652 disagreeing molecules. The last column counts cliff pairs where
both molecules were measured more than once. It then says how many of those have a gap
smaller than the two spreads added together.

| rule | gap | multiple of median repeat spread | pairs with both repeated, of which gap below their own spreads |
|---|---|---|---|
| hu2015 | 2.000 pIC50 | 12.0 | 115, of which 1 |
| calibrated to 15 per cent | 1.868 pIC50 | 11.2 | 142, of which 3 |
| moleculeace | 1.000 pIC50 | 6.0 | 136, of which 16 |
| n01 | 0.980 pIC50 | 5.9 | 596, of which 70 |

The same question on datasets with no repeats, using Kramer's 0.54 pKi as the floor.
Hold ECFP4 Tanimoto at 0.55 and the gap at twice the median absolute deviation. The
share of cliff pairs whose gap falls inside that floor is then 0.0 per cent on hERG Ki
and 0.0 per cent on MDR1-MDCK efflux. On Caco-2 it is 31.1 per cent. At one times the
median absolute deviation the three shares are 0.0, 64.0 and 30.7 per cent. Read from
the three local sweep files by a Claude session on 2026-11-04 and not re-read here. Her
thesis states the floor at line 1047: "Independent p$K_i$ measurements of the same
compound against the same target differ by $0.44$ log units on average, with a standard
deviation of $0.54$ \citep{Kramer2012}".

### Not measured

Nothing measures how the chosen share changes the noise-mitigation result, which is the
second of her two claims.

Whether the share is set on training molecules, held-out molecules or both is her open
DECIDE item above.

Five of her six small-molecule datasets have no repeat measurements, so the noise-floor
check falls back to one published number on those.

### After she fixes the share

`gap_solved_to_target` exists in `cliff_rules.py` and `gap_grid` in `cliff_census.py`
already sweeps the target at 5, 10, 15, 20, 25 and 30 per cent of molecules. It reports
the achieved share rather than the target, because the bisection reaches the target only
approximately, and its docstring states that every caller must report the achieved
share. The one missing piece is the same scoring function against an external label
named under TBD3 and TBD4.

---

## 7. TBD7, training her own cliff model on MoleculeACE

### What she asked

"is this a reasonable task? Is this straightforward?"

### Her numbers

The data is on the drive. `/Volumes/seagate/moleculeace_benchmark/` holds 30 CSV files,
one per target, fetched 2026-11-02. Each file carries the fields `smiles`, `exp_mean
[nM]`, `y`, `cliff_mol`, `split` and `y [pEC50/pKi]`.

Counted from `tests/results/comparisons/opig_moleculeace_cliff_check.json`, field
`per_target`: 48,714 molecules over the 30 targets, 615 to 3,657 molecules per target.
The per-molecule cliff flag is set on 18,831 of the 48,714. Per target the share of
flagged molecules runs from 8.8 per cent (64 of 731 molecules, CHEMBL4203_Ki) to 54.0
per cent (794 of 1,471 molecules, CHEMBL4792_Ki). The median over the 30 targets is 39.2
per cent.

What the flag is: one published rule, quoted under TBD5.

The shipped `split` column marks 38,912 molecules train and 9,802 test. The run read it
and discarded it, with its reason recorded in the same file's `split_column_ignored`
field: "no split is needed to compare two labellings of the same molecules, and this lab
does not use published or random splits."

### Not measured

Nothing has been trained. No classifier anywhere in the repository takes `cliff_mol` as
a target.

The shipped split is a published split and her pipeline discards it. No scaffold-grouped
folds of MoleculeACE exist in the repo.

There are no representations for those 30 targets' molecules, and no Bemis-Murcko
scaffold groups. `tests/experiments/similarity_metrics/build_moleculeace.py` finds its
CSVs only through the installed package or `$MOLECULEACE_DATA`, and the `MoleculeACE`
package is not installed: `cliff_rules.moleculeace_state()` returns
`ModuleNotFoundError: No module named 'MoleculeACE'`.

There is no baseline. Nothing records what error a model reaches on that flag, so there
is no number a trained model would be compared against.

### After she decides whether to do it

The labels are loadable now without the package and without the network.
`cliff_rules.moleculeace_cliff_labels(csv_path)` returns SMILES, the `y [pEC50/pKi]`
column and the `cliff_mol` flag, and `opig_moleculeace_cliff_check.py --no-fetch` reads
the Seagate folder directly. `CHEMBL214_Ki.csv` holds 3,317 rows, of which 1,222 carry
`cliff_mol` set.

---

## 8. The two tasks she assigned

Both are in section 0, in her words. Neither is a question for a chat reply, and neither
has been done.

### "Help me identify published methods"

A literature search. Its scope is what the published literature uses to decide that a
pair is a real activity cliff rather than a measurement error. That covers the
similarity measures and cutoffs, the potency gaps, the repeat-measurement and curation
rules, and any prospective confirmation. On 2026-11-04 a Claude session reduced this to
a two-option question in a chat reply, and then, after she objected, started a
literature search without being told to. Neither was right. It runs when she says so.

What already sits in the code as a published source, from the module records in
`cliff_rules.py`. One line is one source.

- Hu, Furtmann, Stumpfe and Bajorath 2015, for ECFP4 Tanimoto at 0.55.
- van Tilborg, Alenicheva and Grisoni 2022, for the any-of-three rule at 0.90 with a 1-log-unit gap.
- Dablander et al. 2023, for a matched pair with a 2.0-unit gap.
- Kramer, Kalliokoski, Gedeck and Vulpetti 2012, for a 0.54 pKi repeat spread.
- A 2605 arXiv preprint, recorded second-hand through a research session with no paper read.
- One gap rule attributed to the Bajorath group by a research session, marked unverified in the code.

### "Help me brainstorm anything else"

A brainstorm of further ways to decide that a pair is a cliff. Material has accumulated,
none of it hers and none of it decided. One line is one candidate.

- Assay type as a filter before detection.
- Whether any one dataset mixes IC50, Ki and Kd internally. Unchecked.
- Censored values, and whether any dataset ships them into the cliff calculation. Unchecked.
- Restricting cliff pairs to molecules measured in the same document, which hERG IC50 records.
- Stereoisomer pairs, a criterion in the code that finds 0 pairs on all 30 MoleculeACE targets.
- Crossing a cliff mask with the rows the noise experiments corrupted on purpose.
- Dablander's band of 1 to 2 log units dropped from both sides, a third gap option not in her list of two.
- The ChEMBL curator flag, already run, which covers 6 molecules in all of hERG IC50.

It runs when she says so.

---

## 9. Her other open items

Her list of confirmation tests is not closed. In section 0 she writes "Actually two
tests I can run (and should run):" and then lists three, and then asks for the
brainstorm. A Claude session treating it as a closed set of three is what made it ask
her to pick one of the three.

The per-molecule cliff counts scattered through the old experiments. Her words: "Random
stats from experiments: - How many molecules set in at least one cliff pair" and "I want
to get rid of these numbers (see the re-done noise experiments)". A Claude session built
a 310-row table of exactly those counts and reported it to her as progress.

Whether the definition is recomputed per fold. Her words, recorded 2026-11-04: "So
doesn't it just need to be recomputed per fold? So there isn't a dataset wide activity
difference threshold?"

Whether a binary hybrid takes only 0/1 features, or also counts such as the graph
kernel's. No quote is recorded. `tests/experiments/cliffs/binary_hybrid.py` builds both
restricted lists already.

Whether Paper 3 adopts MoleculeACE's 1-log-unit repeat-spread curation rule. No quote is
recorded. It changes every count downstream.

Whether the Tversky weights move off 0.7 and 0.3. No quote is recorded. Those two
weights come from `simeval.dist_tversky` and the score is averaged over both orders. At
0.5 and 0.5 the formula is Dice.

Whether `mmpdb` gets installed. Her list includes "mmpdb (one named chemical change)".
`cliff_rules.mmp_state()` returns `mmpdb_installed: False`.

Whether rat microsomal clearance's uncached representations get built, and whether the
QM9 download path gets fixed. Both block classifying those datasets. No quote is
recorded for either.

---

## 9b. What was built on 2026-11-04 evening

A new experiment category, `cliffs`. It is a category of its own because section 0 says
"Make this a separate set of AC experiments"; s04 and s12 are untouched.

Every one of the seven settings is a named input with no value written in the code. They
live in one file, `tests/experiments/cliffs/CLIFF_SETTINGS.json`, which carries her TBD
number and her own sentence beside each one. A unit whose setting has no value prints
which setting it waited for and writes a skip marker, so the ledger holds it and starts it
by itself once the file says. Three have no value as it stands: TBD7, and the two methods
below.

`"all"` on an axis means every value the code can run. That is what the file ships with on
nine of the twelve axes, because on those running every value covers the question rather
than choosing. It is not a choice made here.

### The four units, in the order the ledger runs them

One line is one unit. Each is a unit of the Papers 2 and 3 ledger on every molecular and
antibody dataset in Paper 3's list.

- `cliff_sets` — which molecules are in a cliff pair, under every definition, per fold.
  It writes the per-molecule flags the others read.
- `definition_agreement` — each definition against a cliff set it did not produce.
- `error_by_cliff_status` — held-out error on cliff molecules against non-cliff molecules.
- `cliff_aware_cleanup` — training on D, D' and D''.

Two more are scripts rather than per-dataset units, because their data is 30 targets and
not one dataset: `cliffs/moleculeace_agreement.py` and
`cliffs/moleculeace_cliff_model.py`.

### What a definition is now, which is what TBD1 and TBD2 needed

A coefficient on one named representation, cut either at a fixed value or at a share of
pairs. Before today a coefficient could reach a cliff set only as an overlap score
(tanimoto, dice, tversky, soergel) or as 1/(1+distance) on every representation glued
together. The f34 sweep's best coefficients on all four datasets it covers are manhattan,
minkowski and euclidean, and none of them is an overlap score, so none of them could name
a single similar pair on a single representation. `cliff_rules.coefficient_criterion` is
the one door now, and thirteen coefficients go through it: braycurtis, canberra,
chebyshev, correlation, cosine, dice, euclidean, manhattan, minkowski p=0.3, minkowski
p=0.5, soergel, tanimoto and tversky.

A fixed cutoff on an overlap score or on cosine means the same thing on two
representations of different widths. On 1/(1+distance) it does not, because the distance
grows with the number of features. Every row carries `cutoff_is_bounded` for that reason,
and the share-of-pairs form is on every coefficient.

### The hybrid defines a cliff, per fold

Each fold's two restricted hybrids are searched on that fold's training molecules only,
through the same `_shared.fold_searches` door the rest of the suite uses, and handed to
the criteria as named representations. One per decided model, so a dataset's two models
each contribute one. The kind of the assembled matrix is checked and not assumed.

Under `definition_scope` 'per_fold' the gap's median absolute deviation and the pairing
are both computed on the fold's training molecules, so no held-out label reaches the
definition. 'whole_dataset' is what the thesis runs did and still runs. Both values run.

### Two measured costs

`cliff_sets` on 1,412 molecules with 13 representations: 12,740 definitions in 15.8
minutes for one fold, 0.33 GB of memory. Five folds is 1.3 hours. The cost is in the
square of the molecule count, so a dataset of 9,978 molecules is about 50 times that per
fold unless the settings narrow the definitions.

Two things got it there from 37 minutes a fold and 72 GB:

The n-by-n similarity matrices are dropped when the group that reads them finishes. One
matrix is 8 MB at 1,412 molecules and 398 MB at 9,978, and 13 coefficients on 14
representations ask for 182 of them.

The gap that puts a chosen share of molecules in a cliff pair is now read off the sorted
per-molecule largest gap to a similar partner, which is exact, instead of bisecting a
continuous interval 80 times. `cliff_rules.gap_curve` returns the whole share-against-gap
curve for the price of one target.

`tests/experiments/cliffs/_smoke_definitions.py` checks 1,140 definitions both ways —
nine counts and the per-molecule flag each, on a regression label, on a dataset with three
rows of one molecule, and on a class label — so the fast path is checked against
`cliff_sets` rather than taken on trust.

### The two settings the downstream test waits on

`cliff_aware_cleanup` needs her M and her L: "Look at the best outlier detection methods,
call one M, and then subsequent best label handling methods, call one L". Both are "the
best" of the noise experiments' own results. It names them and stops.

It also prints the number of model fits the settings ask for before fitting anything, and
stops above 20,000 rather than starting. Nothing is dropped when it stops: the message
says which settings multiply the number.

### MoleculeACE

`build_moleculeace.py` takes `--data-dir` as of today. The 30 CSVs have been on the drive
since 2026-11-02 and that file could previously find them only through the installed
MoleculeACE package, which is not installed. So the representations and our own
Bemis-Murcko scaffold groups can be built for all 30 targets without the package and
without the network, which is what the hybrid criteria and any scaffold-grouped fold of
those targets needed.

`moleculeace_cliff_model.py` ran on one target as a crash check, which is a local run and
not a result. On CHEMBL4203_Ki, 731 molecules over 5 scaffold-grouped folds: their own
rule re-implemented reaches Matthews correlation 1.0000 against their flag, every molecule
called not-a-cliff reaches 0.0000, lightgbm on the descriptor block has a median of 0.2902
over the five folds and ECFP4 Tanimoto at 0.55 with twice the median absolute deviation
has 0.1663. It took 151 seconds for one target.

### Her mmpdb entry now reaches a cliff set on the three datasets that have one

Her outline lists "mmpdb (one named chemical change)" as one of the ways a pair counts as
similar. `cliff_rules.sim_mmp` reads a pair list the loader ships, the three ChEMBL targets
of Dablander et al. 2023 ship one, and the pipeline loader was dropping it:
`find_best_hybrid_molecular.load_dataset` took the SMILES, the labels and the task type and
nothing else, so `matched_molecular_pairs` was excluded on every dataset with the recorded
reason "the loader shipped no matched pairs".

The pairs are now carried through, past the InChIKey dedup. The dedup drops rows and the
pairs are row indices, so they are remapped against the kept rows and a pair whose either
molecule was dropped is removed and counted rather than left pointing at whatever now sits
at that index. `deduplicate_by_inchikey` takes `return_keep` for that, and
`_load_data_impl` refuses a pair naming a row outside the ones the loader returned.

On `chembl_qsar_d2`, 6,333 molecules: all 35,484 matched pairs arrive, 0 dropped by the
dedup, every index inside the rows. The `dablander` rule then gives 456 cliff pairs and 476
cliff molecules, with 3,806 pairs in the 1-to-2-log-unit band dropped from both sides.

Two things disagree by 4 pairs and both are on disk. The loader prints "460 activity cliffs
(|delta pActivity| >= 2.0)" and the rule here counts 456. Of the 35,484 pairs, 4 have a
potency difference of exactly 2.000: `cliff_rules._far` is a strict `gap > cutoff` and the
loader's count is at or above. So the two differ in one thing, whether a pair exactly at the
published threshold is a cliff, and in nothing else. Which one Paper 3 uses is not decided
here.

### The published cliff-prediction models

`definition_agreement` reads a per-molecule cliff flag from
`/Volumes/seagate/external_cliff_flags/<source>__<dataset>.json`, matched by SMILES and
not by row order, and scores every definition against it. No file exists for GraphCliff,
SemiMol or the 2025 foundation-model comparison, so every run records which were looked
for and not found, and no row against them is written.

### Two faults in the full-run code, fixed

The cliff cutoff written down was not the cutoff used. `n42_similarity_noise_sweep.py`
reused the whole-dataset similarity matrix without passing `structure_info`, so
`n25_cliff_exemption.py` fell back to cutting at the closest 2 per cent of pairs while the
cache key recorded `sim_threshold: 0.55`. n25's own docstring says `structure_info` was
added on 2026-10-06 to prevent this and n25 passes it to itself. `n44_detect_and_clean.py`
called the same function the same way. Both now pass it, and the cutoff actually used is in
the cache key, so a mask stored under the old behaviour is not served for the new one.

The random put-back's shortfall was discarded. `n25.random_exempt_mask` returns how many
non-cliff molecules it managed to draw and draws fewer when there are not enough; both
callers wrote `rand, _n = ...`. Both now record the shortfall per pass, so a pass whose
control arm is smaller than its cliff arm can be told from one where they match.

### Two files no longer carry a name she did not give

`cliffs/ac1_definition_counts.py` and `cliffs/_smoke_ac1.py` are now
`whole_dataset_definition_counts.py` and `_smoke_whole_dataset_definition_counts.py`. The
first computes the same counts as `cliff_sets` over all the molecules at once, where a
hybrid's features were chosen using every label including the held-out ones. Whether it is
kept is hers; it has still never produced output.

### Two smoke files were already broken before today

`noise/_smoke_n42.py` and `noise/_smoke_n25.py` both fail at
`AttributeError: module ... has no attribute 'get_best_search_for_downstream'`. Checked by
running them against the unmodified files: the failure is the same before and after
today's change.

---

## 10. The state of the code

Everything in this section was read in the repository on 2026-11-04.

### What holds the definitions

`tests/experiments/noise/cliff_rules.py` holds 44 similarity criteria, 10 gap criteria
and 10 named pairings of one of each. Counted by importing the module. Twelve of the 44
read a hybrid.

How the 44 similarity criteria group. One line is one group, and the number counts
criteria.

- 8 put an absolute cutoff on a fingerprint overlap.
- 4 take a share of pairs on ECFP4.
- 1 uses a neighbour count on ECFP4.
- 6 take a share of pairs on every representation z-scored and concatenated.
- 12 put an overlap score on `hybrid_bits` or `hybrid_counts`.
- 3 put cosine on one learned representation, as a share of pairs.
- 1 reads a supplied list of molecule pairs.
- 1 is on stereochemistry.
- 9 are on sequence edit distance, for the antibody sets.

Three pairs of criteria compute the same thing under two names. `morgan_tanimoto_090`
and `ecfp4_tanimoto_090` give identical molecule counts and identical Matthews
correlations at every gap over the 30 MoleculeACE targets: 1,133 molecules and 0.19355
at a 1-unit gap. The named rules `hu2015` and `n14` are the same two criteria, and reach
Jaccard 1.0000 on hERG Ki. `absolute_2` and `absolute_2_half_cliffs_dropped` give
identical counts over the 30 targets. The second returns the cliff side only. It counts
the 1-to-2-unit band separately and does not remove it from the other side.

`stereoisomer_pairs` finds 0 pairs on every one of the 30 MoleculeACE targets.

### What the sweep varies

`cliff_census.sweep()` in `tests/experiments/noise/cliff_census.py` carries 75
similarity points and 28 gap points, counted by running the two grid builders. That is
2,100 rows per dataset, which matches the 2,100 rows in
`/Volumes/seagate/cliff_census_sweep/herg_ki__sweep.json`. Two further grids add 28
points when representations are loaded and 19 when the dataset has sequences.

What it moves on the similarity side is four fingerprint scores at 14 cutoffs each and
MoleculeACE's union at 6. It also moves a share of ECFP4 pairs at 7 points and a
neighbour count at 6. It varies no overlap score on a hybrid or on any one named
representation. It varies no cosine on a learned representation, no matched pairs, no
stereoisomer pairs, no Tversky and no Soergel.

Per point it writes pair counts, molecule counts, the share of molecules and the gap in
label units. It also writes the gap over that dataset's own measured repeat spread,
where the loader exposes repeats, and the gap over 0.54 pKi. No model is fitted. No fold
is held out. No external label is read.

### What scores a definition against an external label

One file does: `tests/experiments/opig_moleculeace_cliff_check.py`.
`cliff_rules.score_against_labels` is called nowhere else in `tests/` or `src/`.

It iterates the named criteria rather than the sweep's points, so no swept cutoff has
been scored. It builds a context from SMILES and a label and loads no representations,
so 13 of the 44 criteria reached its pooled table and no hybrid criterion did.

### D, D' and D'' in `n44_detect_and_clean.py`

Every row it writes carries three scores of the same held-out fold, in that dataset's
own metric. They are listed in `d_vs_dprime.ROW_FIELDS`. `score_D` is fitted on the
damaged training labels. `score_Dprime` is fitted on what a cleanup returned.
`score_clean` is fitted on the clean training labels. Two further runs exist:
`n42._Unit.emit` writes rows with the cliff molecules restored and with as many
non-cliff molecules restored at random.

One cliff definition is reachable, and it cannot be swapped.
`n42_similarity_noise_sweep.py` line 1117 calls `n25.cliff_mask_for_dataset` with a
fixed 0.55 Tanimoto cutoff and a fixed multiple of 2.0. Neither
`n44_detect_and_clean.py` nor `n42_similarity_noise_sweep.py` imports `cliff_rules`. So
none of the 44 similarity criteria and none of the 10 gap criteria can be reached from
there, and no two definitions can be compared.

The cutoff recorded and the cutoff used are not the same. That call at line 1117 passes
`structure=` and no `structure_info=`. Without `structure_info`,
`n25_cliff_exemption.py` line 746 sets the share of pairs to `n01.SIMILAR_PERCENTILE`,
which is 2.0. Line 835 then takes the cutoff as a percentile of the distribution instead
of 0.55. The cache key still records `sim_threshold: 0.55`. So the cliff set is the
closest 2 per cent of pairs while the setting written down is 0.55. Two different
cutoffs, one of them recorded. `n25`'s own docstring says `structure_info` was added on
2026-10-06 to prevent this.

The random draw's shortfall is discarded. `n25.random_exempt_mask` returns the mask and
the number of molecules it managed to draw, and draws fewer when there are not enough
non-cliff molecules. Line 1120 of `n42_similarity_noise_sweep.py` writes `rand, _n =
n25.random_exempt_mask(...)` and throws the count away.

No row can carry how many molecules were spared. `d_vs_dprime.make_row` raises on any
field outside `ROW_FIELDS` and `OPTIONAL_ROW_FIELDS`, and neither list holds a spared
count or a definition name. `n25.n_spared` exists and is not called.

`cliff_rules.subset_context` is called by nothing but its own smoke file,
`tests/experiments/noise/_smoke_subset_context.py`.

### Matched molecular pairs

`mmpdb` is not installed. `cliff_rules.mmp_state()` returns `mmpdb_installed: False` and
`can_generate_pairs: False`. It also records what cannot be recovered from a shipped
pair list. The literature restricts a matched pair so that the exchanged fragment is
small relative to the molecule. That restriction is a parameter of mmpdb's fragment
generation.

Three datasets ship a pair list, all six files cached in
`/Users/apunt/.cache/kirby/chembl_qsar/`. One row is one dataset.

| dataset | molecules | rows in the pair file |
|---|---|---|
| chembl_dopamine_d2 | 6,333 | 35,484 |
| chembl_factor_xa | 3,605 | 21,292 |
| postera_sars_cov_2_mpro | 1,924 | 12,594 |

The pipeline loader drops them. `tests/find_best_hybrid_molecular.py` takes `smiles`,
`labels` and `task_type` from `load_chembl_qsar` and nothing else, and the string
`mmp_pairs` appears in neither that file nor `tests/experiments/_shared.py`. So
`matched_molecular_pairs` is excluded for every dataset, with the recorded reason "the
loader shipped no matched pairs", and the named rule `dablander` built on it is
unavailable everywhere. One file receives the pairs:
`tests/experiments/opig_activity_cliff_qsar.py`, which calls `load_chembl_qsar`
directly.

### Two files on disk carry a name she did not give

`tests/experiments/cliffs/ac1_definition_counts.py` and
`tests/experiments/cliffs/_smoke_ac1.py`. The name came from a Claude session. That file
has never produced output; `/Volumes/seagate/ac1_definition_counts` does not exist. Its
hybrid rows would carry `similarity_used_labels: True`. A hybrid's features are chosen
using the labels. The gap criterion then reads the same labels over the same molecules,
with no fold held out.

### Two other recorded code facts

`n25_cliff_exemption.py` refuses to run on a dataset with no structural reference
outside the representations, as of 2026-11-04. Until that date it built the cliff set
from Euclidean distance over every representation glued together and ran anyway.

`cliff_rules.py` records two timings. Computing Tanimoto through matrix multiplication
rather than scipy's `jaccard` took 1.75 seconds against 214.61 on CHEMBL214_Ki's 3,317
molecules at 2,048 features, with the two matrices bit-identical. Building the generic
framework of hERG Ki's 1,412 molecules takes 1,410 seconds.

---

## 11. Where the numbers live

Her own five-fold results: `/Volumes/seagate/kirby_results_2026-10-10b/similarity/`,
holding s01, s02, s03, s04, s05, s06, s08, s10, s11, s12 and s13, written 7 to 10
October. The thesis tables were built from this folder.

Her server sweep, in the repo: `tests/results/framework/f34_full_sweep/` and
`tests/results/framework/f34_importance_sweep/`.

Her thesis: `thesis.tex`, with the cliff material in `sec:flaws:similarity` and the
tables in `figures_ch5_7/`.

The external comparison, in the repo:
`tests/results/comparisons/opig_moleculeace_cliff_check.json`.

Local runs written by Claude sessions on 2 to 4 November, which are debug runs and not
results: `/Volumes/seagate/cliff_census/`, `/Volumes/seagate/cliff_census_sweep/`,
`/Volumes/seagate/cliff_gap_vs_repeats/`, `/Volumes/seagate/cliff_curator_flag/` and
`/Volumes/seagate/ac1_smoke/`.

MoleculeACE's own data: `/Volumes/seagate/moleculeace_benchmark/`, 30 CSVs plus a run
log.

---

## 12. Two places her own records disagree

The fingerprint-against-structure correlation. `THESIS_OUTLINE.md` lines 915 to 916
record "maximum-common-substructure distance at Spearman 0.043 on herg_ki; the RDKit
fingerprint at -0.334. On logd, 0.37 (ECFP4-count) and 0.46 (torsion)."
Table `tab:flaws:neighbor_overlap`
in `thesis.tex` line 1205 gives 0.20 on hERG Ki and 0.51 on LogD, for binary ECFP4. The two
use different fingerprint variants and which number belongs in which place is not
established here.

The cliff comparison flips with the comparator. Table `tab:similarity:cliffs` in
`thesis.tex` has each model's first hybrid beating its best single representation on
cliff molecules. That holds on LogD, Caco-2, hERG Ki and hERG blocking. That table's
single representation is unweighted.
`/Volumes/seagate/kirby_s09_out/s09_cross_dataset_summary/results.json`, written
2026-09-23 on three datasets with Manhattan only, has the hybrid losing to the best
importance-weighted single representation on cliff molecules on all three.
