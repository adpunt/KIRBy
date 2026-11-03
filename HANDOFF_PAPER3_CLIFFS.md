# Handoff: activity cliffs for Paper 3

Version 3, 2026-11-02. Supersedes version 2 of the same day, which listed the same work in
four places — an 18-item list, a register, nine numbered measurements, and a decisions
section — so nothing could be read off it. Here every piece of work is one row of one
table, in section A. Everything after that is reference material for doing it, and
contains no tasks.

Three issues, and they are separate.

0. **Telling a cliff apart from a wrong label.** The question of 2026-11-01, and the one
   the first two versions of this document lost. Section C. Nine options, none decided.

1. **How a cliff is computed.** Not settled. Both halves are open: what counts as similar,
   what counts as a large label difference.
2. **Where cliffs enter the restructured noise experiments.** Six places. Each needs to be
   told which computation from issue 1 it uses.

## How to work this document

Three rules, written because the session of 2026-11-02 broke all three.

**Nothing lives in a chat reply.** If Adelaide raises something, add a row to section A
before writing the reply, with how it gets done and who owns it. Thirteen things were
raised in a reply and lost, which she had to count herself.

**Do the measurement rather than offer it.** If a measurement is cheap and section A asks
for it, run it. Only a scope cut, a dependency to install, or a dataset to drop needs her
first.

**One item at a time, and say which row.** Take the lowest-numbered row that is not
blocked, do it, report that one, stop. Do not bring in a result from another thread, such
as s08 or s10, unless that row needs it.

---

# A. The work

One row is one piece of work. Nothing is listed anywhere else in this document. Rows 1 to
4 are measurement and must finish before row 5, because row 5 is chosen from them.

| # | What | How | Owner | State |
|---|---|---|---|---|
| 1 | The MoleculeACE check: how many of its `cliff_mol` molecules each of our rules recovers, over its 30 targets | `tests/experiments/opig_moleculeace_cliff_check.py` | this thread | **done 2026-11-02**. 30 targets, 48,714 molecules, 2h14m, none failed. Our `moleculeace` rule recovers 18,823 of their 18,831 cliff molecules and adds 21, Matthews 0.9987 pooled and exactly 1.0000 on 18 of the 30 targets, so the implementation is right. The thesis's own rule, ECFP4 Tanimoto 0.55 with 2.0 median absolute deviations, lands at 0.3172 pooled and 0.0169 on its worst target while calling 19,064 molecules against their 18,831 — nearly the same count, largely different molecules. Of 120 pairings that produced a correlation, 56 anti-correlate with MoleculeACE on at least one target. **This is agreement between two definitions, not a ranking: `cliff_mol` is MoleculeACE's rule, not nature.** Written up in `CLIFF_DEFINITIONS_2026-11.md` |
| 2 | The threshold sweep. Every cutoff in `cliff_rules.py` was copied from this repo or a paper; three or four points on an axis is not a search | Section E's grid, through `cliff_census.py` with each cutoff swept. Nearly free: one similarity matrix is built once and thresholded many times | this thread | **done** on hERG Ki, Caco-2, MDR1-MDCK, 2026-11-02. 2,100 settings each, 268s / 965s / 1,094s, nothing errored. Findings file section 3c. Headline: at ECFP4 Tanimoto 0.90 the three datasets have 103, 143 and 6 similar pairs, so the 15% share is unreachable and the gap returns 0.000 |
| 3 | Each dataset's own repeat spread, in label units, so a gap can be a multiple of what that assay repeats to instead of Kramer's borrowed 0.54 pKi | `noise/_ground_truth_providers.build_replicate_groups:164` and `compute_replicate_stds:201`. A dataset with no repeats gets no number and is reported as having none | this thread | **built and run**, 2026-11-02. All three datasets expose no repeat structure to the loader, so all three report none and fall back to Kramer's 0.54, which is also the wrong unit for Caco-2 (log permeability) and MDR1-MDCK (log efflux ratio). A dataset with repeats is still needed to settle section C |
| 4 | The census on every remaining dataset: counts, achieved shares, where the solved gap misses its target, agreement between rules, and the share of each rule's cliff pairs inside the noise floor | `cliff_census.py <dataset>`. Three datasets done (hERG Ki, Caco-2, MDR1-MDCK) but their `moleculeace` row is wrong on disk and must be recomputed after the scaffold fix. QM9 needs the similarity matrix chunked: 9,978 molecules is 400 MB a matrix and several are held at once | this thread | 3 of N datasets, and those three partly wrong |
| 5 | **Pick the rule**: one rule, or two or three reported side by side, or a sweep over rules | From rows 1 to 4. Nothing below row 8 starts until this is picked | **Adelaide** | blocked on 1–4 |
| 6 | **Decide which feature spaces cliffs are computed in**: ECFP4 alone, or ECFP4 plus the best single representation and the best hybrid; and whether features are multiplied by their importance first (section G) | — | **Adelaide** | open |
| 7 | **Decide the lead unit, pair or molecule.** Both are in use and they are not interchangeable | Then every sweep point gets reported both ways | **Adelaide** | open |
| 8 | Read every file that mentions a cliff and record per file which rule it builds or only names | 84 files, 67 of them Python, not the 54 of version 2 | this thread | **done 2026-11-02**, written up in `CLIFF_DEFINITIONS_2026-11.md` section 1. Eleven files build a rule, six call one built elsewhere, fifty name something without defining it. Five findings, in the two rows below and in section B |
| 8a | The two target shares in `similarity_helpers.py`. The similarity cutoff was tuned to put 10 per cent of molecules in the cliff set and the gap was then solved to put 15 per cent in, so the second undid the first | **Decided by Adelaide 2026-11-02: the quotas move together.** One constant, `CLIFF_TARGET_PCT`, read by both sites. The pair of cutoffs that hits a share is not unique, so the order is fixed and recorded in the code: the similarity cutoff first under the standard gap, then the gap solved for what is left. Two paths were affected, both the ones with no ECFP4 — the antibody sequence rule (`similarity_helpers.py:875`) and s12's concatenation rule (`s12:180`) | done | **done 2026-11-02**. Any stored s04 or s12 number from before this used the old 10-then-15 pair |
| 8b | Add the criteria the suite runs and the module lacked, so row 2's sweep covers what s01, s02 and s12 actually compute | Three gaps, not two. The concatenation rule was in the module under Euclidean only, and s12 runs it under Manhattan, cosine and Minkowski p=0.5 as well. The sequence rule was at 5 edits only, and s01 and s02 sweep 1, 2, 3 and 5. The CDR-only scope was absent altogether | this thread | **done 2026-11-02**. `cliff_rules.SIMILARITY_CRITERIA` goes from 21 criteria to 32. Checked on made-up sequences: 1 edit gives 5 pairs and 2 edits gives 15, which is the 6 near-duplicates all within 2 edits |
| 8c | A trap in the CDR scope, found while doing 8b. Three functions in a row fall back to the whole sequence instead of failing, and none tells its caller. `antibody.get_cdr_residue_indices` returns every index when ANARCI cannot number a sequence; `_extract_cdr_sequences` only counts a sequence as failed when that call raised, so its "more than half failed, give up" guard never fires. Measured: ANARCI is installed here, and on a random 21-residue peptide every index came back, so the CDR scope equalled the full chain for every sequence while reporting success | `cliff_rules.cdr_sequences` now counts the strings that came back unchanged, records the count, and returns None when all of them did, so the criterion is left out rather than silently duplicating the full-chain one. `antibody.py` is unchanged, because falling back to the whole sequence is right for the pooled representations it was written for. **What is still open:** s01 and s02 call the extractor directly, so on a nucleotide dataset their CDR scope is a copy of their full-chain scope and is reported as a second scope. **Fixed 2026-11-02 by a second session, commit 27c85df and the one after it.** (a) `similarity_helpers._extract_cdr_sequences` now counts a chain returned whole as failed, also when read from cache, gives up past half, and stores `n_fallback_full_chain` in the result. (b) On `cdr_flexibility_l3` the label is the light-chain CDR3 loop, and both sequence scopes compared heavy chains; they now compare light chains (`LIGHT_CHAIN_LABEL_DATASETS`). That dataset has 112 rows and 111 distinct heavy chains, so heavy chains were not identical, only the wrong chain. **Measured on the loaders, 40 chains a dataset:** no chain returned whole on Koenig, AbAgym, ddg_combined, abcov, Adams or any Jain set; 2 of 40 on `flab_thermo_tresanco`, which is not in Paper 3. No nucleotide dataset is in Paper 3. Result: `/Volumes/seagate/cdr_scope_check/result.json` | this thread | **done 2026-11-02** |
| 9 | Point s04 and s12 at the chosen rule, by name from `cliff_rules.RULES` | Both call `similarity_helpers.calibrated_cliff_rule` today. They share the same two calls, so both move together or neither moves. What must not change: section F | this thread | blocked on 5 |
| 10 | Replace the two rules inside n47 with the chosen one | It uses ECFP4 at 2.0 median absolute deviations at `:202` and an absolute 2 label units at `:277`, and both write a field called `n_cliff_compounds` into the same result file | this thread | blocked on 5 |
| 11 | Build the cliff put-back modifier in n44 against the chosen rule, plus its random control | Lift from `n25 put_back` and `n25 random_exempt_mask`. The control cannot run under today's rule on Caco-2, hERG blocking or LogD (`thesis.tex:4148`); whether the chosen rule fixes that is one of row 4's numbers | this thread | blocked on 5, 6 |
| 12 | Does the cliff set move with the representation, and how far | Hold the rule fixed, load ECFP4, MACCS, RDKit fingerprint, ChemBERTa, MolFormer and Chemeleon, match the number of closest pairs as `thesis.tex:1123` did. Extends its 138-to-267 on hERG Ki to every dataset; report the Geometry preprint's 55-fold swing beside ours | this thread | **not started**; no representations have been loaded |
| 13 | How much the cliff molecules overlap what each detector flags | For each detector in list B and each rule, how many flagged molecules are cliff molecules. One table a dataset. This is the number `thesis.tex:1265` asserts and never gives | this thread | **not started**; same reason |
| 14 | **Decide whether `ActivityCliffDetector` becomes a detector in list B or is deleted as a duplicate** | `noise_mitigation_methods.py:742`. No noise experiment calls it. **Correction 2026-11-02:** it does run inside the `sali_cliff` representation (`molecular.py:1616`), with ECFP4 Tanimoto 0.5 and a gap at the 90th percentile of pairwise label differences, and any error there is silently passed. `sali_cliff` is in the full molecular pool (`find_best_hybrid_molecular.py:162`) but in no dataset's chosen set in `complexity_helpers.DATASET_REPS`, so it reaches no current result. It has two free thresholds | **Adelaide** | open |
| 15 | **Decide where s01, s02, s04, s09, s12, `simeval` and `nary_similarity` land in the restructured layout** | — | **Adelaide** | open |
| 16 | **Decide what the continuous scores are for.** Section B's third table is built and printed and nothing uses it. Either one of them defines a cliff — a cutoff on SALI, or the cliff quadrant of a SAS map — and joins the sweep, or they are reported as landscape descriptions only | Then either a new entry in `cliff_rules.RULES` or one sentence in the paper | **Adelaide** | built, unapplied |
| 17 | **Settle Tanimoto and Dice in `distance_config.py`.** | **Done 2026-11-02.** The suite's distance list goes from four names to six: `tanimoto_minmax` and `dice_minmax`, with `tanimoto`, `dice`, `soergel` and `ruzicka` as spellings of them. The min/max form is wired up, not the dot-product form, because on 0/1 features it is the ordinary Tanimoto of the two bit sets — checked against scipy's boolean jaccard on 30 random fingerprints, identical to 0.0, and against a straight double loop on continuous non-negative features, largest absolute difference 3.3e-16. They are built from one Manhattan matrix and the row totals (`preprocessing.overlap_distance`), since scipy has no continuous Tanimoto. **Three consequences.** They need features at or above 0, so on a z-scored hybrid `require_nonnegative` refuses rather than shifting the features up; a sweep point there is undefined, not cut. sklearn has no metric string for them, so they reach a neighbour search only as a precomputed matrix, and `sklearn_metric` raises instead of returning a name that would compute something else. A block of m features is divided by m, the same divisor as Manhattan, because both formulas are sums of per-feature terms; that follows the family table and was not separately measured. `tests/test_f34_scoring_paths.py` passes, 31 of 43 cases, with the 12 raw-feature comparisons skipped because no raw-feature path exists for these two. **Scope decided by Adelaide 2026-11-03: fingerprints and count vectors only.** On a standardised feature space they are reported as undefined and the features are NOT shifted up to make them non-negative. Two measurements behind that, both 2026-11-03. On z-scored MolFormer features, 600 molecules of the Mpro set, every one of the 179,700 pairs lands outside the 0-to-1 range the formula is supposed to return, between 1.617 and 2.587. And a shifted Tanimoto ranks pairs the way Manhattan does: Spearman 0.999995 against plain Manhattan at the smallest shift that reaches 0, 1.000000 by a shift of +100, and 0.999074 then 1.000000 on the 200 physicochemical descriptors — so a shifted Tanimoto would put Manhattan's ranking in the sweep a second time. Between 45 and 66 per cent of the entries of a z-scored block are below 0, measured on four datasets, so shifting is not a small correction. **One claim in the repo was wrong and is now dated as wrong:** `require_nonnegative` said an offset of +3 against +1000 changes the nearest neighbour of 26 per cent of points, measured on 200 random points in 20 dimensions. On MolFormer's 768 features over 1,924 molecules that same pair of offsets changes 0.1 per cent of molecules' nearest neighbour | this thread | **done 2026-11-02** |
| 18 | **Decide whether `mmpdb` is installed.** Without it, matched pairs can only be read from a shipped list, and the restriction on transformation size that the literature applies is a parameter of generating pairs and cannot be recovered from a list made without it | Either install and generate per dataset, or state in the paper that matched-pair cliffs exist only for Dablander's three targets | **Adelaide** | open |
| 19 | **Decide whether ACNet is downloaded.** Zhang et al. 2023: more than 400,000 matched pairs over 190 targets. It would be a second external answer key after MoleculeACE | Either download and score it like MoleculeACE, or state why it is out | **Adelaide** | open |
| 20 | **Decide whether M_cores and M_train are built in `opig_activity_cliff_qsar`,** or whether the paper states it runs on two of Dablander's four pair settings. It has M_inter and M_test only (`:418`). M_cores is the pairs where both molecules are held out and their shared core appears nowhere in training, and Dablander reports it as the hardest of his four | — | **Adelaide** | open |
| 21 | **Decide whether the half-cliff band, 1 to 2 log units, is carried into the noise experiments** or stays only in the Dablander replication | — | **Adelaide** | open |
| 22 | **Decide whether the similar side may be computed in a hybrid's own distance, and at what cutoff; and whether the molecules the put-back spares may come from the same features the detector reads** | The two open parts of her 2026-11-02 decision in section G | **Adelaide** | open |
| 23 | Report the cliff-call Matthews correlations `opig_activity_cliff_qsar` already computes. The thesis reports its prediction errors and none of them | — | this thread | computed, unreported |
| 24 | Write section 3 of `thesis_provenance/CLIFF_DEFINITIONS_2026-11.md` into Paper 3. The counts are in the findings file; the prose is not written | — | this thread | blocked on 4 |
| 25 | Count how many molecules in each rule's cliff set carry a ChEMBL curator's suspect-record flag | `n22_curator_flag` reads `data_validity_comment`, and reports that hERG IC50 is the one dataset keeping one row per record. One table: rule by flagged-and-cliff count, flagged-not-cliff, cliff-not-flagged. Section C option 2 | this thread | **not started**; both halves exist |
| 26 | Count how many molecules in each rule's cliff set are molecules this suite corrupted on purpose | `n45_detector_grading` knows the corrupted rows by construction, so the two masks cross directly. One table a dataset and injection level. This is the cheapest version of Adelaide's classifier idea of 2026-11-01, and its limit is that injected noise is not the noise the data came with. Section C option 6 | this thread | **not started**; both halves exist |
| 27 | Compare each rule's gap with the spread of the same compound's own repeat measurements, per dataset | The spread comes from the repeat measurement above; this row is the comparison, in label units, per rule rather than per dataset. Section C option 1 | this thread | blocked on the repeat spread |
| 28 | **Decide whether dropping a named molecule and refitting is run, and on which dataset** | Adelaide's own first idea of 2026-11-01, and the only option in section C that answers it. Nothing in the repo does it per molecule: `n27` drops a random share and n44 drops a whole flagged set. The honest version costs one fit per molecule per fold, so it is a dataset-size decision, not a yes-or-no one. The free substitute, each molecule's own held-out residual, does not answer the question | **Adelaide** | open, new 2026-11-02 |
| 29 | **Decide whether ACNet and the Bajorath sets are downloaded as a second external cliff set** | Already asked below as its own row for ACNet. Recorded here too because section C option 7 is where it bites: no published set carries a wrong-label column, so none of them can train a cliff-against-noise classifier on its own | **Adelaide** | open |

---

# B. Issue 1: the options a rule is built from

A rule is three choices: what makes two molecules similar, what makes their labels far
apart, and whether the answer is a pair or a molecule. One row is one option. "In repo"
means the piece exists, not that it has ever been used for cliffs.

## What counts as similar

| Option | In repo | Where |
|---|---|---|
| ECFP4 Tanimoto at or above 0.55 (Hu et al. 2015) | yes, used | `similarity_helpers.py:980`, `n01:235` |
| Morgan Tanimoto at or above 0.90 | fingerprint yes, cutoff no | the Geometry paper's strongest setting |
| MACCS Tanimoto at or above 0.85 | fingerprint yes, cutoff no | `representations/molecular.py` |
| RDKit fingerprint under Dice | fingerprint yes, **Dice absent** | `distance_config.py:66` holds four distances, no Dice |
| MoleculeACE: 0.90 on any one of ECFP Tanimoto, generic-scaffold Tanimoto, scaled Levenshtein on the SMILES string | yes, added 2026-11-02 | van Tilborg et al. 2022 |
| Matched molecular pair, transformation-size restricted | pair lists ship with the Dablander data; `mmpdb` not installed (row 18) | `chembl_qsar.py` |
| Top k per cent of pairs by similarity | yes, used | `n01:112` at 2.0 per cent, `similarity_helpers.py` at 1.0 |
| Hamming distance of 1, 2, 3 or 5 edits on a sequence, at the full V-region and at the CDR only | yes, used, and it is the one similarity axis in the suite that is already swept | `similarity_helpers.py:1512`, called by s01 and s02. `cliff_rules.py` has one point, `EDIT_DISTANCE_MAX = 5`, and no CDR scope |
| 1/(1+d) on the standardised concatenation of every representation, similar meaning the top 1 per cent of pairs | yes, used wherever no ECFP4 exists | `similarity_helpers.py:1301`, `s12:319` for Manhattan, cosine and Minkowski p=0.5. Not in `cliff_rules.py` |
| Among the k nearest neighbours, no cutoff | yes, used | `molecular.py:1467`, k=10 at Tanimoto 0.5 |
| Closest 10 per cent of pairwise distances, whatever the features | yes, used | `ActivityCliffDetector`, `noise_mitigation_methods.py:742` |
| A hybrid's own distance: Manhattan, Euclidean, cosine, Minkowski p=0.5 | distances yes, **cutoff undecided** | `distance_config.py:66`; the open decision at `n25:696` |
| ChemBERTa, MolFormer or Chemeleon cosine | all three present, never used for cliffs | `molecular.py`, `forward_compat_molecular.py` |
| Stereoisomer pairs | yes, added 2026-11-02 | named by the Geometry paper as what fingerprints miss |
| Binding mode from a crystal structure | no, and no structures are loaded | Hu, Furtmann and Bajorath |

## What counts as a large label difference

| Option | In repo | Where |
|---|---|---|
| 2.0 times the median absolute deviation of that dataset's labels | yes, used | `n01:235`, `n25:569`, `n35:213`, `thesis.tex:3675` |
| Gap solved so a target share of molecules are cliff molecules, target 15 per cent | yes, used | `similarity_helpers.py:1101`, `solve_activity_gap` |
| Absolute 2 log units, 100-fold | yes, used | `n14:89`, `chembl_qsar.py:13` |
| Absolute 2 log units with 1 to 2 dropped as half-cliffs | yes, used | `opig_activity_cliff_qsar.py:379` |
| Absolute 1 log unit, 10-fold | yes, added 2026-11-02 | MoleculeACE |
| 90th percentile of pairwise label differences | yes, used | `ActivityCliffDetector` |
| Different class, for a classification task | yes, used | `n01:262` and four other files |
| Mean plus two standard deviations of that target's own pairwise differences | yes, added 2026-11-02 | attributed to the Bajorath group, **unverified** |
| A gap that clears the measurement-noise floor | floor present, never used as a gap rule | `n14:101`, `KRAMER_NOISE_FLOOR_PKI = 0.54` |

## Continuous instead of yes-or-no

All of these are now computed by `cliff_rules.CONTINUOUS_SCORES` and nothing uses the
output; that is row 16. SARI, the rivality index with MODI, and ROGI each carry a block in
the code naming the part that is not verified against its source paper.

| Option | Source |
|---|---|
| SALI, the label difference over one minus the similarity | Guha and Van Drie 2008 |
| Roughness per molecule, in the spirit of iCliff | `similarity_metrics/nary_similarity.py:126` |
| SARI, a continuity and a discontinuity score per dataset | Peltason and Bajorath 2007 |
| MODI, RMODI for regression, the rivality index, and ROGI for roughness | Luque Ruiz and Gómez-Nieto; Aldeghi et al. arXiv:2207.09250. MODI is cited at `thesis.tex:978` |
| SAS map: similarity against label difference, with a cliff quadrant | Medina-Franco and co-workers |
| SALI averaged over several representations | the research session's answer to representation dependence |

A defect in SALI itself, found 2026-11-02 and not a code fault: on hERG Ki the largest
SALI is 1,140 against a median of 4.917, and it comes from one of 35 pairs whose
fingerprints are identical, where one minus the similarity is clipped rather than zero. The
maximum SALI over a dataset reports the clip, not the landscape. `sali_clipped_pairs()`
counts them.

## Pair or molecule

`chembl_qsar.py` and `opig_activity_cliff_qsar.py` score pairs. Every noise and similarity
experiment reduces to a per-molecule flag, by asking whether a molecule is in at least one
cliff pair. A molecule-level rule inherits whatever the pair rule did and hides it. Row 7.

## Sets that already carry cliff labels

Not computations. Something to measure a computation against.

| Set | State |
|---|---|
| Dablander's three targets, matched pairs with cliff labels | on disk, used by `opig_activity_cliff_qsar` |
| MoleculeACE, 30 targets with a `cliff_mol` column | on disk; `build_moleculeace.py:10` reads `smiles` and `y` only and drops the labels. Row 1 reads them |
| ACNet | not downloaded, row 19 |
| The Bajorath group's published cliff sets | not downloaded |

---

# C. Telling a cliff apart from a wrong label

This is the question of 2026-11-01 that the earlier versions of this handoff lost: a
cliff and a bad measurement look the same from the outside. Both are a molecule whose
label its neighbours do not predict. Every option below is a way of separating them.
Each says what it computes, what the repo already has, and what it needs that the repo
does not have. Nothing here is decided.

Two molecules that differ by one atom and by three log units of potency are either real
chemistry or a transcription error. A detector that flags molecules its neighbours fail
to predict flags both, and no cutoff on that score separates them, because the score is
the same quantity in both cases. Something outside the labels has to break the tie.

## Nine ways to break the tie

**1. The gap against what the same compound's repeats disagree by.** Where one compound
was measured more than once, the spread of its own repeats is how far two measurements of
one thing sit apart in that assay. A gap larger than that cannot be explained by the
assay alone. In the repo: `_ground_truth_providers.drug_target_repeats` groups rows that
share a drug and a protein, and `compute_replicate_stds:201` returns each group's spread.
It needs the loader to expose `replicate_groups`, or per-row drug and protein ids. hERG
IC50 is in Paper 3's dataset list because it keeps one row per measurement. Needs: the
spread per dataset, which is already one of the measurements in section A. This is the
only option whose unit is label units.

**2. Whether the molecule's label is one a curator already doubted.** ChEMBL writes a
field called `data_validity_comment` on records its own curators marked as suspect. That
is a label-is-wrong flag that nobody here invented. In the repo: `n22_curator_flag`, and
it reports that only hERG IC50 keeps one row per record, so it is the one dataset where
the flag survives. Needs: counting how many molecules in each rule's cliff set carry one.
Cheap, and it is a direct answer rather than a proxy.

**3. Whether two independent groups disagreed about the molecule.** Four published
collections each made their own active-or-inactive call on hERG molecules, and where two
of them covered the same molecule either they agree or one is wrong. In the repo:
`_ground_truth_providers.cross_curator`. The outcome is a flip rather than a magnitude, so
it scores as AUROC. Needs: nothing new to compute it; it covers hERG only.

**4. Whether the pair comes from one paper or two.** Two measurements made in different
papers differ partly because the labs differ. A pair drawn from one document is more
comparable than a pair drawn from two. In the repo: hERG IC50 names the document each
measurement came from, and `n29_document_bias` fits one offset per document. Needs: a
same-document restriction on which pairs may be called cliffs, which is one line where a
rule builds its pair list, and it applies to that one dataset.

**5. Whether dropping the molecule helps or hurts the model.** Adelaide's own first
idea, 2026-11-01: use a detector to find an outlier, then ask whether including it
improves or degrades prediction, averaged over folds. **Nothing in the repo computes
this per molecule.** `n27_removal_cost_and_propagation` removes a random share of the
training molecules, not named ones, and n44 drops a whole flagged set. Both measure a
set, so neither can attribute a change to one molecule. Three shapes it could take, at
very different costs: one fit per molecule per fold, which is the honest version and
costs as many fits as there are molecules; the set version, which is the put-back control
already planned and answers a different question; or each molecule's own held-out
residual, which is already computed by every fold and is free, but says only that the
molecule is badly predicted, which is where this started. **The free version does not
answer her question.** The expensive version is the only one that does.

**6. A classifier trained on molecules that are known cliffs and molecules that are
known wrong.** Adelaide's third idea, 2026-11-01, run after outlier detection. It needs
examples of both classes. Cliff examples exist: MoleculeACE's own column, on 30 targets.
Wrong-label examples do not exist as published labels anywhere on disk. **But they exist
synthetically**: `n45_detector_grading` grades every detector against noise this suite
injected, so the corrupted molecules are known by construction. Crossing the two gives
every molecule one of four states: injected-corrupt, in the cliff set, both, or neither.
Needs: the two masks in one table, and both already exist. This is the cheapest route to
her idea, and its limit is that injected noise is not the noise the data came with.

**7. Whether the published cliff sets agree with ours.** Adelaide's second idea,
2026-11-01: is there a documented source of real cliffs. Four exist. MoleculeACE's
`cliff_mol` column is on disk and scored. Dablander's three targets ship matched pairs
with cliff labels and are on disk. ACNet is not downloaded. The Bajorath group's sets are
not downloaded. **What none of them carry is a wrong-label column**, so they can say a
molecule is a cliff and cannot say a molecule is noise. On their own they cannot train the
classifier in option 6.

**8. Whether the representations disagree about the pair.** From the research session of
2026-11-01, attributed there to the Geometry preprint: a pair that is close under
fingerprints and far apart under learned embeddings is a different kind of pair from one
every representation calls close. In the repo: every representation named there is
present. Needs: the measurement of how far the cliff set moves when the representation
changes, which is already one of the measurements in section A, read a second way — per
pair rather than per count.

**9. Whether the pair has a named chemical change.** A matched molecular pair is two
molecules related by one substitution, named. Noise has no reason to respect chemistry, so
a cliff with a named transformation is harder to explain as an error. In the repo:
Dablander's three targets ship pair lists. `mmpdb`, which generates pairs for a dataset
that has no list, is not installed, and the restriction on how large a transformation may
be cannot be recovered from a list made without it.

## What this adds up to

Three of the nine cost almost nothing, because both halves already exist: the curator
flag against each rule's cliff set (option 2), the injected-noise mask against each rule's
cliff set (option 6), and the repeats spread against each rule's gap (option 1).

One of the nine is the question she actually asked and is expensive: whether dropping a
named molecule changes the score (option 5).

Five are restrictions on which pairs may be called cliffs rather than tests after the
fact: same paper, same assay, named transformation, representations agreeing, an external
set agreeing.

**What nobody has**: a dataset where both labels are known — this molecule is a real
cliff, that one is a bad measurement — on data that was not injected. Every option above
is a way of approximating one half of that.

---

# D. Does it make sense to tune the rule to a dataset

Asked in the prompt of 2026-11-01. What the census measured, per half of a rule. The
choice is Adelaide's (rows 5 and 8a); this section holds the numbers, not a verdict.

**Corrected by a second session, 2026-11-02.** This section used to end "a rule tuned per
dataset to a share is out". That was Claude's conclusion, not hers, and it contradicted
her row 8a decision of the same day, which keeps both cutoffs tuned to one shared target
share. Headings now state what was measured.

**The similarity side: one fixed cutoff gives very different pair counts.** "Similar" has no absolute
meaning across different chemistry. hERG Ki has 10,809 pairs at or above ECFP4 Tanimoto
0.55; MDR1-MDCK has 459, on a dataset half again as large, because it carries 2,318
scaffolds among 2,641 molecules. One fixed cutoff therefore gives one dataset a working
cliff set and another almost none, and that is a property of the chemistry, not of the
rule. A share of pairs keeps the similar set a bounded fraction everywhere, which is why
`n01` and `s12` already fall back to one. Its cost is that a share carries no unit and
cannot be compared with a published number, so report both: the share for comparability
across datasets, the absolute Tanimoto for comparability with the literature.

**The gap side, tuned to a target share: where the gap lands against the noise floor.** This is the
`calibrated` rule. To put 15 per cent of MDR1-MDCK's molecules in a cliff set it drops the
gap to 0.196 label units, which is 0.36 of Kramer's 0.54 pKi, and 54.2 per cent of the
pairs it then calls cliffs differ by less than two independent measurements of one compound
would. On Caco-2 the 2.0-median-absolute-deviation rule does the same thing for a
different reason: gap 0.417, 0.77 of the floor, 31.1 per cent of its 6,909 cliff pairs
inside it. Any rule that fixes the share must move the gap to reach it, and on a dataset
with narrow labels the only place to move it is into the noise. On these two
datasets, a large part of the cliff pairs a fixed share produces differ by less than
repeat measurements of one compound would.

**The gap side, tuned to the dataset's own assay: the alternative.** Whether the
literature does this is unchecked. A
cliff should be a label difference larger than the assay can repeat. That number
legitimately differs per dataset and is not arbitrary, which is exactly what a fixed share
is not. 0.54 pKi is Kramer's figure for public Ki data and it is the wrong assay for
Caco-2's log permeability and MDR1-MDCK's log efflux ratio; the census prints those two
columns against a borrowed floor and says so. Row 3 measures the right number where a
dataset has repeat measurements. hERG IC50 is in Paper 3's dataset list precisely because
it carries them.

**Measured fact for the choice:** on Caco-2 and MDR1-MDCK an absolute 2-unit gap flags
no molecule at all. Where a dataset has no repeat measurements, a gap set from
repeatability cannot be computed and needs a fixed value instead.

---

# E. The threshold sweep: the grid

For row 2. Every axis, every dataset, no dataset shrunk. A similarity matrix is built once
per fingerprint and then thresholded as many times as you like, and since the dot-product
Tanimoto of 2026-11-02 one matrix on 3,317 molecules at 2,048 bits takes 1.75 seconds
against scipy's 214.61. The sweep is thresholding, not computing. Do not prune it for cost.

| Axis | Points |
|---|---|
| ECFP4, Morgan, MACCS Tanimoto cutoff | 0.30 to 0.95 in steps of 0.05 |
| ECFP4, RDKit Dice cutoff | 0.30 to 0.95 in steps of 0.05 |
| MoleculeACE cutoff, all three measures together | 0.70 to 0.95 in steps of 0.05 |
| top share of pairs by similarity | 0.1, 0.25, 0.5, 1, 2, 5, 10 per cent |
| k nearest, no cutoff | 1, 3, 5, 10, 20, 50 |
| gap as a multiple of the median absolute deviation | 0.5, 1, 1.5, 2, 2.5, 3, 4 |
| gap in label units | 0.5, 1, 1.5, 2, 2.5, 3 |
| gap solved to a target share | 5, 10, 15, 20, 25, 30 per cent |
| gap as a percentile of pairwise differences | 75, 80, 90, 95, 99 |
| gap as the mean plus n standard deviations | 1, 2, 3 |

Every point reports: cliff pairs, cliff molecules, the achieved share of molecules, the gap
in label units, the gap as a multiple of that dataset's own repeatability (row 3) and of
Kramer's 0.54, and the share of that point's cliff pairs inside each.

Three readings are the point of running it, rather than the table itself. Where a rule
saturates: the cutoff below which nearly every pair is called similar. Where it empties:
the cutoff above which too few cliff molecules are left for a held-out fold to score,
which `similarity_helpers` puts at 30 a fold. And where the gap crosses the dataset's own
repeatability, which is the line below which the rule is calling noise.

---

# F. Issue 2: the six places cliffs enter the noise experiments

Each needs one thing: which rule from issue 1 it runs on. Code locations are the places to
lift from, never the places to change — n01, n14, n25, n35 and n42 are being retired.

| Place | What it is | Needs a rule | Lift from |
|---|---|---|---|
| **DC**, modifier | Cliff put-back: after a cleanup, undo it for cliff molecules. A modifier on any cleanup, not a cleanup | yes | `n25 put_back`, `cliff_mask_for_dataset` |
| **DC**, control | Random put-back: put back as many non-cliff molecules as cliff molecules | yes, the same one | `n25 random_exempt_mask` |
| **DC**, list B | Cliff-aware gap, registry `activity_cliff_aware`. Scores the gap from the neighbours' mean over the neighbours' own spread | **no.** No cutoff and no cliff list inside it | already a detector |
| **DC**, list B | `ActivityCliffDetector`, called by no experiment | yes, and it has two free cutoffs | `noise_mitigation_methods.py:742` |
| **ST** | Close-pair label disagreement, and its counting of how many flagged close pairs are cliff pairs | yes | `n35 count_cliff_pairs`, `intercept_estimate` |
| **MS** | The cliff measurements that train no cleanup: cliff-stratified prediction error, landscape geometry, and the counts of rows 4 and 12 | yes, all | `n01`, `n14` |

Also moving in: s04 and s12 split a dataset into cliff and other molecules and score
representations on each part; s01 and s02 run a sequence-based cliff sweep on antibody
datasets; s09 reads s04's cliff numbers; `simeval` uses a cliff set as the target a
distance is scored on finding; `nary_similarity` computes roughness per molecule. Where
they land is row 15.

## What is wrong today, before any new work

**The thesis uses two different rules in different chapters and does not say so.**
`thesis.tex:1120` and `:2922` solve the gap on each dataset to aim at 15 per cent of
molecules. `:3675`, cliff put-back, instead calls a label far when it is more than twice
the median absolute deviation. On hERG Ki the second gives 451 of 1,412 molecules against
the first's 211 of 1,412.

**The aimed-at share is not the achieved share, and the thesis prints the target as though
it were reached.** `solve_activity_gap` returns the gap whose share is nearest 15 per
cent, not one that reaches it, and where few pairs are similar it misses by a lot. On QM9
only 1,093 pairs among 9,978 molecules are similar at all (`thesis.tex:1120`), and `:4148`
reports 19 of 7,982 training molecules as cliff molecules. Anything written about this uses
achieved shares, per dataset, never the target.

**Four more things that bite on the way in.** The random put-back control cannot run where
most training molecules are cliff molecules, recorded as impossible on Caco-2, hERG
blocking and LogD at `thesis.tex:4148` under twice the median absolute deviation. Cliff
put-back has never run on anything but ECFP4 at 0.55 (`thesis.tex:3675`), because nobody
has decided what cutoff means similar in a Manhattan or forest distance (`n25:696`). Every
setting of n25's old sweep put back the same ECFP4 cliff set, so a modifier that varies
with the feature space is new behaviour, not a port. And three faults fixed in n25 between
2026-09-23 and 2026-10-06 change what a stored number means: the cliff set was once chosen
from the clean labels, a training molecule's cliff status once depended on held-out
molecules, and a reused similarity matrix was once cut two different ways inside one run.
A fourth decision, 2026-10-06: two rows of the same molecule are never a cliff pair.

---

# G. Already settled, so it is not reopened

**Everything in KIRBy that touches activity cliffs is handled in this thread**, s04 and s12
included (Adelaide, 2026-11-02).

**What must not change in s04.** The cliff set is built per outer fold from that fold's
training molecules only, and a held-out molecule counts as a cliff when at least two
training molecules similar to it differ in label by more than the gap, so no held-out label
is read. Both decided 2026-10-06. s12 shares the same two calls, so the two files split
each fold identically. Row 9 changes only which rule is named.

**The hybrid is out of the definition tests and required in the experiments** (Adelaide,
2026-11-02). The census that chooses a rule runs on fingerprints and on the loader's own
pair lists, never on a searched hybrid space, so no rule is chosen using the features it
will later be used to grade. But every noise experiment that acts on cliffs must run in the
hybrid as well as the single representations; a cliff-aware arm that only ever ran on ECFP4
is not finished.

**The cliff distance should probably be importance-weighted, and nothing does it today.**
s10 read a distance twice over, once on the scaled features and once with every feature
multiplied by its importance. The neighbour prediction improved; the Gaussian process and
the support vector regressor did not. A cliff is a neighbour statement, so it sits on the
side that improved. Two caveats: s10's result was read from its file by another thread on
2026-11-02 and has not been re-run here, and it cannot apply to the ECFP4 reference, which
is a fixed fingerprint with no importance attached and is deliberately fixed so that no
representation is graded against cliffs drawn from its own features. Part of row 6.

**MoleculeACE's `cliff_mol` column is the only external check that exists.**
`opig_moleculeace_cliff_check` is in `PAPER_3_OUTLINE.md`'s inventory and run list.

## What is already measured, so it is not re-measured

- **The literature.** A research session of 2026-11-01, saved at `~/Downloads/cliffs.md`.
  Anything taken from it is marked as from it, in every document.
- **The code inventory and the literature, written up.**
  `thesis_provenance/CLIFF_DEFINITIONS_2026-11.md` sections 1 and 2.
- **Section 3 of the same file**, rewritten 2026-11-02: every named rule measured on hERG
  Ki, Caco-2 and MDR1-MDCK, with the gap against the Kramer floor and the caveat that 0.54
  is the wrong assay for two of the three.
- **One thing the thesis already measures** that neither of those found first,
  `thesis.tex:1123`: keeping each dataset's gap, replacing ECFP4 with each other single
  representation and calling the same number of closest pairs similar, hERG Ki's cliff
  molecule count runs from 138 under atom pairs to 267 under the WL graph kernel. Of the
  pairs ECFP4 calls similar, 57 to 74 per cent are among another representation's closest
  pairs; on QM9, 1 to 24 per cent.
- **One rule reproduces an external answer key exactly.** Our `moleculeace` rule recovers
  all 1,222 of MoleculeACE's cliff molecules on CHEMBL214_Ki, none either way, Matthews
  1.0000, at both 1,024 and 2,048 bits. It did not at first: using the plain Bemis-Murcko
  scaffold rather than the generic framework gave 1,821 molecules and Matthews 0.5566. Both
  numbers are kept in the code as dated blocks so the wrong one cannot later read as a
  measurement. On that target the three parts contribute 1,055 molecules from Levenshtein,
  658 from the generic scaffold and 148 from ECFP Tanimoto, so their rule is mostly a
  string-similarity rule.

## Where things live

- `tests/experiments/noise/cliff_rules.py` — every cutoff, criterion, named rule and
  continuous score in one place, so two experiments cannot disagree silently. 21 similarity
  criteria, 10 gap criteria, 10 named rules.
- `tests/experiments/noise/cliff_census.py` — runs the grid per dataset, writes to
  `/Volumes/seagate/cliff_census/`.
- `tests/experiments/opig_moleculeace_cliff_check.py` — row 1.
- `thesis_provenance/CLIFF_DEFINITIONS_2026-11.md` — the write-up.

## Before you start

- Read repo-root `CLAUDE.md` in full. It sets how to write to Adelaide.
- Read `HANDOFF_NOISE_RESTRUCTURE.md` sections 1 to 4, 8, 9 and 10. The 37 old experiments
  n01 to n42 become n43 to n47: n01 and n14 go to MS, n25 and n42 to DC, n35 to ST. **Do
  not write a fix against an old file.**
- From `MEMORY.md`, the rules that bite here: no patient data, ever; one heavy local job at
  a time; everything on `/Volumes/seagate`; no significance tests; never cut a definition,
  dataset or setting for compute; local runs check code and are not results; post a line
  every few minutes during long work because her terminal disconnects; she decides, so give
  options with the numbers attached.

## Paper 3 placement

- `opig_activity_cliff_qsar` is in Paper 3, not Paper 2, and is in `PAPER_3_OUTLINE.md`.
- Its thesis result: on D2, factor Xa and Mpro the hybrid had lower held-out mean absolute
  error than the top-ranked single representation on 5 of 6 target-and-model rows, the
  exception being factor Xa with LightGBM.
- The research session reports a linear relationship between prediction error and the
  Matthews correlation on the cliff call across Dablander's nine models. If it holds here,
  the thesis's error result predicts a cliff-call result that is already computed and
  unreported — row 23.
