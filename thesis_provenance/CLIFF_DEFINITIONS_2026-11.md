# What the code calls an activity cliff

Started 2026-11-01. Answers questions 1 and 2 of `HANDOFF_PAPER3_CLIFFS.md`. Questions 3
to 5 are not measured; their headings say what is there.

Section 1 comes from code read on 2026-11-01 in this session, at the file and line given.
Nothing in it is from a memory file or an earlier document. No counts were re-measured:
where a count appears it is quoted from a comment in the code and labelled as that.

Section 2 comes from `~/Downloads/cliffs.md`, a research session Adelaide ran and
downloaded 2026-11-01. None of those papers were read in this session, so section 2 is a
claim at second hand throughout. Its numbers are that document's numbers. Section 2's
last two subsections are the exception: they were read from this repo's code.

---

## 1. The definitions in the code

Nine rules. Each decides two things separately: when two molecules count as similar,
and when their labels count as far apart. They differ on both sides.

### The two live families

**A. Fixed cutoffs: ECFP4 Tanimoto 0.55, label gap 2.0 median absolute deviations.**

`tests/experiments/noise/n01_activity_cliff_framework.py:235`, `identify_cliff_pairs`.
A pair is a cliff when ECFP4 Tanimoto is at or above 0.55 and the absolute label
difference is more than 2.0 times the median absolute deviation of that dataset's own
label column. A molecule is a cliff molecule when it is in at least one such pair. On a
dataset with no fingerprint the similar side becomes the closest 2.0 per cent of
off-diagonal pairs (`n01:112`, `SIMILAR_PERCENTILE = 2.0`) instead of the 0.55 cutoff.

The same two numbers are written out again in three more files:
`n25_cliff_exemption.py:569-570`, `n35_near_duplicate_disagreement.py:212-213`, and
`similarity_helpers.py:980` (`identify_cliff_compounds`, the function Adelaide's earlier
`tests/similarity_experiments.py:200` also carries a copy of).

**B. Calibrated: ECFP4 Tanimoto 0.55 kept, label gap solved per dataset.**

`tests/experiments/similarity/similarity_helpers.py:1151`, `calibrated_cliff_rule`,
with `solve_activity_gap` at `:1121`. The similar side is unchanged at Tanimoto 0.55.
The label side is no longer a fixed multiple of the median absolute deviation: it is
solved, per dataset, so that the rule flags the same share of that dataset's molecules
everywhere. The share is 15.0 per cent of molecules (`CLIFF_TARGET_PCT`,
`similarity_helpers.py:1101`, overridable by `KIRBY_CLIFF_TARGET_PCT`). The solved gap
is reported back divided by the median absolute deviation, so it is comparable with the
2.0 of family A, and the family-A count is stored alongside it in every result.

The block comment above it (`similarity_helpers.py:1044-1100`) records why it was
written and what it measured on 2026-09-19. Under family A at Tanimoto 0.55 and a gap
of 2.0 median absolute deviations: 1,625 of 2,158 Caco-2 molecules were called cliff
molecules, 451 of 1,412 hERG Ki molecules, and 125 of 2,641 MDR1-MDCK molecules. As
shares of each dataset, 75.3, 31.9 and 4.7 per cent. Those three counts are quoted from
that comment and were not re-measured here.

### Which experiments use which

Read from the call sites, 2026-11-01:

| Rule | Used by |
|---|---|
| B, calibrated | `s04_activity_cliffs.py`, `s12_cross_domain_cliffs.py`, and nothing else |
| A, fixed 0.55 and 2.0 | `n25_cliff_exemption.py`, and through n25: `n35`, `n42`, `n44` |
| A, fixed, own copy | `n01_activity_cliff_framework.py`, `tests/similarity_experiments.py` |
| MMP list, 2.0 log units | `opig_activity_cliff_qsar.py` (the Paper 3 experiment) |
| MMP list and fixed, side by side | `n14_activity_cliff_landscape.py` |

**No noise experiment calls the calibrated rule.** `calibrated_cliff_rule` has two
callers, both in `tests/experiments/similarity/`. Every cliff-aware noise experiment is
on family A's fixed 0.55 and 2.0.

This is the answer to the worry in the handoff, and it is narrower than "one definition
only". Two things disagree. The similarity experiments solve the label gap per dataset so
that 15 per cent of each dataset's molecules are cliff molecules. The noise experiments
hold the gap at 2.0 median absolute deviations, which gave 75.3 per cent of Caco-2
molecules and 4.7 per cent of MDR1-MDCK molecules when it was last measured. So an
s-experiment and an n-experiment on the same dataset partition it differently, and the
fix written on 2026-09-20 reached only the s-experiments.

### The remaining rules

**C. Matched molecular pairs, 2.0 log units.** `src/kirby/datasets/chembl_qsar.py:13`.
A cliff is a pair, not a molecule: a matched molecular pair whose pActivity differs by
more than 2.0 log units. The similar side is not a threshold at all — the pair list
ships with the Dablander et al. 2023 data. Pairs differing by 1.0 to 2.0 log units are
dropped from scoring rather than called non-cliffs
(`opig_activity_cliff_qsar.py:377`). Used by `opig_activity_cliff_qsar.py`.

**D. Matched molecular pairs or fingerprint, absolute 2.0 log units.**
`n14_activity_cliff_landscape.py:150` (`_cliff_mask_from_pairs`) and `:172`
(`_cliff_mask_from_similarity`), with `AC_POTENCY_THRESHOLD = 2.0` at `:89` and
`ECFP4_SIM_THRESHOLD = 0.55` at `:92`. n14 is the one experiment that runs two rules
side by side: the matched-pair list where the dataset has one, and Tanimoto 0.55 with
an absolute 2.0-log-unit gap where it does not. Its gap is in log units, not median
absolute deviations, and the docstring says so explicitly against n01.

**E. Edit distance on sequences.** `similarity_helpers.py:830`,
`identify_cliff_compounds_editdist`. Similar means Hamming or Levenshtein distance at
or below 5 edits, or the closest `similar_percentile` per cent of nonzero pairwise
distances. The label gap is 2.0 median absolute deviations. Used for antibody datasets
by `n01:480` and `n25:715`. n25 raises an error if asked for an absolute gap on an
antibody set, because this function takes only a multiplier of the median absolute
deviation (`n25:709-714`).

**F. Stacked representations, percentile similarity.** `s12:127`,
`identify_cliff_pairs`. Similar means the top `similar_percentile` per cent of pairwise
similarity on the standardised concatenation of every representation, where similarity
is 1/(1+d) and d is whatever `kirby.similarity.distance_config` holds (Manhattan since
2026-09-04, despite the `_euclidean` in the wrapper's name at `:279`). **Retired for
results.** The docstring at `:132` records that `run()` stopped calling it on
2026-09-20, that it is the rule that disagreed with s04 — 295 of 1,412 hERG Ki
molecules against s04's 451 — and that it is kept only because
`smoke_tests_broken_fixes.py` exercises it.

**G. Two free thresholds, defaults from percentiles.**
`src/kirby/utils/noise_mitigation_methods.py:742`, `ActivityCliffDetector`. Both sides
are arguments with no default. Unset, similar becomes the closest 10 per cent of
pairwise distances and the label gap becomes the 90th percentile of pairwise label
differences, both measured on the data in front of it. Distance is Euclidean unless
told otherwise, on representation features, not fingerprints. Registered as
`activity_cliffs` at `:1231`. Its only live caller is the SALI feature factory below.

**H. SALI features.** `src/kirby/representations/molecular.py:1467`,
`make_sali_cliff_factory`. Not a cliff definition used to label anything — it builds
four features per molecule (`is_cliff`, local SALI, neighbour disagreement, mean
neighbour similarity) at Tanimoto 0.5 over 10 nearest structural neighbours, fitted on
training rows only. Worth listing because its `is_cliff` column is a ninth threshold,
0.5 rather than 0.55, and it is a model input.

**I. The metric yardstick's target.** `tests/experiments/similarity_metrics/simeval.py:218`,
`cliff_ground_truth`, at Tanimoto 0.55 and 2.0 median absolute deviations — family A's
numbers again, as the target a similarity metric is scored on finding.

### One thing that is not a cliff definition

`activity_cliff_aware` (`src/kirby/utils/noise_estimation.py:557`,
`ActivityCliffAwareEstimator`) is a **detector**, not a definition. It never labels a
molecule a cliff. It scores how likely a label is noise, as the molecule's deviation
from its neighbours' mean label divided by one plus the neighbours' own spread, both
over the label standard deviation. It appears in a dozen noise experiments under that
name and is the thing most likely to be mistaken for a cliff rule when reading the
result files. It is already an attempt at question 5: its whole premise is that a
molecule whose neighbours also disagree with each other is a cliff rather than noise.

### Two settings held fixed where the handoff expected a sweep

Read in `n25_cliff_exemption.py:688-704`: n25's sweep runs over representation spaces,
but "every setting of the sweep exempts the same ECFP4 cliff set". The comment names the
two decisions that stop it varying, and they are the same two the Paper 3 outline's open
note names: which threshold means similar in a Manhattan or forest distance, and whether
the spared molecules may come from the same features the detector reads. So the answer
is yes, n25 ran with one definition, and the file says why in a comment rather than
leaving it silent.

Two faults already fixed in n25, recorded at `:582-640`, both worth keeping for the
write-up because they change what a stored number means:

- Before 2026-09-23 the cliff set was chosen once from the clean label column, so the
  exemption was told which labels were right — the question n25 asks.
- Before 2026-09-23 a training molecule's cliff status depended on its similarity to
  held-out molecules, because the mask was built over all molecules and sliced afterwards.
- Before 2026-10-06 a reused similarity matrix was cut at the top 2 per cent of pairs in
  per-fold calls while the whole-dataset call cut at Tanimoto 0.55: two definitions of
  similar inside one run.
- Decided 2026-10-06: two rows of the same molecule are never a cliff pair. On hERG
  IC$_{50}$, loaded at 12,008 rows for 9,802 molecules, two repeat measurements of one
  molecule have Tanimoto 1.0 and were being counted as a cliff pair.

### A labelled cliff set already in the repo, unused

`tests/experiments/similarity_metrics/build_moleculeace.py:10` records that the
MoleculeACE CSV files carry a `cliff_mol` column. The loader reads `smiles` and
`y [pEC50/pKi]` and discards it. So the 30 ChEMBL targets of van Tilborg et al. 2022
are already downloaded with their published cliff labels, and nothing in the repo reads
those labels. That is the cheapest item on question 5's list of documented cliff sets.

### Every file in the suite that mentions a cliff, read 2026-11-02

`grep -rl cliff tests/experiments` returns 84 files, 67 of them Python. The earlier count
of 54 in the handoff was from a narrower search. Read in full on 2026-11-02. Three groups:

**Eleven files build a rule.** `similarity_helpers.py` (three of them, below),
`n25_cliff_exemption.py:582` (`cliff_mask_for_dataset`, ECFP4 Tanimoto 0.55 and 2.0 times
the median absolute deviation), `n35_near_duplicate_disagreement.py:213` (the same two
constants again, its own copy), `n01_activity_cliff_framework.py`,
`n14_activity_cliff_landscape.py` (absolute 2 label units), `s12_cross_domain_cliffs.py`
(the concatenation rule, four distances), `simeval.py:218`, `cliff_rules.py`,
`opig_activity_cliff_qsar.py`, `opig_moleculeace_cliff_check.py`, and the `chembl_qsar`
loader, which reads Dablander's shipped pair labels rather than computing anything.

**Six call a rule built elsewhere.** n42, n44, n47, s04, s12 and the smoke scripts all
call `n25.cliff_mask_for_dataset` or `similarity_helpers.calibrated_cliff_rule`.

**The remaining fifty name something without defining it**: the registry key
`activity_cliff_aware`, the experiment ids `n01`, `n14`, `n25`, the representation name
`sali_cliff`, a help string for the distance argument, or a result field. None of them is
a definition. `activity_cliff_aware` was cut as a detector on 2026-09-12 and came back
into n23's main table on 2026-10-06; that is a detector, not a cliff rule, and it holds no
cutoff and no cliff list.

### Two target shares inside one module, one chasing the other

`similarity_helpers.py` carries two constants whose names differ by a word order.
`CLIFF_TARGET_PCT = 15.0` at `:1096` is what `solve_activity_gap` moves the label gap to
reach. `TARGET_CLIFF_PCT = 10.0` at `:867` is a local inside the sweep that picks the
**similarity** percentile, choosing the candidate whose cliff share is nearest 10 per cent.
`s12_cross_domain_cliffs.py:176` holds its own copy of the same 10.0. On any dataset
without ECFP4 both run in sequence: the similar side is tuned to put 10 per cent of
molecules in the cliff set, and then the gap is tuned to put 15 per cent in. Section 3
measures what tuning either side to a share costs.

**Fixed 2026-11-02 (Adelaide's decision: the quotas move together).** Both sites now read
`CLIFF_TARGET_PCT`. Because the pair of cutoffs that hits a share is not unique, the order
is fixed in the code and stated there: the similarity cutoff is picked first under the
standard gap, then the gap is solved for whatever share is left over. Any s04 or s12 number
stored before this date used the old 10-then-15 pair.

`CLIFF_TARGET_PCT` also reads an environment variable, `KIRBY_CLIFF_TARGET_PCT`, so the
target a stored result used is not knowable from the code alone. The comment at `:1092`
says the value is written into each result file, which makes it recoverable per run.

### Two similarity criteria in use that `cliff_rules.py` does not have

**The concatenation rule.** For a dataset with no ECFP4 — no SMILES, or a non-molecular
domain — the similar side becomes 1/(1+d) on the standardised concatenation of every
representation, and similar means the top `CLIFF_SIMILARITY_PERCENTILE = 1.0` per cent of
pairs (`similarity_helpers.py:1301`). s12 generates the same thing for each of its
distances — Manhattan, cosine, Minkowski p=0.5 — at `:319`. The reason it exists is
recorded in s12's own docstring at `:159`, measured by an earlier session rather than here:
on standardised high-dimensional embeddings the pairwise Euclidean distance is large, so
1/(1+d) never reaches 0.55, and s12 found no cliffs at all in 13 of 16 datasets.

**The sequence rule.** `run_antibody_cliff_analysis` at `:1512` sweeps a Hamming distance
of 1, 2, 3 and 5 edits at two scopes: the full V-region, and the CDR only, extracted with
ANARCI and cached. The gap is 2.0 times the median absolute deviation, or a class flip for
a classification dataset. s01 and s02 both call it and store it as `_sequence_cliffs`.
`cliff_rules.py` has one edit-distance point, `EDIT_DISTANCE_MAX = 5`, and no CDR scope.
So the antibody side of the suite already sweeps its similarity axis across four points and
two scopes, and the molecular side does not sweep at all.

### The CDR-only scope can silently be the whole sequence

Found 2026-11-02 while adding the sequence rule to `cliff_rules.py`. Three functions in a
row fall back to the whole sequence rather than failing, and none of them reports it.

`kirby/representations/antibody.get_cdr_residue_indices` returns every residue index in
three cases: ANARCI is not importable, ANARCI returns no numbering, and the numbering lands
on no CDR position. `similarity_helpers._extract_cdr_sequences` then counts a sequence as
failed only when that call raised or returned nothing, so none of the three counts, and its
own guard — give up if more than half failed — never fires.

Measured here. ANARCI is installed. `get_cdr_residue_indices` on a random 21-residue
peptide returned all 21 indices, and `_extract_cdr_sequences` on 30 random 20-residue
peptides returned 30 strings each equal to its input while reporting success.

This is correct for the pooled antibody representations the function was written for, where
pooling over the whole sequence is a reasonable fallback. It is wrong for a cliff rule,
which would report a CDR scope that is a copy of the full-chain scope. `cliff_rules.py` now
counts the strings that came back unchanged, records the count in its info, and leaves the
criterion out when every one did. `antibody.py` is unchanged.

**Open.** s01 and s02 call `_extract_cdr_sequences` directly through
`run_antibody_cliff_analysis`, so on a dataset ANARCI cannot number — the nucleotide
datasets in their own list — their CDR scope is the full sequence and is stored as a second
scope. Not fixed, because where s01 and s02 land in the new structure is not decided.

### A third copy of the n01 rule, in code nothing runs

`simeval.py:218`, `cliff_ground_truth`, computes ECFP4 Tanimoto at or above 0.55 with a gap
of 2.0 times the median absolute deviation — the n01 rule a third time, after n25 and n35.
Its only caller is `run_battery` at `:470`, and `run_battery` is called by nothing in the
repo. `feature_ablation.py` and `triplet_similarity.py` import simeval's distance functions
and not its cliff code. So simeval defines a cliff rule and an AUROC on it, and neither has
ever run.

### The two Tanimoto implementations, compared

They were never compared before 2026-11-02. On binary bit vectors
`simeval.dist_tanimoto` and `cliff_rules.tanimoto` agree to 7.3e-09, which is simeval's
float64 against `cliff_rules`' float32, on 200 molecules at 256 bits. They are not
comparable on count vectors: simeval's form uses the inner products, which is the
continuous Tanimoto and correct for counts, while `cliff_rules` sums the vector as if its
entries were bits. That is not a fault in either, because `cliff_rules` only ever builds
binary fingerprints, but it means simeval's `ecfp4_count` distances must not be swapped for
`cliff_rules`' matrix without converting to bits first.

### Where the generic framework is taken from, and what it costs

`sim_moleculeace`'s second measure turns every atom to carbon and every bond to single. The
comment above `_bits_from` said that transform was applied to the molecule's Bemis-Murcko
scaffold. It is not, in our code or in theirs: `MakeScaffoldGeneric` is applied to the whole
molecule, with the plain Murcko scaffold as the fallback where it raises. Read in their own
source on 2026-11-02 (`/Volumes/seagate/moleculeace_src/cliffs.py:157-169`,
`get_scaffold_matrix`). The comment in `cliff_rules.py` now says this and cites those lines.
Nothing about the cliff sets changes: the rule as implemented already reproduced their
`cliff_mol` column exactly, which is section 3b.

That transform is the most expensive thing in the suite. Making a chemically impossible
all-carbon, all-single-bond copy of a drug-sized molecule forces RDKit to strip hydrogens and
re-check valences repeatedly. hERG Ki's 1,412 molecules take 1,410s, against about 2s for a
whole 28-point gap axis on an ordinary fingerprint.

**The fingerprints were being rebuilt for every cutoff, fixed 2026-11-02.** `CliffContext.sim`
stores each similarity matrix under a name and builds it once. The bit matrices those are
computed from were not stored: `sim_moleculeace` called `_bits_from` directly on every call.
The threshold sweep asks for that criterion at six cutoffs, so one dataset was set to spend
2h21m building six identical copies of one matrix for 24 minutes of distinct work. The
fingerprints now go through `CliffContext.bits`, whose key is the fingerprint type **and its
width**, because this criterion asks for 1,024 bits and every other asks for 2,048 and a
type-only key would have returned the wrong width. Checked on 20 molecules: 1.51s at the
first cutoff, 0.00s at the next two, 30 similar pairs at all three. The sweep's first launch
was stopped 35 minutes in and restarted on the fixed code; its log is kept as
`/Volumes/seagate/cliff_census_sweep/herg_ki.log.before_bits_cache_fix`.

---

## 2. What the literature uses

### Two families, and they are not interchangeable

**Matched molecular pairs.** Two molecules differ by one chemical transformation at one
site. There is no similarity threshold at all — the pair list is enumerated. Introduced
as MMP-cliffs by Hu, Hu, Vogt, Stumpfe and Bajorath, J. Chem. Inf. Model. 2012, 52:1138.
This is the Bajorath group's preferred definition. Binary, and interpretable as
chemistry.

**Fingerprint similarity.** Tanimoto on a 2D fingerprint at or above a cutoff. The
document gives ECFP4 at 0.55 and MACCS at 0.85 as the conventional pairs of fingerprint
and cutoff. Continuous, and dependent on which fingerprint.

The concept is traced to Maggiora 2006, J. Chem. Inf. Model. 46:1535, "On outliers and
activity cliffs — why QSAR often disappoints".

### The label gap, by source

| Source | Gap | Similar side |
|---|---|---|
| Bajorath group, classic | 2 log units (100-fold), usually with one molecule sub-micromolar | matched molecular pair |
| Dablander et al. 2023 | 2 log units; 1 to 2 log units is a half-cliff and is dropped | matched molecular pair, transformation-size restricted |
| ACNet (Zhang et al. 2023) | 2 log units in pKi | matched molecular pair |
| MoleculeACE (van Tilborg et al. 2022) | 1 log unit (10-fold) | 0.90 on any one of three measures |

MoleculeACE's rule differs from this repo's family A in both halves at once, and in
opposite directions. Its similar side is stricter: 0.90 against 0.55. Its label side is
looser: 1 log unit against the 2 median absolute deviations of family A. So it is not
reachable from family A by moving one number.

MoleculeACE's three measures, any one of which at 0.90 makes a pair similar: Tanimoto on
ECFP, Tanimoto on the scaffold's ECFP, and a scaled Levenshtein distance on the SMILES
string.

Dablander's matched pairs are restricted by transformation size: the shared core is at
least twice the heavy-atom count of either variable part, a variable part is at most 13
heavy atoms, the two variable parts differ by at most 8 heavy atoms, and cuts are at
single exocyclic bonds.

### The continuous scores

- **SALI**, Guha and Van Drie, J. Chem. Inf. Model. 2008, 48:646. The absolute label
  difference divided by one minus the similarity. Per pair.
- **SARI**, Peltason and Bajorath, J. Med. Chem. 2007, 50:5571. A continuity score and a
  discontinuity score combined. Per dataset, not per pair.
- **SAS maps**, pairwise structural similarity plotted against label difference, with a
  cliff quadrant (Medina-Franco and co-workers).
- **MODI**, Golbraikh, Muratov, Fourches and Tropsha, J. Chem. Inf. Model. 2014, 54:1.
  The share of nearest-neighbour pairs in the same class. The document gives 0.65 as the
  cutoff between modelable and not, for binary tasks, and says a nearest-neighbour pair
  in two different classes is formally a cliff. **RMODI** extends it to regression and
  **ROGI** measures landscape roughness; the document says SARI, MODI, RMODI and ROGI are
  all implemented in the `coleygroup/rogi` package.
- **Averaging SALI across several representations**, which the document names as the
  existing way of reducing how much the answer depends on one representation. This is a
  third shape for question 3 and the handoff does not list it.

### Datasets that ship cliff labels

- **MoleculeACE**, van Tilborg, Alenicheva and Grisoni, J. Chem. Inf. Model. 2022,
  62:5938. 48,707 molecules across 30 ChEMBL v29 targets, of which the document says
  35,632 are cliff-relevant. Per-target sizes run from 615 molecules (JAK1) to 3,658
  (dopamine D3). Labels are Ki or EC50. The share of each target's molecules that are
  cliff molecules runs from 9 to 52 per cent. Supplies an RMSE computed on cliff
  molecules alone. **Two published corrections in 2023**, one of them a split bug from
  mislabelled cliff pairs: the document says to use the current GitHub data and not the
  original tables. Its own split is spectral clustering on ECFP into 5 clusters, then an
  80/20 split holding the cliff share constant — a split this repo does not use, because
  `build_moleculeace.py` computes its own Bemis-Murcko scaffold groups instead.
- **ACNet**, Zhang et al. 2023, arXiv:2302.07541. More than 400,000 matched molecular
  pairs against 190 targets, of which more than 20,000 are cliffs and about 380,000 are
  not, from ChEMBL v28. Framed as pair classification rather than regression. Includes a
  small-data subset of 13 tasks and 835 samples.
- **Dablander's three sets**, the ones this repo already loads. Dopamine D2: 6,333
  molecules, 35,484 matched pairs, 461 of those pairs cliffs. Factor Xa: 3,605 molecules,
  21,292 pairs, 1,896 cliffs. SARS-CoV-2 Mpro: 1,924 molecules, 12,594 pairs, 521
  cliffs. Pairs enumerated with `mmpdb`.
- **Bajorath group collections** from ChEMBL, covering matched-pair cliffs, coordinated
  cliffs, cliff clusters and cliffs defined on crystal structures (Stumpfe and Bajorath,
  J. Med. Chem. 2012, 55:2932; Stumpfe, Hu, Dimova and Bajorath, J. Med. Chem. 2014,
  57:18; Stumpfe and Bajorath, J. Chem. Inf. Model. 2012, 52:2348; Stumpfe, Hu and
  Bajorath, ACS Omega 2019, 4:14360).

### Dablander, and what of his design this repo already carries

The document treats him as the priority. Markus Ferdinand Dablander took a DPhil at
Oxford's Mathematical Institute, thesis submitted October 2023, supervised by Garrett M.
Morris of the Oxford Protein Informatics Group, Renaud Lambiotte, and Thierry Hanser of
Lhasa Limited. Thesis at arXiv:2411.13688. The Lhasa connection is worth noting because
this repo already loads a Lhasa hERG set (`tests/experiments/federated/lhasa_herg_loader.py`).

His flagship paper is "Exploring QSAR Models for Activity-Cliff Prediction", J.
Cheminform. 2023, 15:47. Its design, per the document:

- **Nine models**, every pairing of three representations and three model types. The
  representations are ECFP at 2,048 bits and radius 2, a 200-dimensional physicochemical
  descriptor vector normalised to the unit hypercube by the training set's own cumulative
  distributions, and a max-pooled graph isomorphism network. The model types are random
  forest, k-nearest neighbours and a multilayer perceptron.
- **Four settings for the matched pairs**, not pooled. M_train has both molecules in
  training. M_inter has one molecule held out and one in training, so one measured label
  is known, and he classifies on the gap between the known label and the predicted one.
  M_test has both molecules held out. M_cores is the subset of M_test whose cores never
  appear in training, which the document puts at 7 to 13 per cent of M_test pairs.
  M_inter is about twice the size of M_train and of M_test.
- **Metrics**: Matthews correlation on the cliff call as the primary one, then
  sensitivity and precision on the cliff call, then accuracy on which of the two
  molecules is more potent.
- **Results**: Matthews correlation above 0.5 on M_inter for many models, falling to
  about 0.3 on M_test and M_cores, and the fall is in sensitivity rather than precision.
  Graph isomorphism networks were generally best on Matthews correlation. ECFP with a
  multilayer perceptron was consistently best on plain prediction error. Accuracy on
  which molecule is more potent exceeded 0.9 on M_inter and 0.8 on M_test and M_cores,
  restricted to pairs predicted to be cliffs.
- **A linear relationship between prediction error and the cliff call**, which the
  document reports as holding consistently: lower mean absolute error went with higher
  Matthews correlation on the cliff call.
- The document quotes his reading of why graph isomorphism networks do well: they "learn
  an embedding of chemical space in which the distance between two compounds is
  reflective of activity difference rather than structural difference".

His second method is a twin network (thesis chapter 4, and a 2021 poster at the RSC
AI in Chemistry symposium). One shared featuriser embeds each molecule. A cliff head
takes the componentwise maximum of the two embeddings, which makes it indifferent to
which molecule comes first, and predicts one of cliff, half-cliff or not. A second head
takes the difference of the embeddings through a multilayer perceptron with no bias terms
and arctan activations, which makes it flip sign when the two molecules swap, and
predicts which is more potent. On Mpro the document reports cliff precision of 0.110 for
an ECFP-based twin against 0.131 for a graph-based one, with accuracy on which molecule
is more potent of 0.683 against 0.737.

His third is Sort & Slice (J. Cheminform. 2024, arXiv:2403.17954): instead of hashing
ECFP substructures into a fixed number of bits, sort substructures by how often they
occur in the training set and keep the most frequent. No bit collisions. The document
says it beat hash-based folding across tasks, models and ECFP settings.

### Earlier work on predicting cliffs

- **Support vector machines on matched-pair kernels**, Heikamp, Hu, Yan and Bajorath,
  J. Chem. Inf. Model. 2012, 52:2354, which the document reports at 80 to 90 per cent
  accuracy. Tamura, Jasial, Miyao and Funatsu, Molecules 2021, 26:4916, added an
  applicability domain and interpretability to the same kernel.
- **Condensed graph of reaction on matched pairs**, Horvath, Marcou, Varnek, Kayastha,
  de la Vega de León and Bajorath, J. Chem. Inf. Model. 2016, 56:1631. A matched pair is
  encoded as one pseudo-molecule: the shared core, the two exchanged substituents, and
  hypothetical zero-order bonds joining them. This repo already builds condensed graphs
  of reaction for reaction data (`src/kirby/representations/reaction.py`).
- **Convolutional networks on molecular images**, Iqbal, Vogt and Bajorath, Artif.
  Intell. Life Sci. 2021, 1:100022.
- **Structure-based prediction** from binding modes, Husby, Bottegoni, Kufareva, Abagyan
  and Cavalli, J. Chem. Inf. Model. 2015, 55:1062.
- **A large-scale comparison**, J. Cheminform. 2022, 13:66, over support vector machines,
  gradient boosting, random forests and a fully connected network on matched-pair
  representations.
- **Recent cliff-aware deep learning**, which the document flags as provisional and in
  some cases not peer reviewed: ACANet (contrastive learning, 2024), ACtriplet (triplet
  loss, 2024 to 2025), MaskMol (masked molecular images, 2024 to 2025) and MTPNet (2025).

### Fingerprints beat deep models on cliff-rich data

The document reports this as the finding of both benchmarks, independently. In
MoleculeACE a support vector machine on ECFP was best on average, with no significant
difference from random forests or gradient boosting on ECFP under a Wilcoxon rank-sum
test. Every graph model (GAT, GCN, AFP, MPNN) and every string model (transformer, CNN,
LSTM) underperformed a plain ECFP multilayer perceptron. Among the deep models an LSTM
with transfer learning beat ChemBERTa, which had been pretrained on 10 million PubChem
molecules. The document's summary is that the choice of representation mattered more than
the choice of algorithm. ACNet reported the same advantage for ECFP, and that graph
networks failed to generalise when the split was by target.

The named cause is representation collapse: near-identical embeddings for near-identical
structures whose potencies differ.

### How much the count moves with the representation

Two separate measurements, both bearing on question 3.

Bajorath group, across 128 ChEMBL activity classes: 18,749 cliffs under matched pairs,
61,524 under ECFP4, and 79,338 under MACCS. As a share of qualifying pairs they report
5.47 per cent under ECFP4, 5.47 per cent under FCFP4, and 6.78 per cent under MACCS.

Dabrowski-Tumanski, Topolski, Plewczynski and Jetka, "The Geometry of Activity Cliffs",
arXiv:2605.30831. Fifteen combinations of embedding and distance on Dablander's three
datasets — the same three this repo loads, taking his Ki values and his pre-enumerated
matched-pair lists. The embeddings are Morgan, RDKit and MACCS fingerprints under
Tanimoto and Dice, plus ChemBERTa (ChemBERTa-77M-MLM), MolFormer (MoLFormer-XL-both-10pct)
and Chemeleon. Its six steps are pairwise distance geometry, cliff enrichment, the
distribution of activity gradients, persistent homology of the cliff subspace, predictive
benchmarking, and a matched-pair and stereoisomer analysis.

**The number of cliff pairs on one fixed dataset moved by up to 55 times across those
combinations**: 142 cliff pairs at 0.90 Morgan-Tanimoto, 752 under Chemeleon cosine, and
7,903 under RDKit-Dice. The document quotes the paper as finding the representation
effect larger than any difference between the three protein targets.

Three further results from it. Morgan-Tanimoto gave the strongest cliff enrichment and
the best generalisation across scaffolds. MolFormer under cosine was the only combination
with meaningful sensitivity to stereochemistry. ChemBERTa failed throughout, through
embedding collapse. No combination was sensitive to both stereochemistry cliffs and
matched-pair cliffs at once.

The document flags this paper as a preprint, future-dated, and not peer reviewed, and
says to treat its numbers as provisional. The handoff notes it is already cited in
`thesis.tex`.

### The measurement-noise floor

Kramer, Kalliokoski, Gedeck and Vulpetti, J. Med. Chem. 2012, 55:5165, on heterogeneous
public Ki data: a mean error of 0.44 pKi units, a standard deviation of 0.54 pKi units,
and a median error of 0.34 pKi units. The document says this implies a highest reachable
R² of about 0.81 on large sets.

Their number required filtering first, and the document names what for: unit
transcription errors, stereoisomers not distinguished from each other, and single
measurements cited repeatedly, which it puts at 90 per cent of all pairs in the raw data.

The document draws the consequence that a 2-log-unit gap sits at roughly four times the
0.54 standard deviation, which is the argument for 2 log units rather than 1. It adds
that mixed IC$_{50}$ data is noisier (Kalliokoski et al., PLOS ONE 2013) and that
combining Ki and IC$_{50}$ from different sources adds more (J. Chem. Inf. Model. 2024,
4c00049). It quotes the position that matched-pair and cliff analysis on individual data
points requires going back to the original literature to check each value is comparable.

ChEMBL's own flags, per the document: a potential-duplicate flag when the pChEMBL
difference between the earliest and a later reference is under 0.02, and a
transcription-error flag at differences of exactly 3 or 6 log units.

### Cliffs against noise, in the literature's own terms

The document records that cliffs are sometimes dismissed as measurement error, and that
Dablander's position is that deleting them loses real structure-activity information. Its
framing of the task is to separate the two rather than to delete either. That is question
5's framing, arrived at independently.

On how much of prediction error is noise against cliffs against method, it names
Cortés-Ciriano and Bender, Mol. Inform. 2017, "experimental error, kurtosis, activity
cliffs, and methodology: what limits the predictivity of QSAR models?" as the
decomposition. It also reports Sheridan as finding cliff density a strong predictor of
how modelable a dataset is.

### What of this exists in the repo, read 2026-11-01

Faithful to the papers:

- **The half-cliff band and the 1.5-log-unit cutoff.** `opig_activity_cliff_qsar.py:379`
  drops pairs whose true gap is at or above 1.0 and under 2.0 log units. `:329-332` and
  `:367` score a predicted cliff by 2·arctan(|Δpred|/1.5)/π, which crosses 0.5 exactly
  when the predicted gap passes 1.5 log units, and the comment names that as Dablander's
  cutoff.
- **Kramer's floor.** `n14_activity_cliff_landscape.py:101`,
  `KRAMER_NOISE_FLOOR_PKI = 0.54`, the standard deviation above. n14 already compares
  each dataset's residual spread against it (`:144-146`).
- **The molecule counts.** `chembl_qsar.py` expects 6,333 D2 molecules, 3,605 factor Xa
  and 1,924 Mpro (`:44`, `:53`, `:62`), the same three counts the document gives. The
  matched-pair and cliff-pair counts above are asserted nowhere in the code and were not
  measured here.

Present as representations, so the Geometry paper's sweep is reachable with what is
already built: ECFP4 and MACCS and an RDKit fingerprint and a graph isomorphism network
(`src/kirby/representations/molecular.py`), ChemBERTa and MolFormer (same file),
Chemeleon (`src/kirby/representations/forward_compat_molecular.py`), a physicochemical
descriptor vector under the name `pdv`, and condensed graphs of reaction
(`src/kirby/representations/reaction.py`).

Absent:

- **Two of Dablander's four pair settings.** `opig_activity_cliff_qsar.py:418` builds
  `M_inter` and `M_test` and nothing else. M_cores, his hardest setting, and M_train are
  not there. The docstring at `:338-346` explains the two it has and does not mention the
  two it lacks.
- **Dice, and Tanimoto, as hybrid distances.** `src/kirby/similarity/distance_config.py:66`
  holds four: Manhattan, Euclidean, cosine and Minkowski at p=0.5. The cliff rules compute
  ECFP4 Tanimoto directly and separately. So the Geometry paper varied the distance as well
  as the embedding, and two of its distances are not in this repo's decided list.
- **Sort & Slice.** No ECFP built by sorting substructures on training-set frequency.
- **MoleculeACE's `cliff_mol` labels**, read past by the loader (section 1).

### What the research document itself recommends

Recorded as that document's own suggestions, generated by a research session on
2026-11-01, not measured here and not decided by anyone:

- Report under two definitions side by side, matched pairs and a similarity rule, and do
  not assume they agree.
- Keep the gap at 2 log units for strict cliffs and keep 1 to 2 log units as an
  ambiguous band rather than forcing it either way.
- Use the disagreement between several embeddings and distances on one pair as a signal:
  a genuine cliff is close under a structural fingerprint and far apart in label, in the
  same assay, while noise is likelier where the embeddings disagree or the gap sits near
  the measurement floor.
- Build a measurement-noise null model at about 0.5 log units and trust only cliffs whose
  gap is distinguishable from it, preferring pairs from one assay.
- Add MolFormer under cosine for stereochemistry cliffs; be wary of ChemBERTa.
- If a dataset's MODI is under 0.65, expect poor modelability.
- If cliff counts swing by more than ten times across representations, treat the labels
  as largely an artefact of the representation and commit to the matched-pair definition.
- Model at the level of the pair, with a twin network, and take GIN-MLP, GIN-RF,
  ECFP-MLP and SVM-ECFP as the baselines any such model must beat.
- Use Dablander's four pair settings, to keep the one-label-known case apart from the
  both-held-out case.

Two of those sit outside what KIRBy changes. A twin network and the four baselines are
model changes, and KIRBy modifies the representation. Noted so the gap is a decision
rather than an oversight.

---

## 3. What to do about counts differing between datasets

### Measured 2026-11-02: every named rule on three datasets

`tests/experiments/noise/cliff_census.py`, run locally on all the molecules each loader
returns, no fold held out, no model fitted. Result files in
`/Volumes/seagate/cliff_census/`. The rules are the named combinations in
`tests/experiments/noise/cliff_rules.py`, each one a similarity criterion and a label-gap
criterion from a named source. A local run is for settling definitions and is not a
result.

The three counts already in section 1 reproduce exactly, which is the check that these
are the rules the repo ran: hERG Ki 451 of 1,412 and Caco-2 1,625 of 2,158 and MDR1-MDCK
125 of 2,641 under Tanimoto 0.55 with a gap of 2.0 median absolute deviations, and hERG
Ki 211 of 1,412 under the solved gap.

One row is one rule. "Cliff molecules" is molecules in at least one cliff pair, out of
every row the loader returns. "Gap against the floor" is the rule's label-gap cutoff
divided by 0.54, Kramer et al. 2012's standard deviation for two independent public Ki
measurements of one pair; under 1.00 means the rule calls pairs cliffs whose labels differ
by less than that. "Pairs inside the floor" is the share of that rule's own cliff pairs
whose label difference is under 0.54.

| Dataset | Rule | Cliff molecules | Cliff pairs | Similar pairs | Gap, label units | Gap against the floor | Pairs inside the floor |
|---|---|---|---|---|---|---|---|
| hERG Ki, 1,412 rows | n01 (ECFP4 Tanimoto 0.55, 2.0 median absolute deviations) | 451 (31.9%) | 710 | 10,809 | 1.210 | 2.24 | 0.0% |
| | moleculeace (any of three measures at 0.90, 1 log unit) | 316 (22.4%) | 302 | 4,099 | 1.000 | 1.85 | 0.0% |
| | calibrated (0.55, gap solved at a 15% target) | 211 (14.9%) | 211 | 10,809 | 1.760 | 3.26 | 0.0% |
| | hu2015 and n14 (0.55, absolute 2 label units) | 119 (8.4%) | 106 | 10,809 | 2.000 | 3.70 | 0.0% |
| | geometry_rdkit_dice (RDKit fingerprint Dice 0.90, 2 units) | 46 (3.3%) | 33 | 4,745 | 2.000 | 3.70 | 0.0% |
| | geometry_morgan (Morgan Tanimoto 0.90, 2 units) | 5 (0.4%) | 3 | 103 | 2.000 | 3.70 | 0.0% |
| Caco-2, 2,158 rows | n01 | 1,625 (75.3%) | 6,909 | 24,858 | 0.417 | 0.77 | 31.1% |
| | moleculeace | 393 (18.2%) | 544 | 19,954 | 1.000 | 1.85 | 0.0% |
| | calibrated | 324 (15.0%) | 430 | 24,858 | 1.209 | 2.24 | 0.0% |
| | hu2015 and n14 | 0 | 0 | 24,858 | 2.000 | 3.70 | — |
| | geometry_rdkit_dice | 0 | 0 | 13,469 | 2.000 | 3.70 | — |
| | geometry_morgan | 0 | 0 | 143 | 2.000 | 3.70 | — |
| MDR1-MDCK, 2,641 rows | calibrated | 397 (15.0%) | 251 | 459 | 0.196 | 0.36 | 54.2% |
| | n01 | 125 (4.7%) | 71 | 459 | 0.777 | 1.44 | 0.0% |
| | moleculeace | 108 (4.1%) | 120 | 1,054 | 1.000 | 1.85 | 0.0% |
| | hu2015 and n14 | 0 | 0 | 459 | 2.000 | 3.70 | — |
| | geometry_rdkit_dice | 0 | 0 | 242 | 2.000 | 3.70 | — |
| | geometry_morgan | 0 | 0 | 6 | 2.000 | 3.70 | — |

`dablander` needs the loader's matched pairs, which only Dablander's three ChEMBL targets
ship. `s12` and `activity_cliff_detector` read the representations and were not run in
this pass.

### What the table says, and what it cannot say

The spread on one dataset is wider than the spread between datasets. On hERG Ki the named
rules run from 5 to 451 of the same 1,412 molecules. Section 1's widest pair between
datasets, under a fixed rule, is 1,625 of 2,158 against 125 of 2,641.

Two rules go below the Kramer floor, and they are the two the thesis uses. n01's gap on
Caco-2 is 0.417 label units, 0.77 of the floor, and 31.1% of the 6,909 pairs it calls
cliffs differ by less than 0.54. The solved gap on MDR1-MDCK is 0.196 label units, 0.36 of
the floor, and 54.2% of its 251 cliff pairs are inside it. The solved gap reaches 15.0% on
MDR1-MDCK by lowering the gap to 0.50 of a median absolute deviation, which
`similarity_helpers.py`'s own comment already flagged as weak before the floor was brought
into it.

The absolute 2 label units flags no molecule at all on Caco-2 or MDR1-MDCK, under every
one of the three fingerprints. `similarity_helpers.py` records why for Caco-2: its median
absolute deviation is 0.2085 and a 2-unit gap is 9.6 of them.

**The floor comparison is against the wrong assay on two of the three datasets.** 0.54 is
a pKi number. Caco-2's label is log permeability and MDR1-MDCK's is a log efflux ratio,
and no measured repeatability figure for either is in this repo. So the two columns are
the right quantity measured against a floor borrowed from another assay. What each of
those assays repeats to is an open question and a decision about what to measure next.

The solved gap reaches its 15% target more often than expected and misses in a stated way.
Of 12 similarity criteria on hERG Ki it lands within one point of 15% under 10, missing at
Morgan Tanimoto 0.90 (10.5%, with 103 similar pairs of 996,166 to work with). On MDR1-MDCK
it misses under Morgan Tanimoto 0.90 (0.5%, 6 similar pairs) and ECFP4 Dice 0.90 (2.6%, 34
pairs). The pattern is that it reaches the target wherever enough pairs are similar and
misses where they are scarce, which is the same mechanism as QM9's 19 of 7,982.

### The three things noted before any of this was measured

The counts in section 1 were quoted from code comments. They have now been re-run and
agree.

Section 2 adds that the count also moves with the representation on one fixed dataset,
by up to 55 times across fifteen combinations of embedding and distance, measured on the
same three Dablander datasets this repo loads. So there are two spreads, not one: between
datasets under a fixed rule, and between representations on a fixed dataset. Option A and
option B in the handoff both address the first. Neither addresses the second.

Section 2 also supplies a third shape the handoff does not list: average SALI across
several representations, which the document names as the existing way of reducing how
much the answer depends on one representation.

## 3b. Measured against MoleculeACE's published labels, 2026-11-02

`tests/experiments/opig_moleculeace_cliff_check.py`, 30 ChEMBL targets, 48,714 molecules,
8,035.8 seconds, no target failed. Full table in
`tests/results/comparisons/opig_moleculeace_cliff_check.json`; 130 pairings of a similarity
criterion with a gap criterion, 120 of which produced a correlation.

⛔ **What this does and does not measure.** `cliff_mol` is van Tilborg et al.'s rule, not
nature. The first row below is a check on our own code. Every other row is the distance
between two definitions, and a low number there says the two disagree, not that either is
wrong.

| Similarity | Gap | Cliff molecules, ours | Theirs | Matthews, pooled | Median over targets | Worst target |
|---|---|---|---|---|---|---|
| `moleculeace_any_090` | `absolute_1` | 18,844 | 18,831 | 0.9987 | 1.0000 | 0.9809 |
| `moleculeace_any_090` | `mad_1x` | 21,083 | 18,831 | 0.8304 | 0.8369 | 0.2266 |
| `moleculeace_any_090` | `kramer_noise_floor` | 29,173 | 18,831 | 0.6497 | 0.6441 | 0.3596 |
| `moleculeace_any_090` | `solved_target_15pct` | 7,309 | 18,831 | 0.5201 | 0.5228 | 0.3883 |
| `ecfp4_top_0.5pct` | `absolute_1` | 26,771 | 18,831 | 0.4679 | 0.4929 | 0.2349 |
| `ecfp4_tanimoto_055` | `mad_2x` | 19,064 | 18,831 | 0.3172 | 0.3238 | 0.0169 |
| `ecfp4_tanimoto_055` | `solved_target_15pct` | 7,313 | 18,831 | 0.2132 | — | −0.0907 |

**Our implementation of their rule is right.** It recovers 18,823 of their 18,831 cliff
molecules, adds 21 of its own and misses 8, over all 30 targets. Matthews is exactly 1.0000
on 18 of the 30 and its lowest is 0.9809. The earlier run of the same code gave 0.5566 on
one target because it used the plain Bemis-Murcko scaffold rather than the generic
framework; that is recorded in `cliff_rules.py` as a dated block.

**The rule the thesis uses disagrees with theirs almost completely, at nearly the same
count.** ECFP4 Tanimoto 0.55 with a gap of 2.0 median absolute deviations calls 19,064
molecules cliffs against their 18,831 — within 1.2 per cent on the count — and Matthews is
0.3172. On its worst target it is 0.0169, which is no better than drawing the same number
of molecules at random. A matching count is not a matching set.

**The gap matters more than the similarity here.** Holding their similarity criterion fixed
and changing only the gap moves Matthews from 0.9987 to 0.5153. Holding the gap at an
absolute 1 log unit and changing only the similarity moves it from 0.9987 to 0.4679. Their
labels are reproducible only with both halves of their rule, and the string-similarity part
is what does most of the work: on CHEMBL214_Ki the Levenshtein part alone accounts for
1,055 cliff molecules, the generic scaffold 658, and ECFP Tanimoto 148.

**56 of the 120 pairings anti-correlate with MoleculeACE on at least one target**, the
lowest being RDKit Dice at 0.90 with an absolute 2 log units at −0.1935 and MACCS Tanimoto
at 0.85 with 1.0 median absolute deviations at −0.1473. A negative Matthews means that on
that target the rule's cliff molecules are less likely to be theirs than a molecule picked
at random would be.

## 3c. The threshold sweep, hERG Ki measured 2026-11-02

`cliff_census.py --sweep`, 75 similarity points by 28 gap points, 2,100 settings, 268s on
hERG Ki's 1,412 molecules after InChIKey dedup. No setting errored. Result file
`/Volumes/seagate/cliff_census_sweep/herg_ki__sweep.json`, log kept at
`.../scratchpad/sweep_logs/herg_ki.log`. hERG Ki exposes no repeat structure to the loader,
so every gap here is reported against Kramer's 0.54 pKi, which is borrowed.

### The similarity criterion alone moves the count from 1 per cent to 100

Share of the 1,412 molecules called a cliff, holding the gap at twice the labels' median
absolute deviation (1.21 pKi) and moving only the similarity:

| Criterion | Share of 1,412 molecules |
|---|---|
| ECFP4 Tanimoto 0.90 (the published value) | 1% |
| MoleculeACE, any of three, 0.90 | 13% |
| MACCS Tanimoto 0.90 | 14% |
| RDKit Dice 0.90 | 18% |
| top 1% of pairs by ECFP4 | 31% |
| 5 nearest neighbours by ECFP4 | 37% |
| MACCS Tanimoto or RDKit Dice at 0.50 or below | 100% |
| ECFP4 Tanimoto 0.95 | 0% |

MACCS Tanimoto and RDKit Dice saturate: every molecule is a cliff anywhere below 0.50, so
those cutoffs separate nothing. ECFP4 Tanimoto empties at the other end, 1 per cent by 0.85.

### At the published similarity cutoff hERG Ki has 103 similar pairs

ECFP4 Tanimoto 0.90 leaves 103 similar pairs out of 996,166. At the standard gap of twice
the median absolute deviation, 5 of those 103 are cliffs, which is 0.6 per cent of molecules.
Any result conditioned on that rule on this dataset rests on five pairs.

### A target share can be unreachable, and then the gap collapses to zero

`gap_solved_to_target` returns the gap whose achieved share is NEAREST the target, not one
that reaches it (its docstring says so, with QM9's 19 of 7,982 molecules as the example).
Measured here: at ECFP4 Tanimoto 0.90 the largest reachable share is 10.5 per cent, because
only 103 pairs are similar at all. Asking for 15, 20 or 30 per cent therefore returns a gap
of **0.000 pKi** and 96 of the 103 pairs, 10.5 per cent of molecules, in all three cases.
At a zero gap "cliff" means a similar pair whose labels differ at all, so the definition
stops depending on activity. 80.2 per cent of those pairs differ by less than Kramer's 0.54.

At a 10 per cent target the gap is 0.031 pKi, 6 per cent of the noise floor, with 79.1 per
cent of the called pairs inside it. The same solver on the looser MoleculeACE similarity
behaves: 15 per cent is reached at a gap of 1.137 pKi, 2.1 times the floor, with none of the
238 pairs inside it. The failure is not the solver, it is asking a share of a similarity
criterion that has too few similar pairs to supply it.

### Where the gap crosses the noise floor

In 170 of the 2,100 settings the gap required to call a cliff is below 0.54 pKi, so a repeat
measurement of the same pair could produce the difference by itself. All 170 are at
saturating similarity cutoffs. At ECFP4 Tanimoto 0.90, the gap criteria that land above the
floor are 1.0 median absolute deviations and up (0.605 pKi, 1.12 times the floor) and every
absolute cutoff from 1 log unit (1.85 times). The two that land below it are 0.5 median
absolute deviations (0.302 pKi, 0.56 times) and an absolute 0.5 log units (0.93 times).

### Caco-2, measured 2026-11-02

2,158 molecules, 2,327,403 pairs, 964.7s for the same 2,100 settings, no setting errored.
Caco-2 also exposes no repeat structure to the loader. ⛔ Its label is log permeability, not
pKi, so Kramer's 0.54 is not merely borrowed here, it is the wrong unit; the sweep's count of
486 of 2,100 settings "below the floor" should not be read as a noise statement for this
dataset. Its own labels are much tighter than hERG Ki's: twice the median absolute deviation
is 0.417 label units against hERG Ki's 1.210.

Share of molecules called a cliff at twice the median absolute deviation, Caco-2 against
hERG Ki at the same cutoff:

| Criterion | Caco-2 | hERG Ki |
|---|---|---|
| ECFP4 Tanimoto 0.90 | 1% | 1% |
| MoleculeACE, any of three, 0.90 | 49% | 13% |
| MACCS Tanimoto 0.90 | 64% | 14% |
| RDKit Dice 0.90 | 51% | 18% |
| top 1% of pairs by ECFP4 | 74% | 31% |
| 1 nearest neighbour by ECFP4 | 25% | 11% |

Caco-2 is the denser set: 8,377 pairs are similar at the MoleculeACE cutoff against hERG Ki's
2,594, and its tighter labels mean the same relative gap is a smaller absolute one. So the
two datasets agree on the published ECFP4 cutoff and disagree by a factor of three or four
everywhere else.

### The unreachable target share repeats on the second dataset

| Criterion | Similar pairs | Gap returned for a 15% target | Share actually reached |
|---|---|---|---|
| hERG Ki, ECFP4 Tanimoto 0.90 | 103 of 996,166 | **0.000** | 10.5% |
| Caco-2, ECFP4 Tanimoto 0.90 | 143 of 2,327,403 | **0.000** | 10.8% |
| hERG Ki, MoleculeACE 0.90 | 2,594 | 1.137 pKi | 15.0% |
| Caco-2, MoleculeACE 0.90 | 8,377 | 1.011 log units | 15.0% |
| hERG Ki, top 1% of ECFP4 pairs | 9,994 | 1.710 pKi | 15.1% |
| Caco-2, top 1% of ECFP4 pairs | 23,546 | 1.194 log units | 15.0% |

Two datasets, same shape: at ECFP4 Tanimoto 0.90 there are too few similar pairs for a 15 per
cent share to exist, the solver returns a zero gap, and the rule stops depending on activity.
Every criterion with thousands of similar pairs reaches the target at a gap of about one label
unit.

### MDR1-MDCK, measured 2026-11-02, and it is the sparse one

2,641 molecules, 3,486,120 pairs, 1,094.0s, no setting errored. 2,318 Bemis-Murcko scaffolds
for 2,641 molecules, close to one scaffold per molecule, and the sweep shows what that does.
Its label is a log efflux ratio, so Kramer's 0.54 is the wrong unit here too. No repeat
structure exposed to the loader. Twice the median absolute deviation is 0.777 label units.

**At ECFP4 Tanimoto 0.90 this dataset has six similar pairs and zero cliffs.** Not a small
number, six. MoleculeACE's rule at 0.90 leaves 282 similar pairs and 31 cliff pairs, 2.2 per
cent of molecules. Every fingerprint criterion collapses on it: ECFP4 Tanimoto is at 1 per
cent of molecules by 0.70, and MACCS Tanimoto, which saturated at 100 per cent on the other
two datasets up to 0.50, is at 1 per cent by 0.90.

### All three datasets, the published cutoff against the share-based ones

One row is one dataset at one similarity criterion, gap held at twice the median absolute
deviation. "Similar pairs" is out of every pair of molecules in the dataset.

| Dataset | Criterion | Similar pairs | Cliff pairs | Cliff molecules |
|---|---|---|---|---|
| hERG Ki, 1,412 mols | ECFP4 Tanimoto 0.90 | 103 of 996,166 | 5 | 0.6% |
| Caco-2, 2,158 mols | ECFP4 Tanimoto 0.90 | 143 of 2,327,403 | 7 | 0.6% |
| MDR1-MDCK, 2,641 mols | ECFP4 Tanimoto 0.90 | 6 of 3,486,120 | 0 | 0.0% |
| hERG Ki | MoleculeACE 0.90 | 2,594 | 183 | 12.6% |
| Caco-2 | MoleculeACE 0.90 | 8,377 | 2,383 | 48.9% |
| MDR1-MDCK | MoleculeACE 0.90 | 282 | 31 | 2.2% |
| hERG Ki | top 1% of ECFP4 pairs | 9,994 | 655 | 30.5% |
| Caco-2 | top 1% of ECFP4 pairs | 23,546 | 6,512 | 74.5% |
| MDR1-MDCK | top 1% of ECFP4 pairs | 37,826 | 12,836 | 88.9% |
| hERG Ki | 5 nearest neighbours, ECFP4 | 4,785 | 524 | 37.1% |
| Caco-2 | 5 nearest neighbours, ECFP4 | 7,471 | 1,928 | 66.2% |
| MDR1-MDCK | 5 nearest neighbours, ECFP4 | 9,618 | 3,035 | 83.5% |

The fixed cutoff gives a comparable share across datasets (0.6, 0.6, 0.0 per cent) by giving
almost nothing to work with. The share-based criteria give a usable count by making the
similarity cutoff itself dataset-dependent, and their shares then disagree between datasets
by a factor of three or more. That is the trade, measured, and it is the choice behind
picking a rule.

### What the 15 per cent target does on each of the three

| Dataset, criterion | Gap returned | Share reached |
|---|---|---|
| hERG Ki, ECFP4 Tanimoto 0.90 | **0.000** | 10.5% |
| Caco-2, ECFP4 Tanimoto 0.90 | **0.000** | 10.8% |
| MDR1-MDCK, ECFP4 Tanimoto 0.90 | **0.000** | 0.5% |
| MDR1-MDCK, MoleculeACE 0.90 | **0.009** | 15.0% |
| hERG Ki, MoleculeACE 0.90 | 1.137 | 15.0% |
| Caco-2, MoleculeACE 0.90 | 1.011 | 15.0% |
| hERG Ki, top 1% of ECFP4 pairs | 1.710 | 15.1% |
| Caco-2, top 1% of ECFP4 pairs | 1.194 | 15.0% |
| MDR1-MDCK, top 1% of ECFP4 pairs | 2.103 | 15.0% |

Two ways the share target fails, both on measured data. It can be unreachable, and then the
gap goes to zero and a cliff is any similar pair whose labels differ at all. Or it can be
reached at a gap of 0.009 label units, which is MDR1-MDCK with MoleculeACE's similarity: 282
similar pairs can only cover 15 per cent of 2,641 molecules if nearly every one of them
counts, so the solver buys the share by abandoning the gap. Neither case raises an error.
Both report a 15 per cent cliff share that means nothing about activity.

The guard this implies: a share target is only usable when the similarity criterion supplies
enough similar pairs, and `achieved_pct` alone does not catch it — MDR1-MDCK reports 15.0.
The gap in label units has to be reported with it, which is what the sweep rows carry.

## 4. Whether hybrids change which pairs are cliffs

Not started. The two open decisions are named in section 1 under n25.

Section 2 bears on the circularity risk the handoff raises. Dablander's reading of why
graph isomorphism networks predict cliffs well is that their embedding puts molecules at
distances reflecting activity difference rather than structural difference. That is the
same mechanism as the handoff's worry, stated in the literature as a strength rather than
a confound. It is an argument for the handoff's rule that the hybrid is searched on
training rows only and cliffs counted on held-out pairs.

The Geometry paper's result that no combination is sensitive to both stereochemistry
cliffs and matched-pair cliffs is the closest published answer to this question, on the
same three datasets. The handoff's testable idea — a pair that is a cliff under ECFP4 but
not in a hybrid is one whose label difference the hybrid's features explain — would be a
second reading of the same effect.

Section 2 also names a specific first check: MolFormer under cosine was the only one of
fifteen combinations with meaningful stereochemical sensitivity, and MolFormer is already
in the repo.

## 5. Cliffs against what a neighbour-based detector flags

Not measured. Five things that belong here:

- `ActivityCliffAwareEstimator` already separates the two by neighbour spread
  (section 1).
- The MoleculeACE `cliff_mol` labels are already on disk and unread (section 1).
- `KRAMER_NOISE_FLOOR_PKI = 0.54` at `n14:101` is already the null model the research
  document recommends building, at least for Ki data.
- The literature's own framing is to separate cliffs from noise rather than delete
  either, and names Cortés-Ciriano and Bender, Mol. Inform. 2017, as the decomposition of
  prediction error into noise, cliffs and method. Unread here.
- ChEMBL's duplicate and transcription-error flags, and the finding that single
  measurements cited repeatedly make up 90 per cent of pairs in raw public Ki data, are
  curation records of the kind the handoff lists as flagging noise without using
  structure. n22 already reads one such field.
