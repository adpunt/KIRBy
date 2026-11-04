# Handoff: the activity cliff study

Version 1, 2026-11-04. Written to Adelaide's own structure, which she set out on
2026-11-04 after rejecting the previous handoff's framing.

**Supersedes `HANDOFF_PAPER3_CLIFFS.md`**, which is kept only for the measurements already
recorded in it. Its section A holds eight finished pieces of work with their numbers, and
its sections E, F and G hold the sweep grid, the places cliffs enter the noise experiments,
and the settled decisions. Nothing in it should be treated as the plan.

`PAPER_3_OUTLINE.md` is the documentation of record. This handoff exists to get the work
**implemented and ready to run inside the ledger, on the available datasets.** Nothing
here replaces the outline; where the two disagree the outline wins and this file is wrong.

---

## 1. What this study is for, and the two ways it ends

Three claims, in Adelaide's order:

1. **That activity cliffs can be detected accurately**, on a criterion more flexible than
   ECFP4 Tanimoto at a fixed cutoff, which does not work on every dataset.
2. **That detecting them makes noise mitigation work better.**
3. **That the study can offer a clearer definition of an activity cliff, stated twice —
   with hybrids and without** (Adelaide, 2026-11-04). One definition that takes the
   existing ones into account and says why it is better. Conditional on 1 and 2.

Claim 3 is the writing goal, and it has a constraint that decides whether it can be made
at all. "Better" cannot be asserted; it has to be the measured outcome of section 4, on
all three counts: closer agreement with the one external cliff set that exists, a cliff
set that survives changing the representation, and a larger gain in section 4.3 than
sparing the same number of molecules at random. A definition that wins on none of those is
a definition with no claim behind it, however much simpler it reads.

The paper needs both statements, because they do different jobs. **Without hybrids** the
definition runs on a fingerprint, so it is comparable with the published definitions and a
reader with none of this machinery can reproduce it. **With hybrids** it uses the searched
feature space, which is where any advantage over the published definitions would come
from. Which of the two is the paper's headline is Adelaide's.

**If 1 and 2 both hold well enough, Paper 3 splits into two papers:** noise methods with
hybrid features, and cliff detection with cliff-aware noise mitigation.

**If they do not hold, or hold weakly,** a smaller part of this study is reported inside
Paper 3, confined to how cliffs connect to noise — including the outcome that cliffs can
be identified but not in a way that helps.

Both endings are reportable. The design below has to produce the number that decides
which, and that number is in section 4.

---

## 1b. The defect this overhaul exists to fix

Adelaide, 2026-11-04: *"this never should have been made. This is why the results in the
thesis are not what I intended."*

**What the code does, read 2026-11-04.**
`similarity_helpers.cliff_structural_similarity(data, reps, ...)` takes the dataset and its
representations. On a molecular dataset whose SMILES rdkit can parse, it returns ECFP4
Tanimoto over 2,048 bits at radius 2 and never touches `reps`. The representations are
reached only in the `except` branch, when the SMILES fail to parse.

**Who calls it.** `s04_activity_cliffs.py:185` and `s12_cross_domain_cliffs.py:494`.
`n01` and `n25` build ECFP4 the same way in their own code.

**The consequence.** No hybrid has ever defined a cliff on any molecular dataset in this
repo. Every cliff set behind every cliff number in the thesis is an ECFP4 cliff set, and
the hybrid appears only as something scored on cliffs that ECFP4 picked. The hybrids were
built to find cliffs; they have never been used to find one.

**Nobody decided this on the record.** `HANDOFF_PAPER3_CLIFFS.md` carried a line saying the
hybrid is kept out of choosing a cliff definition, attributed to Adelaide and dated
2026-11-02. Provenance checked 2026-11-04: it entered that file in commit 71a49716, whose
subject is about dataset names and which mentions neither cliffs nor hybrids. She did not
recognise it and has said it should never have been made. It is marked unsourced in that
file.

**So the first thing this study has to produce is a cliff definition computed in a hybrid.**
That is TBD2, and it is the point of the study, not an extension of it. The one real
constraint on it is in section 5 and it touches one experiment.

## 2. What is already decided, so it is not reopened

**Scope: small molecules only** (Adelaide, 2026-11-04). Paper 3 as a whole keeps small
molecules and antibodies; the cliff work is small molecules. Searched 2026-11-04: no
published work applies the term to antibodies, biologics or protein variants, and the same
phenomenon in protein engineering is written about as epistasis or a rugged fitness
landscape. No external antibody cliff set exists, so the agreement tests of section 4
cannot run on one. Consequences are recorded in `HANDOFF_PAPER3_CLIFFS.md` section G.

**No dataset outside drug discovery** (Adelaide, 2026-11-04). Kepler and DroneRF have
nothing outside their features to measure similarity with, so their only cliff set would be
built from the representations being compared. The reaction sets are Paper 2 only. Five of
Matbench's six targets are computed and carry no measurement noise.

**A cliff set built from the representations being compared is refused, not flagged**
(Adelaide, 2026-11-04, implemented the same day). Two files each held a copy of the same
fallback and both now raise. One of the two paths was undocumented: a molecular dataset
whose fingerprint calculation threw fell silently into the circular reference.

**The quotas move together.** One constant, read by both the similarity cutoff and the
gap, with the order fixed and recorded (Adelaide, 2026-11-02).

**The hybrid is required in the experiments.** Every cliff-aware arm runs in the hybrid as
well as the single representations; an arm that only ever ran on ECFP4 is not finished
(Adelaide, 2026-11-02).

---

## 3. The definitions, as Adelaide set them out

### Which pairs count as similar

**1. A fixed similarity score above a cutoff.**

| Version | What it is | Unknowns |
|---|---|---|
| One coefficient on one representation | The general form. Not a single rule but a set | TBD1 coefficient, TBD2 representation, TBD3 cutoff |
| MoleculeACE, original | A pair is similar if **any one** of three scores clears 0.90: fingerprint overlap, overlap of the core framework with side groups stripped, or how many single-character edits turn one SMILES into the other | TBD4 cutoff |
| MoleculeACE, hybrid | The same shape with **one** overlap score on a hybrid instead of three | TBD5 cutoff, separate from TBD4 |
| `mmpdb` | A pair is similar when one molecule becomes the other by swapping one chemical group. No score and no cutoff | needs installing |

**2. The top share of pairs by any score from (1).** Per dataset, so "similar" means a
different score on every dataset.

### How far apart the labels have to be

1. Twice the median absolute deviation of that dataset's labels.
2. The gap moved until some share of molecules come out as cliffs. TBD6.

**Not an option, and the previous handoff had this wrong:** the spread of repeat
measurements of one compound. Its only job is to say whether a chosen gap is larger than
the assay's own noise. It is a check on choice 2.

### Models that predict cliffs directly

Not definitions. They predict a cliff label somebody else assigned, and their role is in
section 4.

| | What | State |
|---|---|---|
| GraphCliff | Short-long range gating, 2025. Reports cliff-specific error on MoleculeACE's 30 targets | published numbers only |
| SemiMol | Semi-supervised, 2026. Reports on the same 30 | published numbers only |
| The 2025 foundation-model comparison | Tested descriptor-based foundation models on 29 of the 30, and reports them separating cliff from non-cliff on only 4 of 29. One model it tested, Chemeleon, is in this repo's representation pool | published numbers only |
| `opig_activity_cliff_qsar` | Dablander's replication, **already running here** on dopamine D2, factor Xa and SARS-CoV-2 Mpro. Predicts each pair's activity difference from the two molecules' predicted activities and scores the cliff call | built, its cliff-call correlations computed and never reported |
| A model trained here on MoleculeACE's cliff labels | TBD7 | not assessed |

---

## 4. How a definition gets checked. This is the centre of the study.

Adelaide, 2026-11-04: *"How do I confirm if I actually got cliffs? Well, I can't
conclusively but I have some options."*

**What the literature does, searched 2026-11-04.** No published method tests whether a
cliff is real. What exists is curation applied *before* detection, which prevents false
cliffs rather than confirming true ones:

- Equilibrium constants only, never IC50, Ki and Kd pooled, because mixing assay types
  manufactures cliffs out of unit differences.
- Exact relations only; censored values dropped.
- Records carrying ChEMBL's own data-validity flag excluded.
- A potency gap of at least two orders of magnitude for a "high-confidence" cliff.
- **MoleculeACE drops a compound when the standard deviation of its own repeat
  measurements exceeds 1 log unit.** Verified against van Tilborg et al. 2022 on
  2026-11-04; it removed about 4.33 per cent of molecules across their 30 targets. This is
  the only published rule found for discarding a compound whose own measurements disagree
  too much, and it is applied BEFORE cliff detection, as prevention.
  **Measured here the same day:** on hERG IC50 that rule drops 69 of the 1,218 molecules
  measured more than once, which is 5.67 per cent of those and 0.70 per cent of all 9,802.
  The paper does not say why 1 log unit, and this dataset's own median disagreement is
  0.17 pIC50 with a 90th percentile of 1.07, so their threshold sits near this assay's
  90th percentile. Whether Paper 3 adopts it is open; it is a curation decision applied to
  the data, so it changes every number downstream.

The one wet-lab validation in the literature is prospective: the SAR Matrix method
predicted an MMP-1 cliff, the compounds were synthesised and retested, and the predicted
partner came out 60-fold more potent. Not available here.

**So the functional test in 4.3 is not a weaker substitute for a published method. There
is no published method.** That is worth stating in the paper.

### 4.1 Agreement test one: against MoleculeACE's own labels

Run each definition on their 30 targets and count how many of their cliff molecules it
recovers and how many it adds. Already built and run for the ten named rules; it
established that our copy of their rule reproduces theirs almost exactly and the thesis's
own rule does not.

**What is new:** sweep the cutoff rather than scoring one point, and read where agreement
peaks. That turns TBD3, TBD4, TBD5 and TBD6 from judgements into measurements on 30
targets.

**What it cannot do:** their labels come from their rule, so perfect agreement means
reproducing their definition.

### 4.2 Agreement test two: against the published cliff-prediction models

Put our cliff-specific error beside GraphCliff's, SemiMol's and the foundation-model
comparison's, on the same public splits. **And beside plain ECFP4 Tanimoto at the standard
cutoffs**, which the setup already produces and which has to be named explicitly as the
baseline rather than left implicit (Adelaide, 2026-11-04).

### 4.3 The functional test: D, D', D'' — Adelaide's design, 2026-11-04

Her words, kept as the specification:

> Look at the best outlier detection methods, call one M, and then subsequent best label
> handling methods, call one L. Take dataset D. Run M on D, giving you a list of outliers.
> Now we have to determine if the outliers are noise or ACs. Take the AC detection method
> in question and run it, cross-check with the list of outliers. Now first, apply L without
> AC-awareness to get D'. Then apply L with AC-awareness to get D''. Compare the results of
> training on D, D', and D''. Assuming M and L work, for a higher-is-better metric, we
> should see D'' > D' > D. Potentially D > D' if the removal of ACs was actually damaging.
> But the important thing being D'' should be the best. And then the difference between D''
> and D and D'' and D' should indicate the relative performance of different AC detection
> methods.

**One thing has to be added or the comparison cannot be read.** D'' differs from D' in two
ways at once, not one: the molecules the cliff definition names are spared, **and** fewer
molecules are cleaned in total. A gain could come from either. So every D'' needs a twin in
which the same *number* of molecules is spared, chosen at random. Call it D'''.

- D'' better than D''': the gain comes from WHICH molecules were spared, so the definition
  identified something.
- D'' no better than D''': the gain comes from cleaning fewer molecules, and the
  definition contributed nothing.

So the four trained datasets are D (nothing done), D' (cleaned, no cliff-awareness),
D'' (cleaned, the cliff definition's molecules spared) and D''' (cleaned, the same NUMBER
of molecules spared, chosen at random). D''' is a named arm of the experiment, not a check
run afterwards, and a result reported without it cannot be read.

The code for both exists, in a retired file: `n25 put_back` and `n25 random_exempt_mask`.
Lift, never edit.

**The number that decides the paper split** is D'' minus D''', per dataset and per
definition. Claim 1 of section 1 rests on 4.1 and 4.2; claim 2 rests on this.

**Known limit, measured.** The random twin cannot run where most training molecules are
cliff molecules, recorded as impossible on Caco-2, hERG blocking and LogD under twice the
median absolute deviation. Whether a chosen definition fixes that is one of the census
numbers. This is a reason to prefer a definition with a bounded cliff share, and it is a
measurement, not a verdict.

### 4.4 Two more functional tests, cheaper and weaker

**Are the molecules predicted worse?** Split the held-out molecules into the definition's
cliff set and the rest, compare the error. A definition finding real chemistry gives a
bigger gap; one picking arbitrary pairs gives none.

**Does the cliff set survive changing the representation?** Hold the definition fixed, swap
the representation, count how many molecules stay cliffs. The thesis already measures a
version of this on one dataset: holding the gap and the number of closest pairs fixed, the
cliff molecule count runs from 138 to 267 across representations.

---

## 5. The seven unknowns, and where each answer comes from

Two of the seven are answered by reading results that already exist. Four are answered by
one sweep. One is Adelaide's judgement.

**TBD1, which similarity coefficients. A set, not one.** Answer from her own results.
`tests/results/framework/f34_importance_sweep/` holds per-dataset, per-representation,
per-coefficient scores for predicting a held-out molecule from its nearest training
molecules, over four datasets, and
`f34_neighbour_weighting_cross__four_coefficients_2026_09_28/` holds the four-coefficient
cross. Memory records that framework local results count as evidence for the
nearest-neighbour question from f34 onwards, which is exactly this question.
**What to produce:** one table, one row per coefficient, the score per dataset in that
dataset's own metric, so the set can be chosen on numbers and justified in the outline by
naming that result rather than a convention.

**TBD2, which representations. A set, and hybrids are the reason the hybrids were built.**
Same files, plus `s02_hybrid_similarity`, `s05_knn_prediction`, `s06_applicability_domain`,
`s08_optimal_dimension` and s10 under `tests/results_server/`.
**Two complications.**

*One is mechanical and has to be solved before anything is built.* An overlap score needs
features at or above zero, and a hybrid's are standardised and go negative. Measured
2026-11-03: between 45 and 66 per cent of a standardised block's entries are below zero,
on four datasets, so this is not a rounding problem. Either the score changes or the
features go into a non-negative form, and which is **Adelaide's**.

*One applies to one experiment only.* AC4 scores each representation by how well it
predicts cliff molecules. If the cliffs were found in the hybrid, the hybrid is scored on
cliffs it chose. That affects AC4's reading and nothing else: AC1, AC2, AC3, AC5 and the
D''/D''' arm are unaffected, because none of them compares representations against each
other on a cliff set.
**This is a constraint on one experiment, not a ban on computing cliffs in a hybrid.** An
earlier session wrote the ban into `HANDOFF_PAPER3_CLIFFS.md` as a decision of Adelaide's
dated 2026-11-02; provenance checked 2026-11-04, it entered that file in a commit about
dataset names and she did not recognise it. It is not her decision and does not block the
hybrid overlap score.

**TBD3 and TBD4, the cutoff, and whether it is global, per domain or per dataset.** The
sweep in 4.1 answers it: where agreement with MoleculeACE's labels peaks, per target.
Whether that peak is one number across 30 targets or thirty different numbers is itself the
answer to the global-or-per-dataset question, and it is a measurement.
**What is already measured and bears on it:** at ECFP4 Tanimoto 0.90, hERG Ki, Caco-2 and
MDR1-MDCK have 5, 7 and 0 cliff pairs, so a fixed high cutoff is comparable and empty. The
top 1 per cent of pairs gives 655, 6,512 and 12,836 pairs at 30.5, 74.5 and 88.9 per cent
of molecules, so a share is usable and means something different each time. That is the
trade, measured.

**TBD5, the cutoff for the single hybrid overlap.** Separate from TBD4 because one score
replaces three. Measured on one target: of MoleculeACE's 1,222 cliff molecules there, 1,055
came from the string score and 148 from the fingerprint, so their 0.90 is mostly a cutoff on
text similarity and does not transfer.

**TBD6, the share a moved gap aims at.** The same sweep. What is already measured: a share
target can be unreachable, and the gap then goes to zero so a cliff is any similar pair
whose labels differ at all; or it can be reached at 0.009 label units. Both happen on real
datasets and neither raises an error. Any guard has to read the gap in label units, not the
achieved share.

**TBD7, whether to train a cliff-prediction model here.** Adelaide's.
**What is known:** the training data ships with MoleculeACE, a binary cliff label per
molecule on 30 targets, so it costs nothing to assemble, and it is straightforward — a
binary classifier on features this repo already computes.
**What is unclear:** a model trained on their labels and scored on their labels measures
their definition twice. It becomes worth doing if it is scored on something else: our
datasets, where there is no cliff label, with 4.3 as the score.

---

## 6. The experiments, and where they live

Adelaide, 2026-11-04: *"Make this a separate set of AC experiments"*, and the cliff numbers
come **out** of the restructured noise experiments.

### 6.1 A separate family

The restructured noise suite is NA, DC, DG, ST and MS — built as n43 to n47, with n15, n20
and n36 standing. The cliff measurements currently sit inside MS (n47), lifted from n01 and
n14, and they are to be removed from there.

**Proposed, and the naming is Adelaide's to approve:** a new family for this study, so the
cliff work is one set of files with one definition module behind it, and the only thing left
inside the noise suite is the one arm that needs the detector-by-cleanup machinery.

| | What it runs | Lifted from | Section |
|---|---|---|---|
| AC1, the definitions and their counts | Every definition on every dataset: cliff pairs, cliff molecules, the achieved share, the gap in label units, the gap against the dataset's own repeat spread where it has one | `cliff_census.py`, and the cliff half of n47 | 3 |
| AC2, agreement with MoleculeACE | Every definition against their labels on 30 targets, **with the cutoff swept** | `opig_moleculeace_cliff_check.py` | 4.1 |
| AC3, against the published models | Our cliff-specific error beside GraphCliff's, SemiMol's and the foundation comparison's, and beside plain ECFP4 Tanimoto at the standard cutoffs | `opig_activity_cliff_qsar.py`, which already computes the cliff-call correlations and reports none | 4.2 |
| AC4, predicted worse or not | Held-out error on cliff against non-cliff molecules, per representation and per hybrid | s04, and the cliff half of n47 | 4.4 |
| AC5, does the set survive the representation | One definition, every representation, how many molecules stay cliffs | `thesis.tex:1123`'s measurement, extended | 4.4 |

### 6.2 The one arm that stays in the noise suite

The D/D'/D'' test is a detector crossed with a cleanup plus a modifier, which is exactly
what DC is. It stays in `n44_detect_and_clean` as:

- the cliff-awareness modifier on a cleanup, which produces D'', and
- D''', sparing the same number of molecules chosen at random.

Both lift from `n25 put_back` and `n25 random_exempt_mask`. This is the only place the
cliff work touches the noise suite, and it is the place the paper split is decided.

### 6.3 Hybrids

AC4 and AC5 are where TBD2 is answered on cliffs rather than on neighbours, and s09's
aggregation is the shape to reuse: per dataset, one hybrid minus one representation per
held-out fold, in the input's own quantity, on all molecules, cliff molecules and
non-cliff molecules separately.

### 6.4 Writing claim 3

The proposed definition, in both forms, is written from AC1 to AC5 and the D/D'/D'' arm
together — not from any one of them. What each contributes to the case:

| | What it licenses the definition to claim |
|---|---|
| AC1 | that it gives a usable cliff set on every dataset, with the gap in label units and the share bounded, which the fixed published cutoff does not |
| AC2 | how close it comes to the one external cliff set that exists, and whether its cutoff is one number or thirty |
| AC3 | where it stands against the published cliff-prediction models and against plain ECFP4 Tanimoto at the standard cutoffs |
| AC4 | that the molecules it names are predicted worse, which is the behaviour a cliff is supposed to have |
| AC5 | that the set is not an artefact of one fingerprint |
| D'' against D''' | that naming those molecules improves a cleanup, which is the only test here that no published method does |

Until those exist the definition cannot be written, and a definition written before them
would be a verdict dressed as a result.

### 6.5 What comes out of the noise suite

The cliff counts and the cliff-split error now in n47 move to AC1 and AC4. n47 keeps the
noise measurements and loses the cliff ones. Note before touching it: n47's two blocks
currently disagree on the gap — one uses twice the median absolute deviation, the other a
flat 2 log units, and both write a field called `n_cliff_compounds` into the same result
file.

---

## 7. Getting it into the ledger

The target is **implemented and ready to run inside the ledger, on the available
datasets**, not a design. `HANDOFF_PAPERS_2_3_LEDGER.md` holds the five steps and the gate;
read it in full with the files it names before changing any of it.

The AC family runs at **Step 5**, with Paper 3's experiments, and depends on Steps 1 to 3
for the hybrids it needs. Two parts of it do not: AC2 and AC3 run on MoleculeACE's and
Dablander's own data and splits, so they can run before the gate.

**What has to be settled for a ledger entry to exist**, in the order it blocks:

1. The definition set. One entry per definition, or one entry sweeping them.
2. Which datasets. Paper 3's small molecules: the three QM9 targets, hERG Ki, hERG IC50,
   Caco-2, LogD, rat and human microsomal clearance, MDR1-MDCK, hERG blocking, and the
   three Dablander targets.
3. Whether each AC experiment is one unit per dataset or one per dataset and definition,
   because the ledger resets and reruns one unit at a time.
4. The memory class per unit. AC1 builds a pairwise matrix, which is the cost: 9,978
   molecules is about 400 MB a matrix and several are held at once, so QM9 needs chunking.

---

## 8. Other things worth considering

Raised here as options with what each needs, not as recommendations.

**Assay type as a filter before detection.** The literature's first curation rule is not to
pool IC50, Ki and Kd, because mixing them manufactures cliffs out of unit differences. Two
datasets here are hERG Ki and hERG IC50 and they are kept separate, which is right. Worth
checking whether any single dataset pools types internally.

**Censored values.** The literature drops them before detecting cliffs. A censored label is
a bound, not a measurement, so a pair involving one has an unknown gap. This repo already
has a censoring noise condition, so it knows what censoring is; whether any dataset ships
censored labels into the cliff calculation is unchecked.

**Same paper, same lab.** Two measurements from different papers differ partly because the
labs differ. hERG IC50 names the document each measurement came from, so a same-document
restriction on which pairs may be called cliffs is one line where the pair list is built.

**Stereoisomer pairs.** Named in the Geometry preprint as the kind of cliff fingerprints
miss entirely. Already added as a criterion, never run as a definition.

**The half-cliff band.** Dablander drops pairs 1 to 2 log units apart from both sides rather
than calling them non-cliffs. That is a third gap option not in the list in section 3, and
it changes what the cliff call is scored against.

**Cliff molecules against the molecules this suite corrupted on purpose.** The detector
grading experiment knows which rows it damaged by construction, so the two masks cross
directly. It is the cheapest version of asking whether a definition separates chemistry from
error, and its limit is that injected noise is not the noise the data came with.

**The curator flag, already run and weak.** Every definition's cliff molecules carry
ChEMBL's suspect-record mark at about the rate the whole dataset does. The "potential
transcription error" text covers 6 molecules in all of hERG IC50, so it cannot separate
anything. Worth one sentence in the paper and no more work.

---

## 9. How to work this document

Written because three sessions failed on this before, in ways that are recorded.

**State the decision, not the evidence for it.** Lead with what has to be decided, what the
situation is, and what the options are. The measurement is the evidence for the options, not
the message.

**Never refer back to a list, option number or name she cannot see.** Restate the options in
full in the message that asks her to choose.

**Say what a thing is before asking about it.** Never name code, a dataset or a score she
has not used first.

**Every number carries its unit and what it is a share of.** Molecules, pairs, folds,
label units. A table needs a line above it saying what one row is.

**No verdict written into any document of hers.** Later it cannot be told from a measured
result. Headings state what was measured.

**Do the measurement rather than offer it.** Only a scope cut, a dependency to install or a
dataset to drop needs her first.

**Nothing lives in a chat reply.** If she raises something, it goes in this file before the
reply is written.

### Before starting

- Repo-root `CLAUDE.md` in full. It sets how to write to her.
- `PAPER_3_OUTLINE.md`. The documentation of record.
- `HANDOFF_PAPERS_2_3_LEDGER.md` and the files it names.
- `HANDOFF_NOISE_RESTRUCTURE.md` sections 1 to 4, 8, 9 and 10. The old experiments n01,
  n14, n25, n35 and n42 are retired: lift from them, never edit them.
- `HANDOFF_PAPER3_CLIFFS.md` sections A, E, F and G, for the measurements already made.
- From `MEMORY.md`: no patient data; one heavy local job at a time; everything on
  `/Volumes/seagate`; no significance tests; never cut a definition, dataset or setting for
  compute; local runs check code and are not results; post a line every few minutes because
  her terminal disconnects; she decides, so options come with numbers attached.
