# Handoff: activity cliffs for Paper 3

Rewritten 2026-11-02. Supersedes the version of 2026-11-01, which framed this as five
research questions to be answered by experiments. That was wrong. Adelaide is settling
these now, as decisions, so that cliffs can be built into the new noise experiments once
and used consistently. All of it goes in Paper 3.

There are two issues and they are separate.

1. **How a cliff is computed.** There is no settled answer. Both halves are open: what
   counts as similar, and what counts as a large label difference. This needs a lot of
   measurement before a choice.
2. **Where cliffs enter the noise experiments.** Six places in the new structure. Each
   needs a decision about which computation from issue 1 it uses.

## Before you start

- Read repo-root `CLAUDE.md` in full. It sets how to write to Adelaide. Follow it.
- **The noise experiments have been restructured.** Read `HANDOFF_NOISE_RESTRUCTURE.md`
  sections 1 to 4, 8, 9 and 10. The 37 old experiments n01 to n42 become n43 to n47.
  **Do not write a fix against n01, n14, n25, n35 or n42.** They are being retired:
  n01 and n14 go to MS, n25 and n42 to DC, n35 to ST. Anything below that names an old
  file names it as the place to lift code from, never as the place to change.
- The similarity experiments move into the new structure too (Adelaide, 2026-11-02).
  s01, s02, s04, s09, s12, `similarity_metrics/simeval` and
  `similarity_metrics/nary_similarity` all compute cliffs today.
- From `MEMORY.md`, the rules that bite here:
  - No patient data. TCGA, ANNITIA and NHANES are never opened or counted.
  - One heavy local job at a time. All outputs, logs and temp files on `/Volumes/seagate`.
  - No significance tests. Per-fold scores and their spread.
  - Do not cut a definition, dataset or setting to save compute.
  - Local runs are for checking definitions and code. They are not results.
  - Post a short line every few minutes during long work. Her terminal disconnects.
  - She decides. Give options with the numbers attached. Do not design alone.

## What is already done, so it is not redone

- **The literature.** A research session of 2026-11-01, saved at `~/Downloads/cliffs.md`.
- **The code inventory and the literature, written up.**
  `thesis_provenance/CLIFF_DEFINITIONS_2026-11.md`, sections 1 and 2. Section 1 is read
  from the code. Section 2 is at second hand from that research session and says so.
- **One thing the thesis already measures**, which neither of the above found first:
  `thesis.tex:1123`. Keeping each dataset's gap and replacing ECFP4 with each other
  single representation, calling the same number of closest pairs similar, the count of
  hERG Ki cliff molecules runs from 138 under atom pairs to 267 under the WL graph
  kernel. Of the pairs ECFP4 calls similar, 57 to 74 per cent are among another
  representation's closest pairs; on QM9, 1 to 24 per cent.

## Two things that are wrong today, before any new work

**The thesis uses two different cliff rules in different chapters and does not say so.**
`thesis.tex:1120` and `:2922` choose the gap on each dataset to aim at 15 per cent of
molecules being cliff molecules. `:3675`, Cliff put-back, instead calls a label far when
it is more than twice the median absolute deviation. On hERG Ki the second gave 451 of
1,412 molecules against the first's 211 of 1,412.

**The aimed-at share is not the achieved share, and the thesis prints the target as
though it were reached.** `similarity_helpers.solve_activity_gap` returns the gap whose
share is nearest 15 per cent, not one that reaches it. Where few pairs are similar it
misses by a lot. On QM9 only 1,093 pairs of 9,978 molecules are similar at all
(`thesis.tex:1120`), and `:4148` reports 19 of 7,982 training molecules as cliff
molecules. Anything written about this must use achieved shares, per dataset, never the
target.

---

# Issue 1: how a cliff is computed

A cliff rule is three choices: what makes two molecules similar, what makes their labels
far apart, and whether the answer is a pair or a molecule. Below is every option found in
the code or in the research session. One row is one option. "In repo" means the piece
exists; it does not mean it has ever been used for cliffs.

## 1a. What counts as similar

| Option | In repo | Where |
|---|---|---|
| ECFP4 Tanimoto at or above 0.55 (Hu et al. 2015) | yes, used | `similarity_helpers.py:980`, `n01:235` |
| Morgan Tanimoto at or above 0.90 | fingerprint yes, threshold no | Geometry paper's strongest setting |
| MACCS Tanimoto at or above 0.85 | fingerprint yes, threshold no | `representations/molecular.py` |
| RDKit fingerprint under Dice | fingerprint yes, **Dice absent** | `distance_config.py:66` holds four distances, no Dice |
| MoleculeACE: 0.90 on any one of ECFP Tanimoto, scaffold ECFP Tanimoto, scaled Levenshtein on the SMILES string | no | van Tilborg et al. 2022 |
| Matched molecular pair, transformation-size restricted | pair lists ship with the Dablander data; whether `mmpdb` can be run here is **not checked** | `chembl_qsar.py` |
| Top k per cent of pairs by similarity | yes, used | `n01:112` at 2.0 per cent; `similarity_helpers.py` at 1.0 per cent |
| Edit distance at or below 5 on sequences, or the closest k per cent | yes, used | `similarity_helpers.py:830` |
| Among the k nearest neighbours, no threshold | yes, used | SALI factory, `molecular.py:1467`, k=10 at Tanimoto 0.5 |
| Closest 10 per cent of pairwise distances, whatever the features | yes, used | `ActivityCliffDetector` default, `noise_mitigation_methods.py:742` |
| A hybrid's own distance, Manhattan or Euclidean or cosine or Minkowski p=0.5 | distances yes, **threshold undecided** | `distance_config.py:66`; the open decision at `n25:696` |
| ChemBERTa, MolFormer or Chemeleon cosine | all three present, never used for cliffs | `molecular.py`, `forward_compat_molecular.py` |
| Stereoisomer pairs | no | named by the Geometry paper as the kind fingerprints miss |
| Binding mode from a crystal structure | no, and no structures loaded | Hu, Furtmann and Bajorath |

Tanimoto and Dice are to be added to `distance_config.py:66` (Adelaide, 2026-11-01).

## 1b. What counts as a large label difference

| Option | In repo | Where |
|---|---|---|
| 2.0 times the median absolute deviation of that dataset's labels | yes, used | `n01:235`, `n25:569`, `n35:213`, and `thesis.tex:3675` |
| Gap solved so a target share of molecules are cliff molecules | yes, used, target 15 per cent | `similarity_helpers.py:1101`, `solve_activity_gap` |
| Absolute 2 log units, 100-fold | yes, used | `n14:89`, `chembl_qsar.py:13` |
| Absolute 2 log units with 1 to 2 log units dropped as half-cliffs | yes, used | `opig_activity_cliff_qsar.py:379` |
| Absolute 1 log unit, 10-fold | no | MoleculeACE |
| 90th percentile of pairwise label differences | yes, used | `ActivityCliffDetector` default |
| Different class, for a classification task | yes, used | `n01:262` and the same in four other files |
| Set from that target's own distribution of pairwise differences, for example mean plus two standard deviations | no | attributed to the Bajorath group, **unverified** |
| A gap that clears the measurement-noise floor | floor present, never used as a gap rule | `n14:101`, `KRAMER_NOISE_FLOOR_PKI = 0.54` |

## 1c. Continuous instead of yes-or-no

| Option | In repo | Where |
|---|---|---|
| SALI, the label difference over one minus the similarity | yes, as model features | `molecular.py:1467` |
| iCliff-style roughness per molecule | yes | `similarity_metrics/nary_similarity.py:126` |
| SARI, a continuity and a discontinuity score per dataset | no | Peltason and Bajorath 2007 |
| MODI, and RMODI for regression, and ROGI for roughness | no; `coleygroup/rogi` packages all four | MODI is cited at `thesis.tex:978` |
| SAS map, similarity against label difference, with a cliff quadrant | no | Medina-Franco and co-workers |
| Average SALI across several representations | no | the research session's answer to representation dependence |

## 1d. Pair or molecule

Both are in use and they are not interchangeable. `chembl_qsar.py` and
`opig_activity_cliff_qsar.py` score pairs. Every noise and similarity experiment reduces
to a per-molecule flag by asking whether a molecule is in at least one cliff pair. A
molecule-level rule inherits whatever the pair rule did and hides it.

## 1e. Sets with cliff labels already attached

Not computations. Ground truth to measure a computation against.

| Set | State |
|---|---|
| Dablander's three targets, matched pairs with cliff labels | on disk, used by `opig_activity_cliff_qsar` |
| MoleculeACE, 30 targets with a `cliff_mol` column | **on disk and discarded**: `build_moleculeace.py:10` reads `smiles` and `y` only |
| ACNet, more than 400,000 matched pairs against 190 targets, more than 20,000 cliffs | not downloaded |
| Bajorath group's published cliff sets | not downloaded |

## 1f. The measurement needed to choose

Every one is local, about definitions rather than hybrids, time-boxed at normal settings,
outputs on `/Volumes/seagate`. Do not shrink a dataset to make one finish; record what was
slow.

1. **Counts under every option.** For each non-patient dataset, every pairing of 1a and
   1b: cliff pairs, cliff molecules, and the achieved share of molecules. One row per
   dataset and rule. Print the spread of pairwise similarity and of pairwise label
   difference beside it, so a count can be read against them.
2. **Where a rule cannot reach its target.** For the solved gap, the achieved share
   against 15 per cent, per dataset, and the number of similar pairs available. QM9 is the
   known failure and must appear as one.
3. **Agreement between rules.** For each dataset, for each pair of rules, how many
   molecules both call cliffs and how many only one does. A table per dataset.
4. **Agreement across representations at a fixed rule.** `thesis.tex:1123` did this for
   hERG Ki and QM9 by matching the number of closest pairs. Extend it to every dataset and
   to the embeddings the Geometry paper used, which are all present: ECFP4, MACCS, RDKit
   fingerprint, ChemBERTa, MolFormer, Chemeleon. Report the published 55-fold swing beside
   our own.
5. **Against the labelled sets.** On MoleculeACE's 30 targets, how many of the molecules
   its `cliff_mol` column marks does each of our rules recover, and what else does each
   flag. This is the only check against a label nobody here chose.
6. **Against the noise floor.** Per dataset, the share of cliff pairs under each rule whose
   label gap is under 0.54 pKi, Kramer's standard deviation. A rule that mostly calls pairs
   inside the measurement floor is calling noise.
7. **Overlap with what a detector flags.** For each detector in list B and each rule, how
   many of the flagged molecules are cliff molecules. One table per dataset, one row per
   detector and rule. This is the number the thesis asserts at `:1265` and never gives.

---

# Issue 2: where cliffs enter the new noise experiments

Six places. Each needs one decision: which rule from issue 1 it runs on. Code locations
are from `HANDOFF_NOISE_RESTRUCTURE.md` sections 3 and 4 and are the places to lift from.

| Place | What it is | Needs a cliff rule | Lift from |
|---|---|---|---|
| **DC**, modifier axis | Cliff put-back: after a cleanup, undo it for cliff molecules. Audit block m28. A modifier on any cleanup, not a cleanup. | yes | `n25 put_back`, `cliff_mask_for_dataset` |
| **DC**, control | Random put-back: put back as many non-cliff molecules as cliff molecules | yes, the same one | `n25 random_exempt_mask` |
| **DC**, list B | Cliff-aware gap, registry `activity_cliff_aware`, audit block m5. Scores the gap from the neighbours' mean over the neighbours' own spread. | **no.** No threshold and no cliff list inside it | already a detector |
| **DC**, list B | `ActivityCliffDetector`, in `noise_mitigation_methods.py` and used by no experiment. Section 4.2 says to check each such class against the registry, delete a duplicate and add anything new to list B. | yes, and it has two free thresholds | `noise_mitigation_methods.py:742` |
| **ST** | Close-pair label disagreement, audit block m25, and its counting of how many flagged close pairs are cliff pairs | yes | `n35 count_cliff_pairs`, `intercept_estimate` |
| **MS** | The cliff measurements that train no cleanup: cliff-stratified prediction error, cliff landscape geometry, and the counts of 1f | yes, all of them | `n01`, `n14` |

Two further places, from the similarity experiments moving in: s04 and s12 partition a
dataset into cliff and other molecules and score representations on each part; s01 and s02
run a sequence-based cliff sweep on antibody datasets; s09 reads s04's cliff numbers;
`simeval` uses a cliff set as the target a distance is scored on finding;
`nary_similarity` computes roughness per molecule. Where these land in the new layout is
not decided (section "Decisions that are Adelaide's" below).

## What breaks on the way in

- **The random put-back control cannot run where most training molecules are cliff
  molecules.** `thesis.tex:4148` records it as impossible on Caco-2, hERG blocking and
  LogD. That is under twice the median absolute deviation. Whether another rule fixes it
  is for measurement 1 to say, not for a handoff to assert.
- **Cliff put-back has never been run on anything but ECFP4 at 0.55.** `thesis.tex:3675`
  says so in the thesis, and `n25:696` says why: nobody has decided what threshold means
  similar in a Manhattan or forest distance, and nobody has decided whether the molecules
  put back may come from the same features the detector reads.
- **Every setting of n25's old sweep put back the same ECFP4 cliff set** (`n25:696`). If
  the modifier axis in DC is to vary with the feature space, that is new behaviour, not a
  port.
- **Three faults in n25 were fixed between 2026-09-23 and 2026-10-06** and change what a
  stored number means: the cliff set was once chosen from the clean labels; a training
  molecule's cliff status once depended on held-out molecules; a reused similarity matrix
  was once cut two different ways inside one run. A fourth decision, 2026-10-06: two rows
  of the same molecule are never a cliff pair.

---

# Decisions that are Adelaide's

One per line. None of these is to be settled by whoever picks this up.

- Which rule, or which few rules reported side by side, the put-back modifier runs on.
- Whether a dataset may have its own rule, or one rule holds everywhere, or results are
  reported under two or three named rules. The research session's own suggestion is the
  third; it is an AI suggestion and is marked as one in the findings file.
- Whether the similar side may be computed in a hybrid's distance, and at what threshold.
- Whether molecules the put-back spares may come from the same features the detector reads.
- Whether `ActivityCliffDetector` becomes a detector in list B or is deleted as a duplicate.
- Where s01, s02, s04, s09, s12, `simeval` and `nary_similarity` land in the new layout.
- Whether MoleculeACE's `cliff_mol` labels are loaded, and whether ACNet is downloaded.
- Whether M_cores and M_train are built in `opig_activity_cliff_qsar`. It has M_inter and
  M_test only (`:418`). M_cores is the pairs where both molecules are held out and their
  shared core appears nowhere in training; Dablander reports it as his hardest of four.
- Whether the half-cliff band, 1 to 2 log units, is carried into the noise experiments or
  stays only in the Dablander replication.

# Paper 3 placement

- `opig_activity_cliff_qsar` is in Paper 3, not Paper 2. It is already in
  `PAPER_3_OUTLINE.md`'s inventory and run list as of 2026-11-01.
- Its thesis result is the indicator that hybrids help on cliff-rich targets: on D2, factor
  Xa and Mpro the hybrid had lower held-out mean absolute error than the top-ranked single
  representation on 5 of 6 target-and-model rows, the exception being factor Xa with
  LightGBM. The experiment also scores the cliff calls by Matthews correlation and the
  thesis reports none of them.
- The research session reports a consistent linear relationship between prediction error
  and the Matthews correlation on the cliff call across Dablander's nine models. If that
  holds here, the thesis's error result predicts a cliff-call result that is already
  computed and unreported.

# What to bring back

- The counts of 1f, in the findings file, with achieved shares and never a target share.
- One message to Adelaide. Lead with the counts and which rules disagree by how much. Then
  the decisions above, each with the numbers beside it and a recommendation.
- Nothing written into `PAPER_3_OUTLINE.md` or `thesis.tex` beyond what is already there.
- No AI verdict in any of her documents. Mark anything from the research session as from it.
