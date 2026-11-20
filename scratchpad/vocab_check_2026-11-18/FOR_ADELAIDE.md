# Vocabulary and accuracy check, 2026-11-18

Files changed but not committed. Nothing in thesis.tex or Paper 1 was edited.

## What was changed

One row is one document. The number is the edits made after two separate checks confirmed each one.

| document | edits |
|---|---|
| `PAPER_2_OUTLINE.md` | 156 |
| `PAPER_3_OUTLINE.md` | 71 |
| `PAPER_3_NOISE_EXPERIMENTS.md` | 146 |
| `AC_GRID_TEST_PLAN.md` | 231 |
| `NEW_DATASETS.md` | 108 |
| `scratchpad/precompute_cuts/DECISIONS.md` | 157 |
| `tests/slurm_scripts/runbook/PAPERS.md` | 34 |
| `tests/slurm_scripts/runbook/SCREEN.md` | 11 |
| `tests/slurm_scripts/runbook/RUN_BOTH_LEDGERS.md` | 7 |
| `SEARCH_SETTINGS.md` | 64 |
| `HANDOFF_PROTEINGYM.md` | 43 |
| `VOCABULARY.md` | 15 |
| `thesis_provenance/CLIFF_DEFINITIONS_2026-11.md` | 142 |

Total: 1185 edits. 3991 claims were checked against the code, thesis.tex, git and your messages. 443 were about server results and could not be checked here.

Not touched:

- Text another chat wrote today in `PAPER_3_NOISE_EXPERIMENTS.md` and `AC_GRID_TEST_PLAN.md` (the recount, "Her settings for the eight-hybrid comparison", "The same training set is scored under several names"). It still uses "piece" and "draws".
- Code comments, and the documents not kept on 2026-11-17.
- `scratchpad/paper3_grid_counts/count_n44.py`: one checking agent edited it against instructions (two lines, adding `**_k` to two stand-in functions). I was not allowed to undo it. It is uncommitted.

## Items that need you

120 items. Each was found, then re-checked by a second agent against the current file. Grouped by document.

### `PAPER_2_OUTLINE.md` (30)

**Line 21. No agreed word.**

- The document says: "single feature set" at lines 21, 36, 57, 148 and 177, against "single view" at 429, 513, 537 and 554.
- What the code, thesis or your messages say: The word list's row for "single feature set" is scoped to PAPER_3_OUTLINE.md, so this document has no settled word for one feature set scored on its own.
- Checked in: VOCABULARY.md:157; VOCABULARY.md:181
- Your choice: Her Paper 2 word for one feature set scored on its own, for all of those lines.

**Line 76. The document is wrong, and the fix needs your choice.**

- The document says: Evaluator settings sit within 0.005 on small molecules, and differ up to 0.08 Spearman on antibodies in no fixed direction.
- What the code, thesis or your messages say: The thesis records QM9 with LightGBM at 0.363 RMSE against 0.346 for the matched evaluator, a gap of 0.017. It also records the matched evaluator best with both models on EGFR binding, HIC retention, expression titer and SMAC colloidal stability.
- Checked in: thesis.tex:2571; thesis.tex:2575; c23 result files at /Volumes/seagate/kirby_results_2026-10-10b/complexity/c23_evaluator_model_matching/
- Your choice: Whether the paper keeps the claim that the cheaper evaluator is enough, given the QM9 exception and the four antibody sets where the matched one wins.

**Line 109. Two places disagree.**

- The document says: The two Adams scans use random folds, the only random split in either paper.
- What the code, thesis or your messages say: Matbench is listed as a Paper 2 dataset at line 146. Its loader uses Matbench's own shuffled KFold test folds, which are random and ungrouped.
- Checked in: src/kirby/datasets/matbench.py:16-21; PAPER_2_OUTLINE.md:146; the TODO at PAPER_2_OUTLINE.md:95-97 says Matbench is not decided
- Your choice: Whether Matbench stays in this dataset list, which settles whether the Adams scans are the only random split.

**Line 216. The document is wrong, and the fix needs your choice.**

- The document says: On the target fold four things are fitted, then three are named, then a fifth score is named.
- What the code, thesis or your messages say: The fold-to-fold code records five scores per ordered fold pair: the three-fold search, the pair settings with features re-picked, the pair's fixed features, and the source fold's own settings and fixed features.
- Checked in: tests/experiments/complexity/c24_blueprint_transfer.py:28-52
- Your choice: Which five scores the Methods section names, and in what grouping.

**Line 219. No agreed word.**

- The document says: "differs from the fold's own blueprint only in training molecules", and the same phrase at line 559.
- What the code, thesis or your messages say: The word list says molecules in Paper 3 and samples in Paper 2. Both sentences are general method statements in a Paper 2 document. The file says samples elsewhere, at line 200 and line 430.
- Checked in: VOCABULARY.md:109; VOCABULARY.md:180; PAPER_2_OUTLINE.md:200, :430, :559
- Your choice: Whether a general method sentence in this paper says samples even where the datasets are molecular.

**Line 222. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: her words, 2026-10-31: "run all of it".
- What the code, thesis or your messages say: She wrote "oh we went over this, i thought i was doing all of it, its a quick test no ?" on 2026-10-31. The string "run all of it" appears only in a Claude-written docstring line and in assistant tool inputs.
- Checked in: her message, /Users/apunt/.claude/projects/-Users-apunt-repos-KIRBy/0ab3513e-113d-4949-b0df-cdb1f11dce6a.jsonl, 2026-10-31T02:44:33Z; tests/experiments/complexity/c24_blueprint_transfer.py:28; line 557 carries the same attribution
- Your choice: Whether she approves every ordered fold pair and the five scores, in her own sentence.

**Line 290. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: The question kept is how many of the picked features come from the two mutation-difference views; scoring them alone is the supporting number.
- What the code, thesis or your messages say: The file marks the standalone comparison as the owner's question, the one that matters. It records the feature-count question as dropped by the owner on 2026-09-11.
- Checked in: tests/experiments/opig_scaffold_hybrid.py:5-17; the feature counts are recorded again since 2026-10-05 at :48-50
- Your choice: Which of the two questions this subsection reports, and whether the dropped one is back in.

**Line 302. No agreed word.**

- The document says: "the identical row of numbers", "still share a row", at lines 302, 310, 331, 333 and 336.
- What the code, thesis or your messages say: "rows" is a rejected word. Its replacements are molecules, samples and folds. One molecule's vector of numbers in a view has no entry in the word list.
- Checked in: VOCABULARY.md:124; VOCABULARY.md:109; VOCABULARY.md:101 uses "feature vectors" in a definition only
- Your choice: Her word for one molecule's vector of numbers in a view.

**Line 334. The document is wrong, and the fix needs your choice.**

- The document says: The features of the merging view stay; the partners' features are added beside them.
- What the code, thesis or your messages say: The hybrid selects a budget's worth of the target view's features, not all of them. That is why each fold reports the tie count three ways. The same bullet says one fold kept 244 of the graph kernel's 276 features.
- Checked in: tests/experiments/opig_shared_vectors_remedy.py:94-104; PAPER_2_OUTLINE.md:345-346
- Your choice: Whether this bullet reports the width control as well, since the merging view's features do not all stay.

**Line 358. No agreed word.**

- The document says: "16 rows from the thesis run" at line 358 and "no rows" at 373, and "one row per dataset and downstream model" in the table captions.
- What the code, thesis or your messages say: One is a line of recorded results, the other a table row. Both are the rejected word in senses the word list's replacements do not cover.
- Checked in: VOCABULARY.md:124; PAPER_2_OUTLINE.md:373, :428, :450, :512, :517, :565, :581
- Your choice: Whether "row" stays for a line of results and for a table row, or each gets its own word.

**Line 434. The document is wrong, and the fix needs your choice.**

- The document says: the three ChEMBL targets (also at line 101).
- What the code, thesis or your messages say: Dopamine D2 and factor Xa come from ChEMBL. The SARS-CoV-2 main protease comes from the COVID Moonshot project.
- Checked in: thesis.tex:2084; tests/experiments/dataset_policy.py:361 records SARS-CoV-2 Mpro (PostEra)
- Your choice: What the paper calls these three datasets as a set.

**Line 435. Two places disagree.**

- The document says: These three are the only datasets here with a published score on the same molecules and labels.
- What the code, thesis or your messages say: Antibody humanness carries eight published AUROC values on the same therapeutics, and this outline cites one of them at line 448.
- Checked in: tests/experiments/opig_humanness.py:181-192; PAPER_2_OUTLINE.md:448; humanness_therapeutic is a Paper 2 dataset, tests/experiments/papers/plan.py:88
- Your choice: Whether the published-comparison claim covers antibody humanness too, or is narrowed to datasets with published errors in the same metric.

**Line 448. The document is wrong, and the fix needs your choice.**

- The document says: germline identity alone gives 0.968, beside five-fold means for random forest and XGBoost.
- What the code, thesis or your messages say: The published value is Germline content, 0.968 AUROC, from a threshold sweep on the whole set with no train and test split, with heavy and light chain scores averaged. This work scores the heavy chain only.
- Checked in: tests/experiments/opig_humanness.py:178-192; thesis.tex holds no occurrence of "germline" (case-insensitive grep)
- Your choice: Whether that published value stays beside the five-fold numbers, and what caveat sits with it.

**Line 471. No agreed word.**

- The document says: so this is a missing piece, not a side point.
- What the code, thesis or your messages say: "piece" is a rejected word. The word list replaces it with a configuration and its fold, noise level and repetition, which is not this use.
- Checked in: VOCABULARY.md:123; VOCABULARY.md:143
- Your choice: Whether the rejected word covers this figurative use, and what the sentence says instead.

**Line 490. Two places disagree.**

- The document says: on CHO polyreactivity it is ahead everywhere.
- What the code, thesis or your messages say: Line 117 says CHO polyreactivity is out. No CHO polyreactivity dataset is in the Paper 2 or Paper 3 dataset lists in the run plan.
- Checked in: PAPER_2_OUTLINE.md:117; tests/experiments/papers/plan.py:77-91
- Your choice: Whether a thesis finding on a dataset this paper drops may be reported in the Results.

**Line 492. The document is wrong, and the fix needs your choice.**

- The document says: mRMR beats the hybrid wherever its feature cap lets it run, which is antibodies only.
- What the code, thesis or your messages say: The cap is 200 features. The Doyle plate run has budgets of 100, 250 and 500 features, and hERG blocking starts at 100, so both run under the cap and neither is an antibody. Over 60 dataset-and-budget cells recorded in the code, mRMR is best in 2 and the hybrid in 29.
- Checked in: tests/experiments/complexity/c17_information_theoretic_decomposition.py:615-630; tests/experiments/decided_grid.json, bh_doyle_plates general budgets [100, 250, 500], herg general [100, 250, 500, 1000, 2000]
- Your choice: What the paper claims about mRMR against the hybrid, and on which datasets the cap lets it run.

**Line 496. No agreed word.**

- The document says: held-out score against the number of views.
- What the code, thesis or your messages say: The same searched setting is called the representation count at lines 45 and 71 of this file. The word list sends "number of representations" to "representation count" and "representation" to "view" in Paper 2 documents.
- Checked in: VOCABULARY.md:148; VOCABULARY.md:181; PAPER_2_OUTLINE.md:45, :71
- Your choice: Her Paper 2 word for the setting whose values are 1, 2, 3, 5 and all.

**Line 533. The document is wrong, and the fix needs your choice.**

- The document says: on the small-molecule ADME sets the score rests on five or six views, none carrying more than about a third; on the antibody sets it rests on two or three.
- What the code, thesis or your messages say: The thesis says the drop spreads about equally over five to seven representations on the small-molecule sets, and that antibody hybrids draw on about one and a half to five.
- Checked in: thesis.tex:2114; thesis.tex:2182
- Your choice: Whether the paper restates the thesis ranges or waits for the Paper 2 run's own numbers.

**Line 539. The document is wrong, and the fix needs your choice.**

- The document says: In the thesis this is nowhere, about the narrowed prediction intervals.
- What the code, thesis or your messages say: The thesis reports the 90% conformal coverage and says the hybrid's intervals are narrower, for example on LogD with the SVM, and points to an Additional file for the coverage.
- Checked in: thesis.tex:2647; thesis.tex:2067; thesis.tex:2016
- Your choice: What this bullet adds beyond the thesis, now that the thesis carries both coverage and width.

**Line 540. Two places disagree.**

- The document says: Coverage stays out of the comparison: split conformal meets its 90% whatever the features are, so the conditions cannot be told apart by it.
- What the code, thesis or your messages say: Lines 541 to 543 then say coverage falls well below 90% on three antibody binding scans. The thesis says the hybrid's intervals cover fewer held-out antibodies than the single representation's on several small antibody sets.
- Checked in: PAPER_2_OUTLINE.md:541-543; thesis.tex:2647
- Your choice: Whether coverage is part of this comparison or left out of it.

**Line 547. Two places disagree.**

- The document says: It runs the other way on nanobody thermostability.
- What the code, thesis or your messages say: Line 117 says nanobody thermostability is out. No nanobody dataset is in the Paper 2 or Paper 3 dataset lists in the run plan.
- Checked in: PAPER_2_OUTLINE.md:117; tests/experiments/papers/plan.py:77-91; the example comes from thesis.tex:2494-2497
- Your choice: Whether the SVM regularization paragraph keeps an example from a dataset this paper drops.

**Line 578. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Her words on what the budget being tracked is, date not recorded: "there is no singular 'winning budget'..." and "It's the best set of features for that model AND budget".
- What the code, thesis or your messages say: Neither sentence appears in any of her messages in this project's transcripts. "winning budget" appears there only in assistant text and session summaries.
- Checked in: search of "type":"user" lines across /Users/apunt/.claude/projects/-Users-apunt-repos-KIRBy/*.jsonl for both strings: no hit; assistant hits in e0c2452b at 2026-10-30T01:56:29Z and 16:19:19Z
- Your choice: Whether these are her sentences, and what the quantity c03 correlates is called.

**Line 639. The document is wrong, and the fix needs your choice.**

- The document says: Limitations: each dataset's grid was cut by hand; adaptive pruning exists and was not used.
- What the code, thesis or your messages say: The Paper 2 run runs three searches per dataset per fold: the full search space with calibrated pruning, the fast tier, and the full space with no pruning. She picks which one the experiments read.
- Checked in: tests/experiments/papers/plan.py:13-15, :26-29; tests/experiments/orchestrate/ledger.py:81-83
- Your choice: Which of the three searches the paper reports, which settles whether either limitation still holds.

**Line 658. Two places disagree.**

- The document says: Written from the results that exist; not being rerun. Source c07.
- What the code, thesis or your messages say: Line 690 says everything re-runs except c15. Her messages exempt only c15, on 2026-10-30 and 2026-11-02, after she wrote "Everything needs to be re-run".
- Checked in: PAPER_2_OUTLINE.md:690; her messages, e0c2452b 2026-10-30T19:21:59Z and 19:26:43Z, 75b352ed 2026-11-02T05:39:46Z
- Your choice: Whether the easier-and-harder-half split experiment reruns, or is exempt like c15.

**Line 663. Two places disagree.**

- The document says: The run list names c01 through c33 and the OPIG scripts, without c05 or c07.
- What the code, thesis or your messages say: Neither is in the dismissed list at lines 669 to 685. c05 is cited as a source at lines 38 and 563, and c07 at line 659.
- Checked in: PAPER_2_OUTLINE.md:38, :563-564, :659, :664-685; tests/experiments/papers/plan.py:98-100 lists neither
- Your choice: Whether the sample-count null and the easier-and-harder-half split run, are dismissed, or are reported from existing results.

**Line 669. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Her words on what the c13 experiment should be, date not recorded: "Forward compatibility was supposed to showcase..." and "It should be able to run this experiment for recently published new embeddings."
- What the code, thesis or your messages say: Neither sentence appears in any of her messages in this project's transcripts.
- Checked in: search of "type":"user" lines across /Users/apunt/.claude/projects/-Users-apunt-repos-KIRBy/*.jsonl: the only file holding "Forward compatibility was supposed" is this session's own transcript
- Your choice: Whether these are her sentences, and whether the dismissal stands on them.

**Line 693. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: c18 — rerun from scratch on all the datasets (Adelaide, 2026-10-30).
- What the code, thesis or your messages say: No message of hers says c18 reruns from scratch on all the datasets. Her nearest sentences are "This is the end of this chat, but I need c18 finished" and "Everything needs to be re-run".
- Checked in: her messages, 2199a5fd 2026-10-30T18:49:50Z and e0c2452b 2026-10-30T19:21:59Z; search of her messages for "from scratch" returns no c18 hit
- Your choice: Whether the experiment on where the gain comes from reruns on every dataset, and in her own sentence.

**Line 708. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: 1,000 resamples of molecule groups is the form she is open to.
- What the code, thesis or your messages say: Her sentence names 1,000 and nothing else: "the ban was for expecting to see significance across 5 folds, if this is done correctly with 1000 I'm fine adding it back in". What is resampled was named by Claude after her message.
- Checked in: her message, 0ab3513e 2026-11-01T17:23:16Z; the assistant's molecule-groups explanation follows at 17:23:43Z
- Your choice: What a bootstrap resamples, if one goes beside those numbers.

**Line 729. Two places disagree.**

- The document says: The generator experiment's random null is described two ways, and the two disagree on what is drawn at random.
- What the code, thesis or your messages say: The Methods bullet at line 254 says a random Gaussian matrix of the blueprint's width. The code builds the same number of features as the blueprint, drawn at random from the views the blueprint drew from, and says a random Gaussian matrix cannot be run because a predictor fitted on noise cannot score a molecule the generator invents.
- Checked in: PAPER_2_OUTLINE.md:254-255; tests/experiments/opig_generator_scoring_swap.py:294-306; tests/experiments/opig_generator_scoring/README.md:43, :51-58
- Your choice: Which of the two is the null for this experiment: the numbers drawn at random, or which features are taken.

**Line 738. No agreed word.**

- The document says: Arm A of the generator experiment is dropped.
- What the code, thesis or your messages say: "arm" is a rejected word and the list says to write out what each one is. The files record the reason for the drop but never state what that run handed the predictor.
- Checked in: VOCABULARY.md:123, :186; tests/experiments/opig_generator_scoring/README.md:36-49, which says the three published components are kept in every arm
- Your choice: What the dropped run is called by what it computes.

### `PAPER_3_OUTLINE.md` (13)

**Line 29. No agreed word.**

- The document says: Introduction: 'a cleanup that spares a molecule because its neighbors agree'.
- What the code, thesis or your messages say: VOCABULARY.md:84 and :171 rule out 'spares' and map it to Cliff put-back. Cliff put-back undoes a cleanup for molecules whose similar partner has a label far from their own (thesis.tex:3675). That is neighbors disagreeing. The n44 cleanup `reliability_weight` weights each molecule from its neighbors' labels (n44_detect_and_clean.py:391, :736).
- Checked in: PAPER_3_OUTLINE.md:28-30; VOCABULARY.md:84, :171; thesis.tex:3675; n44_detect_and_clean.py:391, :736
- Your choice: Which cleanup this sentence means, and your word for it.

**Line 37. No agreed word.**

- The document says: Methods: 'The six noise conditions and the five flip rules'.
- What the code, thesis or your messages say: VOCABULARY.md:66 defines noise condition as the thesis's seven (Gaussian to Censoring). Paper 3 keeps six code conditions: `gaussian`, `outlier_p05`, `grouped_shifted`, `censoring`, `shift_only_p10`, `hetero_seen`. A different six was decided on 2026-10-05 and included `sparse_dprime_up_p10`, cut on 2026-11-04.
- Checked in: VOCABULARY.md:66; PAPER_3_NOISE_EXPERIMENTS.md:47-55; noise_helpers.py:2096 PAPER3_CUT_CONDITIONS = ('dense_oofmse', 'hetero_hidden', 'sparse_dprime_up_p10')
- Your choice: Does 'noise condition' also name Paper 3's six code conditions, and does line 37 mean the six left after the 2026-11-04 cut?

**Line 46. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: The trimmed refit was cut on 2026-10-31 because 'it raised the top score no more often than deleting the same number at random.' No quote from you.
- What the code, thesis or your messages say: Your messages that day: 'trimmed refit / did it work / I didn't ask if it was better than random deletion', then 'how is it compared ot other methods for outliers and shifts', then 'it doesn't seem to be worth the extra computation no?', then 'yes do that'. The random-deletion reason is also written in n44_detect_and_clean.py:83-92.
- Checked in: Transcript 3544bfe7, 2026-10-31T01:05:24, 01:08:57, 01:12:22, 01:14:36; PAPER_3_OUTLINE.md:46-48; n44_detect_and_clean.py:83-92
- Your choice: Which reason the outline records for cutting the trimmed refit: its computation cost against deleting once, or the comparison with random deletion.

**Line 46. No agreed word.**

- The document says: 'repeating find-and-delete (the trimmed refit)'.
- What the code, thesis or your messages say: VOCABULARY.md:130-132 and :173 list the trimmed refit as not named yet. In the code it is `rounds_delete`: the detector re-scored on the kept molecules and the flagged share deleted, three times. It is cut. Multistage cleaning (three rounds, each deleting the 10% with the largest in-sample error) still runs whole in n43.
- Checked in: VOCABULARY.md:86, :130-132, :173; n44_detect_and_clean.py:83-92; n43_noise_aware_training.py:188-191
- Your choice: Is the cut trimmed refit the same method as Multistage cleaning, and if not, what is it called?

**Line 51. The document is wrong, and the fix needs your choice.**

- The document says: After a method is split into its test and its action, 'The published method still runs whole.'
- What the code, thesis or your messages say: In n43, Multistage cleaning is the only split method that also runs whole. Small-loss selection, confident learning, distance selection, Jo-SRC weighting and co-teaching run only as their detector-and-cleanup pairing. Your 2026-10-30 message: 'if the published method is covered by the detector/cleanup sweep, just make note of which combination maps to the original methodology'.
- Checked in: n43_noise_aware_training.py:76-78, :188-191, :215-222; transcript f65dbf9e 2026-10-30T15:04:48
- Your choice: Should the outline say that a split method covered by the detector-and-cleanup experiment is reported only as its pairing, naming the pairing?

**Line 59. Two places disagree.**

- The document says: The first of the two supporting checks, for every method, is shuffled detector scores.
- What the code, thesis or your messages say: Shuffled detector scores were built into n44 on 2026-11-18. VOCABULARY.md:166 records 'remove it' for `random_twin` in the detector and cleanup experiment only. `random_twin` is still written by n07, n23, n24, n25, n32, n35, n36, n38, n42 and the n15/n22 code, and n47's smoke test reads it.
- Checked in: n44_detect_and_clean.py:108-111, :1359-1360; VOCABULARY.md:81, :166; grep random_twin over tests/experiments/noise/*.py
- Your choice: Is shuffled detector scores the first supporting check in every experiment and for your two lists in `#s6`, or only in the detector and cleanup experiment?

**Line 73. Two places disagree.**

- The document says: The Adams folds are 'the only random split in either paper'. Line 106-107 repeats that no random split is used except the two Adams scans.
- What the code, thesis or your messages say: PAPER_2_OUTLINE.md:271 runs the MODNet experiment 'on Matbench's own five folds'. NEW_DATASETS.md:344-345 says those folds are KFold(5, shuffle=True), 'Random, not grouped'. PAPER_2_OUTLINE.md:95-97 marks Matbench not yet decided for Paper 2.
- Checked in: PAPER_3_OUTLINE.md:73-74, :106-107; PAPER_2_OUTLINE.md:95-97, :271; NEW_DATASETS.md:344-345
- Your choice: Does Matbench enter Paper 2 with its own random folds, which would make the Adams folds not the only random split?

**Line 96. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: 'Two rules fixed on 2026-10-06 and not reopened', including that s04's cliff set is built per outer fold from training molecules only.
- What the code, thesis or your messages say: s04 and s12 record the per-fold rule as 'DECIDED 2026-10-06 (Adelaide delegated)'. No message of yours from 2026-10-06 is in the transcripts. The outline carries no quote.
- Checked in: s04_activity_cliffs.py:15-21; s12_cross_domain_cliffs.py:348-350; search of her messages 2026-10-04 to 10-08 returned none
- Your choice: Is building the cliff set per outer fold from training molecules only your decision, or open?

**Line 96. The document is wrong, and the fix needs your choice.**

- The document says: Two measurements of the same molecule are never a cliff pair, and 's12 shares the same two calls'.
- What the code, thesis or your messages say: The same-molecule rule is in noise/cliff_rules.py (n25). s04 and s12 build cliffs through `fold_cliff_partition`, which calls `calibrated_cliff_rule`. That removes only a molecule paired with itself. Two repeat measurements of one molecule, as in hERG IC50, can still form a cliff pair there.
- Checked in: cliff_rules.py:45-48; s12_cross_domain_cliffs.py:348-374; similarity_helpers.py:1159, :1188, :1193 (np.fill_diagonal only)
- Your choice: Should s04 and s12 also treat two measurements of one molecule as never a cliff pair?

**Line 126. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Inside the paragraph dated to you on 2026-11-01: 'Keeping a random 90% of molecules, best of 20 seeds, is the yardstick for any best-of-several-settings result.'
- What the code, thesis or your messages say: No message of yours mentions best of 20, 90% or a yardstick. Your 2026-11-01 message sets QM9's two parts only. The sentence came in with commit d3528199 on 2026-11-02. 'yardstick' is a rejected word (VOCABULARY.md:165). n47 has a removal curve at several shares with 20 seeds each, with no single share fixed.
- Checked in: Search of her messages for 'best of 20|90 ?%|yardstick'; transcript 3544bfe7 2026-11-01T15:49:38; n47_noise_measurements.py:72-76; VOCABULARY.md:165
- Your choice: Is a random 90% of molecules, best of 20 seeds, a control for best-of-several-settings results, and is it the shuffled detector scores control or a separate one?

**Line 156. The document is wrong, and the fix needs your choice.**

- The document says: Change to s02 (Adelaide, 2026-11-02): choose the single representation by the correlation on the fold's training molecules; this 'is already measured in s01, so this changes which representation is named, not what is computed.'
- What the code, thesis or your messages say: s01 computes the correlation over pairs of the fold's held-out molecules, not training molecules. Your approval was 'Yep that sounds good', answering a reply that made the same claim.
- Checked in: s01_similarity_baseline.py:8-11, :115; transcript 75b352ed 2026-11-02T08:14:51 and the assistant reply before it
- Your choice: Keep the change, knowing it needs a new correlation computed on each fold's training molecules?

**Line 177. Two places disagree.**

- The document says: s05 names 'the kNN hybrid (the top hybrid of the kNN search)'. Lines 189-191 give thesis-run results for it.
- What the code, thesis or your messages say: VOCABULARY.md:47 says the kNN search fits kNN only, from your 2026-11-04 decision. That is an uncommitted edit in decided_grid.py. The thesis-run search added kNN to the dataset's two models (line 201 says so). thesis.tex:2808 describes it as kNN as the downstream model.
- Checked in: PAPER_3_OUTLINE.md:177-178, :189-191, :201; VOCABULARY.md:38, :47; git diff tests/experiments/decided_grid.py; thesis.tex:2808
- Your choice: Should lines 189-191 say the thesis-run kNN hybrid came from a search that also fitted the two downstream models?

**Line 274. Two places disagree.**

- The document says: s08 thesis run: the two sides 'disagreed in 149 of 184 comparisons', with prediction wanting more features in 158, the same in 22 and fewer in 4.
- What the code, thesis or your messages say: 158, 22 and 4 add to 184, which makes 162 disagreements, not 149. The outline does not say what one of the 184 comparisons is. The count rests on server results, which are not checkable here.
- Checked in: PAPER_3_OUTLINE.md:273-276
- Your choice: What one of the 184 comparisons is, and whether the disagreement count is 149 or 162.

### `PAPER_3_NOISE_EXPERIMENTS.md` (5)

**Line 32. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Line 32 records as decided that large datasets run levels 0 and 1.0 and that datasets of 5,000 molecules or fewer keep four levels. Line 488 repeats the 5,000-molecule rule. The only quotes are "LARGE DATASETS EXCLUDING QM9 RUN LESS" and "except qm9 that's a really important baseline to have".
- What the code, thesis or your messages say: Her messages never give 5,000 molecules or the levels. Claude asked whether Caco-2-size datasets (2,158 molecules, four levels) should keep four levels. Its 'No' option was 'only QM9 keeps four levels'. She replied "LARGE DATASETS EXCLUDING QM9 RUN LESS" (2026-11-05 00:59). Claude recorded that datasets of 5,000 or fewer keep four, and that nothing changes. QM9 now loads 4,995 molecules, so the 5,000 rule keeps its four levels with or without her exception. Section 4 counts Caco-2 at four levels (49 combinations).
- Checked in: PAPER_3_NOISE_EXPERIMENTS.md:32, :187, :488; session d1386d9a, 2026-11-05T00:22:12 (question), 00:58:25 (Yes/No options), 00:59:20 (her reply), 00:59:49 (Claude's record); tests/experiments/noise/noise_helpers.py:2142-2145 (level 0 and 1.0 for the larger datasets, dated to her decision of 2026-10-06, not checked: no transcript before 2026-10-14) and :2156 ('more than 5000'); tests/experiments/papers/plan.py:117-120
- Your choice: Which datasets "LARGE DATASETS" means: more than 5,000 molecules, so Caco-2 keeps four levels, or every dataset except QM9, so Caco-2 runs levels 0 and 1.0.

**Line 38. Two places disagree.**

- The document says: Lines 38-41, headed 'the wording she approved', say the second run adds Cliff put-back, Random put-back, 'plus the random and shuffled twins'.
- What the code, thesis or your messages say: She approved these two sentences on 2026-11-04 ("Yes I approve those two sentences at the top"). On 2026-11-18 she answered "remove it" to keeping `random_twin`. Since commit 7d07b139, n44 writes no `random_twin`. It now writes shuffled cleanup results and shuffled detector scores. So the approved wording names a control that her later answer removed, and omits one she asked to build ("build it").
- Checked in: PAPER_3_NOISE_EXPERIMENTS.md:38-41, :301; her messages 2026-11-04T20:49:46 (session 780d4c6a), 2026-11-18T15:24:00 "build it" and 15:38:17 "remove it" (session af128977); tests/experiments/noise/n44_detect_and_clean.py:1358-1360 (no_random_twin=True); VOCABULARY.md:81, :166
- Your choice: Whether lines 38-41 stay as the record of what she approved on 2026-11-04, with a dated note below, or are rewritten to the controls n44 runs now.

**Line 130. The document is wrong, and the fix needs your choice.**

- The document says: The open question asks whether `grouped_shifted` belongs in `DETECTION_IS_DEFINED`. It says `grouped_shifted` moves every molecule in an affected scaffold family and leaves the rest alone, so a set of moved molecules exists.
- What the code, thesis or your messages say: The injector draws an offset for every scaffold family and a separate error for every molecule, and adds both. Every label moves. No set of moved molecules exists, as with `gaussian`. A code comment says 'grouped_shifted moves every molecule'. The thesis table gives 'All' for labels moved.
- Checked in: /Users/apunt/repos/NoiseInject/noiseInject/core.py:715-739 (_draw_grouped_shifted: b for every group, e for every molecule); tests/experiments/noise/noise_helpers.py:478, :515-520; thesis.tex:3520-3522 (from the earlier pass, not re-read this run)
- Your choice: Whether this question stays on her open list, now that the injector moves every molecule under `grouped_shifted`.

**Line 135. The document is wrong, and the fix needs your choice.**

- The document says: The open question is what n36's re-measurement priority ranks molecules by. It says both `measurement_spread` and `training_label_error` are in `SPREAD_SCORES`, with `measurement_spread` the default.
- What the code, thesis or your messages say: `SPREAD_SCORES` and `SPREAD_SCORE = 'measurement_spread'` are in n17_replicate_removal.py. There they set what n17's ceiling ranks molecules by. n36 ranks molecules by its own rules: random, disagreement, leverage, leverage times disagreement, and sparsest. n36 uses neither spread measure. n17 is not in the Paper 3 run list; n36 is.
- Checked in: tests/experiments/noise/n17_replicate_removal.py:243-266; tests/experiments/noise/n36_remeasure_priority.py:189-190 (RULES), :281; tests/experiments/papers/plan.py:106 (n43-n47, n15, n20, n36)
- Your choice: Whether this question is about n17's ceiling, which Paper 3 does not run, or is removed from the Paper 3 list.

**Line 541. No agreed word.**

- The document says: The cut cleanup `rounds_delete` is called 'repeating find-and-delete', and its evidence is given for 'the trimmed refit'. Line 544 then says Multistage cleaning stays whole in the noise-aware training experiment.
- What the code, thesis or your messages say: VOCABULARY.md lists 'trimmed refit' and 'repeating find-and-delete' as having no agreed word, and whether it is Multistage cleaning as open. The code treats them as two methods. `rounds_delete` is n32's trimmed refit applied to any detector. n32's trimmed refit is ITLM (Shen and Sanghavi 2019): keep 80, 90 or 95% of molecules, 4 rounds. Multistage cleaning deletes 10% per round for 3 rounds. She wrote "trimmed refit did it work" on 2026-10-31.
- Checked in: PAPER_3_NOISE_EXPERIMENTS.md:537-545; VOCABULARY.md:86, :130-132, :173; tests/experiments/noise/n44_detect_and_clean.py:83-92, :319-320; tests/experiments/noise/n32_iterative_and_dual_selection.py:10-11, :20-21; her message 2026-10-31T01:05:24 (session 3544bfe7)
- Your choice: What word names `rounds_delete` in this paragraph, given that the code keeps it apart from Multistage cleaning.

### `AC_GRID_TEST_PLAN.md` (8)

**Line 164. Two places disagree.**

- The document says: Line 143 gives the prediction-objective binary search no n_reps setting. Line 144 puts n_reps=all on the similarity binary search only, and lines 185 and 1273 repeat that. Line 164 says "n_reps is fixed to all, on the binary searches only", plural.
- What the code, thesis or your messages say: Her first sentence: "general tier, similarity objective, binary input only (n_reps=all, not swept over for both)". Her second: "n_reps=all - for binary only. Because not many reps are binary only." Neither says which of the two binary searches the fixed value covers.
- Checked in: AC_GRID_TEST_PLAN.md:143-144, 164-166, 184-185, 1273; transcript 37fa5e85, user messages 2026-11-04T23:55:51Z and 23:57:25Z.
- Your choice: Whether n_reps is fixed to all on both binary searches, or on the similarity-objective binary search only.

**Line 444. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Heading: "TWO THINGS CLAUDE INVENTED AND PUT IN THIS FILE AS HER DECISIONS. BOTH CUT 2026-11-07." The two are correlating each metric's ranking with a model's score on each hybrid, and the similarity-to-distance conversion.
- What the code, thesis or your messages say: Commit df987119 removed text that recorded both as open, not as her decisions. Its removed list under "Open, for her" held both items. Her messages from 02:28 to 02:43 on 2026-11-07 ask what the first item is. None says to cut either one. At 02:43 she wrote "do not change anything".
- Checked in: AC_GRID_TEST_PLAN.md:444-459; git show df987119 -- AC_GRID_TEST_PLAN.md (removed lines 34, 42, 86-90); transcript 37fa5e85, user messages 2026-11-07T02:28:15Z to 02:43:44Z.
- Your choice: Whether these two items are cut, or go back on her open list.

**Line 454. The document is wrong, and the fix needs your choice.**

- The document says: The conversion "is not one: the distance comes straight from the hybrid's feature table with the same `scipy_cdist` call `cliff_rules.py:1092` already makes, and nothing is converted." Line 463 then says nothing is open.
- What the code, thesis or your messages say: The scipy_cdist route covers euclidean, manhattan, minkowski, chebyshev, cosine, correlation, canberra and braycurtis only. Tanimoto and dice are two of her five TBD1 coefficients. They are not on that route. In simeval.py, dist_tanimoto and dist_dice compute a similarity and return 1.0 minus it.
- Checked in: AC_GRID_TEST_PLAN.md:124, 454-465; tests/experiments/noise/cliff_rules.py:1062-1095; tests/experiments/similarity_metrics/simeval.py:48-72.
- Your choice: Whether the similarity-to-distance conversion for tanimoto and dice is a setting she sets, or stays as 1 minus the similarity.

**Line 881. No agreed word.**

- The document says: "the decided grid with it" (line 881), "n43 to n47 on the decided grid" (line 1087), "runs her decided grid for n43 to n47" (line 1413).
- What the code, thesis or your messages say: Here the phrase means the noise grid the Papers ledger runs with the promote flag: five folds and every level. VOCABULARY.md:151 maps "decided grid" to search grid, which is the hybrid search's settings. The thesis has no "noise grid". She wrote "the full noise grid" on 2026-11-16. The file also uses "larger" and "smaller noise grid" for the two dataset-size groups.
- Checked in: AC_GRID_TEST_PLAN.md:881, 932-933, 1087, 1413; VOCABULARY.md:34, 151; transcript 13883162, user message 2026-11-16T14:25:12Z.
- Your choice: Her word for the noise grid the ledger runs with every fold and level, as against the screen.

**Line 973. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "She decided five folds, and four levels on datasets of 5,000 molecules or fewer." Line 1413-1414 repeats the 5,000 cutoff. Line 829 says "Five folds and the dataset's two decided models are unchanged (Adelaide, 2026-10-30)" with no quote.
- What the code, thesis or your messages say: Her folds sentence: "just do the 5 folds - that rule was made in the interest of time". Her levels sentence: "LARGE DATASETS EXCLUDING QM9 RUN LESS", which names no molecule count. No message of hers in these transcripts gives 5,000. The code takes 5,000 from the kNN dataset groups. KNN_SEARCH_SPACE_RECORD.md credits that split to her on 2026-10-05; not checked, as the transcripts start 2026-10-14. The two decided models are not checked.
- Checked in: AC_GRID_TEST_PLAN.md:829, 973-974, 1413-1414; transcript 3544bfe7, user message 2026-10-30T19:27:06Z; transcript d1386d9a, user message 2026-11-05T00:59:20Z; tests/experiments/noise/similarity_space.py:196-204, 233; tests/experiments/framework/KNN_SEARCH_SPACE_RECORD.md:224-240.
- Your choice: Whether 5,000 molecules is the size above which a dataset runs fewer noise levels.

**Line 1003. No agreed word.**

- The document says: Line 1003: "Each runs six label-flip noise conditions at six levels with 3 repetitions." Line 1286: "The classification grid: label-flip noise conditions, levels...".
- What the code, thesis or your messages say: VOCABULARY.md:66 lists seven noise conditions, all for regression. The thesis calls the six ways of flipping labels "label-flipping strategies" at line 363 and "rules" at line 3533. No agreed word covers them.
- Checked in: AC_GRID_TEST_PLAN.md:1003, 1286; VOCABULARY.md:66; thesis.tex:363, 3533.
- Your choice: Her word for the six ways the classification labels are flipped.

**Line 1270. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Heading: "Named in her outline and not built, carried here so they are not lost." It lists the kNN search, the binary similarity search, and the Matthews correlations `opig_activity_cliff_qsar` computes.
- What the code, thesis or your messages say: Her pasted outline names none of the three. The kNN-only search and the binary searches come from her separate 2026-11-04 message. No message of hers asks for the Matthews correlations. That item appears only in Claude's lists, one of which she pasted back.
- Checked in: AC_GRID_TEST_PLAN.md:1270-1276; transcript 37fa5e85, user messages 2026-11-07T02:38:04Z (outline) and 2026-11-04T23:55:51Z; transcript aa8f8e5d, user paste 2026-11-02T09:37:29Z.
- Your choice: Whether reporting the Matthews correlations of `opig_activity_cliff_qsar` stays on her list.

**Line 1978. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "Her own record, `THESIS_OUTLINE.md` lines 2269 to 2273, marked CHECKED 2026-10-09" and "Her open decision on that, `THESIS_OUTLINE.md` lines 2274 to 2275, verbatim".
- What the code, thesis or your messages say: Those lines entered in commit 3575fcf69 of 2026-10-09, which carries "Co-Authored-By: Claude Opus 5.5". The note cites "Source: CH6.md §B 6.5", a scratchpad file. Nothing read this run shows she wrote either note.
- Checked in: AC_GRID_TEST_PLAN.md:1978-1988; THESIS_OUTLINE.md:2269-2275; git blame -L 2269,2275 THESIS_OUTLINE.md; git log -1 3575fcf6.
- Your choice: Whether the held-out cliff rule question in that note is one of her open decisions.

### `NEW_DATASETS.md` (11)

**Line 83. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "DroneRF goes into Paper 2 after its view screen (Adelaide, 2026-11-02)." It adds that DroneRF being in neither outline is expected.
- What the code, thesis or your messages say: No message of hers says DroneRF goes into Paper 2. On 2026-11-02 she was asked whether it is in Paper 2. She wrote: "Eh I'll add stuff to the outline after the screening/view selection, leave it". On 2026-11-04 she wrote: "fine as long as it all ends up in the right places".
- Checked in: NEW_DATASETS.md:83-84; route step 4 at NEW_DATASETS.md:48-51 makes in or out her call; transcript 4a62cdb5, 2026-11-02T16:05:14Z; transcript c75742b8, 2026-11-04T18:58:15Z
- Your choice: Whether DroneRF goes into Paper 2, or stays open until she has seen its view screen.

**Line 103. Two places disagree.**

- The document says: The table of where each candidate stands puts ProteinGym v1.3 at "2, reproduced", waiting for a view screen.
- What the code, thesis or your messages say: Step 2 is the published score rerun under the published protocol. The ProteinGym section says "Reproduction: Not done." Its benchmark files say none of them reproduces a published number, because the published baselines convolve over residues. The route says nothing moves a step until she says so.
- Checked in: NEW_DATASETS.md:34-35 and 40-42 (route); :674-677 (Reproduction); :627-629 and :741-742 (convolution over residues); :103
- Your choice: Whether ProteinGym stays at step 1 until ProteinNPT's published baseline is rerun, or whether the mean-pooled ESM-1v run counts as its step 2.

**Line 253. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "Kept as it is (2026-11-02)": the cdr_flexibility folds keep loop clusters at 80% identity. The reason given is that lowering the threshold on a 9-residue loop merges loops differing at one position. It says to write in the paper that the folds hold out loops rather than families.
- What the code, thesis or your messages say: No one is named and no quote is given. On 2026-11-02 this was put to her as open. Her replies were "but this was fixed elsewhere wasn't it" and "give me the concensus on those two datasets". The stated reason disagrees with the code. Identity is 1 minus edits over the longer loop's length. At 80% a 9-residue loop already merges with a loop one position away (0.89); at 70% it merges two positions away (0.78).
- Checked in: NEW_DATASETS.md:249-257; tests/experiments/_shared.py:2939-2943 (identity) and :5242 (identity_threshold=0.80); transcript 3544bfe7, 2026-11-02T14:04:17Z and 14:04:48Z; the 80% value traces to the comment 'option A (user 2026-07-19)' at _shared.py:5198, older than the transcripts and not checked
- Your choice: Whether the cdr_flexibility folds keep loop clusters at 80% identity, against the other two options put to her: a lower threshold, or grouping on the whole Fv.

**Line 262. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "tz_her2 split, fixed 2026-11-02 (Adelaide)." The folds group CDRH3 loops at 80% identity, single linkage. Line 275 says 80% rather than Chinery et al.'s 70%.
- What the code, thesis or your messages say: No quote of hers gives 80% or the change from 70%. On 2026-11-01 she asked how OPIG members handle the dataset. On 2026-11-02 she wrote "check the clonotype split". Then: "I don't know what this means, just get it done right". The file's measured numbers: at 70% one group holds 3,950 of 5,000 samples; at 80% the largest holds 284; at 90%, 4,646 of 4,809 groups hold one sample.
- Checked in: NEW_DATASETS.md:262-280; transcript 3544bfe7, 2026-11-01T21:19:37Z; transcript 62112545, 2026-11-02T08:18:49Z, 08:21:30Z, 08:39:14Z
- Your choice: Which CDRH3 identity threshold groups the tz_her2 folds: 70% as in Chinery et al., 80%, or 90%.

**Line 583. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "Only modulo and contiguous are used here." Both ProteinGym fold schemes are used.
- What the code, thesis or your messages say: Her recorded choice is modulo: "sure do modulo, as long is that is an industry standard solution" (2026-10-23). No message of hers adds contiguous. The loader records modulo only. HANDOFF_PROTEINGYM.md says only modulo has been checked against the loader.
- Checked in: NEW_DATASETS.md:580-583 and :1884; transcript e812739e, 2026-10-23T03:56:04Z; src/kirby/datasets/proteingym.py:15 ('Fold scheme (Adelaide, 2026-10-23): modulo.'); HANDOFF_PROTEINGYM.md:22-23
- Your choice: Whether ProteinGym uses contiguous folds as well as modulo, or modulo only.

**Line 660. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Not all 217 assays. A rule picks 13: at least 800 single mutants, estimated ESM-1v cost at most 3 hours on the laptop, then the cheapest assay per selection type and taxon.
- What the code, thesis or your messages say: No quote of hers gives this rule. On 2026-10-28 she wrote "Not 217 tasks". She also wrote: "I do want ot use a few standard ones using published github protocols that match the pbulished results". The 800, 3-hour and cheapest-per-combination values are not hers. The cost is scaled from one measured assay.
- Checked in: NEW_DATASETS.md:658-672; transcript e812739e, 2026-10-28T15:28:03Z and 15:30:50Z
- Your choice: Whether the 13 assays this rule picks are the ProteinGym assays used.

**Line 881. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Under "Object, models and candidate views (owner, 2026-10-30)": every view is computed on the 100,000-sample chunk, the object of the published split.
- What the code, thesis or your messages say: No quote of hers chooses the chunk. Her quotes that night cover models: "I would rather compare against published papers than other datasets in my own expeirment". They also cover views: "I'll select up to 10 (ideally less)". The file names two other objects papers use: the whole 0.25 s file and the 10,000-sample frame.
- Checked in: NEW_DATASETS.md:879-883; transcript 35aa60ea, 2026-10-30T03:15:30Z and 03:16:46Z
- Your choice: Whether the 100,000-sample chunk is the object every DroneRF view is computed on.

**Line 884. Two places disagree.**

- The document says: Almasri 2021's removal of low-power chunks before training is "Not done", because it changes the set of chunks.
- What the code, thesis or your messages say: A brief she pasted on 2026-10-30 says: "Add an option to drop low-power chunks below a PSD threshold (Almasri’s γ) and log how many chunks per class survive. Run with and without." No message of hers drops it.
- Checked in: NEW_DATASETS.md:884-886; transcript 35aa60ea, user message 2026-10-30T02:31:55Z
- Your choice: Whether DroneRF runs both with and without Almasri's low-power chunk removal, or without it only.

**Line 892. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Three DroneRF model settings the papers do not state are filled in: xgboost defaults, an svm grid step of 2^2, and k = 5 for knn. The heading attributes them to the owner on 2026-10-30.
- What the code, thesis or your messages say: No quote of hers gives any of the three values. The table itself marks each as not stated in its source paper. The svm is also not in the screen, because no smaller grid has been chosen.
- Checked in: NEW_DATASETS.md:879 and :889-896; :977-980 (svm left out of the screen); transcript 35aa60ea, 2026-10-30 search for k = 5 and 2^2 found no message of hers
- Your choice: Which values the DroneRF screen uses for the xgboost settings, the svm grid step and the knn k.

**Line 922. Two places disagree.**

- The document says: "the raw 20,000 samples as a vector" is left out of the candidate views, because no paper uses them with a vector model.
- What the code, thesis or your messages say: The object is a 100,000-sample chunk per band, which is 200,000 values across both bands. 20,000 values is one 10,000-sample frame across both bands. Her words on views: "Sorry just test everything" (2026-10-30).
- Checked in: NEW_DATASETS.md:823, :881, :915, :918-922; tests/experiments/benchmarks/b06_dronerf_views.py:134-136 (L = 100,000; FRAME = 10,000); transcript 35aa60ea, 2026-10-30T03:16:46Z
- Your choice: Whether a raw-sample vector is a DroneRF candidate view, and if so of the chunk or of the 10,000-sample frame.

**Line 1856. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "New condition from this (owner, 2026-10-30)": a dataset must leave room above its published best, measured in test samples, well above the test set's resolution.
- What the code, thesis or your messages say: Her words that day were "So there is almost no room to improve on it" and "I need things where there's actually remove to improve". No quote of hers says "measured in test samples" or "well above the test set's resolution". The file sets no margin for "well above".
- Checked in: NEW_DATASETS.md:1856-1858 and :39 (route step 1); transcript 35aa60ea, 2026-10-30T13:05:48Z and 13:06:18Z
- Your choice: What margin above the test set's resolution counts as enough room, if resolution is the measure.

### `scratchpad/precompute_cuts/DECISIONS.md` (13)

**Line 1. No agreed word.**

- The document says: "# Decisions from the earlier searches of the whole dataset (single record)"
- What the code, thesis or your messages say: The committed title was "# Precompute decisions (single record)". VOCABULARY.md:152 maps "precompute" to "earlier search of the whole dataset". Several sections did not come from those searches. The ChEMBL pairs and QM9 size (lines 79-104) come from tests/experiments/model_choice_test.py. Step 10 (lines 803-831) and New datasets (from line 833) do not either.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:1, :79-104, :803-831, :833; git show HEAD:scratchpad/precompute_cuts/DECISIONS.md line 1; VOCABULARY.md:36, :152
- Your choice: What this file's title is.

**Line 22. Two places disagree.**

- The document says: Line 22: "ridge is not a downstream model: removed everywhere on 2026-07-16". Line 107: "Ridge: not yet answered."
- What the code, thesis or your messages say: tests/experiments/_shared.py:509 and :570 record the 2026-07-16 removal. _shared.py:604-629 defines RIDGE_MODELS = DOWNSTREAM_MODELS + ['ridge'] for c16, the feature-selection-as-regularization experiment. Step 1 lists ridge among the top three models at lines 36, 52, 56, 64 and 66. Her transcripts begin 2026-10-14, so her 2026-10-03 rules cannot be checked.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:22, :36, :52, :56, :64, :66, :107; tests/experiments/_shared.py:509, :570, :604-629
- Your choice: Whether ridge is out as a downstream model everywhere, or still unanswered as line 107 says.

**Line 72. Mentions patient data.**

- The document says: Mentions patient data at lines 72, 73, 153, 154, 201, 202, 249, 250, 297, 298, 345, 346, 393, 394, 459, 460, 508, 509, 556, 557, 604, 605, 653, 654, 698, 699, 746, 747, 796 and 797. The counts of "41 datasets" at lines 3 and 753 include those two datasets.
- What the code, thesis or your messages say: CLAUDE.md bans patient data from "any figure, table, count or handoff". It also says older documents still mention them: "Ignore those parts." The Step 1 table (lines 35-75) has 41 dataset entries; two of them mention patient data.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:3, :35-75, :753 and the 30 lines listed; CLAUDE.md "No patient data, anywhere"
- Your choice: Whether these 30 table lines and the count of 41 are removed from this file, or left as an older document.

**Line 101. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "The models stay dnn + lightgbm on all three targets, from the same test". The test is the fixed-blueprint run at 10,000 molecules.
- What the code, thesis or your messages say: Her message of 2026-11-04 17:27 reads "D2 - dnn, svm | fxa - svm, lightgbm | mpro - lightgbm, rf | And 5k for QM9". It names no QM9 models. At 17:09 she called the fixed-blueprint scores "a shit hybrid". dnn+lightgbm is the 2026-10-03 pair at line 70, unchanged in decided_grid.json. On the gap, the output she pasted at 16:02 has svm 0.3596 rmse and lightgbm 0.3603.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:70, :97-104; tests/experiments/decided_grid.json qm9, qm9_mu, qm9_alpha general models [dnn, lightgbm]; transcript c75742b8 2026-11-04T16:02:46, 17:09:16, 17:27:34
- Your choice: Which two downstream models each QM9 target (gap, mu, alpha) uses.

**Line 261. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: 16 decision entries read "(2026-10-03, revised after the internal evaluator was no longer limited to the two chosen models; was ...)": lines 261, 271, 275, 277, 281, 292, 299, 356, 357, 363, 375, 376, 385, 425, 428, 434. None names Adelaide. Line 515 says the earlier Step 5 "wrongly limited the internal evaluator to the two chosen models".
- What the code, thesis or your messages say: The other decision entries in those tables carry "Adelaide". These 16 have been unattributed since the file entered git in commit 31bf504a (2026-10-06). Her transcripts begin 2026-10-14, so no message of hers on these revisions or on "wrongly" can be checked.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:261-299, :356-385, :425-434, :515; git show 31bf504a:scratchpad/precompute_cuts/DECISIONS.md lines 210, 220, 224
- Your choice: Whether these 16 revised values, and the word "wrongly" at line 515, are hers.

**Line 363. Two places disagree.**

- The document says: Line 42 sets the bh_doyle_plates pair to rf+lightgbm (2026-10-07, Adelaide; was rf+dnn). Lines 123 and 574 still label bh_doyle_plates rf+dnn. Line 363 records "cut treeshap" for the importance method. Line 352 says that table uses the chosen pair.
- What the code, thesis or your messages say: Recomputed in this pass from the slim copy, budgets 100-500. rf+dnn with the internal evaluator limited to the pair gives deeplift 15.4162 rmse (22 of the 30 best), lightgbm_gain 15.7082 (2), treeshap 15.6630 (6). That matches lines 315 and 363 exactly. rf+lightgbm with no limit gives deeplift 15.0697 (30), lightgbm_gain 15.6974 (0), treeshap 15.6064 (0). Under the line-352 rule at 1%, both lightgbm_gain and treeshap are then cut. decided_grid.json runs deeplift and lightgbm_gain in the general search. An earlier pass found the Step 7 entry at line 671 also follows rf+dnn; not re-run here.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:42, :123, :267, :315, :352, :363, :574, :671; tests/experiments/decided_grid.json bh_doyle_plates general (models rf, lightgbm; importance_methods deeplift, lightgbm_gain); /Volumes/seagate/slim_precompute/bh_doyle_plates.slim.db recomputed with scratchpad sk2.py
- Your choice: Whether bh_doyle_plates' cuts are recomputed on rf+lightgbm, or the cuts computed on rf+dnn stand.

**Line 567. Two places disagree.**

- The document says: abagym_Ang2 internal evaluators: rf 0.462 (5 of the 30 best), lightgbm 0.444 (3), dnn 0.452 (4), match 0.492 (18). Cut: none. Decision: "keep all four evaluators" (2026-10-03, Adelaide).
- What the code, thesis or your messages say: Recomputed in this pass with the cuts recorded now: pair gp+svm, budgets 25-100, deeplift cut. lightgbm gives 0.446 spearman with 1 of the 30 best; match gives 0.492. That is 9.3% below match, over the 3% the rule allows. So the rule at line 563 now cuts lightgbm. rf moves to 0.479 (8).
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:260, :356, :563, :567; scratchpad ax/ev3.py run on /Volumes/seagate/slim_precompute/abagym_Ang2.slim.db with scratchpad/precompute_cuts/scripts/axes.py decided()
- Your choice: Whether the lightgbm internal evaluator is cut on abagym_Ang2, or all four evaluators are kept.

**Line 755. Two places disagree.**

- The document says: Line 8: "No extra scripts for searches of the whole dataset." Line 108 and line 755 ("Decision (Adelaide 2026-10-03)") say searches of the whole dataset with knn as the downstream model are run.
- What the code, thesis or your messages say: In git HEAD line 8 read "No extra precompute scripts" and line 755 read "run the knn downstream precompute". The uncommitted word change made both say "of the whole dataset". Lines 10-11 say knn searches "run for the experiments that need them, when they come up". The similarity experiments now take knn's top hybrid from a search on each outer fold's training molecules.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:8-11, :108, :755; git show HEAD:scratchpad/precompute_cuts/DECISIONS.md lines 8, 110, 755; tests/experiments/similarity/similarity_helpers.py:321-330
- Your choice: Whether the knn runs in Step 9 are searches of the whole dataset or the per-fold kNN searches the similarity experiments run.

**Line 814. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "One row is one domain. Adelaide, 2026-11-04." The table gives run or cut per domain, with a reason for each, then per benchmark.
- What the code, thesis or your messages say: She asked "which of those domains should similarity be run on" (2026-11-04 18:50). The yes/no list and the reasons are the assistant's reply at 18:50. Her answer at 18:54: "Okay well this is going to be have to something that's made per-dataset in decisions.md | record those". The table is per domain, not per dataset.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:814-828; transcript c75742b8 2026-11-04T18:50:09, 18:50:22, 18:54:18
- Your choice: Whether the similarity run-or-cut decision is recorded per dataset or per domain.

**Line 903. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "tz_her2 grouping FIXED 2026-11-02 (Adelaide)." The folds group CDRH3 loops at 80% identity, where Chinery et al. 2024 use 70%.
- What the code, thesis or your messages say: Her messages that morning were "check the clonotype split" (08:21) and "I don't know what this means, just get it done right" (08:39). Neither names a threshold. The assistant measured 70%, 80% and 90% at 08:28 and reported 80% as done at 08:47. tests/antibody_baseline_survey.py:420 calls it "ONE DEVIATION".
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:903-910; tests/antibody_baseline_survey.py:420; transcript 62112545 2026-11-02T08:21:30, 08:28:35, 08:39:14, 08:47:58
- Your choice: Which CDRH3 identity threshold groups the tz_her2 folds: 70% as published, or 80%.

**Line 919. Two places disagree.**

- The document says: Under "Which datasets are in": "Benchmarks outside drug discovery: Matbench v0.1 (6 tasks), ProteinGym v1.3 (13 assays), DroneRF." No date and no name.
- What the code, thesis or your messages say: Line 828 of the same file says DroneRF is "not in either outline yet; it joins after its view screen". Lines 840-845 say a dataset reaches an outline only after it is let in. NEW_DATASETS.md:100-103 puts Matbench and DroneRF at step 3 and ProteinGym at step 2. NEW_DATASETS.md:109-111 says "whether they are in is Adelaide's call". None is in tests/experiments/papers/plan.py:77-92.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:828, :840-845, :919-920; NEW_DATASETS.md:100-103, :109-111; tests/experiments/papers/plan.py:77-92
- Your choice: Whether Matbench, ProteinGym and DroneRF are in Paper 2 now, or only after their view screens.

**Line 932. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "New condition from these two (Adelaide, 2026-10-30): a dataset must leave room above its published best, measured in test samples, well above the test set's resolution."
- What the code, thesis or your messages say: Her sentence (2026-10-30 13:06) was "I need things where there's actually remove to improve". The parts "measured in test samples" and "well above the test set's resolution" are not in her messages.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:932-933; transcript 35aa60ea 2026-10-30T13:03:26, 13:05:48, 13:06:18
- Your choice: Whether the measured form of this condition is her rule, or only room to improve on the published best.

**Line 960. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Under "Decided per benchmark (Adelaide)": "DroneRF (2026-10-30): the object is the 100,000-sample chunk."
- What the code, thesis or your messages say: The assistant proposed the chunk at 03:13 ("Take this set. Say yes, or change any line."). Her reply questioned the model set instead. At 03:15:52 the assistant listed the chunk as "Settled so far". On 2026-10-29 she had asked "each claassification task is for each time chunk, right?" No message of hers states the choice.
- Checked in: scratchpad/precompute_cuts/DECISIONS.md:954-963; transcript 35aa60ea 2026-10-29T22:27:02, 2026-10-30T03:13:39, 03:14:40, 03:15:52
- Your choice: Whether one DroneRF sample is the 100,000-sample chunk.

### `tests/slurm_scripts/runbook/PAPERS.md` (3)

**Line 62. Other.**

- The document says: The variant cleanups run starts only when `KIRBY_N44_SECOND_RUN` names a JSON file. Nothing in this ledger sets that variable. If a job exports it, every n44 unit that job runs is a variant cleanups run.
- What the code, thesis or your messages say: Once a unit's standard cleanup run is `ok`, the ledger does not claim that unit again. run_all also skips n44 when `<dataset>__<condition>.json` already exists. The variant cleanups run writes its rows under `__second_run`, but its summary file has that same name. So exporting the variable after the standard cleanup run starts nothing in this ledger. Exporting it before the standard cleanup run makes the variant cleanups run take its place. The ledger then records the unit done, and detector and cleanup pairs the JSON file does not name are never written. On 2026-11-04 you wrote: "Put-back and twins - okay yes, modify the experimental plan (plan.md) and the paper 2/3 ledger to refelct this. First the results, then the checks for what I need." You have not said how the variant cleanups run is started on the server.
- Checked in: tests/experiments/papers/ledger.py:325-345 (finish_unit reads the result file), tests/experiments/papers/plan.py:299-314 (result file is `<dataset>__<condition>.json`), tests/experiments/noise/noise_helpers.py:1850-1879 (skip when that file exists), tests/experiments/noise/n44_detect_and_clean.py:1163-1165 (`__second_run` only on rows and checkpoint), n44_detect_and_clean.py:816-823 (unnamed pairs not written), tests/experiments/papers/worker.py:14-16; her message, transcript 780d4c6a, 2026-11-04T20:30
- Your choice: Whether the variant cleanups run is started through this ledger, which needs a code step, or by hand outside it.

**Line 84. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Puts "Every dataset searches the full space, with adaptive pruning on" in quotation marks as your words of 2026-11-02.
- What the code, thesis or your messages say: That sentence is not in any of your messages. It appears only in text written by Claude. Your words on 2026-11-02 were "all run on addaptive pruning no?" and "full with adaptive pruning if that wasn't clear - that should be in the handoff". On 2026-11-04 you wrote: "I want to run all 3 for most datasets, then decide which one of the three to use everywhere. It will probably be adaptive pruning but it may be fast tier. We'll see." The same sentence is also quoted as yours in SEARCH_SETTINGS.md:136 (not checked this run).
- Checked in: tests/slurm_scripts/runbook/PAPERS.md:84-85; her messages, transcript 75b352ed 2026-11-02T05:36 and transcript af128977 2026-11-04T01:53; grep of user messages in all project .jsonl files finds no match for the quoted sentence
- Your choice: Which sentence of yours goes in this quotation, or whether the quotation marks and attribution come off.

**Line 91. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Puts "Adaptive pruning is not being re-tuned now (Adelaide, 2026-11-01). She will check at the Step 3 gate that it works as intended." in quotation marks, then says that gate is the choice of `experiments_search`.
- What the code, thesis or your messages say: The quoted text is in the third person and is not in any of your messages. Your sentence on 2026-11-01 was: "Okay in that case for that checkpoint I'll check the pruning is working as intended but I won't re-tune it right now". You said "that checkpoint", not "Step 3 gate". On 2026-11-04 you asked "what is the step 3 check?".
- Checked in: tests/slurm_scripts/runbook/PAPERS.md:91-92; her messages, transcript 75b352ed 2026-11-01T20:45 and transcript af128977 2026-11-04T01:53
- Your choice: Whether your 2026-11-01 sentence replaces the quoted text as written, and what to call that checkpoint in place of "Step 3 gate".

### `tests/slurm_scripts/runbook/SCREEN.md` (3)

**Line 4. Two places disagree.**

- The document says: Line 4: a screen scores each view "under the models the dataset's papers used". Line 49: the reactions row uses `decided_models(ds, 'general')`, rf and lightgbm on all three tasks.
- What the code, thesis or your messages say: The reactions row reads its models from KIRBy's search grid (Reactions._models), not from the reaction datasets' papers. Her words of 2026-10-30: "I would rather compare against published papers than other datasets in my own expeirment". No message from her chooses rf and lightgbm for the reaction screen. Which models the reaction datasets' papers used: not checked.
- Checked in: tests/experiments/screen/datasets.py:282-283, 304-306; SCREEN.md:4, 49; her message 2026-10-30T03:15 in session 35aa60ea
- Your choice: Does the reaction screen keep rf and lightgbm, or use the models the reaction datasets' papers used?

**Line 52. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: DroneRF svm is planned but not in the rows. Kılıç's grid over C and gamma with inner CV is "about 460,000 fits". It is "Not added until a smaller grid is chosen." (SCREEN.md:52-53)
- What the code, thesis or your messages say: The code grid is 16 C values x 16 gamma values (2^-15 to 2^15, step 2^2) with inner 3-fold CV. That is 768 inner fits per outer fold. DroneRF has 720 outer folds: 12 views x 3 tasks x 2 splits x 10 folds. The total is 552,960 inner fits plus 720 refits. 460,000 matches 10 views, not 12. The 460,000 first appears in a chat reply of 2026-10-31, which offered a 4 x 4 grid, svm on the small views only, or no svm. No message from Adelaide picks one. Her words of 2026-11-17: "for the main code source I need to ensure I'm testing those models (in this case ResNet with those specific configurations)". Time for the full grid: not measured.
- Checked in: tests/experiments/benchmarks/_xgb_worker.py:62-65; tests/experiments/screen/datasets.py:86-97 (comment: "the owner has not chosen a smaller one yet"); session 35aa60ea assistant reply 2026-10-31T00:25; her messages 2026-10-31T00:25 ("i don't have capacity for this locally") and 2026-11-17T17:21 in session 947237f0
- Your choice: Does DroneRF svm run on Kılıç's full grid, on a smaller grid, or not at all?

**Line 190. Two places disagree.**

- The document says: Lines 77-78 and 181-182: imports on the login node died with "Illegal instruction", so run them in an srun. Lines 89-96 run three import checks under "On the server, login node:". Lines 170-171 and 189-191: if compute nodes have no internet, run the LSST download and its four pretrained-embedding builds, and the BEANS fetch and first build of each encoder, on the login node.
- What the code, thesis or your messages say: The LSST pretrained-embedding builds import torch. The BEANS encoder builds import torch, then avex, torchaudio or transformers. These are the kind of import the document says died on the login node. Whether compute nodes reach the internet is not checked. Whether these interpreters crash on the login node is not checked. Both are server facts.
- Checked in: SCREEN.md:77-78, 89-96, 170-171, 181-182, 189-191; tests/experiments/benchmarks/b12_lsst_views.py:231-294; tests/experiments/benchmarks/beans_bats_views.py:136-183
- Your choice: Where do the import checks and the first downloads and builds run: login node, srun, or another route you name?

### `tests/slurm_scripts/runbook/RUN_BOTH_LEDGERS.md` (1)

**Line 6. No agreed word.**

- The document says: The Papers 2 and 3 ledger "builds representations" (line 6). Line 160 says "Representation builds always run". Line 164 says a dataset waits "until its representations have built".
- What the code, thesis or your messages say: One build step makes the feature sets for every dataset in this ledger. Those datasets include Paper 2-only ones: nucleotide sets, reactions and antibody sets. VOCABULARY.md gives Paper 2 "view" and Paper 3 "representation", quoting her: "for a paper talking about ML more generally, it's views." No rule covers a runbook for a step that serves both papers. The server-words section does not list either word.
- Checked in: RUN_BOTH_LEDGERS.md:6, :160, :164 (read this run). VOCABULARY.md:9-14 and :154-158. tests/experiments/papers/plan.py:12-13 (the reps step builds each dataset's files) and :77-92 (PAPER2_ONLY holds nt_*, gb_human_enhancers_cohn, buchwald_hartwig, bh_doyle_plates, humanness_therapeutic and the two Adams sets). Line 5 of the runbook says "views" for the screening sweep.
- Your choice: In this runbook, should the feature sets built for both papers be called views, representations, or something else?

### `SEARCH_SETTINGS.md` (5)

**Line 74. Two places disagree.**

- The document says: "The budget is not a setting chosen inside the search." Lines 76-77 say to report one row per budget and never keep a winning budget. Lines 13-20 of the same file list budgets as one of the six settings.
- What the code, thesis or your messages say: thesis.tex:1484 names the budget as one of the six settings searched, and VOCABULARY.md:24 follows it. thesis.tex:2076 says "Each fold picks its own budget". The code comment at src/kirby/find_best_hybrid.py:11502-11504 agrees with the file: "which is why it is not an axis inside the search".
- Checked in: SEARCH_SETTINGS.md:13-20, :74-77; thesis.tex:1484, :2076; src/kirby/find_best_hybrid.py:11502-11504; tests/experiments/decided_grid.py AXES includes 'budgets'.
- Your choice: Does each fold's search pick one budget, as thesis.tex:2076 says, or is every budget reported separately, as this file says?

**Line 88. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Adaptive pruning is "Off by default since 2026-09-13, owner's decision".
- What the code, thesis or your messages say: No quote from her is given. The code comment it cites says the same with no quote. The transcripts available here start 2026-10-14, so her message of 2026-09-13 is not checked.
- Checked in: SEARCH_SETTINGS.md:88-89; src/kirby/find_best_hybrid.py:8489-8492.
- Your choice: Do you confirm that adaptive pruning is off unless a caller passes it, and with which sentence of yours?

**Line 94. No agreed word.**

- The document says: The file names the two parts of adaptive pruning "Setting votes" (line 94) and "Mid-flight drops" (line 118), in bold as headings.
- What the code, thesis or your messages say: Neither name is in VOCABULARY.md. thesis.tex:1609 describes both without naming them: each setting "is then voted on", and at budget 100 the worse half of allocation-and-importance pairs is "not searched at larger budgets".
- Checked in: SEARCH_SETTINGS.md:94, :118; VOCABULARY.md (no entry); thesis.tex:1609.
- Your choice: Do these two parts of adaptive pruning get names from you, or are they written out each time?

**Line 108. Mentions patient data.**

- The document says: Each pruning rule was calibrated on the f33 record, five datasets with 25 model-by-budget conditions each. Line 113 says that after one fold alone no top_n and threshold pair is safe on any setting.
- What the code, thesis or your messages say: Mentions patient data. The code comments on the calibration show one of the five datasets is a patient dataset. The 'not safe after one fold' damage recorded for the importance-method and internal-evaluator votes is on that dataset only.
- Checked in: src/kirby/find_best_hybrid.py:3888-3893 (five datasets, fold-0 damage), :3919-3921 and :3945-3946 (fold-0 damage on the importance and evaluator votes); SEARCH_SETTINGS.md:108, :113.
- Your choice: Should the count drop to the four other datasets and line 113 be restated on them, or should this calibration text come out of the file?

**Line 132. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: "Per-fold searches run unpruned by default (Adelaide, 2026-10-05, handoff 2.1)."
- What the code, thesis or your messages say: No quote from her is given. The code comment at tests/experiments/_shared.py:1436-1438 carries the same attribution with no quote. thesis.tex:1605 says the per-fold searches in that chapter ran without pruning. Her 2026-10-05 message is before the transcripts available here, so it is not checked.
- Checked in: SEARCH_SETTINGS.md:132; tests/experiments/_shared.py:1436-1438; thesis.tex:1605.
- Your choice: Is there a sentence of yours to quote for per-fold searches running unpruned by default, or should the attribution to you come off?

### `HANDOFF_PROTEINGYM.md` (9)

**Line 22. Two places disagree.**

- The document says: Line 22 says 'Random is not to be used (Adelaide's standing rule)'. Line 23 says modulo and contiguous are both hers to use. Line 284 says 'No random splits.'
- What the code, thesis or your messages say: Her pasted brief of 2026-10-22 says "Do not use random." She then chose modulo: "sure do modulo, as long is that is an industry standard solution" (2026-10-23). On 2026-10-29 she wrote "Random splits are fine if that's how they did it in the source code. The random split ban rule applies only to molecular data btw". ProteinGym ships `fold_random_5` (line 21). Whether protein variants count as molecular data, she has not said.
- Checked in: Transcript e812739e, 2026-10-22T23:58:06 and 2026-10-23T03:56:04; transcript 35aa60ea, 2026-10-29T14:36:29. HANDOFF_PROTEINGYM.md:21-23, 284.
- Your choice: Does the random split ban cover ProteinGym, given her 2026-10-29 sentence that it applies only to molecular data? Time: not measured.

**Line 31. No agreed word.**

- The document says: The handoff says 'representation' throughout, for example lines 28, 31, 230 and 248.
- What the code, thesis or your messages say: VOCABULARY.md:13-14 says Paper 2 uses 'view' and Paper 3 uses 'representation'. Neither list names proteins. PAPER_2_OUTLINE.md:95-96 says ProteinGym is named for Paper 2 but not decided. Her 2026-10-28 message places proteins in the cross-domain paper: "I want to add the existing things in NEW_DATASETS ... as well as proteins."
- Checked in: VOCABULARY.md:9-16. PAPER_2_OUTLINE.md:95-96, 146. Transcript e812739e, 2026-10-28T15:30:50.
- Your choice: Is ProteinGym in Paper 2 (the handoff says 'view') or Paper 3 (it keeps 'representation')?

**Line 43. Two places disagree.**

- The document says: Lines 43-45 say one-hot is degenerate under position-disjoint folds. They call that 'a stated reason to drop it from those folds, and the only one on record'. The table at line 34 gives one-hot as undefined.
- What the code, thesis or your messages say: ProteinGym publishes a One-Hot score for this assay under modulo, -0.021 (line 42). When the earlier chat told her 'One-hot cannot be scored on these folds', she replied (2026-10-26): "the whole point of this was to replicate what these other papers are doing". No message from her drops one-hot.
- Checked in: HANDOFF_PROTEINGYM.md:34, 38-45. Transcript e812739e, her message 2026-10-26T14:51:02, answering the assistant text just before it. Their one-hot config: /Volumes/seagate/ProteinNPT/proteinnpt/baselines/model_configs/OHE_not_augmented.json:5,7.
- Your choice: Is one-hot left out on modulo folds, or reproduced the way their linear one-hot baseline scores it? Time: not measured.

**Line 62. Two places disagree.**

- The document says: Line 62 gives their ESM-1v baseline as layer 33 mean-pooled over residues. Line 188 says their head is one nn.Linear per target. Lines 206-207 say a plain embedding differs from their column 'in two ways at once, not one'. Line 84 calls the plus_zs scores the comparable ones.
- What the code, thesis or your messages say: All three of their embedding configs set `target_prediction_model: CNN`. The kernel is 7 for ESM-1v and Tranception and 9 for MSA Transformer. In their code, layer norm, dropout and a convolution over residues run before the mean, then a linear head. The local scores at lines 78-81 fit LightGBM or SVM to mean-pooled features. So the local run differs from their column in three ways: no convolution, a different downstream model, and how the zero-shot score enters. scratchpad/precompute_cuts/DECISIONS.md:945 already records this score as 'not a reproduction: a different model and pooling'. Her words (2026-10-26): "I don't give a flying fuck what they got if I can't reproduce it."
- Checked in: /Volumes/seagate/ProteinNPT (ba50f15): proteinnpt/baselines/model_configs/Embeddings_ESM1v_final.json:10-11, Embeddings_MSAT_final.json:15-16, Embeddings_Tranception_final.json:10-11; proteinnpt/baselines/model.py:61-63, 80-85, 165-191. HANDOFF_PROTEINGYM.md:62-64, 78-84, 188, 206-207. scratchpad/precompute_cuts/DECISIONS.md:945. Transcript e812739e, 2026-10-26T14:53:43.
- Your choice: Do the three embedding baselines get their own head (convolution over residues, linear head, zero-shot score added outside the network), or her downstream models on mean-pooled embeddings? Time for either: not measured.

**Line 160. Other.**

- The document says: Line 160 says 'Adelaide picks this one' between two ways to compute MSA Transformer embeddings: 64 variants per batch (`fast_MSA_mode`), or one variant per batch on the GPU server. Line 156 says it is not checked which of the two their published column used.
- What the code, thesis or your messages say: At ProteinNPT commit ba50f15, their embedding script runs `batch_size=1` and never passes `--fast_MSA_mode`. Only scripts/embeddings.py defines that flag, and it is off by default. Line 149-151 says 64 variants in one batch see each other, so the numbers differ from one variant per batch. Measured costs: one variant per batch on the laptop, 625 s per variant, about 377 hours for 2,172 variants. `fast_MSA_mode`: 34 forward passes, time not measured. GPU server: not measured.
- Checked in: /Volumes/seagate/ProteinNPT/scripts/embeddings_subs.sh:5, 13-25; scripts/embeddings.py:39, 112; tests/experiments/benchmarks/b06_proteingym_msat_embeddings.py:55; HANDOFF_PROTEINGYM.md:142-160.
- Your choice: Which way MSA Transformer embeddings are computed: one variant per batch on the laptop (about 377 hours), one variant per batch on the GPU server (not measured), or `fast_MSA_mode` (not measured).

**Line 176. Two places disagree.**

- The document says: Line 176 says 'Adelaide picks the layer' for Tranception, between hidden state index 0 and index -1.
- What the code, thesis or your messages say: The script already takes one. tests/experiments/benchmarks/b07_proteingym_tranception_embeddings.py:19 says it follows embedding_utils.py and takes hidden_states[-1], 1,280 features, and line 173 does so. Their model.py takes hidden_states[0], the input embedding layer before any transformer block. Their embedding_utils.py:300 takes hidden_states[-1], the last layer. No message from her picks one.
- Checked in: tests/experiments/benchmarks/b07_proteingym_tranception_embeddings.py:15-19, 173. /Volumes/seagate/ProteinNPT/proteinnpt/baselines/model.py (Tranception branch, `output.hidden_states[0]`). HANDOFF_PROTEINGYM.md:166-176.
- Your choice: Which Tranception hidden state the embeddings use: index 0 (their model.py) or index -1 (their embedding_utils.py, and b07 as written). Time: not measured.

**Line 227. The document is wrong, and the fix needs your choice.**

- The document says: Lines 227-228 say three ways to make a search cheaper are in SEARCH_SETTINGS.md and Adelaide picks which one this search uses. Lines 225-227 say the lost example script's grid was one entry per setting (not checked). Lines 237-238 say no ProteinGym assay has a DECISIONS.md entry, so `search_on_fold` needs `fixed_grid=`.
- What the code, thesis or your messages say: The three ways are her list (2026-10-26): "Options: 1. Adaptive pruning 2. Fast tier 3. Hand-coded smaller grid". On the earlier run she wrote: "the grid is 3 budgets and one each of importance method, allocation method and internal evaluator. - so there's no search grid". SEARCH_SETTINGS.md:132 says per-fold searches run unpruned by default (Adelaide, 2026-10-05), with no quote. SEARCH_SETTINGS.md:136-137 quotes 'Every dataset searches the full space, with adaptive pruning on' for the Papers 2 and 3 run. Her actual words were "full with adaptive pruning if that wasn't clear - that should be in the handoff" (2026-11-02). PAPER_2_OUTLINE.md:95-96 says ProteinGym is not decided for Paper 2. No budgets or search grid values for ProteinGym are recorded.
- Checked in: Transcript e812739e, 2026-10-26T10:00:25 and 10:04:37; transcript 75b352ed, 2026-11-02T05:36:05 and 05:36:25. SEARCH_SETTINGS.md:132-137. PAPER_2_OUTLINE.md:95-96. tests/experiments/decided_grid.py:14, 43; tests/experiments/_shared.py:1350-1355. HANDOFF_PROTEINGYM.md:225-238.
- Your choice: Which of her three ways (adaptive pruning, fast tier, hand-coded smaller grid) the ProteinGym per-fold search uses, and with which budgets and search grid. Time per option: not measured.

**Line 274. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Lines 273-276 say the hybrid result 'does not test the sentence from the thesis about antibody representations carrying the same information, which is what this whole exercise was for.'
- What the code, thesis or your messages say: thesis.tex:2192 says the antibody representations 'contain similar information', not the same information. 'Carrying the same information' is the wording of her pasted brief of 2026-10-22. Her later sentence of the purpose is different (2026-10-28): "I am trying to write a paper that shows find_best_hybrid is a valuable method across domains". She also wrote: "I do want ot use a few standard ones using published github protocols that match the pbulished results. THen I'll add find_best_hybrid methodology to that and compare."
- Checked in: thesis.tex:2192. Transcript e812739e, 2026-10-22T23:58:06 and 2026-10-28T15:30:50. HANDOFF_PROTEINGYM.md:273-276.
- Your choice: Is the ProteinGym work for the thesis sentence at thesis.tex:2192, for the cross-domain paper, or both?

**Line 287. Two places disagree.**

- The document says: Line 287 says 'No significance testing. Per-fold scores and spread only.' It gives no quote.
- What the code, thesis or your messages say: On 2026-11-01, the day before this file was committed, she wrote: "the ban was for expecting to see significance across 5 folds, if this is done correctly with 1000 I'm fine adding it back in". Whether a 1,000-resample test is added is not decided anywhere.
- Checked in: Transcript 0ab3513e, her message 2026-11-01T17:23:16. HANDOFF_PROTEINGYM.md:287.
- Your choice: Does the ProteinGym comparison carry a 1,000-resample test, or per-fold scores and spread only?

### `VOCABULARY.md` (13)

**Line 15. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: The method keeps the name adaptive hybrid representation in both papers. The line credits this to Adelaide, 2026-11-17, with no quote.
- What the code, thesis or your messages say: Her message of 2026-11-17T17:40:18 says "2. adaptive hybrid represenation is good". No sentence of hers is on the line.
- Checked in: VOCABULARY.md:15; transcript af128977, user message 2026-11-17T17:40:18
- Your choice: Is "adaptive hybrid represenation is good" the sentence that records this name for both papers?

**Line 47. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: The kNN search searches kNN and nothing else, her decision of 2026-11-04. The line gives no quote, only a pointer to AC_GRID_TEST_PLAN.md. It says the search was built by that plan's first prompt.
- What the code, thesis or your messages say: Her message of 2026-11-04T23:55:51 says "this should literally just search over knn and nothing else". In the working tree, decided_grid.py:173-178 sets both the downstream model and the internal evaluator to kNN for this search. That change is not committed. VOCABULARY.md:155 still says the kNN-only search is not built.
- Checked in: VOCABULARY.md:47-48, :155; transcript 37fa5e85, user message 2026-11-04T23:55:51; tests/experiments/decided_grid.py:158, :163, :166-178 (git diff, uncommitted)
- Your choice: Is "this should literally just search over knn and nothing else" the sentence this line records?

**Line 55. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Top base is the best-ranked base representation for a model, not the same thing as a base. The source column says "her words, AC_GRID_TEST_PLAN.md" with no quote. Line 54 defines a base representation as one representation scored on its own, as a baseline.
- What the code, thesis or your messages say: Her message of 2026-11-17T17:53:53 says "Base = an existing representation. ... top base = top scoring base represntation." AC_GRID_TEST_PLAN.md:174 uses "the top base plus ecfp4".
- Checked in: VOCABULARY.md:54-55; transcript af128977, user message 2026-11-17T17:53:53; AC_GRID_TEST_PLAN.md:174
- Your choice: Should lines 54-55 use your 17:53 definitions, "Base = an existing representation" and "top base = top scoring base represntation"?

**Line 66. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Noise condition is the word, and "noise kind", "damage kind" and "noise type" are banned. All of it is credited to Adelaide, 2026-11-17, with no quote.
- What the code, thesis or your messages say: Her message of 2026-11-17T17:40:18 says only "1. noise condition". The three banned words are not in her message.
- Checked in: VOCABULARY.md:66; transcript af128977, user message 2026-11-17T17:40:18
- Your choice: Does "1. noise condition" also ban "noise kind", "damage kind" and "noise type", or only choose the word?

**Line 80. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: A flagging method hands back a yes-or-no list of molecules. It feeds only cleanups that act on a list, and is not graded like a detector. All of it is credited to Adelaide, 2026-11-17.
- What the code, thesis or your messages say: Her words were "I don't think they're being used as detectors. If not, call them flagging methods" (17:53:53). The two-part description came from Claude's reply at 17:54:39, not from her.
- Checked in: VOCABULARY.md:80; transcript af128977, user 2026-11-17T17:53:53, assistant 2026-11-17T17:54:39
- Your choice: Do you accept the description written in Claude's reply as the definition of flagging methods?

**Line 81. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Random deletion is the shuffled detector scores control with a deleting cleanup, not a separate one. This is credited to Adelaide, 2026-11-17, with no quote.
- What the code, thesis or your messages say: Her messages at 2026-11-17T18:05 say "so the random deletion is just the shuffled scores with the cleanup method being deletion?" and "That's just shuffled with deletion." The "build it" quote on the same line is about building the control (15:24:00).
- Checked in: VOCABULARY.md:81; transcript af128977, user messages 2026-11-17T18:05:05, 18:05:49, 2026-11-18T15:24:00
- Your choice: Is "That's just shuffled with deletion." the sentence this line records?

**Line 82. The document is wrong, and the fix needs your choice.**

- The document says: Shuffled cleanup results shuffles the cleanup's result across molecules: which molecules were deleted, and each molecule's new weight or label.
- What the code, thesis or your messages say: For a label cleanup, the code shuffles each molecule's label change, y' - y, and adds it to another molecule's own label. It does not move new labels between molecules. Deletion and weights are shuffled as written. The definition she approved at 18:06 was Claude's wording, which said "new weight or label".
- Checked in: tests/experiments/noise/controls.py:175-200 (lines 197-200 for labels); transcript af128977, assistant 2026-11-17T18:06:26, user 18:06:52
- Your choice: For label cleanups, should this control shuffle the new labels, as approved, or the label changes, as the code does?

**Line 82. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: The name shuffled cleanup results is credited to Adelaide, 2026-11-17, with no quote.
- What the code, thesis or your messages say: She wrote "shuffled cleanups scores" (18:06:24). Claude proposed "shuffled cleanup results" (18:06:26, 18:06:46). She replied "fantastic, record that" (18:06:52).
- Checked in: VOCABULARY.md:82; transcript af128977, user 2026-11-17T18:06:24, 18:06:52; assistant 18:06:26, 18:06:46
- Your choice: Is "fantastic, record that" the sentence that approves the name shuffled cleanup results?

**Line 85. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Random put-back is the name of the control for Cliff put-back. It is credited to Adelaide, 2026-11-18, with no quote.
- What the code, thesis or your messages say: Claude asked whether to use the thesis header "Random molecules put back" (15:15:21). Her reply was "as long as you're consistent with put-back and put back (idk which is better, I'm not a writer) that's fine". Claude then chose "Random put-back" (15:19:52).
- Checked in: VOCABULARY.md:85; thesis.tex:4157 (header, cited by the file, not opened this run); transcript af128977, assistant 2026-11-18T15:15:21, 15:19:52, user 15:19:36
- Your choice: Is the name Random put-back, or the thesis header wording Random molecules put back?

**Line 103. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Cliff share is the share of molecules that are cliffs, never "target share". It is credited to Adelaide, 2026-11-17, with no quote.
- What the code, thesis or your messages say: Her message of 2026-11-17T17:40:18 says "4. cliff share not target share".
- Checked in: VOCABULARY.md:103; transcript af128977, user message 2026-11-17T17:40:18
- Your choice: Is "4. cliff share not target share" the sentence this line records?

**Line 130. No agreed word.**

- The document says: The trimmed refit has no word yet. Line 130 equates it with `rounds_delete` in the detector-and-cleanup experiment. Whether it is Multistage cleaning is open (lines 131-132, 173).
- What the code, thesis or your messages say: The code treats these as three settings. The trimmed refit keeps 0.8, 0.9 or 0.95 of the molecules for 4 rounds. `rounds_delete` deletes a detector's flagged share 3 times. Multistage cleaning deletes the 10% with the largest in-sample error, 3 times. `rounds_delete` is cut, "author's decision 2026-10-31".
- Checked in: VOCABULARY.md:130-132, :173; tests/experiments/noise/n44_detect_and_clean.py:83-92, :318-321; thesis.tex:3730
- Your choice: Are the trimmed refit and `rounds_delete` Multistage cleaning, or one method, and if neither, what are they called?

**Line 166. No agreed word.**

- The document says: The random twin is removed from the detector and cleanup experiment, quoting "remove it". The line gives no word for it anywhere else.
- What the code, thesis or your messages say: Her "remove it" (2026-11-18T15:38:17) answered a question about that one experiment. The shared control writer still writes this control by default (n07_remediation.py:2370, :2429), and n42 passes the default (n42:455). n23, n32 and n38 have their own versions. Claude told her at 15:38:57 that she had not said whether these keep it.
- Checked in: VOCABULARY.md:166; transcript af128977, user 2026-11-18T15:38:17, assistant 15:38:57; tests/experiments/noise/n07_remediation.py:2370, :2429; n42_similarity_noise_sweep.py:455; n23_uncertainty_cleaning.py:594; n32_iterative_and_dual_selection.py:611; n38_noise_relative_to_model_error.py:342
- Your choice: Do the other noise experiments keep this control, and if they do, what is it called?

**Line 184. Two places disagree.**

- The document says: Line 184 replaces "stage" in prose with "step". Line 119 says "step", never "stage". Line 123 bans "stage" only for runs. None of the three lines quotes her.
- What the code, thesis or your messages say: Her ban on "stage" is recorded only in a memory file, quoting a 2026-10-05 message about runs. That message is not in the kept transcripts, which start 2026-10-22. No message of hers chooses "step". She wrote "stage" herself on 2026-11-09 ("If it makes it to this stage") and 2026-11-17 ("from the stage of lit review").
- Checked in: VOCABULARY.md:119, :123, :184; memory/feedback_banned_words_round_stage.md; user messages 2026-11-09T15:46:06, 2026-11-17T17:21:01
- Your choice: Is "stage" banned only for runs, or in all prose, and is "step" the word that replaces it?

### `thesis_provenance/CLIFF_DEFINITIONS_2026-11.md` (6)

**Line 164. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: 'Decided 2026-10-06: two measurements of the same molecule are never a cliff pair.' No one is named and no words are quoted.
- What the code, thesis or your messages say: The n25 docstring says 'DECIDED 2026-10-06 (Adelaide delegated)', with no quote from her. The stored chat transcripts start 2026-10-14, so her message of that date was not checked.
- Checked in: n25_cliff_exemption.py:621-627; ~/.claude/projects/-Users-apunt-repos-KIRBy/*.jsonl earliest start 2026-10-14
- Your choice: Whether 'a molecule's own repeat measurements are never a cliff pair' is her decision.

**Line 176. Two places disagree.**

- The document says: The heading and line 178 date this section 2026-11-02. They give 84 files, 67 of them Python, in groups of eleven, six and fifty. The six list now names n35, n42, n44, s12, `_smoke_n25.py` and s04. Line 193 says n47 called `n25.cliff_mask_for_dataset` until 2026-11-16. Lines 58-69 are dated 2026-11-01. They say every cliff-aware noise experiment uses ECFP4 Tanimoto 0.55 with a gap of 2.0 median absolute deviations, n44 among them.
- What the code, thesis or your messages say: The six list was edited after 2026-11-02. n35 and s12 now sit in both the eleven and the six, so the three groups hold 65 distinct files, not 67. n47 is in no group. The committed six list already had s12 in two groups. The eleventh file, the `chembl_qsar` loader, is in src/kirby/datasets/, outside the grep. Committed n47 (HEAD 7d07b139) still calls `n25.cliff_mask_for_dataset` at :208. Only the uncommitted n47 drops the call, and the date 2026-11-16 comes from its own comments. Run today, the grep returns 108 files besides one patient-data file, 91 of them Python. Since 2026-11-09, n44's variant cleanups run reads every definition in tests/experiments/cliffs/definitions.py. It computes the n25 definition only when named. n14, in the noise folder, uses an absolute 2.0-log-unit gap.
- Checked in: CLIFF_DEFINITIONS_2026-11.md:58-69, :176-200; git show HEAD:thesis_provenance/CLIFF_DEFINITIONS_2026-11.md lines 190-191; git show HEAD:tests/experiments/noise/n47_noise_measurements.py :208; n47_noise_measurements.py:17-20, :188 (working tree); n44_detect_and_clean.py:187-195, :1137-1147; n14_activity_cliff_landscape.py:89; src/kirby/datasets/chembl_qsar.py; grep -rl cliff tests/experiments (109 files, 92 .py, one with a patient-data name)
- Your choice: Keep lines 58-77 and 176-200 as records of 2026-11-01 and 2026-11-02, with the six list put back to that date, or recount both against today's 91 Python files (cost not measured).

**Line 207. The document is wrong, and the fix needs your choice.**

- The document says: Lines 208-210: on any dataset without ECFP4, the similarity cutoff is tuned to a 10 per cent cliff share, then the gap is solved to 15 per cent. Line 213 records the fix as her decision. Lines 216-217: s04 and s12 numbers stored before 2026-11-02 used the 10-then-15 pair.
- What the code, thesis or your messages say: No code runs the two one after the other. The gap is solved only inside `calibrated_cliff_rule`. Without ECFP4, its similarity side is fixed at the top 1.0 per cent of pairs. The 10 per cent sweep runs only when a list of percentiles is passed. No caller passes a list. So s04 and s12 never used a 10-then-15 pair. Her words, 'the quotas must move together', answered an assistant message saying 'On any dataset without ECFP4 both run in sequence.'
- Checked in: similarity_helpers.py:877-878 (sweep only for a list), :1110-1116 (CLIFF_SIMILARITY_PERCENTILE = 1.0), :1208 (only call of solve_activity_gap); callers pass one value: n01_activity_cliff_framework.py:491, n25_cliff_exemption.py:718, :829, n35_near_duplicate_disagreement.py:868, smoke_tests_broken_fixes.py:84; session aa8f8e5d, 2026-11-02T09:51:25Z (pasted text) and 09:51:46Z (her reply)
- Your choice: Whether the 2026-11-02 change, both cliff shares reading one constant, stands as her decision, given the two shares were never applied one after the other.

**Line 213. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: 'Fixed 2026-11-02 (Adelaide's decision: the quotas move together).' The next sentence fixes an order: the similarity cutoff is picked first under the standard gap, then the gap is solved for the share left over.
- What the code, thesis or your messages say: Her words were 'the quotas must move together'. The order was written by the assistant 13 seconds later, in its own reply. No message from her mentions the order. The code comment records both under '2026-11-02 (Adelaide)'.
- Checked in: session aa8f8e5d-9341-4de1-88aa-f13779e0870f.jsonl: user 2026-11-02T09:51:46Z 'the quotas must move together'; assistant 09:51:59Z 'the order has to be fixed and recorded'; similarity_helpers.py:868-874
- Your choice: Whether the fixed order, similarity cutoff first and then the gap, is hers, or is taken out as written without her.

**Line 223. No agreed word.**

- The document says: 'the concatenation rule' (lines 186, 225) and 'the sequence rule' (lines 234, 244). The heading at line 223 calls both 'similarity criteria'. The document uses 'similarity criterion' elsewhere too, for example lines 641 and 708.
- What the code, thesis or your messages say: VOCABULARY.md:177 maps 'cliff rule (a cliff definition)' to 'definition of an AC'. VOCABULARY.md has no word for the similarity side on its own. Its nearest entry is 'cutoff', her word for the similarity limit. The concatenation item sets only a similarity side: the top 1 per cent of pairs on all representations joined. The sequence item sets a Hamming cutoff and a gap.
- Checked in: VOCABULARY.md:102, :177; CLIFF_DEFINITIONS_2026-11.md:223-237, :244
- Your choice: What to call the similarity side alone, such as the top 1 per cent of pairs on all representations joined: 'similarity criterion' or another word.

**Line 951. A decision is credited to you, with no quote or with words that are not yours.**

- The document says: Lines 951-952 cite 'the handoff's rule that the hybrid is searched on training molecules only and cliffs counted on held-out pairs.' Lines 947-958 also cite the handoff's worry and its testable idea.
- What the code, thesis or your messages say: That sentence was written by an assistant into HANDOFF_PAPER3_CLIFFS.md on 2026-11-01, with no quote from her. No committed version of the handoff holds it. The handoff was deleted on 2026-11-17. In the code both halves are her open settings, `definition_scope` and `held_out_cliff_rule`. CLIFF_SETTINGS.json now holds ['per_fold'] and ['same_as_training']; who chose those was not checked.
- Checked in: transcript 75b352ed-3f96-45e7-bcf2-f043e19556fb.jsonl, assistant Write 2026-11-01T19:36:07Z; git log -S'training rows only' -- HANDOFF_PAPER3_CLIFFS.md (no hits); commit 1e73396d; tests/experiments/cliffs/settings.py:31-38, :217-228; tests/experiments/cliffs/CLIFF_SETTINGS.json
- Your choice: Whether 'hybrid searched on training molecules only, cliffs counted on held-out pairs' is her rule, or stays as two open settings.

## Paper 1 (report only, nothing edited)

**1. Which files are Paper 1**

No local record shows that Paper 1 was submitted, so I cannot name a submitted file.
- `submission/SUBMISSION_PLAN.md:1-12` is the newest status note, written on 2026-11-06. It says "You have not yet read the merged copy or approved it."
- Steps 23 to 27 of that plan, which build the package and submit, are still open.
- `JCHEMINF_STATUS.md` and `JCHEMINF_SUBMISSION_NOTES.md` were last changed on 2 and 1 June. They describe `paper.tex` as it was then.
- `KIRBy/README.md:6` does say "submitted to the Journal of Cheminformatics". It was last changed in commit 71a49716 on 2026-11-02.

These are the most recent versions, with the evidence:
- **Main text: `/Users/apunt/repos/qsar_qm_models/paper_from_thesis.tex`** (850 lines, changed 22 Oct 15:50). `SUBMISSION_PLAN.md:9` says "The thesis wording was merged into a separate file, `paper_from_thesis.tex`, on 2026-10-22". Commit b862c5c (2026-10-22) is titled "Submission batch: … in the merged copy".
- `paper.tex` was last changed on 8 Oct and does not hold the thesis wording (`SUBMISSION_PLAN.md:10`). Step 10, which replaces it with the merged copy, has not been done. `paper.tex` still has 9 uses of "noise type" or "corrupt-".
- **Additional files: `/Users/apunt/repos/qsar_qm_models/additional_files/additional_files.tex`**. It was staged on 22 Oct by `scripts/stage_additional_files.py` (`additional_files/MANIFEST.md:3`).
- Its text matches the root `additional_files.tex` (8 Oct). The only difference is the `\graphicspath` line (line 26).

**2. Old terms from the VOCABULARY.md replacement table used in the table's sense**

Lines are in `paper_from_thesis.tex` unless the line says `additional_files.tex`.

Row "corrupted → damaged" (a label carrying injected noise):
- 197: "which labels are corrupted?"
- 356: "ranks the corrupted labels"
- 443: "the labels get increasingly corrupted with noise"
- 631: "its training labels are corrupted with noise"
- 714: "which labels had been corrupted"
- 822: "ranks the corrupted labels"
- `additional_files.tex:1924`: "how well the corrupted labels are ranked"

Lines 164 ("corrupted predictions the most") and 704 ("corrupt QSAR model predictions the most") use the word about predictions, not labels. They fall outside the row's sense.

Row "noise kind, kind, damage kind, noise type → noise condition":
- 164 (abstract): "Relative to all other noise types".

Row "noise draw, draw, draws → repetition":
- 351: "the same noise draw for every model and representation". Thesis line 545 has the same sentence.

Row "training rows, held-out rows → molecules":
- 218: "NGBoost holds out its own scaffold-grouped fifth of the rows it is handed". Thesis line 452 has the same sentence.

Rejected words in the rejected sense:
- `additional_files.tex:285`: "cell" means one model on one representation: "a two-way comparison across model and representation cannot be read if each cell carries its own hyperparameters". `thesis_appendix/noise/items/hyperparameters_tuned.tex` has the same sentence.
- 204: "10 independent draws from one pool". Here "pool" means all QM9 molecules. Thesis line 436 has the same sentence. Whether this counts depends on the rejected list, which says "pool" means nothing except global pool.

These matches were checked and are outside the table's sense:
- 256: "one draw $b$ per family", a random value in an equation.
- 204 and 420: "draws of $N=10{,}000$", which are samples of molecules.
- 327 and 329: "clean-label $R^2$". The table's row is "clean-label check".
- 732: "boosting stage".
- 688: "a band spanning the folds".
- 394: "endpoint columns", which are data-file columns, not features.
- Rows, columns and cells of tables and figures (428, 506, 602, 686, and `additional_files.tex:1151-1844`).
- "cell lines" (193, 285).
- "neural network(s)". Paper 1's NNs are called "NN" in thesis Chapter 3 as well, so the row "neural net → DNN" does not apply.

None of these appear in either file: dose, piece, recipe, axis, precompute, neighbour, oracle, twin, put-back, spares, yardstick, scorer, knob, chain, read-across, SAC, decoy.

**3. Places where Paper 1's word differs from thesis.tex**

I compared each Paper 1 sentence with thesis lines 55-967 (Chapters 2 and 3). Where the two share a sentence, the wording is the same apart from these:
- 481 uses "SNS" where thesis 650 uses "Sort \& Slice".
- 187 says repeat measurements of "the same compound-target pair" where thesis 742 says "the same compound".

Sentences that appear only in the paper use these words. The thesis word for the same thing is beside each:
- 164 "noise types" and 172 "types of label noise". The thesis says "noise conditions" (209, 267).
- 164 "ADMET datasets". The thesis says "assay datasets" throughout.
- 164 "quantile regression forests". The thesis says QRF (Quantile Regression Forests at 450). Line 216 of the paper agrees with the thesis.
- 777 "five targeting patterns". Thesis 336 says "a targeting rule".
- 777 "label-flip strategies". Thesis 363 says "label-flipping strategies".

Two of the terms above are words the VOCABULARY.md table later replaced, and thesis Chapter 3 uses them too:
- "corrupted": thesis 396, 400 and 860 use it. "damaged" is the Chapter 7 word (VOCABULARY cites 3606).
- "noise draw": thesis 342 and 545 use it.

**VOCABULARY.md cites the wrong place for "repetition".** It gives thesis.tex 342 as the source. Line 342 reads "the same at every noise level while the noise draws change". The word "repetition" does not occur anywhere in thesis.tex (count 0).

I also compared the Additional files with the thesis appendix copies in `thesis_appendix/noise/items/`. The only word difference is "standardisation" (`additional_files.tex:408`) against "standardization" in the appendix.

**4. Paper 1 content used in the Paper 3 documents**

`PAPER_3_OUTLINE.md` does not cite Paper 1 or thesis Chapter 3. `PAPER_3_NOISE_EXPERIMENTS.md` uses Paper 1 content in five places.
- **Line 62:** "Noise shape changed little in Chapter 3". This matches paper 479 and 704, and thesis 941.
- **Line 68:** "Hardest condition in Chapter 3" for grouped-shifted. Paper 704 and thesis 941 name grouped-shifted and censoring together: "Grouped-shifted and censored noise … corrupt QSAR model predictions the most." Neither names grouped-shifted alone.
- **Line 64:** `outlier_p05` is "5% of molecules get error 3 times wider". Paper 1's outlier condition widens the error on a random 10% of molecules (paper 243, table row 283). Both are in the code: `src/kirby/noise_spec.py:98` defines `outlier_p01`, `outlier_p05` and `outlier_p10`. So the outlier used for Paper 3 is not the one Paper 1 reported.
- **Line 57:** the "Assay source" column is "Claude's reading of thesis Table `tab:noise_sources`" (thesis 200).
- **Lines 32 and 173:** the noise levels (0, 0.5, 1, 1.5) and censoring levels (0, 0.1, 0.25, 0.4) are all among Paper 1's levels (paper 261).

`PAPER_3_OUTLINE.md` uses one idea from Paper 1 without citing it. Lines 60 and 122-128 say QM9's labels are computed, so QM9 has no lab error. Paper 1's abstract (line 164) says the same: "whose labels are computed and thus clean".