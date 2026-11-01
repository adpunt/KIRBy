# Handoff: the remaining complexity experiments for PAPER_2_OUTLINE.md

Written 2026-10-31 at the end of a chat that went through c19 to c23. Everything
below was read in that chat; re-read the code before repeating any of it.

## What the chat is for

Adelaide is going through every complexity experiment in
`tests/experiments/complexity/` and deciding what each contributes to
`PAPER_2_OUTLINE.md`. For each one she wants, briefly:

1. the question it set out to answer, in her words, not the file's;
2. whether it reached `thesis.tex` (grep the id and the quantity it measures);
3. what the results say, from the 10-06 server run at
   `/Volumes/seagate/kirby_results_2026-10-10b/complexity/<folder>/<dataset>.json`
   (skip every `tcga_*` file: patient data);
4. a one-sentence takeaway. If it needs more than a sentence it is not a big takeaway.

She then probes with short questions and decides. Only after she decides, add it to
the outline the way the others are added: a Methods bullet of a few sentences, a
Results bullet stating shape (table / figure / one sentence) and finding without
numbers, "Source cXX" at the end, and the id appended to "## Experiments to run".
Run-status lines do not belong in "## Needs work": everything re-runs except c15.
Nothing is committed; the working tree is hers.

## Done (in the outline): c01, c02, c03, c04, c05, c06, c07, c08, c09, c11, c15, c16, c17,
c18, c19, c21, c22, c23, c24, c33, opig_blueprint_transfer.

Decisions made 2026-10-31 that the outline already carries:
- c19 runs at every decided budget, not a fixed 100 (the 10-06 run mixed the two).
- c21 is a TODO in Methods and Results: it has to be refined and run. If adding the
  extra features works, that is a headline result on the datasets where it does.
  The 10-06 run tested extras on two Ginkgo sets only, the decoys were admitted and
  random extras matched chosen ones. Refining it is a separate chat.
- c22 was cut to two statistics: features per representation and the shuffling drop
  (grouped permutation importance). SHAP and the per-feature list are gone from
  `c22_explainability.py`; result identity changed; the file before the cut is in the
  old chat's scratchpad only. One sentence in main text, table in supplement.
- c16 (2026-10-31): every decided model now has a regularisation sweep (LightGBM and
  XGBoost depth, DNN dropout, GP length scale fixed at multiples of sqrt(features));
  the 10-06 run had SVM and RF only. Result identity renamed. The new sweep values
  were set by Claude and not yet run. One sentence beside c06 in Results.
- c24 (2026-10-31): "run all of it". Every ordered pair of folds now records five
  scores on the target fold (docstring lists them): the 3-fold fresh search as before,
  the 3-fold settings re-picked on the target's training rows, the 3-fold exact
  features, and fold k's own settings / exact features, the last two marked as having
  seen the target's rows. No new search; fits only. Result identity renamed. Results
  bullet is a TODO until it runs.
- c33 (2026-10-31): her question is "does the feature selection overfit", answered two
  ways: the search's own score of its pick against the held-out score, and the held-out
  score against the same number of random features. c33 now fits that random partner at
  every budget, fold and model (width = what the hybrid built), and stores per fold the
  search score, held-out score and random score at the fold's own pick
  (picks_per_fold). Result identity renamed; checked locally on herg_ki 10-31. Added to
  the outline 10-31 in her words: score on training data, score on held-out set, random
  score on held-out set; "overfitting", never "over-promised".
- c23: one sentence beside the internal evaluator in Methods, table in supplement.
  Note for the thesis, not yet acted on: `thesis.tex:2575` says matched is best with
  both models on EGFR, HIC retention, titer and colloidal stability; in the current
  files the wider-DNN setting beats matched for the DNN on HIC retention and titer.

## Left to go through, in order

| id | file | state 10-06 run | note |
|---|---|---|---|

Decided 2026-11-01: c31 and c32 dropped; c18 covers mixing against the best single
representation, and the allocation methods are justified from the per-fold searches
in the analysis script (HANDOFF_PAPER2_ANALYSIS.md). c43 deleted 2026-11-01; its
measurement of the CGR features a hybrid picks, scored alone, moved to
opig_cgr_comparison.

Dismissed by her, do not bring back unless she asks: c13 (skipped by her 10-06),
c14 (not used in the thesis, 10-09), c20 (never in the run list; "not important",
2026-10-31). c12 is TCGA only: patient data, leave it out of every count.

## How the earlier ones were read, so the pattern holds

- Result layout differs per file; open one json and print its keys first.
- Numbers in the dataset's own metric, mean over folds, with the fold spread beside
  any gap. No tallies, no significance tests.
- The id's docstring often states a "Conclusion" written before any run. Do not
  repeat it as a finding.
- `/Volumes/seagate/kirby_results_2026-10-10b/analysis/cXX.md` is a dump of stored
  keys, not an analysis; useful only to see which datasets have final files.

## Writing to her

Repo-root `CLAUDE.md` applies. In this chat the replies she accepted were: the
question in one or two sentences; thesis yes/no with line numbers; three or four
result bullets; one-sentence takeaway; where it fits. The replies she rejected
were ones that invented a statistic ("agree more often with each other"), used a
threshold she had not seen ("representations above a tenth of the drop"), or
explained a run-plan mechanism when she asked why something did not run. When she
says "plain English", give one sentence.

## Next: the opig experiments (from 2026-11-01)

Every complexity experiment has been through. The same pass now goes through
`tests/experiments/opig_*.py`, same format, same rules. Already in the outline:
opig_blueprint_transfer; opig_cgr_comparison (TODO, waits for the reaction screen,
`<run>/reactions/b10_views/summary.md`). Left, in file order:

opig_activity_cliff_qsar, opig_cdr_flexibility, opig_generator_scoring_swap,
opig_ginkgo_developability, opig_graphinity_ddg, opig_humanness,
opig_kingsmith_component_splits, opig_modnet_feature_scores,
opig_modnet_hybrid_features, opig_scaffold_hybrid, opig_scaffold_similarity,
opig_shared_vectors_census, opig_shared_vectors_remedy, opig_trastuzumab_enrichment.

10-06 results: `/Volumes/seagate/kirby_results_2026-10-10b/opig/` holds ginkgo,
graphinity, qsar (final + partial), trastuzumab (partial only), blueprint_transfer,
shared_vectors_census, shared_vectors_remedy, cgr_comparison (partial only). An
experiment with no file there did not run on 10-06; say so, and check the ledger
(`ledger.json` in that folder) for why. Some opig experiments may belong to another
paper (noise, similarity) rather than Paper 2; say which, and she decides.
