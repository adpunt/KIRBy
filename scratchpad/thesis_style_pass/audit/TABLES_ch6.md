# Chapter 6 tables built for the cut numbers (2026-10-10)

Chapter 6 (`chap:similarity`) had 17 `% TODO: table needed` comments. All 17 are resolved by 13 new tables. None is left in the chapter.

Every number comes from the result files in `/Volumes/seagate/kirby_results_2026-10-10/similarity/` (and `complexity/c04_importance_as_metric/`), read only. The 9 Oct files that also exist in the 10 Oct folder are byte-identical. The script that reads them is `scratchpad/thesis_style_pass/audit/ch6_tables_gen.py`. Each table has a `% SOURCE` comment naming the file and the key path, and a `% TODO: manual review` line.

Scores have three decimals. Two values compared in one row that would print the same keep a fourth decimal. A nonzero value that would print as 0.000 also keeps a fourth decimal. Buchwald--Hartwig RMSE and label gap are percentage points of yield, so they have one decimal. The label-information table has two decimals, as the text gives them (1.63, 1.68). An empty cell means the result file has no value for it. Every empty cell is explained in the table's SOURCE comment.

Each table has at most four bold cells. The bold cells mark the values the paragraph describes.

## Tables

| Label | Rows x number columns | Comments it replaces | Source |
|---|---|---|---|
| `tab:similarity:single_spread` (landscape) | 14 datasets x 5, plus the name of the best representation and distance | best single pair with folds; combinations at or below zero | s01 `by_rep[rep][distance].sac_rho_mean`, `.sac_rho_per_fold` |
| `tab:similarity:predictive_hybrids` | 13 datasets x 2 models, 3 columns | model's best single against its predictive hybrids | s02 `comparisons["<model> hybrid 1/2 - <model> base rank 1"]`, Manhattan, unweighted |
| `tab:similarity:feature_spaces` | 13 datasets x 6 | single and distance hybrid, unweighted and weighted; hindsight pick against both hybrids (with `single_spread`) | s02 `comparisons["knn distance hybrid ..."]` and `["similarity hybrid ..."]`, Manhattan |
| `tab:similarity:feature_count` (landscape) | 10 datasets (network only) x 12 | network hybrid score and Spearman by number of features | s08 `prediction_by_budget.dnn[b].rank1.mean` and `folds[*].prediction.dnn[b].hybrids[0].neighbour...sac_unweighted` |
| `tab:similarity:knn_picked_again` | 14 datasets x 2 | prediction from neighbors with the setting picked again (serves two paragraphs) | s05 `fold_reselection_result.knn` |
| `tab:similarity:joined_weighting` | 13 regression datasets x 4 | all representations joined, unweighted and by each importance method, 10 neighbors | c04 `methods[m].knn["10"].{unweighted,weighted}_dataset_metric` |
| `tab:similarity:knn_by_setting` (landscape, two pages) | 14 datasets x 2 feature spaces, 13 columns | gain by distance, by number of neighbors, unweighted and weighted, per fold | s05 `cells[...]`, single_knn1 and distance_hybrid |
| `tab:similarity:retrieval_by_setting` (landscape, two pages) | 14 datasets x 2 feature spaces, 11 columns | label gap by distance, by number of neighbors, per fold at the fixed setting | s13 `retrieval.{rep_knn1,hybrid_knn_distance1}.features_unweighted` |
| `tab:similarity:cliffs` | 5 datasets x 2 models, 4 columns | cliff table; QM9 rows of it | s12 `summary.models[m].ecfp4_tanimoto` |
| `tab:similarity:error_distance` | 14 datasets x 2 models, 2 columns | distance against error; its antibody and Buchwald--Hartwig rows | s06 `distance_error_correlation.manhattan.k5["<model>_all_representations"]` |
| `tab:similarity:retrieval` | 14 datasets x 3 | label gap with the setting picked again; principal-component column | s13 `fold_reselection_retrieval_unweighted_result`, `..._vs_pca_result` |
| `tab:similarity:clustering` | 14 datasets x 6 | k-means coherence at 3, 5 and 10 clusters | s13 `clustering_delta_by_cluster_count.clustering` |
| `tab:similarity:label_information` | 14 datasets x 3 | kept features, principal components, random subset (nats) | s03 `per_rep[rep].summed_component_mi_{selected,pca,random}.held_out`, median |

The 14 datasets are the 13 in the text plus MDR1-MDCK efflux. The two figure-script tables already in the chapter list MDR1-MDCK efflux too. MDR1-MDCK efflux has no row where its result file is incomplete: s02 and s08 exist only as a partial `.jsonl`. It also has no row in the cliff table, because the text reports cliffs on five datasets.

Two tables do not fit one page. The tables by setting are 42 rows each, so they are landscape longtables over two pages. The table by setting for prediction uses 3pt column gaps. The font size is unchanged in every table. `tab:similarity:feature_spaces` is 11pt wider than the text block. That is about 4 mm into the right margin.

## Paragraphs re-pointed

All of these paragraphs are marked `% TODO: manual review` (AI text). Each got its pointer next to its claim. No number was added to any paragraph.

- Weak distance in a standard representation: `single_spread`.
- Representations at or below zero: `single_spread`.
- Predictive hybrids that track less closely: `predictive_hybrids`.
- Weighting closes the gap: `feature_spaces`. It used to point to `tab:distance_tracks_activity`, which has no weighted values.
- Hindsight pick: `single_spread` and `feature_spaces`.
- Number of features: `feature_count`.
- Fixed setting, fold sentence: `knn_by_setting`. This paragraph had no comment; the fold claim had no table behind it.
- Setting picked again, small molecules: `knn_picked_again`.
- Setting picked again, antibodies: `knn_picked_again`.
- All representations joined: `joined_weighting`.
- Dataset matters more than setting: `knn_by_setting`.
- Retrieval shows the same pattern: `retrieval_by_setting`.
- Cliffs, four datasets: `cliffs`.
- Cliffs, QM9: `cliffs`.
- Distance against error, small molecules: `error_distance`.
- Distance against error, antibodies: `error_distance`.
- Retrieval and clustering, small molecules: `retrieval` and `clustering`.
- Retrieval and clustering, antibodies: `retrieval` and `clustering`.
- Label information: `label_information`.

## Claims the tables do not back

These are claims the tables contradict. In most of them the text names fewer datasets than the tables show. The prose was left as it is for the results chat.

1. "reaches at most 0.279 Spearman, on QM9". MDR1-MDCK efflux reaches 0.281 (`single_spread`).
2. "On the small-molecule datasets and Buchwald--Hartwig, no representation has a correlation at or below zero." MDR1-MDCK efflux has 5 of its 30 pairs at or below zero, the lowest at -0.052.
3. "A hybrid's prediction score peaks at more features than the correlation ... does." This is stated in general. For the network it holds on QM9, Caco-2, LogD, nanobody thermostability and CHO polyreactivity. On hERG K_i both peak at 250 features, and on expression titer both peak at 1,000. On EGFR binding the score peaks at 250 and the Spearman at 500. On SMAC colloidal stability they peak at 500 and 1,000. On binding free energy they peak at 100 and 500.
4. "the hybrid still wins on all five small-molecule datasets" (setting picked again). The table has six. The hybrid also wins on MDR1-MDCK efflux, 0.522 to 0.480 RMSE.
5. "weighting by LightGBM gain improves all four regression small-molecule datasets". MDR1-MDCK efflux is a fifth, and LightGBM gain improves it too. TreeSHAP also loses on SMAC colloidal stability, 0.134 to 0.122, and the text names only LightGBM gain.
6. "Cosine distance gives the hybrid its smallest lead" on the antibody datasets. On EGFR binding the hybrid's weakest distance is Manhattan, where it loses by 0.0009, not cosine.
7. "The features a hybrid keeps carry no more label information than principal components." On QM9 the kept features have 2.68 nats against 1.93 for principal components. On Buchwald--Hartwig they have 7.24 against 5.98.
8. "The distance hybrid retrieves neighbors with closer labels" (setting picked again). In s13 the hybrid side picks among every hybrid of the dataset, not the distance hybrid alone. The caption says so; the sentence does not.

These claims have no table behind them:

9. "a few hundred training antibodies, against one to eight thousand training molecules". These are training-set sizes. They are in the SOURCE comment only (152 to 538 against 1,129 to 7,983).
10. "On LogD and QM9 it picks the same allocation and importance method in every fold, while CHO polyreactivity wins every fold with four different choices." These are search choices, not scores. No table was built.
11. "about half of its combinations ... at or below zero" (SMAC colloidal stability). The table backs it with the median over pairs, -0.0004. The count, 15 of 30, is in the SOURCE comment only.

## Other things found

- The gains in the "dataset matters more" paragraph (LogD 0.078 to 0.023) average the distance and similarity hybrids (`scratchpad/thesis_ch5_8/ch6_conditional_knn.py`). The table shows the distance hybrid only, as scores, not gains.
- The prose has "0.136 to 0.0295" for SMAC colloidal stability with the network. The raw value rounds to 0.030 at three decimals.
- The `TODO(missing result)` comments for QM9 are out of date for two experiments. `s08_optimal_dimension/qm9.json` exists in the 10 Oct folder (written 9 Oct 14:51), and `feature_count` uses it. `s11_bayesian_optimization/qm9.json` also exists (10 Oct 00:34); it was not read. `s10_importance_downstream/qm9.json` is still missing.
- The existing `tab:importance_weighting` has 6 bold cells, over the limit of 4. It was not changed.
- The figure-script table `tab:distance_tracks_activity` has an MDR1-MDCK efflux row. That row is read from `s02_hybrid_similarity/adme_mdr1.partial.jsonl`, because no finished `.json` exists.
- Build: `main.tex` compiled twice with no errors. None of the 13 new tables triggers "Float too large". The two such warnings in the log are at input lines 1944 and 2032, outside Chapter 6.

## Check

Independent check, 2026-10-10. Script: `/private/tmp/claude-501/-Users-apunt-repos-KIRBy/9f687a4a-c71f-4415-b6e9-5c2f5f842ef6/scratchpad/ch6check/check.py`. It re-reads every result file and compares each printed cell with the raw value rounded to the decimals printed.

1. No "table needed" comment is left in Chapter 6.
2. Every `\ref` in the chapter resolves. No `tab:similarity:*` label is defined twice.
3. Numbers. Every cell in all 13 tables was checked. Every cell traces to its named result file. One cell was fixed: hERG blocking, LightGBM, in `predictive_hybrids`. Both of its values printed as 0.0244 but differ: the single representation is 0.02444 and the first hybrid 0.02440. They now print with five decimals.
4. Claims the tables contradicted. All were fixed in the prose. All were AI paragraphs.
   - Best single pair: "at most 0.279, on QM9" is now "at most 0.281, on MDR1-MDCK efflux". The bold moved from QM9 to MDR1-MDCK efflux.
   - Pairs at or below zero: MDR1-MDCK efflux was added to the datasets that have one. "0.0155" is now "0.016", as the table prints it.
   - Predictive hybrids: hERG blocking was added to the datasets where a hybrid tracks less closely (see point 3).
   - Number of features: the general claim held on 5 of the 10 datasets. It now names those five, the two where both peak at the same number of features, and the three where the prediction score peaks at fewer. "0.2315 to 0.1535" is now "0.231 to 0.154", as the table prints it.
   - Setting picked again: "all five" is now "all six" small-molecule datasets.
   - Joined representations: "all four" is now "all five" regression small-molecule datasets. TreeSHAP on SMAC colloidal stability was added as a loss and bolded. The table now has 4 bold cells.
   - Dataset matters more: "LogD 0.078 to 0.023" averaged in the similarity hybrid. It is now 0.089 to 0.025, the distance hybrid alone, from the table's source. A pointer to `tab:datasets` was added after the training-set sizes.
   - Retrieval by setting: cosine gives the smallest lead on antibodies except EGFR binding (where Manhattan does) and Delta Delta G (where Minkowski does). The sentence now names these two exceptions.
   - Error against distance: "0.0295" is now "0.030".
   - Retrieval with the setting picked again: the text said "distance hybrid" and "all five". It now says "the hybrid" and "all six", because the hybrid side picks among every hybrid. MDR1-MDCK efflux was added to the datasets where clusters are more coherent at 3, 5 and 10.
   - Label information: the claim now reads "except on QM9 and Buchwald-Hartwig", in the results and in the conclusions. "summed over single features" is now "summed over components", matching the methods.
   - Cliff shares: the text said the share is lowest on hERG K_i. It is lowest on QM9 (7 to 13 of about 1,996 held-out molecules per fold). Fixed.
   - Conclusions: "five" is now "six" small-molecule datasets for distance, prediction from neighbors and retrieval.
   - Introduction: the sentence said MDR1-MDCK efflux appears only in the two figure-script tables. It now lists the tables where it has no row.
5. Claims with no table behind them.
   - One new table was built: `tab:similarity:svr_weighting`. It shows the support vector regressor on the single representation, unweighted and weighted, from s10. It backs "the regressor also loses" on hERG K_i and "barely moves" on LogD.
   - "0.7115" is now "0.711". The raw value is 0.71145.
   - The Gaussian process was not put in a table. Its fixed and median length-scale starts give different results: hERG K_i weighted is 0.711 with one start and 0.605 with the other. Several antibody folds are nan. Choosing between the starts needs the author.
   - Some claims were verified against the result files but have no table. Each now has a `% Check 2026-10-10` comment with the values: per-fold distance tracking on LogD and Caco-2; LogD cliff gains in every fold; the second cliff run (s04) agreeing in direction; the search choices on LogD, QM9 and CHO; training-set sizes; choosing molecules to measure (s11).
6. Table rules. All 14 new tables have one number per cell and at most 4 bold cells. None uses a smaller font or `adjustbox`. Captions give what one row is, the metric and its direction.
7. Author paragraphs: every paragraph in Chapter 6 has a `% TODO: manual review` marker. None is the author's.

### Left open

- The q-RASAR sentence has no table and no number. It says the direction on LogD depends on the setting: ten neighbors under cosine against twenty. The top-3 settings differ by fold, so a table needs a decision on which settings to show.
- The choosing-molecules sentence has no table or number in the text. Its values are in a check comment only.
- Older tables break the rules. These were not changed:
  - `tab:importance_weighting` has 6 bold cells and uses `\small` with `adjustbox`.
  - `tab:knn_settings` and `tab:similarity_experiments` use `\small`/`\scriptsize` with `adjustbox`.
  - The two figure-script tables use four decimals, `\small` and `adjustbox`.
- `tab:similarity:feature_spaces` is still about 4 mm wider than the text block.
- Text that still says "all 13" or "five small-molecule":
  - In Chapter 6: the methods paragraph and `tab:similarity_experiments` ("All 13"), and the cliff paragraph ("five small-molecule datasets").
  - In Chapter 7: the first paragraph of the introduction ("on the five small-molecule datasets").
- The conclusions say QM9 has no result for the best number of features, but `feature_count` has a QM9 row. The QM9 and other result-pending items belong to the results chat, so this was not edited.
