# Places in the thesis still waiting for something (2026-10-10)

Built from the PDF of `thesis.tex` as it stood on 2026-10-10 (265 pages). "Page" is the PDF page and "line" is the line in `thesis.tex`.

## 1. Printed placeholders (5)

These print in bold in the PDF.

| Page | Line | What it waits for |
|---|---|---|
| 203 | 3284 | `[RESULT NEEDED: ginkgo_pr_cho real censoring, not yet run]` |
| 203 | 3284 | `[RESULT NEEDED: QM9 arm with injected clipping and the not-at-floor control, approved but not built]` |
| 221 | 3636 | `[RESULT NEEDED: ginkgo_pr_cho, the same moves on the labels at 0 ...]` |
| 221 | 3636 | `[RESULT NEEDED: QM9 with injected clipping and the not-at-limit control]` |
| 233 | 3903 | `[RESULT NEEDED: whether the same corrections help on CHO polyreactivity's labels at the floor ...]` |

All five are the same two runs: real censoring on CHO polyreactivity (n30b), and the QM9 injected-clipping run. n30b exists in the code but hasn't been run. The QM9 run is approved but not built.

## 2. Paragraphs whose numbers were moved to a table that doesn't exist yet (57)

On 2026-10-10 the numbers were cut from these paragraphs, to sit in a table, but the tables were never built. So each paragraph now gives a pattern ("most datasets", "little") with no table to back it. The note in `thesis.tex` at that line says what one row of the table is.

| Page | Line | Section | Table it waits for |
|---|---|---|---|
| 160 | 2495 | Against the best single representation | one row per small-molecule dataset and model; columns: best single representation in hindsight (c01) and its held-out score, hybrid held-out score (c02), effective number of representations and top representation's share of the shuffling drop (c22). |
| 160 | 2508 | Against the best single representation | one row per antibody dataset and model; columns: best single representation in hindsight (c01) and its held-out score, hybrid held-out score (c02), share of the shuffling drop carried by the representations built for this thesis (c22), number of representations the hybrid draws on. |
| 162 | 2528 | Against ensembles, and many representations against few | one row per dataset and model; columns: hybrid, same-model ensemble and best-model ensemble held-out scores, folds where the hybrid is ahead (c11). |
| 163 | 2532 | Against ensembles, and many representations against few | one row per dataset, model and budget; columns: search's hybrid, top-ranked representation alone, even split over the same representations (held-out score). |
| 163 | 2537 | Against ensembles, and many representations against few | one row per dataset, model and scaffold half (frequent, rare); columns: number of molecules, best single representation and hybrid held-out score (c07). |
| 164 | 2550 | Overfitting of the search | one row per dataset and model; columns: the search's own score of its pick, the held-out score of the same pick, their difference, folds whose pick was wider than the training set (c33). |
| 164 | 2558 | Overfitting of the search | one row per dataset and model; columns: held-out score of each fold's own search and of the search with one fold fewer, mean difference (c24). |
| 164 | 2567 | Selection at a fixed number of features | one row per dataset, model and budget; columns: hybrid, recursive feature elimination, random features, mRMR (held-out score, c17). |
| 165 | 2573 | Dependence on the downstream model | one row per dataset and model; columns: budget picked on each of the five folds and number of representations drawn on (c02 hybrid labels; Figure fig:best_number_of_features now shows the same per-fold picks as dots under each panel). |
| 165 | 2578 | Dependence on the downstream model | one row per dataset; columns: share of score variance from the downstream model and from the budget; SVM budget picked per fold at each C (c06). |
| 165 | 2602 | Dependence on the downstream model | one row per dataset and model; columns: held-out score and number of features with the matched, near-family and far-family evaluator (c23). |
| 168 | 2609 | Using importance during training | one row per dataset and model; columns: held-out score unweighted, feature-weighted, sample-weighted, both, above-median features only (c19). |
| 168 | 2619 | Using importance during training | one row per dataset and model; columns: coverage and mean width of 90% conformal intervals for the single representation and the hybrid (c08). |
| 169 | 2650 | Blueprints on specific datasets | one row per dataset and model; columns: stability index of the selected features across folds, smallest and largest budget picked. |
| 169 | 2657 | Collaboration studies | one row per target and model; columns: MAE of the search's top single representation and of the hybrid, folds where the hybrid is lower (OPIG qsar). |
| 172 | 2665 | Collaboration studies | one row per representation set added to the graph kernel on hERG Ki and model; columns: RMSE alone and with the added features, molecules still tied. |
| 172 | 2670 | Collaboration studies | one row per model on hERG Ki; columns: best single representation, search on hERG Ki, LogD settings with features reselected, LogD settings and features reused. |
| 172 | 2677 | Features chosen across separate sites | one row per dataset and model; columns: best single representation, best family, one selection over all families, selection per family (RMSE, fed03). |
| 172 | 2686 | Features chosen across separate sites | one row per model and flip level on Caco-2; columns: MCC of pooled selection, per-site selection, the collaborator's two ensembles and student model (fed01). |
| 173 | 2691 | Features chosen across separate sites | one row per model on Caco-2; columns: MCC of global selection, per-cluster features with one model, one model per cluster; rank agreement between clusters and between random parts (fed04). |
| 185 | 2891 | Distance against label difference | one row per dataset; features: dataset, best single representation and distance picked on held-out molecules, mean Spearman (distance against absolute activity difference), lowest and highest fold |
| 185 | 2895 | Distance against label difference | one row per dataset; columns: number of representation and distance combinations, number at or below zero, best single value (Spearman) |
| 185 | 2909 | Distance against label difference | one row per dataset and model; columns: model's best single representation and its first predictive hybrid, Spearman of distance against absolute activity difference |
| 186 | 2916 | Distance against label difference | one row per dataset; columns: single representation and distance hybrid, each unweighted and weighted, Spearman of distance against absolute activity difference (like tab:importance_weighting) |
| 186 | 2920 | Distance against label difference | one row per dataset; columns: hindsight best single representation and distance, distance hybrid, similarity hybrid, Spearman of distance against absolute activity difference |
| 186 | 2926 | Distance against label difference | one row per dataset and number of features; columns: network hybrid RMSE, Spearman of distance against absolute activity difference in the same hybrid |
| 186 | 2938 | Prediction from nearest neighbors | one row per dataset; columns: best single representation and distance hybrid with the setting picked again for each fold, in each dataset's own metric (same rows as tab:nearest_neighbour_prediction); also serves the next paragraph |
| 187 | 2950 | Prediction from nearest neighbors | one row per dataset; columns: all representations joined, unweighted and weighted by each importance method (LightGBM gain, DeepLIFT, ...), 10 neighbors, in each dataset's own metric |
| 187 | 2987 | Prediction from nearest neighbors | one row per dataset; columns: hybrid gain over the single representation by distance (cosine, Manhattan, ...), by number of neighbors, unweighted and weighted, lowest and highest fold, in each dataset's own metric |
| 189 | 2991 | Prediction from nearest neighbors | one row per dataset; columns: difference in neighbors' label gap (single minus hybrid) by distance and by number of neighbors, lowest and highest fold at the fixed setting |
| 189 | 3005 | Activity cliffs | one row per dataset and model; columns: best single representation and first hybrid, on cliff molecules and on the other held-out molecules, in each dataset's own metric |
| 189 | 3009 | Activity cliffs | QM9 rows of the cliff table above (one row per model; cliff and other molecules, single and hybrid, RMSE) |
| 190 | 3015 | Prediction error, retrieval and clustering | one row per dataset and model; columns: single representation and distance hybrid, Spearman of distance to nearest training molecules against the model's error on each held-out molecule |
| 190 | 3019 | Prediction error, retrieval and clustering | rows of the distance-against-error table above for the antibody datasets and Buchwald--Hartwig |
| 190 | 3023 | Prediction error, retrieval and clustering | one row per dataset; columns: neighbors' label gap in single representation and distance hybrid (setting picked again), cluster coherence of each at 3, 5 and 10 k-means clusters |
| 190 | 3027 | Prediction error, retrieval and clustering | rows of the retrieval and clustering table above for the antibody datasets and Buchwald--Hartwig, with a column for principal components of all representations |
| 190 | 3031 | Prediction error, retrieval and clustering | one row per dataset; columns: median label information summed over single features (nats) for kept features, principal components, random subset of the same number |
| 209 | 3419 | Changing a label against deleting a molecule | one row per action (none, delete, move toward neighbors' mean, down-weight, best random deletion) and fold; columns AUROC of xgboost on the best single representation and on the hybrid, hERG confusion-directed flips at 10%, folds 0-2 |
| 210 | 3437 | Changing a label against deleting a molecule | one row per action (none, delete, down-weight, label smoothing, shrink, best random deletion); columns network RMSE on the first single representation, first hybrid, second hybrid; LogD 5% outliers, folds 0 and 1 |
| 211 | 3443 | Changing a label against deleting a molecule | one row per detector (five) plus no cleaning; columns AUROC on single representation and hybrid for hERG flips folds 0 and 1, RMSE of the network hybrid on LogD outliers |
| 211 | 3454 | Changing a label against deleting a molecule | one row per weighting method and dataset (hERG flips, LogD outliers, Buchwald-Hartwig Gaussian, QM9 no noise); columns untreated and weighted score on single representation and hybrid, model named |
| 215 | 3502 | The labels as delivered | one row per dataset, detector, cleanup and configuration; columns no cleaning, cleaned, shuffled seeds 0, 1, 2; labels as delivered, fold 0 |
| 215 | 3508 | The labels as delivered | one row per fold (0-2); columns network RMSE untreated and after label smoothing, hybrid and second single representation; hERG K_i as delivered |
| 216 | 3529 | Detection of the suspect labels | one row per dataset and damage kind; columns average precision of the forest residual, the cliff-aware gap (best and median setting), chance level; fold 0 |
| 217 | 3539 | Detection of the suspect labels | one row per dataset and damage kind; columns best average precision in each of the four distance families, chance level; fold 0 |
| 217 | 3551 | Detection of the suspect labels | one row per dataset and damage kind; columns average precision of the GP, cliff-aware gap, residual, nearest-neighbor Shapley, cleanlab; fold 0 |
| 218 | 3562 | Detection of the suspect labels | one row per dataset, damage kind, flagged share and action; columns detector precision, break-even precision; fold 0 |
| 218 | 3611 | Labels clipped at an assay limit | one row per dataset and fold; columns network RMSE on the hybrid before clipping, clipped, after censored EM; LogD folds 0-3, Caco-2 folds 1-2 |
| 218 | 3616 | Labels clipped at an assay limit | one row per dataset, model and configuration; columns score before clipping, clipped, constant lift, truncated-normal mean, censored EM; level 0.25, fold 0 (fig:noise:censoring_ladder has clean/clipped/censored EM/deleted only) |
| 221 | 3625 | Labels clipped at an assay limit | one row per dataset; columns network RMSE on the hybrid with each molecule's own lift and with shuffled lifts seeds 0, 1, 2; level 0.25, fold 0 |
| 221 | 3642 | A shared offset per scaffold group | one row per dataset and draw; columns network RMSE on the hybrid damaged, after family mean, after random intercept, clean labels; grouped-shifted level 1.0, fold 0 |
| 222 | 3677 | The antibody datasets | one row per antibody dataset x model x representation (single or hybrid) x cleanup; columns: Spearman with no cleaning, after cleaning, and with each of the three shuffled-score controls; fold 0 (n18 numbers of N7-30 and N7-31). |
| 223 | 3683 | The antibody datasets | one row per antibody dataset x model x representation (single or hybrid) x cleanup; columns: Spearman with no cleaning, after cleaning, and with each of the three shuffled-score controls; fold 0 (n18 numbers of N7-30 and N7-31). |
| 223 | 3707 | Repeat measurements on hERG IC\texorpdfstring{$_{50 | one row per model x method on hERG IC50 (count weights, inverse-variance weights, log-count feature, shrink by spread, interval model, one row per measurement); columns: RMSE before and after on each of three draws, fold 0, held-out molecules with 3 or more measurements. |
| 224 | 3716 | Cleaning inside the feature search | one row per dataset x noise condition x model x representation x cleanup; columns: score with no cleaning, cleaning the fit only, cleaning both the search and the fit, cleaning the search only; fold 0 (n20 numbers of N7-35 and N7-36). |
| 224 | 3727 | Cleaning inside the feature search | one row per dataset x model x outer fold; columns: RMSE (Spearman for SMAC) of the second hybrid searched on clean labels and searched on labels with Gaussian noise at level 1.0, both fitted on the damaged labels, mean over 3 draws (n15). |
| 225 | 3735 | Changing the model instead of the labels | one row per dataset x damage level x model x representation; columns: RMSE of the plain model, seed averaging, stacking, retuning on the damaged labels, settings tuned on clean labels reused, and the plain model on clean labels; fold 0 (n40 numbers of N7-38 to N7-40). |

## 3. Sentences that say a result is missing (not placeholders, but they change when runs land)

| Page | What the text says |
|---|---|
| 162 | "We have not yet run the experiment that tests whether the best settings follow ..." (cross experiment c03) |
| 186 | "The weighted comparison is missing on Caco-2, hERG blocking, hERG Ki, expression titer and SMAC ..." (s02 re-run) |
| 152, 162, 163, 164, 165, 168, 173, 231 | "... datasets with a result": true now, and these change when the missing Chapter 5 results land |

## 4. Checked and clean

- The build log has no undefined references or citations, and the PDF has no "??" or "[?]".
- No code names print in the running text except `ginkgo_pr_cho` inside the placeholders above. The blueprint listing on page 123 prints field names on purpose.
- Figures were not checked: text inside an image can't be searched.
