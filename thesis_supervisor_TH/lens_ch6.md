# Lens proposals, Chapter 6 (chap:similarity, thesis.tex lines 2661-3390)

Read in full: thesis.tex 2661-3390, plus figures_ch5_7/tab_distance_tracks_activity.tex and
figures_ch5_7/tab_nearest_neighbour_prediction.tex. Not repeated here: his own Ch6 comments
(question list, flow figure, overlap with Ch5, RF against SVM/LightGBM in the clipping tables,
bold and clipped landscape in old Table 6.10).

Numbers I checked myself this session (paths given in each item):
- s01 representation-distance pairs per dataset: /Volumes/seagate/kirby_results_2026-10-10b/similarity/s01_similarity_baseline/<dataset>.json, by_rep. 10 representations x 3 distances = 30 on most datasets; QM9, LogD, hERG blocking have 20 (no Minkowski); ddg_combined has 8 representations, 24 pairs.
- s11 areas: /Volumes/seagate/kirby_results_2026-10-10b/similarity/s11_bayesian_optimization/<dataset>.json, results.conditions[*].metrics.budget_100.auc_mean. Each dataset has 20 hybrid conditions and one best_single.
- Fold re-selection: tests/experiments/similarity/s05_knn_prediction.py lines 49-52 (docstring): "re-picks ... for each held-out fold from the other folds' held-out scores"; function tests/experiments/_shared.py:2273.

---------------------------------------------------------------------------------------------
## HIGH

### X-ch6-1
Location: line 2666, Introduction (sec. before sec:similarity:methods); Owner: hers
Quote: "The first idea behind the hybrid was to compute distances on the features relevant to the task alone, and this chapter tests that idea."
His comment: The thesis is about noisy labels. Please say in one sentence why similarity matters for noise here, at the start, not only in the last line of the chapter.
Lens item: 1 (also 12)
Proposed change:
old_string: `The first idea behind the hybrid was to compute distances on the features relevant to the task alone, and this chapter tests that idea.`
new_string: `The first idea behind the hybrid was to compute distances on the features relevant to the task alone, and this chapter tests that idea. This matters for label noise. Methods that find a noisy label by comparing it with its neighbors' labels can only be as good as the distance they use, and Chapter~\ref{chap:noise} builds on this chapter.`
Priority: high

### X-ch6-2
Location: line 2670, Introduction, research question 1; Owner: hers
Quote: "Do pairwise distance values between hybrids track their relative label differences better than distance computed using the base representations?"
His comment: Distance between hybrids, or between molecules in a hybrid? And "relative" label difference, while Methods uses the absolute difference.
Lens item: 19 (also 4)
Proposed change:
old_string: `Do pairwise distance values between hybrids track their relative label differences better than distance computed using the base representations?`
new_string: `Do distances between molecules in a hybrid track their label differences better than distances between the same molecules in the base representations?`
Priority: high

### X-ch6-3
Location: line 2719, sec:similarity:knn_settings; Owner: hers
Quote: "Unlike in the similarity score, a training molecule from the held-out molecule's own group (scaffold family, for small molecules) can be one of its neighbors."
His comment: The outer folds are grouped by scaffold (line 2678, and Ch5 line 1684). How can a training molecule share the held-out molecule's group? Is there a train/test overlap here?
Lens item: 8
Proposed change: NEEDS HER. The two statements disagree: the outer folds put a whole group on one side, so no training molecule should share a held-out molecule's group. Check in s05/similarity_helpers which group the neighbor search can see. If the folds are grouped as stated, the edit is:
old_string: `Unlike in the similarity score, a training molecule from the held-out molecule's own group (scaffold family, for small molecules) can be one of its neighbors.`
new_string: `Because the outer folds are grouped, no training molecule shares a held-out molecule's group (scaffold family, for small molecules), so no group needs to be kept out here.`
Priority: high

### X-ch6-4
Location: line 2784, sec:similarity:experiments, "Prediction from nearest neighbors"; Owner: AI
Quote: "The second picks a combination for each held-out fold from the scores on the other four folds."
His comment: The other four folds were trained with this fold's molecules as neighbors. Is the choice then fully separate from the test fold? Same question for the retrieval and clustering tables "picked again for each fold".
Lens item: 8
Proposed change (code checked: s05_knn_prediction.py docstring lines 49-52 picks from the other folds' held-out scores; in those folds, this fold's molecules are training molecules). Full paragraph:
old_string: `We predict each held-out molecule's label from its nearest training molecules, as in Equation~\ref{eq:knn_weights}, and compare hybrid and single representation in two ways. The first comparison, fixed before the run, uses Manhattan distance, 5 neighbors weighted by inverse rank, and no importance weighting. The second picks a combination for each held-out fold from the scores on the other four folds. The hybrid side chooses among the four predictive hybrids, the distance hybrid and the similarity hybrid, weighted or unweighted, at every setting of Table~\ref{tab:knn_settings}, up to 216 combinations in all. The single side chooses among the nearest-neighbor model's two best single representations, unweighted, giving 36. These counts fall to 144 and 24 without Minkowski distance, and to 180 and 36 without the similarity hybrid. The two sides therefore differ in two ways at once. The hybrid side has five or six times as many combinations, and only it can use importance weighting.`
new_string: `We predict each held-out molecule's label from its nearest training molecules, as in Equation~\ref{eq:knn_weights}, and compare hybrid and single representation in two ways. The first comparison was fixed before the run. It uses Manhattan distance, 5 neighbors weighted by inverse rank, and no importance weighting. The second comparison picks a combination for each held-out fold from the held-out scores of the other four folds. This fold's own scores are never used in the pick. In those four folds, however, this fold's molecules were training molecules, so their labels served as neighbor labels. The hybrid side chooses among the four predictive hybrids, the distance hybrid and the similarity hybrid, weighted or unweighted, at every setting of Table~\ref{tab:knn_settings}. This gives up to 216 combinations. The single side chooses among the nearest-neighbor model's two best single representations, unweighted, giving 36. These counts fall to 144 and 24 without Minkowski distance, and to 180 and 36 without the similarity hybrid. The two sides therefore differ in two ways at once. The hybrid side has five or six times as many combinations, and only it can use importance weighting. The comparison fixed before the run has neither difference.`
Priority: high

### X-ch6-5
Location: line 2702, sec:similarity:hybrid_similarity; Owner: hers
Quote: "A representation's weight in the distance is therefore roughly its number of features divided by the budget. Features that only take the values 0 and 1, such as fingerprint bits, add less per feature."
His comment: Does an ECFP bit count as much as a descriptor like logP in this distance? "Less" by how much? The reader needs to know if fingerprints are pushed aside.
Lens item: 10 (also 6)
Proposed change: NEEDS HER to approve the numbers. They are derived by me, not measured: for two molecules drawn at random, a 0/1 feature set in a fraction p of molecules differs by 2p(1-p) on average (at most 0.5); two draws of a standard normal feature differ by 2/sqrt(pi), about 1.13. Real descriptors are not normal, so the 1.13 is an approximation. Alternative: she measures the mean absolute difference per feature block on one dataset.
old_string: `Features that only take the values 0 and 1, such as fingerprint bits, add less per feature.`
new_string: `Features that only take the values 0 and 1, such as fingerprint bits, add less per feature. Between two molecules, a bit set in a fraction $p$ of molecules differs by $2p(1-p)$ on average, which is at most 0.5. A standardized feature with a normal distribution differs by about 1.13. A fingerprint therefore weighs less in the distance than a descriptor block with the same number of features.`
Priority: high

### X-ch6-6
Location: lines 2819 and 2822, sec:similarity:results_similarity (first two paragraphs); Owner: AI
Quote: "even the best of 30 pairs of representation and distance ... reaches at most 0.281 Spearman" and "about half of its pairs are at or below zero"
His comment: Where do the 30 pairs come from? And "half of its pairs": pairs of molecules or of representations? What does 0.28 mean for the reader: one sentence of take-home.
Lens item: 4 (also 2)
Proposed change: 30 is wrong on four datasets (checked: s01_similarity_baseline/<dataset>.json by_rep; QM9, LogD, hERG blocking 20 pairs, no Minkowski; ΔΔG 24 pairs, 8 representations). SMAC 15 of 30 is from the SOURCE comment at line 2827. Two full paragraphs:
old_string: `As in Chapter~\ref{chap:flaws}, distance in a standard representation rises only weakly with label difference, as seen in Table~\ref{tab:similarity:single_spread}. On the small-molecule datasets, even the best of 30 pairs of representation and distance, picked with hindsight on the held-out molecules, reaches at most 0.281 Spearman, on MDR1--MDCK efflux. This best pick also varies widely between folds, for example on hERG $K_i$.`
new_string: `As in Chapter~\ref{chap:flaws}, distance in a standard representation rises only weakly with label difference, as seen in Table~\ref{tab:similarity:single_spread}. For each dataset, we score every pair of one base representation and one distance of Table~\ref{tab:knn_settings}. Most datasets have 10 representations under three distances, giving 30 pairs. QM9, LogD and hERG blocking have 20, as Minkowski distance is left out above 5,000 molecules, and $\Delta\Delta G$ has 24, from 8 representations. On the small-molecule datasets, even the best pair, picked with hindsight on the held-out molecules, reaches at most 0.281 Spearman, on MDR1--MDCK efflux. This best pick also varies widely between folds, for example on hERG $K_i$. In any one representation, how far apart two molecules are says little about how far apart their labels are.`
old_string: `At least one representation is at or below zero on MDR1--MDCK efflux, EGFR binding, CHO polyreactivity, AC-SINS self-association and SMAC colloidal stability. Every representation is above zero on Buchwald--Hartwig and the other five small-molecule datasets. SMAC colloidal stability is the weakest, as its best single representation reaches 0.016 Spearman and about half of its pairs are at or below zero.`
new_string: `Some pairs of representation and distance give a correlation at or below zero, so that molecules further apart are no more different in their labels. This happens on MDR1--MDCK efflux, EGFR binding, CHO polyreactivity, AC-SINS self-association and SMAC colloidal stability. Every pair is above zero on Buchwald--Hartwig and the other five small-molecule datasets. SMAC colloidal stability is the weakest. Its best pair reaches 0.016 Spearman, and half of its 30 pairs of representation and distance are at or below zero.`
Priority: high

### X-ch6-7
Location: line 2866, sec:similarity:results_similarity; Owner: AI
Quote: "Distance in the distance hybrid tracks label differences more closely than in the single representation on all 14 datasets"
His comment: The hybrid is chosen with the labels; ECFP, Mordred and the pretrained embeddings are not. A smoother landscape is expected from a task-learned space. Please say this, and say which comparison is task-learned on both sides.
Lens item: 9
Proposed change: full paragraph. Facts: base representations are fixed or pretrained without the labels; the weighted single representation uses training-label importances (line 2740). If she wants a task-trained embedding as an extra baseline (for example a DNN's last hidden layer), that is NEEDS HER: a new experiment.
old_string: `Distance in the distance hybrid tracks label differences more closely than in the single representation on all 14 datasets, as seen in Table~\ref{tab:distance_tracks_activity}. On LogD the hybrid is higher in every fold, while on Caco-2 it is lower in two of the five.`
new_string: `Distance in the distance hybrid tracks label differences more closely than in the single representation on all 14 datasets, as seen in Table~\ref{tab:distance_tracks_activity}. On LogD the hybrid is higher in every fold, while on Caco-2 it is lower in two of the five. The two sides are not built the same way. The hybrid's features are chosen with the training labels, while every base representation is built without them. The weighted single representation of Table~\ref{tab:similarity:feature_spaces} also uses the training labels, through its feature importances, and is compared below.`
Priority: high

### X-ch6-8
Location: captions of the hand-written tables in sec:similarity:results_* (lines 2858, 2918, 2961, 3003, 3048, 3089, 3126, 3240, 3294, 3330, 3364); Owner: AI
Quote: e.g. Table tab:similarity:svr_weighting bolds 0.630 on hERG Ki, the worse (higher) RMSE; no caption says what bold means.
His comment: Bold usually means best. Here it seems to mark something else, and in one table it marks a worse value. Please say in each caption what bold is.
Lens item: 16
Proposed change: these tables are written by scratchpad/thesis_style_pass/audit/ch6_tables_gen.py (SOURCE comment line 2828), not by tests/analysis; change the script, or edit the .tex by hand if the script is retired. Rule to pick (NEEDS HER): either "Bold: a value quoted in the text." added to every caption, or bold removed. Orphan bold cells, not quoted anywhere in the current text: svr_weighting (hERG $K_i$ 0.630, LogD 0.507; the text quotes no SVM number), clustering (LogD 0.225, SMAC 0.461), retrieval (QM9 0.393), error_distance (SMAC 0.030), single_spread (SMAC median $-$0.0004), feature_count (both LogD cells), cliffs (QM9 0.280, Caco-2 0.288; 0.288 is quoted only in a SOURCE comment). Concrete edit for the worst case:
old_string: `hERG $K_i$ & 0.607 & \textbf{0.630} \\
LogD & 0.508 & \textbf{0.507} \\`
new_string: `hERG $K_i$ & 0.607 & 0.630 \\
LogD & 0.508 & 0.507 \\`
Priority: high

### X-ch6-9
Location: line 3189, sec:similarity:results_knn (choosing molecules to measure); Owner: AI
Quote: "On all three, the best-scoring of these simulations used a hybrid."
His comment: The reader will be frustrated not to know by how much. And how many hybrid simulations against how many single ones?
Lens item: 6 (also 8)
Proposed change: numbers checked in s11_bayesian_optimization/{herg_ki,qm9,buchwald_hartwig}.json, results.conditions[*].metrics.budget_100.auc_mean (also in the comment at lines 3187-3188, except Buchwald--Hartwig best_single 9946.8, read now). Each dataset has 20 hybrid conditions and one best_single condition. On QM9, every hybrid except hybrid_ngboost__lightgbm_h1 (253.4) is above best_single (260.1).
old_string: `We also simulated choosing which molecules to measure next, on hERG $K_i$, QM9 and Buchwald--Hartwig. On all three, the best-scoring of these simulations used a hybrid.`
new_string: `We also simulated choosing which molecules to measure next, on hERG $K_i$, QM9 and Buchwald--Hartwig. The score is the area under the curve of the best label found so far over 100 picks. On hERG $K_i$ and QM9 the search hunts the lowest label, so a smaller area is better. On hERG $K_i$, the best hybrid scored 410.6, the best single representation 418.1 and random picks 431.7. On QM9 the same three scored 253.4, 260.1 and 341.2. On Buchwald--Hartwig, where a larger area is better, they scored 9992.3, 9946.8 and 9901.5. Each dataset ran 20 simulations with hybrids and one with the best single representation. A hybrid being best is therefore partly a matter of more tries. On QM9, every other hybrid simulation scored worse than the best single representation.`
Priority: high

### X-ch6-10
Location: line 2796 (Methods, "Distance against prediction error") and line 3247 (results); Owner: AI
Quote: "\citet{Sheridan2004} found that a molecule's similarity to the training molecules tracks how well it is predicted." / "On hERG blocking the hybrid tracks error worse for both models."
His comment: This is the applicability domain, central for regulatory toxicity prediction; name it. And hERG blocking is a safety endpoint where the hybrid is much worse (0.139 to 0.019): why?
Lens item: 14 (also 19, 21)
Proposed change: Chapter 4 already names it (line 1142, "defining an applicability domain by distance to the training set \citep{Tropsha2010}"). Methods paragraph, full rewrite:
old_string: `\citet{Sheridan2004} found that a molecule's similarity to the training molecules tracks how well it is predicted. We test this by fitting each downstream model on all representations joined and on its first hybrid.`
new_string: `\citet{Sheridan2004} found that a molecule's similarity to the training molecules tracks how well it is predicted. Distance to the training molecules is also a common way to define an applicability domain \citep{Tropsha2010}, as in Chapter~\ref{chap:flaws}. We test this by fitting each downstream model on all representations joined and on its first hybrid.`
(the rest of the paragraph unchanged; it keeps its marker). For the hERG blocking drop: NEEDS HER explanation. One fact that can go in without a verdict: hERG blocking is the only classification dataset here, and its error is one minus the predicted probability of the true class (line 2796).
Priority: high

### X-ch6-11
Location: Tables tab:distance_tracks_activity, tab:nearest_neighbour_prediction, tab:similarity:knn_picked_again, tab:similarity:retrieval, tab:similarity:error_distance; Owner: AI (tables) / script
Quote: "mean over outer folds" with no spread; the text says "smaller than the spread across folds" (line 3173) without giving it.
His comment: Please show the variability next to the means, many differences here are in the third decimal.
Lens item: 16
Proposed change: NEEDS HER: choose the form (lowest and highest fold, as in tab:similarity:single_spread, fits the no-significance-test rule). The two \input tables are generated by tests/analysis/thesis_figures.py, --figure f5 and --figure f6; add the per-fold minimum and maximum columns there. The others come from scratchpad/thesis_style_pass/audit/ch6_tables_gen.py.
Priority: high

### X-ch6-12
Location: line 3057, sec:similarity:results_knn; Owner: AI
Quote: "Multiplying each feature by its importance mostly helps prediction from nearest neighbors ... In the other downstream models, weighting the single representation gains little or loses, as seen for the SVM"
His comment: Same question as before: it helps one model and not the other. Why? Is this discussed?
Lens item: 18
Proposed change: NEEDS HER: one sentence of explanation, or a sentence saying it is not explained. The SOURCE comment (lines 3063-3064) says the Gaussian process is not tabled because its length-scale starts disagree; if the SVM's RBF length scale was not re-tuned after weighting, that is a candidate reason, but I have not checked this in s10.
Priority: high

### X-ch6-13
Location: line 3194, sec:similarity:results_cliffs; Owner: AI
Quote: "Held-out molecules are flagged as cliffs by a looser rule than training molecules. Among them, the share of cliffs is highest on LogD and hERG blocking, nearly half in some folds"
His comment: The gap is set for 15% cliffs in training, but nearly half of the held-out molecules are cliffs? Are these still activity cliffs? The reader will wonder.
Lens item: 21 (also 6)
Proposed change: counts from the comment at lines 3195-3197. Full paragraph:
old_string: `We report activity cliffs on the six small-molecule datasets only. The antibody datasets have too few cliff antibodies to score, for example 11 of the 429 held-out antibodies on $\Delta\Delta G$. Held-out molecules are flagged as cliffs by a looser rule than training molecules. Among them, the share of cliffs is highest on LogD and hERG blocking, nearly half in some folds, and lowest on QM9.`
new_string: `We report activity cliffs on the six small-molecule datasets only. The antibody datasets have too few cliff antibodies to score, for example 11 of the 429 held-out antibodies on $\Delta\Delta G$. The gap is set so that 15\% of training molecules are cliff molecules. Held-out molecules are flagged by a looser rule, so their share of cliffs is higher. On LogD, 393 to 447 of the 1,006 held-out molecules per fold are cliffs, and on hERG blocking 574 to 769 of 1,446 to 1,660. QM9 has the fewest, 7 to 13 of about 1,996 per fold. A held-out cliff molecule is therefore one whose similar training molecules disagree in label, not always one with a single cliff partner.`
Priority: high

### X-ch6-14
Location: line 3389, sec:similarity:conclusions (last paragraph); Owner: AI
Quote: "Methods that find noisy labels by comparing each label with its neighbors' labels rest on the distance studied here. Chapter~\ref{chap:noise} tests whether the hybrid's distance makes noisy labels easier to find and to act on."
His comment: Good link to noise. Please also link to drug safety: two of the datasets are hERG endpoints.
Lens item: 14
Proposed change: facts from Tables tab:distance_tracks_activity (hERG $K_i$ 0.0729 to 0.2625, hERG blocking 0.0257 to 0.0552) and tab:similarity:error_distance (hERG blocking worse for both models). Full paragraph:
old_string: `Methods that find noisy labels by comparing each label with its neighbors' labels rest on the distance studied here. Chapter~\ref{chap:noise} tests whether the hybrid's distance makes noisy labels easier to find and to act on.`
new_string: `Two of the datasets are drug-safety endpoints, hERG $K_i$ and hERG blocking. On both, distance in the hybrid tracks label differences more closely. On hERG blocking, however, it tracks prediction error worse. This matters when distance to the training molecules is used to judge whether a toxicity prediction can be trusted. Methods that find noisy labels by comparing each label with its neighbors' labels also rest on this distance. Chapter~\ref{chap:noise} tests whether the hybrid's distance makes noisy labels easier to find and to act on.`
Priority: high

---------------------------------------------------------------------------------------------
## LOW

### X-ch6-15
Location: line 2666, Introduction; Owner: hers
Quote: "Any method that trusts a molecule's neighbors inherits this"
His comment: "negatively impacted by" rather than "inherits".
Lens item: 19
Proposed change:
old_string: `Any method that trusts a molecule's neighbors inherits this, such as`
new_string: `Any method that trusts a molecule's neighbors is negatively affected by this, such as`
Priority: low

### X-ch6-16
Location: line 2678, sec:similarity:methods; Owner: hers
Quote: "Of its 20 datasets, we leave out six: rat microsomal clearance, both anti-VEGF scans, angiopoietin-2 binding, antibody humanness and HIC retention."
His comment: Why these six? If they were dropped after seeing results, say so; otherwise say the reason.
Lens item: 8 (also 6)
Proposed change: NEEDS HER: one reason per dataset, or one shared reason, appended after this sentence.
Priority: low

### X-ch6-17
Location: line 2685, sec:similarity:spaces; Owner: hers
Quote: "The distance search, in which a k-nearest-neighbor model is used as the downstream model with Euclidean distance and five equally weighted neighbors."
His comment: Why Euclidean in this search, when the similarity score and all results use Manhattan? Is the hybrid then chosen for one distance metric and scored on another?
Lens item: 19 (also 21)
Proposed change: NEEDS HER: the reason. Minimal added sentence once known, e.g. after "...five equally weighted neighbors.": `Its hybrid is therefore chosen under Euclidean distance and scored in this chapter under the distances of Table~\ref{tab:knn_settings}.`
Priority: low

### X-ch6-18
Location: line 2691, sec:similarity:spaces; Owner: hers
Quote: "the top hybrid of the similarity search: the similarity hybrid, on the datasets where that search is defined."
His comment: On which datasets is it not defined, and why? Five antibody datasets have empty cells in Table 6.x.
Lens item: 4 (also 21)
Proposed change: NEEDS HER: name the datasets (from Table tab:similarity:feature_spaces: AC-SINS self-association, SMAC colloidal stability, CHO polyreactivity, expression titer, $\Delta\Delta G$) and the reason.
Priority: low

### X-ch6-19
Location: figures_ch5_7/tab_distance_tracks_activity.tex and tab_nearest_neighbour_prediction.tex (input at lines 2871, 3015); Owner: script (tests/analysis/thesis_figures.py fig_f5, fig_f6)
Quote: column "Hybrid built for distance", rows "Caco-2 efflux", "QM9 HOMO-LUMO gap", "hERG Ki", four decimals, no domain headings; Buchwald--Hartwig listed under RMSE with the small molecules.
His comment: These two tables repeat numbers of Tables tab:similarity:feature_spaces and tab:importance_weighting (same values, other rounding, other names and order). Please make tables of the same comparison consistent.
Lens item: 17 (also 20)
Proposed change: NEEDS HER: either drop the two \input tables and point to the unweighted columns of tab:similarity:feature_spaces and tab:importance_weighting, or change fig_f5/fig_f6 in tests/analysis/thesis_figures.py to the names, order, domain headings and three decimals of the other tables ("Single representation", "Distance hybrid"; Small molecules / Antibodies / Reactions). The f5 caption also says "difference in their labels" and "samples"; the other tables say "absolute label difference" and "molecules".
Priority: low

### X-ch6-20
Location: line 2970, sec:similarity:results_similarity; Owner: AI
Quote: "The DNN's hybrid often predicts best at a different number of features from the one where its distances best track label differences"
His comment: Many numbers; what is the take-home?
Lens item: 2
Proposed change: full paragraph, one sentence added at the end:
old_string: `On LogD, for example, the hybrid predicts better at 2,000 features than at 500, while its distance correlation falls from 0.231 to 0.154 Spearman.`
new_string: `On LogD, for example, the hybrid predicts better at 2,000 features than at 500, while its distance correlation falls from 0.231 to 0.154 Spearman. The best number of features for prediction is therefore often not the best for distance.`
Priority: low

### X-ch6-21
Location: line 3057, sec:similarity:results_knn; Owner: AI
Quote: "Giving each model four extra features taken from a molecule's nearest neighbors helps or hurts depending on the setting. On LogD, the SVM improves with ten neighbors under cosine distance and gets worse with twenty."
His comment: Improves by how much? The reader will be frustrated not to know.
Lens item: 6
Proposed change: NEEDS HER: the two LogD SVM RMSE pairs (s10_importance_downstream, q-RASAR task), then "from X to Y RMSE".
Priority: low

### X-ch6-22
Location: line 2966, sec:similarity:results_similarity; Owner: AI
Quote: "This pick beats the distance hybrid on Caco-2, EGFR binding, expression titer and CHO polyreactivity, as seen in Table~\ref{tab:similarity:feature_spaces}."
His comment: The hindsight pick is in the other table; restate one pair here rather than make me go back and forth.
Lens item: 5
Proposed change: values from Tables single_spread (Caco-2 0.229) and feature_spaces (0.197). Full paragraph:
old_string: `We also compare against a stronger single representation, namely the pair of representation and distance whose distance best follows the label, picked with hindsight as in Table~\ref{tab:similarity:single_spread}. This pick beats the distance hybrid on Caco-2, EGFR binding, expression titer and CHO polyreactivity, as seen in Table~\ref{tab:similarity:feature_spaces}. On Caco-2, the similarity hybrid beats it in turn.`
new_string: `We also compare against a stronger single representation, namely the pair of representation and distance whose distance best follows the label, picked with hindsight. This pick, in Table~\ref{tab:similarity:single_spread}, beats the unweighted distance hybrid of Table~\ref{tab:similarity:feature_spaces} on Caco-2, EGFR binding, expression titer and CHO polyreactivity. On Caco-2, for example, it reaches 0.229 Spearman against 0.197. The similarity hybrid beats it in turn on Caco-2, with 0.277.`
Priority: low

### X-ch6-23
Location: lines 3053 and 3173, sec:similarity:results_knn; Owner: AI
Quote: "On the antibody datasets and Buchwald--Hartwig, the result depends on how the setting is chosen." / "Whether the hybrid wins depends far more on the dataset than on how the neighbors are found"
His comment: Two paragraphs apart, the result depends on the setting, then not on the setting. Which is it?
Lens item: 17 (also 19)
Proposed change: first sentence of the 3053 paragraph:
old_string: `On the antibody datasets and Buchwald--Hartwig, the result depends on how the setting is chosen. At the fixed setting,`
new_string: `On the antibody datasets and Buchwald--Hartwig, the datasets where the hybrid wins change with how the setting is chosen. At the fixed setting,`
and NEEDS HER check that "far more on the dataset than on how the neighbors are found" (3173) still holds against the Additional file table it cites.
Priority: low

### X-ch6-24
Location: line 3204, sec:similarity:results_cliffs; Owner: AI
Quote: "A second, separate run flags the same cliff molecules and agrees in direction on all four datasets."
His comment: What is the second run, and how does it differ?
Lens item: 4
Proposed change: difference from the SOURCE comment at lines 3211-3212 (s04 averages every fold; s12 only folds with at least 10 molecules of each kind).
old_string: `A second, separate run flags the same cliff molecules and agrees in direction on all four datasets.`
new_string: `Averaging over all five folds, including folds with fewer than 10 cliff molecules, gives the same direction on all four datasets.`
Priority: low

### X-ch6-25
Location: line 3374, sec:similarity:results_other; Owner: AI
Quote: "The features a hybrid keeps carry no more label information than principal components of the same number, except on QM9 and Buchwald--Hartwig"
His comment: Methods also compares against random features; what happened there? And if the kept features hold no more information, why does their distance track labels better?
Lens item: 21 (also 17)
Proposed change: NEEDS HER: the random-subset result from app:add:similarity_label_information, one clause; and whether she wants a sentence on why distance can improve without more summed information (Methods line 2810 already says the sum counts shared information twice).
Priority: low

### X-ch6-26
Location: line 3379, sec:similarity:conclusions; Owner: AI
Quote: "Of the nine datasets where the similarity hybrid has a value, it beats the distance hybrid at tracking label differences only on Caco-2 and hERG blocking"
His comment: Unweighted or weighted? The table has both.
Lens item: 19
Proposed change: from Table tab:similarity:feature_spaces: true unweighted; weighted, the similarity hybrid is also above on QM9 (0.462 against 0.460) and LogD (0.312 against 0.309). Full paragraph:
old_string: `Of the nine datasets where the similarity hybrid has a value, it beats the distance hybrid at tracking label differences only on Caco-2 and hERG blocking, as seen in Table~\ref{tab:similarity:feature_spaces}.`
new_string: `Of the nine datasets where the similarity hybrid has a value, it beats the unweighted distance hybrid at tracking label differences only on Caco-2 and hERG blocking, as seen in Table~\ref{tab:similarity:feature_spaces}.`
Priority: low

### X-ch6-27
Location: line 2696, subsection title; Owner: hers
Quote: "\subsection{Similarity in a hybrid}"
His comment: The title reads like the whole chapter; this section defines the similarity search. Name the aim.
Lens item: 11
Proposed change:
old_string: `\subsection{Similarity in a hybrid}\label{sec:similarity:hybrid_similarity}`
new_string: `\subsection{Choosing features by how well neighbors predict}\label{sec:similarity:hybrid_similarity}`
Priority: low

### X-ch6-28
Location: line 2808, sec:similarity:experiments, "Three smaller checks"; Owner: hers
Quote: "The search looks for the lowest label on hERG $K_i$ and QM9, and the highest yield on Buchwald--Hartwig."
His comment: Why the lowest? On hERG this is the safest molecule; say so. And on QM9?
Lens item: 21 (also 14)
Proposed change: the s11 file for QM9 records the direction as "metric direction (NO label direction recorded for this dataset)" (s11_bayesian_optimization/qm9.json, results.hunt.source), so the QM9 choice was a default, not a decision. NEEDS HER for QM9. Edit for hERG:
old_string: `The search looks for the lowest label on hERG $K_i$ and QM9, and the highest yield on Buchwald--Hartwig.`
new_string: `The search looks for the lowest label on hERG $K_i$ and QM9, and the highest yield on Buchwald--Hartwig. On hERG $K_i$, the lowest $\mathrm{p}K_i$ is the weakest hERG binder, the molecule least likely to block the channel.`
Priority: low
