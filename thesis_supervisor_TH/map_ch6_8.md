# Supervisor comments, old Ch6 and Ch7 -> current chap:similarity, chap:noise, chap:discussion

Line numbers are from thesis.tex as read on 2026-10-10 (5093 lines). Old page text from pNNN.txt.

---

### TH-p162a
Comment: Nice set of questions !
Current location: lines 2668-2673 (chap:similarity, Introduction, two numbered research questions); lines 3396-3399 (sec:noise:intro, the noise questions as prose).
Owner: hers (2668-2673, marked "% [author text, restored 2026-10-10]"); AI (3396-3399)
Status: PRAISE
Reasoning: The old list had three questions. The first one survives, widened, as the second similarity question at line 2671 ("Do the methods built on that distance work better in a hybrid? We test prediction from nearest neighbors, ..."). The two noise questions moved to chap:noise and are now prose, not a list: "Can we find the noisy labels in a QSAR dataset, and does acting on them improve a model's held-out score?" (line 3396) and "For each method we therefore ask whether it improves the held-out score, and whether it improves the hybrid more than the best single representation." (line 3399). The praised numbered form survives only in chap:similarity.
Edit: none.

---

### TH-p163a
Comment: Is this going to be replaced?  (on the highlighted sentence "Noise detection has no result yet.")
Current location: sentence removed. Detection results now in sec:noise:res:detectors, lines 4122-4180, Table tab:noise:res:detect_residual (line 4178).
Owner: AI
Status: ALREADY ADDRESSED
Reasoning: The placeholder sentence is gone (grep "no result yet" finds nothing). The detection section now has results, for example: "On Caco-2 under outliers, for example, its average precision is 0.338 against a chance level of 0.051." (line 4127). The table gives average precision of the forest residual and the cliff-aware gap on 10 datasets.
Edit: none.

---

### TH-p163b
Comment: Yes we really need a figure here.  (on "[FIGURE NEEDED: flow from noisy data, to per-fold hybrid and single representations, to scores]")
Current location: lines 3412-3421, sec:noise:methods, Figure fig:noise:flow (figures_ch5_7/d7_noise_flow.png, file exists, 10 Oct 01:48).
Owner: AI
Status: ALREADY ADDRESSED
Reasoning: The figure exists and is cited: "Figure~\ref{fig:noise:flow} shows how one method is run and scored." (line 3412). I opened the PNG. It runs from "Damaged training labels" to "Features, chosen per fold and per downstream model: two best single representations and two best hybrids", then to the selection and action, then to the held-out score, with the controls dashed. That is the flow the placeholder asked for. No "FIGURE NEEDED" placeholder is left anywhere in thesis.tex. The second old placeholder, "[FIGURE NEEDED: D6.1, one search, two scores]" (old p164), is also gone, and chap:similarity has no figure in its place. Lines 2661-3391 hold no \includegraphics.
Edit: none.

---

### TH-p164a
Comment: This section seems to have overlap with chapter 5 and previous chapters aswell (probably because of the paper projection) . For the thesis reader it might feel a bit redundant though.

Note: I am not familiar with the thesis writing protocol in Oxford and this might be perfectly OK
Current location: old 6.2.1 "Similarity in a Hybrid" is now sec:similarity:hybrid_similarity (lines 2696-2710), sec:similarity:knn_settings (2712-2746) and the experiment definitions in sec:similarity:experiments (2748-2811).
Owner: mixed (see each passage)
Status: PARTLY
Reasoning: The section is shorter than the old one, and it already points back for the distance ("the Manhattan distance of Chapter~\ref{chap:flaws}", line 2702) and the modelability index ("discussed in Chapter~\ref{chap:flaws}", line 2710). Six passages still repeat earlier chapters. Each is listed below with the earlier passage it repeats.

Duplicated passages:
1. Line 2700 (hers) repeats line 937 (chap:flaws, hers). Both say Zheng2000 and Itskowitz2005 chose descriptors for nearest-neighbor prediction within one set, and that this thesis selects across several representations.
2. Line 2702, sentences 2-3 (hers) repeat line 1060 (sec:flaws:similarity, hers). Line 2702 says each standardized feature adds a term of similar size to the distance. Line 1060 says the same thing.
3. Line 2698, sentence 2 (hers) repeats the search settings of sec:method:searchspace (line 1395 on) and line 1385. Both list importance method, allocation method, number of representations and budget. The formula after it is needed for q_5 and is not a repeat.
4. Line 2740, sentence 2 (hers) repeats line 1946 (sec:method:comparators, "Feature weighting: standardized features multiplied by the square root of their importance, with the weights normalized to average 1"). The second formula at line 2743 is the same weighting.
5. Line 2780 (AI) defines the pair-distance Spearman measure. Line 1068 (sec:flaws:similarity, hers) defines the same measure ("the Spearman correlation between the distance of a pair and the absolute difference of its labels").
6. Line 2792 (AI) defines similar molecules (ECFP4 Tanimoto at least 0.55, Hu2015) and the 15% gap rule. Line 1063 (sec:flaws:similarity, hers) gives the same two rules.

Line 2710 (AI) has a short repeat of the modelability index from line 933. It is already a cross-reference, and it adds the contrast with q_5, so I propose no cut. Line 2665 (hers) restates the chap:flaws result in one sentence as a link, which is normal.

Edits, AI text (rewritten in full; marker lines above them stay):

Passage 5, line 2780:
```
This experiment asks how well distance in a feature space follows the label. For pairs of held-out molecules, we compute the Spearman correlation between their distance and their absolute label difference. A positive value means that molecules further apart differ more in their labels. We measure it in each representation alone, unweighted, under every distance of Table~\ref{tab:knn_settings}, and then in the feature spaces of Section~\ref{sec:similarity:spaces}, unweighted and weighted.
```
```
This experiment asks how well distance in a feature space follows the label. It uses the measure of Section~\ref{sec:flaws:similarity}, here on pairs of held-out molecules. For each pair, we compute the Spearman correlation between their distance and their absolute label difference. A positive value means that molecules further apart differ more in their labels. We measure it in each representation alone, unweighted, under every distance of Table~\ref{tab:knn_settings}. We then measure it in the feature spaces of Section~\ref{sec:similarity:spaces}, unweighted and weighted.
```

Passage 6, line 2792:
```
An activity cliff is a pair of structurally similar molecules with very different activity \citep{Stumpfe2012,vanTilborg2022}. Only the two hERG datasets measure an activity, but we use the term on all six small-molecule datasets. Following \citet{Hu2015}, two molecules are similar when their ECFP4 Tanimoto similarity is at least 0.55. Their activity gap of two log units flags no molecule on Caco-2, so we set the gap so that 15\% of each outer fold's training molecules are cliff molecules.
```
```
An activity cliff is a pair of structurally similar molecules with very different activity \citep{Stumpfe2012,vanTilborg2022}. Only the two hERG datasets measure an activity, but we use the term on all six small-molecule datasets. We define cliffs as in Section~\ref{sec:flaws:similarity}. Two molecules are similar when their ECFP4 Tanimoto similarity is at least 0.55 \citep{Hu2015}. A gap of two log units flags no molecule on Caco-2. We therefore set the gap so that 15\% of the molecules are cliff molecules. Here the 15\% is counted on each outer fold's training molecules.
```
(The rest of the paragraph, from "A training molecule is a cliff molecule if", is unchanged.)

Edits proposed for her text (minimal; cross-reference only, for her to accept or not):

Passage 1, line 2700:
```
\citet{Zheng2000} and \citet{Itskowitz2005} chose descriptors to maximize nearest-neighbor prediction within one descriptor set. We modify this idea by choosing features across several representations.
```
```
As discussed in Chapter~\ref{chap:flaws}, \citet{Zheng2000} and \citet{Itskowitz2005} chose descriptors to maximize nearest-neighbor prediction within one descriptor set. We modify this idea by choosing features across several representations.
```

Passage 2, line 2702:
```
Each standardized feature adds a term of similar size to this distance.
```
```
Each standardized feature adds a term of similar size to this distance (Section~\ref{sec:flaws:similarity}).
```

Passage 3, line 2698:
```
Each configuration $\theta$ of that search fixes an importance method, an allocation method, a number of representations and a budget $B$.
```
```
Each configuration $\theta$ of that search fixes an importance method, an allocation method, a number of representations and a budget $B$ (Section~\ref{sec:method:searchspace}).
```

Passage 4, line 2740:
```
Before the distance is computed, each feature is multiplied by a weight, $\mathbf{x}' = \mathbf{x} \odot \mathbf{w}$.
```
```
Before the distance is computed, each feature is multiplied by a weight, $\mathbf{x}' = \mathbf{x} \odot \mathbf{w}$, as in the feature weighting of Section~\ref{sec:method:comparators}.
```
Caveat on passage 4: the Ch5 weights are rescaled to average 1. The second formula here, $\mathbf{w} = \mathbf{s}^{1/2}$ (line 2743), is not stated to be rescaled. If they are not the same, the cross-reference should say "a variant of". Unverified in code.

---

### TH-p171a
Comment: Encouraging results :-)   (on old Table 6.4: nearest-neighbor RMSE on hERG Ki, equal against importance-weighted features)
Current location: sec:similarity:results_knn, lines 3008-3165. Table tab:nearest_neighbour_prediction (line 3015, \input figures_ch5_7/tab_nearest_neighbour_prediction.tex, made by tests/analysis/thesis_figures.py --figure f6). Table tab:importance_weighting (line 3165).
Owner: AI
Status: PRAISE
Reasoning: The result survived, now on five folds instead of three. On hERG Ki the distance hybrid predicts from five neighbors at 0.5822 RMSE against 0.6280 for the best single representation, and it wins on all six small-molecule datasets. With importance weighting, hERG Ki goes from 0.628 to 0.597 RMSE on the single representation and from 0.582 to 0.580 on the hybrid (tab:importance_weighting). One part did not survive. Old Table 6.4 showed the similarity hybrid as the best feature space (0.6128 RMSE), but the current kNN prediction tables report only the distance hybrid.
Edit: none.

---

### TH-p174a
Comment: Interesting to see there is signal beyond the clipping threshold and how all learning approaches are impacted.  (on old Table 6.6: hERG Ki, ECFP4, 10% clipped, four models)
Current location: sec:noise:res:censoring, lines 4322-4440. Figure fig:noise:censoring_ladder; Table tab:noise:res:censoring_moves_main.
Owner: AI
Status: PRAISE
Reasoning: Old Table 6.6 is gone. The current runs clip 25% of training labels, not 10%, and use each dataset's two downstream models, not four models on ECFP4. The finding survived: "censored EM lowers the held-out RMSE for every model and configuration, though not to the level before clipping" (line 4326). On hERG Ki, the network on the first hybrid goes from 0.608 RMSE before clipping to 0.731 clipped. Raising the clipped labels brings it to 0.668 with the constant lift and 0.664 with censored EM. Deleting the clipped molecules is still worse than keeping them on hERG Ki (line 4405), as it was in old Table 6.6.
Edit: none.

---

### TH-p175a
Comment: We constantly see that hybrid representation really benefit SVM and LightGBM but not so much (or even detariorate) RF models. Are we discussiong this aspect specifically somewhere?
Current location: not discussed. Checked sec:noise:res:repr (lines 4944-4981), chap:discussion (4999-5075) and sec:method:results_single.
Owner: n/a
Status: NEEDS HER
Reasoning: Nothing in the thesis discusses why the hybrid helps the random forest less. The old example came from hERG Ki. There the random forest scored 0.6121 RMSE on the hybrid against 0.6032 on ECFP4. That three-model hERG Ki run is gone. In chap:noise the random forest is a downstream model only on Buchwald--Hartwig and expression titer. The nearest mention, line 4983, is about cleaning, not the hybrid: "on expression titer the random forest gains more often than the network." Line 5007 says random forests were the most consistent model under noise, but that is about noise robustness, not the hybrid.
What the current tables show (Table tab:hybrid_vs_best_single, figures_ch5_7/tab_hybrid_vs_best_single.tex, held-out score of best single representation against hybrid):
- Random forest, Buchwald--Hartwig: 14.4056 to 15.2383 RMSE (hybrid worse).
- Random forest, antibody humanness: 0.9659 to 0.9614 AUROC (hybrid worse).
- Random forest, anti-VEGF scan binding: 0.4325 to 0.5112 Spearman (hybrid better).
- Random forest, expression titer: 0.2504 to 0.2706 Spearman (hybrid better).
- LightGBM is better on the hybrid on every dataset it runs on: QM9 0.3720 to 0.3463, Buchwald--Hartwig 16.1679 to 15.6211, AC-SINS 0.4631 to 0.5242, hERG blocking 0.8555 to 0.8666.
- SVM is better on the hybrid on most datasets. It is flat on nanobody thermostability (0.6901 to 0.6899) and SMAC colloidal stability (0.3485 to 0.3480).
Ch5 line 2076 discusses only the Buchwald--Hartwig case: "the hybrid lowers the RMSE with LightGBM and raises it with the random forest ..., both by less than the spread across folds."
She needs to decide whether to add a sentence on this to sec:method:results_single or chap:discussion. The random forest is a downstream model on only four datasets, two of them antibody or reaction datasets. Any reason given for the pattern would be new. Nothing in the results tests why it happens.
Edit: none until she decides.

---

### TH-p177a
Comment: This table could benefit from bold best results.
Landscape is clipped.   (on old Table 6.10: twelve methods on nanobody thermostability, one-hot, hybrid and weighted hybrid, two noise levels)
Current location: old Table 6.10 is gone, and no nanobody-thermostability table of the twelve methods is left. The twelve methods are now in sec:noise:res:sweep (lines 4846-4938): Table tab:noise:res:sweep (counts), tab:noise:res:sweep_examples and tab:noise:res:qm9_sweep (line 4937). All three are hand-written in thesis.tex, not \input. None of the chap:noise tables is landscape, so nothing is clipped.
Owner: AI
Status: PARTLY
Reasoning: The clipping is gone because the table is gone. Its closest replacement, tab:noise:res:qm9_sweep, shows the twelve methods with RMSE per method. It has three bold cells, and its caption does not say what bold means. Bold is on the plain network's first hybrid (0.355), distance-based selection (0.368) and DivideMix (0.452). DivideMix is the worst result in its column, not the best. The edit below bolds the lowest RMSE in each column instead, three cells in all, and states this in the caption. No generator script exists for this table (grep in tests/analysis finds none), so the edit is to thesis.tex. Separately, tab:noise:res:sweep_examples (line 4893) also has bold with no caption explanation. That table is outside this comment.
Edit:
```
Plain network & 0.396 & \textbf{0.355} & 0.364 \\
\midrule
Small-loss selection & 0.465 & 0.422 & 0.423 \\
Confident learning & 0.412 & 0.371 & 0.383 \\
Distance-based selection & 0.413 & \textbf{0.368} &  \\
Multistage cleaning & 0.455 & 0.419 & 0.382 \\
Early learning & 0.420 & 0.370 & 0.356 \\
Uncertainty curriculum & 0.444 & 0.380 & 0.354 \\
DivideMix & 0.499 & \textbf{0.452} & 0.405 \\
Meta-Weight-Net & 0.426 & 0.380 & 0.353 \\
MentorNet & 0.425 & 0.392 & 0.343 \\
```
```
Plain network & \textbf{0.396} & \textbf{0.355} & 0.364 \\
\midrule
Small-loss selection & 0.465 & 0.422 & 0.423 \\
Confident learning & 0.412 & 0.371 & 0.383 \\
Distance-based selection & 0.413 & 0.368 &  \\
Multistage cleaning & 0.455 & 0.419 & 0.382 \\
Early learning & 0.420 & 0.370 & 0.356 \\
Uncertainty curriculum & 0.444 & 0.380 & 0.354 \\
DivideMix & 0.499 & 0.452 & 0.405 \\
Meta-Weight-Net & 0.426 & 0.380 & 0.353 \\
MentorNet & 0.425 & 0.392 & \textbf{0.343} \\
```
and
```
One row is one method. Held-out RMSE, lower is better. Outer fold 0. Empty: not run.}
```
```
One row is one method. Held-out RMSE, lower is better; bold is the lowest in each column. Outer fold 0. Empty: not run.}
```

---

### TH-p184a
Comment: " …implicitely assum" ?   (on "assume" in "Real assay labels are not noisy in the way most studies assume.", the first sentence of old Ch7)
Current location: sentence removed. chap:discussion now opens at line 5002 with "This chapter returns to the research questions of Chapter~\ref{chap:intro} ...". Its point now lives only in the recommendation at line 5028.
Owner: AI (chapter opening) / hers (line 8)
Status: GONE
Reasoning: grep finds no "most studies assume" and no "implicitly" in thesis.tex. Two similar claims remain in her text. Line 8 (chap:intro) says "Most work on noise in QSAR modeling assumes Gaussian label noise". Line 620 (chap:robustness) says "Many studies involving experimental label noise rely on the assumption that noise can be modeled by a Gaussian distribution." If she wants his word there, the optional minimal edit is to line 8.
Edit (optional, her text, propose only):
```
Most work on noise in QSAR modeling assumes Gaussian label noise
```
```
Most work on noise in QSAR modeling implicitly assumes Gaussian label noise
```

---

### TH-p192a
Comment: New foundation models or new representations ?
Current location: line 5071, sec:discussion:conclusions (marker "% [AI text, revised 2026-10-10]" at line 5070).
Owner: AI
Status: APPLIES
Reasoning: The sentence is still about foundation models, although the claim is about representations: the search takes in a new representation, not a model. The rewrite opens with new representations and says where they come from. It keeps "can add" (TH-p192b) and every fact.
Edit:
```
A new foundation model appears every few months in molecular and protein modeling, antibodies included. With the hybrid search, a researcher need not choose between their current representation and a new one. They can add the new embedding and rerun the search, which decides whether it contributes any features, and which ones. We tested this on LogD and hERG blocking in Chapter~\ref{chap:method}.
```
```
New representations appear every few months in molecular and protein modeling, antibodies included. Many are embeddings from a newly published foundation model. With the hybrid search, a researcher need not choose between their current representation and a new one. They can add the new representation to those searched and rerun the search. The search then decides whether the new representation contributes any features, and which ones. We tested this on LogD and hERG blocking in Chapter~\ref{chap:method}.
```

---

### TH-p192b
Comment: "…they can add…" ?
Current location: line 5071, sec:discussion:conclusions.
Owner: AI
Status: ALREADY ADDRESSED
Reasoning: The current sentence reads "They can add the new embedding and rerun the search, which decides whether it contributes any features, and which ones." The TH-p192a rewrite keeps "can add".
Edit: none.
