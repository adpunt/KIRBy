# Chapter 4 — suggestions for the author's paragraphs (not edited)

Paragraphs ranked in the top 150 of `benchmark/worklist.csv` that have no review marker above them. Nothing here has been applied. Every number in a suggested edit is copied from the paragraph or from a table in Chapter 4.

## 1. "A distance sums over every feature, so when" (Conclusions, rank 2)
Flag: 5.2 numbers per 100 words (reference conclusions 0.2); every result sentence reports numbers.
Old: "Across 70 combinations of dataset and representation, the Spearman correlation between pair distance and label difference was below 0.10 in 44. [...] Predicting each label from its five nearest neighbors gave a higher Spearman correlation in all 70, so the useful information sits in a few close pairs rather than in the distance as a whole."
New: "We found that distance over all pairs barely tracks label difference. Predicting each label from its five nearest neighbors gave a higher Spearman correlation on every dataset and representation. The useful information sits in a few close pairs."

## 2. "Across the 70 combinations of dataset and representation," (Results, rank 3)
Flag: every sentence carries 3 or more numbers; the table already holds them.
Old: "Across the 70 combinations of dataset and representation, the all-pairs correlation ranged from $-0.04$ to 0.23 and was below 0.10 in 44 of them. The five-neighbor prediction reached 0.29 to 0.89, higher than the all-pairs correlation in all 70 (Table~\ref{tab:flaws:distance_summary}; every representation is in Additional file~\ref{app:add:flaws})."
New: "Table~\ref{tab:flaws:distance_summary} gives both correlations per dataset, and Additional file~\ref{app:add:flaws} every representation. The all-pairs correlation was below 0.10 in 44 of the 70 combinations of dataset and representation. The five-neighbor prediction was higher in all 70."

## 3. "On a single dataset, often very little of" (Conclusions, rank 8)
Flag: 7.6 numbers per 100 words (references 0.2); no "we" in the conclusions.
Old: "On hERG $K_i$, 1,815 of the 2,048 ECFP4 features take one value in at least 95\% of the 1,412 molecules (Section~\ref{sec:flaws:constant})."
New: "On hERG $K_i$, we found that most ECFP4 features take one value in nearly every molecule (Section~\ref{sec:flaws:constant})."
Old: "ECFP4 gives 56 of the 1,412 hERG $K_i$ molecules a vector shared with another molecule, all of them stereoisomers (``Molecules with identical feature vectors'')."
New: "On hERG $K_i$, every molecule that ECFP4 ties is a stereoisomer of another (``Molecules with identical feature vectors'')."

## 4. "A representation that avoids these flaws would need" (Conclusions, rank 17)
Flag: 6.4 hedges per 100 words ("would" five times, "rather").
Old: "Its features would be chosen for the task, using only each fold's training labels. It would be compact, dropping features that do not vary or do not matter, with the number kept searched rather than fixed."
New: "Its features are chosen for the task, using only each fold's training labels. It is compact, dropping features that do not vary or do not matter, and the number kept is searched."
(Same change, "would" to present tense, in the next three sentences.)

## 5. "Figure~\ref{fig:flaws:herg_pairs} shows why the two differ." (Results, rank 28)
Flag: 15.8 numbers per 100 words (19 numbers); the figure shows most of them.
Old: "Mordred shows the same pattern: the closest 5\% of pairs have a median difference of 0.44 p$K_i$, and every other 5\% has 0.78 to 1.11 p$K_i$."
New: "Mordred shows the same pattern (Figure~\ref{fig:flaws:herg_pairs}b): only the closest 5\% of pairs have a small median difference, 0.44 p$K_i$."

## 6. "We first consider how much the representation impacts" (Introduction, rank 31)
Flag: 4.5 numbers per 100 words (reference introductions 0.5).
Old: "\citet{Riniker2013} benchmarked 12 fingerprints and two simple baseline fingerprints for similarity-based virtual screening on 88 targets from three public collections, and found that the screening performance of the 12 fingerprints was highly correlated. Only the two baselines differed significantly from the rest."
New: "In similarity-based virtual screening, the performance of 12 fingerprints was highly correlated, and only two simple baseline fingerprints differed from the rest \citep{Riniker2013}."
Old: "across 15 configurations of representation and similarity measures on 3 datasets, the representations disagreed"
New: "across several pairings of representation and similarity measure, the representations disagreed"

## 7. "Section~\ref{sec:lnl_motivation} reports a small sweep of published" (Results, rank 37)
Flag: 14.6 nominalizations per 100 words; booster "clear-cut".
Old: "Section~\ref{sec:lnl_motivation} reports a small sweep of published noise-handling methods, in which the 20 configurations with the lowest retention under noise all relied on a pairwise distance measurement somewhere in their process. Section~\ref{sec:flaws:neighbors} focuses directly on neighbor-based noise detection (Table~\ref{tab:noise_terms}), the most clear-cut use of similarity."
New: "In a small sweep of published noise-handling methods (Section~\ref{sec:lnl_motivation}), the 20 configurations that kept the least of their $R^2$ under noise all used a pairwise distance. Section~\ref{sec:flaws:neighbors} tests neighbor-based noise detection (Table~\ref{tab:noise_terms}), which uses similarity most directly."

## 8. "We compared the five nearest neighbors of each" (Results, rank 44)
Flag: lexical density 44.8% (references 60.3%); numbers repeat Table 4.5.
Old: "On hERG $K_i$ any two of the three share on average 0.47 to 0.52 of each molecule's five neighbors, and on the three Biogen datasets and QM9 only 0.13 to 0.21 (Table~\ref{tab:flaws:neighbor_overlap})."
New: "Table~\ref{tab:flaws:neighbor_overlap} gives the overlap. Any two of the three share about half of each molecule's neighbors on hERG $K_i$, and under a quarter on the three Biogen datasets and QM9."

## 9. "With the neighbor-based scores used as sample weights" (Results, rank 50)
Flag: 13.8 numbers per 100 words (12 numbers), mostly counts of configurations.
Old: "Of all 860 cleaning configurations, 8 had positive recovery at both noise levels; 6 of these used a single representation, and 4 of those 6 were neighbor-based."
New: "Only 8 of all 860 cleaning configurations had positive recovery at both noise levels, 4 of them neighbor-based."

## 10. "Uncertainty-based methods, which use no neighbors, showed potential" (Results, rank 59)
Flag: 12.5 numbers per 100 words; the four numbers fit the columns of Table~\ref{tab:code_input_neighbor_recovery}.
Suggested: add a row "All scores, sample weighting & $-22.9$ & $-20.9$ & 14.2 & 4.7" to that table (it would then hold more than neighbor-based scores, so its caption and the "Table ... holds only" sentence above it would need changing too).
Old: "Sample weighting did better than label smoothing but usually still hurt, with positive recovery in 14.2\% of configurations at $\sigma = 0.3$ and 4.7\% at $\sigma = 0.6$, and mean recoveries of $-22.9\%$ and $-20.9\%$."
New: "Sample weighting did better than label smoothing but usually still hurt (Table~\ref{tab:code_input_neighbor_recovery}, last row)."

## 11. "Chapter~\ref{chap:background} gives the MoleculeACE definition of an" (Results, rank 70)
Flag: 13 numbers; lexical density 49.2%. Table~\ref{tab:flaws:distance_summary} already holds each gap and cliff count.
Old: "On hERG $K_i$ the gap is 1.76 p$K_i$, and 211 of the 1,412 molecules are cliffs. On QM9 only 1,093 pairs of the 9,978 molecules are similar, and the gap is 0.05~eV, 0.06 times the median absolute deviation of the labels. On the three Biogen datasets (RLM clearance, MDR1--MDCK efflux and HLM clearance) it is 0.20 to 0.34 log units, 0.50 to 0.60 times the median absolute deviation."
New: "Table~\ref{tab:flaws:distance_summary} gives each dataset's gap and number of cliff molecules. On QM9 only 1,093 pairs of the 9,978 molecules are similar, and the gap is 0.06 times the median absolute deviation of the labels. On the three Biogen datasets it is 0.50 to 0.60 times."

## 12. "Single groups can still disagree by a lot." (Results, rank 92)
Flag: 10.7 numbers per 100 words (11 numbers).
Old: "On hERG $K_i$, 6 of the 26 ECFP4 groups and 38 of the 153 MACCS groups differ by more than $0.54$. The widest ECFP4 group is a pair of stereoisomers $1.14$ p$K_i$ apart (Figure~\ref{fig:flaws:stereo_pair}). An ECFP4 model can at best predict their mean, $5.59$ p$K_i$, which leaves each molecule $0.57$ p$K_i$ from its measured value."
New: "On hERG $K_i$, a few ECFP4 groups and about a quarter of MACCS groups differ by more than this. The widest ECFP4 group is a pair of stereoisomers $1.14$ p$K_i$ apart (Figure~\ref{fig:flaws:stereo_pair}), so an ECFP4 model misses each by at least $0.57$ p$K_i$."

## 13. "The substructure-based fingerprints have the most near-constant features." (Results, rank 97)
Flag: 14.9 words per sentence (references 27.3); three short sentences make one point.
Old: "ECFP4 is the worst offender. On hERG $K_i$ (1,412 molecules) 1,815 of the 2,048 features are near-constant. That jumps to 1,997 out of 2,048 near-constant features for the HOMO--LUMO gap on QM9."
New: "ECFP4 has the most: 1,815 of its 2,048 features are near-constant on hERG $K_i$, and 1,997 on QM9."

## 14. "The problem of whether neighboring molecules can predict" (Introduction, rank 104)
Flag: 10.8 nominalizations per 100 words; "the X of the Y" chains.
Old: "The problem of whether neighboring molecules can predict chemical activity has been measured before."
New: "Others have measured how well neighboring molecules predict activity."
Old: "None of these measures ask specifically about whether the specific features within these representations matter for the property at hand."
New: "None of these measures asks whether the features of a representation matter for the property."

## 15. "\citep{Guha2008}, is infinite when their similarity is 1" (Results, rank 131)
Flag: hedges "rather", "may".
Old: "and here the cliff is made by the representation rather than the chemistry. A method that judges a label by its nearest neighbors sees a tied neighbor at distance 0 with a different label, and may flag a correct label as noise"
New: "and here the representation alone makes the cliff. A method that judges a label by its nearest neighbors sees a tied neighbor at distance 0 with a different label, and can flag a correct label as noise"

## 16. "The distances may have failed for two reasons," (Results, rank 135)
Flag: 3.8 hedges per 100 words ("may" three times, "possibly").
Old: "First, a single representation may lack the information a distance needs, with and possibly even without noise."
New: "First, a single representation may lack the information a distance needs, even without noise."
Old: "and any one of these sources alone may not be sufficient for robust similarity estimation."
New: cut this clause; the sentence before it says the same.

## 17. "A similarity coefficient cannot tell you which features" (Results, rank 142)
Flag: lexical density 50.0%; short sentences that make one point; throat-clearing ("we make the following computation").
Old: "A similarity coefficient cannot tell you which features matter to the prediction task at hand. And the SPP holds only relative to such a measure. To understand the impact of irrelevant features on similarity measures, we make the following computation. First, we categorize a representation's features into those relevant to the task, $R$, and those irrelevant to the task, $I$, and take the squared Euclidean distance as a sum over both:"
New: "A similarity coefficient cannot tell which features matter for the task, and the SPP holds only relative to such a coefficient. To see how irrelevant features affect it, we split a representation's features into those relevant to the task, $R$, and those irrelevant, $I$, and write the squared Euclidean distance as a sum over both:"
