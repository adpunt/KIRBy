# Chapter 2: suggestions for the author's paragraphs (worklist top 150)

Not edited. One suggestion each. Rank is the worklist rank.

1. Rank 4. "However, PDVs have limitations. The dimensionality of PDV"
   Flag: nominalizations 15.5 per 100 words (reference mean 4.3); AI-guide word "Additionally".
   Old: "However, PDVs have limitations. The dimensionality of PDV feature spaces can lead to problems with multicollinearity \citep{Todeschini2009}. This can be mitigated by feature selection or dimensionality reduction techniques"
   New: "PDVs have hundreds of features, and many of them are correlated \citep{Todeschini2009}. Feature selection or dimensionality reduction can remove the redundant ones"
   Also: "Additionally, PDVs encode" -> "PDVs also encode".

2. Rank 5. "For a model that returns a predicted standard"
   Flag: one 64-word sentence with 4 numbers.
   Old: "For a model that returns a predicted standard deviation $u_i$ for each sample, the package computes, at each noise level, the Spearman correlation between $u_i$ and the absolute error $|y_i - \hat{y}_i|$, the empirical coverage ... at $k = 1$ and $k = 2$, against Gaussian targets of $68.3\%$ and $95.4\%$, the mean interval width ..., and the expected calibration error (ECE),"
   New: "Some models return a predicted standard deviation $u_i$ for each sample. For these, the package computes four metrics at each noise level. The first is the Spearman correlation between $u_i$ and the absolute error $|y_i - \hat{y}_i|$. The second is the empirical coverage ... at $k = 1$ and $k = 2$, against Gaussian targets of $68.3\%$ and $95.4\%$. The third is the mean interval width ..., and the fourth is the expected calibration error (ECE),"

3. Rank 20. "where $\tau$ is the target standard deviation of the"
   Flag: lexical density 45.5% (reference mean 61.7%); "the X of the Y" chains.
   Old: "The factor $G$ divides both of these out, so that $\epsilon$ has standard deviation $\tau$ in expectation whatever the distribution and whichever molecules are affected."
   New: "Dividing by $G$ removes both, so $\epsilon$ has expected standard deviation $\tau$ for any distribution and any set of affected molecules."

4. Rank 26. "Real-world cheminformatics datasets contain label noise arising from"
   Flag: boosters 3.5 per 100 words ("must").
   Old: "other varieties of noise must be considered when working with experimental data. To properly model experimental noise, we must first understand where the noise is coming from."
   New: "experimental data carry other kinds of noise. To model it, we first describe where it comes from."

5. Rank 30. "We selected the parameter values from published measurement"
   Flag: hedges 5.3 per 100 words ("estimate", "possible"); three subordinate clauses.
   Old: "We selected the parameter values from published measurement errors wherever possible, and Table~\ref{tab:noise_conditions} notes where no published estimate exists."
   New: "We took the parameter values from published measurement errors. Table~\ref{tab:noise_conditions} marks the values with no published source."

6. Rank 34. "Although the general assumption in much of the"
   Flag: AI-guide phrasing ("These studies indicate that"); nominalizations.
   Old: "These studies indicate that the presence of noise may act as a form of regularization in some contexts. It is important to address this potential of regularization from label noise in QSAR and quantitative structure--property relationship (QSPR) models."
   New: "Noise can therefore act as a regularizer. We test whether label noise does so in QSAR and quantitative structure--property relationship (QSPR) models."
   (Second sentence only if Chapter 3 does test this; otherwise cut it.)

7. Rank 40. "Neural fingerprints provide several advantages over traditional fingerprints."
   Flag: AI-guide word "insight".
   Old: "although attention visualization offers some insight \citep{Chithrananda2020}."
   New: "although attention maps show which substructures a model attends to \citep{Chithrananda2020}."

8. Rank 48. "Studies involving noise, and specifically the quantification of"
   Flag: hedges 3.7 per 100 words ("typically", "suggest", "in most cases", "rather"); "evidence to suggest".
   Old: "Studies involving noise, and specifically the quantification of noise robustness, typically approach label noise by adding independent, zero-mean, Gaussian error to the labels. However, there's evidence to suggest that noise from experimental assays takes on different forms."
   New: "Studies of noise robustness usually add independent, zero-mean Gaussian error to the labels. Noise from experimental assays takes other forms."

9. Rank 49. "The package provides six label-flipping strategies: uniform flips,"
   Flag: booster "realized"; hedge "usually".
   Old: "Because the realized flip rate usually differs from the nominal probability, a binary search finds the probability that changes a target fraction of labels."
   New: "The fraction of labels actually flipped differs from the nominal probability. A binary search therefore finds the probability that flips a target fraction of labels."
   (This paragraph carries a "% TODO: manual review" marker above the subsubsection heading but the worklist marks it hers; left unedited.)

10. Rank 60. "A typical MLP for molecular property prediction has"
    Flag: nominalizations 10.5 per 100 words.
    Old: "Training is performed using backpropagation with gradient descent optimizers such as Adam, to minimize a loss function"
    New: "Backpropagation with an optimizer such as Adam trains the network to minimize a loss"

11. Rank 73. "One key issue with QSAR models is the"
    Flag: nominalizations 12.5 per 100 words; booster "in fact"; appraisal word "key".
    Old: "One key issue with QSAR models is the standard ML assumption that labels represent true values"
    New: "Standard ML assumes that labels are true values"
    Also: "is, in fact, an accurate reflection of the population" -> "reflects the population".

12. Rank 96. "Unlike standard deterministic ML methods, Bayesian methods model"
    Flag: lexical density 72.5% (reference mean 61.7%); noun stacks.
    Old: "Unlike standard deterministic ML methods, Bayesian methods model distributions over predictions using Bayes' rule, allowing for the quantification of uncertainty within parameters and labels."
    New: "Bayesian methods use Bayes' rule to give a distribution over predictions. This lets them quantify uncertainty in the parameters and labels."

13. Rank 102. "Gradient boosting methods such as eXtreme Gradient Boosting"
    Flag: lexical density 76.1% (reference mean 61.7%); "a handful of" is vague.
    Old: "We used a handful of gradient-boosted tree ensembles in this thesis including Light Gradient-Boosting Machine (LightGBM) \citep{ke2017lightgbm}, and Natural Gradient Boosting (NGBoost) \citep{Duan2020}."
    New: "We also used Light Gradient-Boosting Machine (LightGBM) \citep{ke2017lightgbm} and Natural Gradient Boosting (NGBoost) \citep{Duan2020}."
    (Check first that XGBoost, LightGBM and NGBoost are the complete list.)

14. Rank 115. "For a given label, every condition except censoring"
    Flag: nominalizations; "the X of the Y" chains.
    Old: "For a given label, every condition except censoring delivers the same expected quantity of noise at a set noise level. These levels are defined as a fraction of the standard deviation of clean training labels."
    New: "At a given noise level, every condition except censoring adds the same expected amount of noise. A noise level is a fraction of the clean training labels' standard deviation."

15. Rank 118. "which is the number of bits set in"
    Flag: lexical density 48.4% (reference mean 61.7%); short sentences.
    Old: "In the Tanimoto coefficient, a bit that is off in both molecules contributes nothing, whereas every bit that is set in either molecule counts. In the Euclidean distance, every feature that differs between the two molecules counts."
    New: "The Tanimoto coefficient ignores bits off in both molecules and counts every bit set in either; the Euclidean distance counts every feature that differs."

16. Rank 146. "Although numerous studies have explored the effects of"
    Flag: hedges 3.2 per 100 words ("rather", "assumed", "could", "argue"); eight subordinate clauses.
    Old: "They assumed experimental noise could be modeled as Gaussian-distributed, using the Central Limit Theorem to argue that aggregated experimental measurements from multiple independent error sources exhibit a Gaussian distribution."
    New: "They modeled experimental noise as Gaussian. Their reason was the Central Limit Theorem: a measurement with many independent error sources has Gaussian error."
