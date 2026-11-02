# Chapter 3: suggested edits to the author's paragraphs (not applied)

This file covers the author's Chapter 3 paragraphs that the benchmark worklist ranks in the top 150. Each entry gives the worklist rank and section, the first eight words, what the benchmark flags, and one suggested edit. None of these edits has been made in thesis.tex.

## 12. Methods: "We controlled noise quantity using noise levels, representing"
- Flag: 13.1 numbers per 100 words. The reference theses average 1.8. 40% of its sentences carry three or more numbers.
- The seven Gaussian levels are listed here and listed again under Equation eq:aucnorm. Keep this list and cut the second one (see entry 19).
- Old: "We controlled noise quantity using noise levels, representing a fraction of the standard deviation of clean training labels."
- New: "A noise level is a fraction of the standard deviation of the clean training labels."

## 19. Methods: "evaluated using the trapezoidal rule over the seven"
- Flag: 12.1 numbers per 100 words. The reference theses average 1.8.
- Old: "evaluated using the trapezoidal rule over the seven noise levels, $\tau \in \{0,\allowbreak 0.2,\allowbreak 0.3,\allowbreak 0.5,\allowbreak 0.75,\allowbreak 1.0,\allowbreak 1.5\}$, in the units of Equation~\ref{eq:dose_matching}."
- New: "evaluated with the trapezoidal rule over the seven noise levels in the Noise conditions and levels subsection, in the units of Equation~\ref{eq:dose_matching}."

## 35. Introduction: "This chapter addresses the questions set out in"
- Flag: one 45-word sentence with three subordinate clauses.
- Old: "This chapter addresses the questions set out in Chapter~\ref{chap:intro}, specifically how the choice of representation and model architecture explain the variance in predictive performance and in noise robustness, what is the impact of the type of label noise, and whether uncertainty estimates track label noise."
- New: "This chapter addresses three questions from Chapter~\ref{chap:intro}. How much of the variance in predictive performance and in noise robustness do representation and model architecture explain? What does the type of label noise change? Do uncertainty estimates track label noise?"

## 36. Methods: "An \aucnorm{} value near 1 indicates that the"
- Flag: AI-guide phrase ("indicates that"). The paragraph also has long "the X of the Y" chains.
- Old: "An \aucnorm{} value near 1 indicates that the QSAR model in question has retained predictive performance with artificial noise, meaning it is more robust to noise."
- New: "An \aucnorm{} near 1 means the model kept its predictive performance under added noise."

## 41. Conclusions: "Second, \aucnorm{} is a ratio to clean $R^2$,"
- Flag: 2.1 numbers per 100 words. Reference conclusions average 0.2. The paragraph is also a limitation. Conclusions are 14.5% limitation sentences, against 0 to 5.5% in the references.
- Old: "so a model that starts from a low clean baseline has less to lose, and a few replicates scored above 1.05."
- New: "so a model that starts from a low clean baseline has less to lose."
- Another option is to merge the limitation paragraphs ("The first limitation...", "Second, ...", "Third, ...") into one short paragraph that keeps only the most important points.

## 45. Results: "While we observe similar patterns between noise conditions,"
- Flag: AI-guide phrases ("This indicates that"). "Significantly" is used with no test behind it.
- Old: "the predictive performance of grouped-shifted noise in particular varies significantly between replicates"
- New: "under grouped-shifted noise, predictive performance varies widely between replicates"
- Old: "This indicates that the offset applied to each scaffold may account for more of the variance in both R$^2$ and \aucnorm."
- New: "The offset given to each scaffold family may therefore account for more of the variance in both."

## 53. Conclusions: "Third, it is hard to draw conclusions that"
- Flag: booster words ("know"). The paragraph addresses the reader as "you". Chapter 3's conclusions use "we" 0.5 times per 100 words, against 1.3 to 1.9 in the references.
- Old: "When working with cheminformatics data, you don't know in advance how a model and representation will perform on the given prediction task. The underlying data is probably noisy, but you don't know how noisy it is or what shape that noise takes."
- New: "On a new task, we cannot tell in advance how a model and representation will perform. The labels are probably noisy, but we do not know by how much or in what pattern."

## 62. Results: "The aleatoric component indicates that label noise increased,"
- Flag: 5.0 hedges per 100 words ("indicates", "roughly", "estimate"). The reference theses average 1.2.
- Old: "The aleatoric component indicates that label noise increased, but it doesn't tell us by how much (Figure~\ref{fig:aleatoric_injected})."
- New: "The aleatoric component rises with label noise, but it does not measure how much noise was added (Figure~\ref{fig:aleatoric_injected})."

## 64. Introduction: "While there have been many investigations into the"
- Flag: booster words ("certainly", "found") and many -tion nouns.
- Old: "The choice of molecular representation certainly affects predictive performance."
- New: "The choice of molecular representation affects predictive performance."

## 74. Results: "Uncertainty estimation breaks under censoring as the process"
- Flag: booster word ("actually") and three subordinate clauses in 44 words.
- Old: "Uncertainty estimation breaks under censoring as the process of clipping the labels narrows their spread, which in turn shrinks the total uncertainty. None of the models tested treat a clipped label as a noisy one, so they actually grow more confident while losing information."
- New: "Uncertainty estimation breaks under censoring. Clipping narrows the spread of the labels, which shrinks the total uncertainty. None of the models tested treats a clipped label as a noisy one, so they grow more confident while losing information."

## 81. Conclusions: "These results follow from the loss function each"
- Flag: low lexical density, 52.9% against a reference mean of 62.7%.
- Old: "The GP is the most affected by error that is correlated with structure, because its kernel measures structure."
- New: "Error correlated with structure affects the GP most, because its kernel measures structure."

## 84. Conclusions: "In this research, we define a noise level"
- Flag: low lexical density, 52.5%. The paragraph is all limitations and method background, which pushes the conclusions' limitation share above the references. It also addresses the reader as "you".
- Old: "You can also use screening data to test for row, column and plate effects, which produce systematic rather than independent error \citep{Caraus2015}. When you lack controls, replicates, and metadata, a label can be checked against the data itself."
- New: "Screening data can be tested for row, column and plate effects, which produce systematic error \citep{Caraus2015}. Without controls, replicates or metadata, each label can be checked against the data itself."

## 91. Results: "While this tells us that the models are"
- Flag: booster words ("certain", "know") and "the X of the Y" chains.
- Old: "While this tells us that the models are reportedly less certain under noisy data, we don't know what the model is attributing this uncertainty to."
- New: "The models report less certainty on noisy labels, but this does not show what they attribute it to."

## 103. Introduction: "The work in this chapter is adapted from"
- Flag: low lexical density, 51.6% against a reference mean of 61.5%.
- Old: "Most of the figures, tables, and text in this chapter are therefore either identical or similar to the content of that manuscript."
- New: "Most of the figures, tables and text in this chapter are therefore identical or close to that manuscript."

## 110. Results: "While it is clear that QSAR model performance"
- Flag: 18 numbers and low lexical density, 48.2% against a reference mean of 60.3%. The worklist suggests moving the retained fractions into a table.
- Option A, a table: one row per dataset. Columns would be the lowest retained share of clean $R^2$ at noise levels 0.3, 0.5 and 1.5 with ECFP4, plus the number of the 13 base models below 90% at 0.3. Only the values already in this paragraph would be used. Then point to it.
- Option B, a smaller edit:
  - Old: "While it is clear that QSAR model performance drops with label noise, different models drop faster than others."
  - New: "Models lose performance under label noise at different rates."

## 124. Methods: "ANOVA relies on the assumption that observations both"
- Flag: 26 numbers in 394 words, 6.6 per 100 words.
- The counts of matching comparisons (out of 24) for each pair of models are not in Additional file app:noise:sec:model_redundancy. That item holds a Spearman table of noise degradation slopes, not these counts. So a small table in the chapter is the option. It would have one row per pair of models, with the number of the 24 comparisons in which they matched.
- Old: "Five families met this criterion: RF with quantile regression forests (QRF; 16 of 24), XGBoost with LightGBM (23 of 24), NN-$\alpha$ with BNN-$\alpha$ (20 of 24), NN-$\beta$ with BNN-$\beta$ (18 of 24), and VBLL-$\alpha$ with VBLL-$\beta$ (20 of 24)."
- New: "Five pairs met this criterion (Table~X): RF with QRF, XGBoost with LightGBM, NN-$\alpha$ with BNN-$\alpha$, NN-$\beta$ with BNN-$\beta$, and VBLL-$\alpha$ with VBLL-$\beta$."

## 128. Results: "As on QM9, the noise distribution does not"
- Flag: booster words ("never", "truly"). The last sentence repeats the measurement-noise point from the opening paragraph of the subsection. It is also a limitation sentence in Results.
- Old: "It's worth remembering that while we are injecting artificial noise, the assay measurements carry their own source of noise whose properties are unknown, so no ``clean'' label is truly clean."
- New: cut the sentence. The subsection's first paragraph already says the injected noise adds to an unknown noise floor.

## 141. Results: "Since Gaussian, Laplace, Student-$t$ and grouped-shifted conditions add"
- Flag: low lexical density (50.0%) and 14 subordinate clauses. The last sentence does not parse.
- Old: "Ironically, per-molecule uncertainty can be used in reverse to identify clipped labels as the labels a model is more certain about, especially when they fall in a certain range, are most likely to have been clipped."
- New: "Per-molecule uncertainty can therefore be used in reverse. The labels a model is most certain about, especially within a narrow range, are the most likely to have been clipped."

## 149. Results: "We include two terms which stem from the"
- Flag: low lexical density (50.0%) and short, chopped sentences.
- Old: "We include two terms which stem from the data rather than the model, the residual and replicate terms."
- New: "Two terms come from the data rather than the model: the replicate and the residual."
