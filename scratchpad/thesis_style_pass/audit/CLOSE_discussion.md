# CLOSE: discussion section type (Chapter 8, AI-marked paragraphs)

1. "This thesis set out to handle label noise" -- link/transition.
   old: "This thesis set out to handle label noise in molecular data. This chapter answers the research questions of Chapter~\ref{chap:intro}, then sets out ..."
   new: "This chapter answers the research questions of Chapter~\ref{chap:intro} on handling label noise in molecular data, then sets out ..."

2. "On QM9 under Gaussian noise, model family explained" -- interpretation (added after it), sentence-length SD.
   old: (no sentence)
   new: "Label noise corrupts the labels and leaves the features untouched, so this split is what one would expect." (from Ch3's own reasoning: label noise does not impact the features)

3. "Second, random forests were the most consistent model" -- comparison -> interpretation; connective "while" removed; sentence length.
   old: "Second, random forests were the most consistent model under noise, while LightGBM and XGBoost predicted well on clean labels and lost the most as noise rose."
   new: "Second, random forests were the most consistent model under noise. LightGBM and XGBoost predicted well on clean labels and lost the most as noise rose, so a model's score on clean labels says little about how it holds up under noise." (Ch3: clean R^2 vs AUCnorm Spearman -0.10)

4. "Chapter~\ref{chap:flaws} then asked why published methods" -- link/transition (question sentence merged with its answer; "then" removed).
   old: "Chapter~\ref{chap:flaws} then asked why ... exploratory study. The methods that compared molecules by a distance did worst."
   new: "Chapter~\ref{chap:flaws} asked why ... exploratory study, and found that those comparing molecules by a distance did worst."

5. "A distance sums over every feature, so" -- interpretation (added after it).
   old: (no sentence)
   new: "Any method that relies on such a distance inherits the problem, which the adaptive hybrid representation was built to repair." (Ch1 overview of Chapters 4-6)

6. "The method also carried over from small" -- connectives.
   old: "The method also carried over ..."
   new: "The method carried over ..."

7. "On antibodies and reactions distance also tracked" -- comparison -> interpretation; formulaic opener; sentence length.
   old: "On antibodies and reactions distance also tracked the label better, but prediction from neighbors was worse on several datasets."
   new: "On antibodies and reactions distance also tracked the label better. However, prediction from neighbors was worse on several of these datasets, so the benefit of a hybrid has to be measured on each one." (Ch1 overview of Chapter 6 says the same)

8. "The ranking of the methods also changed between" -- comparison -> interpretation; connective "also" removed.
   old: "The ranking of the methods also changed between the hybrid and the single representations."
   new: "Even the order of the methods changed between the hybrid and the single representations, so a method chosen on one representation may not be the best on another."

9. "We therefore recommend modeling the noise in a" -- connectives.
   old: "We therefore recommend modeling ..."
   new: "We recommend modeling ..."

10. "The DNN began clipping each prediction to the" -- limitation (minor caveat cut; already stated in Chapter 5's correction paragraph, which names it beside each affected result).
   old: "The DNN began clipping each prediction to the range of its training labels only after most of its results were computed."
   new: (removed)

Not changed: Contributions intro and list (list items are marked hers), Future work, Concluding remarks (conclusion type), remaining limitation sentences.
