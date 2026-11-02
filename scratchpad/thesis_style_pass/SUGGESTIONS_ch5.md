# Chapter 5: suggestions for your paragraphs (not edited)

These are your paragraphs that the benchmark worklist ranks in its top 150. None was changed. Each entry gives the first eight words, what the benchmark flags, and one suggested edit. Rank is the worklist rank.

## 1. "The budget is the maximum number of features" (rank 6, Methods)
Flagged: 8 numbers in a 61-word paragraph (numbers per 100 words far above the references).
Suggested edit, pointing to the table that already lists the budgets:
- Old: "taken from eight values (10, 25, 50, 100, 250, 500, 1,000 and 2,000 features)."
- New: "taken from eight values (Table~\ref{tab:search_tiers})."

## 2. "Every allocation method is paired with every count." (rank 10, Methods)
Flagged: 9 numbers; "the X of the Y" chains. The second half of this paragraph ("The full runs show the same split. On LogD ...") is marked for review and was left as it is, because it runs on from your sentence without a blank line.
- Old: "In preliminary runs, some datasets performed best with 2 or 3 representations and others with all 10."
- New: "In preliminary runs, the best representation count differed between datasets."

## 3. "The task of selecting the best features from" (rank 27, Introduction)
Flagged: hedge ("should"); "the X of the Y" chains.
- Old: "The task of selecting the best features from a set of representations and combining them is not simple."
- New: "Selecting and combining the best features from several representations is not simple."

## 4. "Another caveat of this setup is that the" (rank 51, Introduction)
Flagged: 7 subordinate clauses in 3 sentences; "Additionally" is on your AI-phrasing list.
- Old: "Additionally, if there is only a single base representation that carries signal, there is nothing for the hybrid to improve upon, which is shown in Section~\ref{sec:method:results_single}."
- New: "If only one base representation carries signal, the hybrid has nothing to improve upon (Section~\ref{sec:method:results_single})."

## 5. "While a set of thousands of features may" (rank 57, Introduction)
Flagged: 9 subordinate clauses; hedges ("may", "could").
- Old: "While a set of thousands of features may seem overwhelming and irrelevant, identifying substructures or descriptors that recur among the selected features can point to parts of the molecule that may be involved in binding or activity."
- New: "Even among thousands of selected features, substructures or descriptors that recur can point to parts of the molecule that may be involved in binding or activity."

## 6. "Any representation that gives one fixed-length vector per" (rank 63, Methods)
Flagged: booster ("never"); nominalizations.
- Old: "and is never trained on the labels."
- New: "and is not trained on the labels."

## 7. "The input to the search is $K$ frozen" (rank 66, Methods)
Flagged: two sentences over 30 words (the longest is 67); hedges ("rather", "presumably").
- Old: "It returns a hybrid of at most $B$ features before any augmentation features added outside the budget (Section~\ref{sec:method:augmentation}), where the budget $B$ is a maximum rather than a target, and a blueprint that records the settings necessary to rebuild the hybrid representation on another set of features, presumably another subset within the same dataset."
- New: "It returns a hybrid of at most $B$ features, not counting augmentation features added outside the budget (Section~\ref{sec:method:augmentation}). It also returns a blueprint that records the settings needed to rebuild the hybrid on new samples, such as another subset of the same dataset."
- Check: the old sentence says "on another set of features". The blueprint is rebuilt on other samples, so "samples" may be the word you meant. Please confirm.

## 8. "Not every molecular representation is equal. In fact," (rank 82, Introduction)
Flagged: boosters ("in fact", "we know"); 4 numbers (introductions in the references carry about 0.5 per 100 words).
- Old: "Not every molecular representation is equal. In fact, they hold very different types of information."
- New: "Molecular representations hold very different types of information."
- Also, old: "We know that the choice of representation matters, as is discussed in Chapter~\ref{chap:robustness}, yet" -> new: "The choice of representation matters (Chapter~\ref{chap:robustness}), yet".

## 9. "The most straightforward method of combining representations is" (rank 105, Introduction)
Flagged: boosters ("found", "showed", "consistently"); short sentences that each carry one fragment.
- Old: "showed that the additional features consistently improved predictive accuracy over the original fingerprints across a range of models."
- New: "reported that the added features improved predictive accuracy over the original fingerprints across a range of models."

## 10. "Let $S_k$ be the top $b_k$ features of" (rank 112, Methods)
Flagged: "the X of the Y" chains; hedge ("would").
- Old: "The cut exists because tree importance gives exactly 0 to every feature no tree split on, and filling the budget with those would pick them by their position in the matrix."
- New: "Tree importance gives exactly 0 to every feature no tree split on. Filling the budget with those would pick them by their position in the matrix."

## 11. "Depending on the origin of the feature, some" (rank 119, Introduction)
Flagged: "the X of the Y" chains; fragment sentences.
- Old: "This feature-level explainability fails when features are derived from learned embeddings. That will only tell you if the learned embedding itself is useful as a representation, but nothing about the selected feature."
- New: "This fails for features of learned embeddings. A selected embedding feature shows that the embedding is useful and says nothing about what the feature means."

## 12. "The first method of feature allocation we tried" (rank 125, Methods; a fix note sits above it, so it is treated as yours)
Flagged: 8 numbers; booster ("very").
- Old: "In practice, this is very expensive. With blocks of 10 features and a budget of 500 and 10 representations, it needs up to 500 model fits per configuration and fold against 10 for a score-based method."
- New: "In practice it is expensive: with blocks of 10 features, a budget of 500 and 10 representations, it needs up to 500 model fits per configuration and fold, against 10 for a score-based method."
- Optional: the Spearman pair (0.3600 against 0.4526) could move to Additional file~\ref{app:add:design_experiments}, if that file already carries it. Not checked.
