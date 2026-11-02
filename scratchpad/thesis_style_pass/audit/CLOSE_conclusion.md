# Conclusion-section edits (chapter Conclusions sections and Chapter 8), 2026-10-10

Rows: L = three-item lists per 100 sentences; C = distinct connectives / connectives used; S = sentence over 30 words.
Limitation sentences: no edit. Each chapter already has at most one limitations paragraph in its conclusions, and the Ch8 Limitations section was trimmed by another agent during this pass.

1. Ch2 "A predictor has two parts, the representation and" (S, C): "...and the model, and the representation also shapes what counts as similar: similarity depends on the representation and the coefficient, and activity cliffs and uninformative features break the similar property principle." -> "...and the model. The representation also shapes what counts as similar, since similarity depends on both the representation and the coefficient. Activity cliffs and uninformative features, in turn, break the similar property principle."
2. Ch2 "Real assay noise is often not independent, identically" (L, C): "often not independent, identically distributed or zero-mean, so we defined seven artificial noise conditions." -> "often neither independent nor identically distributed, and it need not be zero-mean. For this reason we defined seven artificial noise conditions."
3. Ch2 "Methods for noisy labels fall into noise detection," (C): "..., and no single method is expected to win on every dataset." -> "... . Whichever group a method belongs to, none is expected to win on every dataset."
4. Ch4 "Several representations also give different molecules, mostly stereoisomers," (C): "Several representations also give" -> "In addition, several representations give"
5. Ch5 "On every small-molecule dataset the hybrid beats the" (C): "and also the representation that was best in hindsight" -> "and it beats even the representation that was best in hindsight"
6. Ch5 "At the same number of features the hybrid" (C): ", which shows that its gain is more than a reduction in width." -> ". Its gain is therefore more than a reduction in width."
7. Ch5 "We found no setting that suits every dataset" (C): "We found no setting ..., so the settings should be searched" -> "Since we found no setting ..., the settings should be searched"
8. Ch5 "The selected features can also be read: on" (C): "The selected features can also be read" -> "Finally, the selected features can be read"
9. Ch5 "On the smaller sets an ensemble of one" (C): "On the smaller sets an ensemble" -> "On the smaller sets, however, an ensemble"
10. Ch5 "We also ran the method on reactions, where" (C): "We also ran the method on reactions, where it gains nothing on the additive split." -> "On reactions the method gains nothing on the additive split."
11. Ch5 "The hybrid also takes in new embeddings added" (C): "The hybrid also takes in" -> "The hybrid can likewise take in"
12. Ch5 "In training, using the importance scores again adds" (C): "In training, using" -> "In training, by contrast, using"
13. Ch6 "Prediction from nearest neighbors improves on all six," (C, S): "..., at the setting fixed before the run and with the setting picked again, and the neighbors retrieved in the hybrid have closer labels." -> "..., both at the setting fixed before the run and with the setting picked again. In addition, the neighbors retrieved in the hybrid have closer labels."
14. Ch6 "The hybrid's distance tracks prediction error worse on" (C): "The hybrid's distance tracks prediction error worse" -> "The hybrid's distance does, however, track prediction error worse"
15. Ch7 "Label smoothing, weighting and changes to the model" (L, C): "Label smoothing, weighting and changes to the model also improved QM9, so their gains ... are largely better training; label smoothing on hERG $K_i$ is the exception." -> "Label smoothing and weighting improved QM9 as well, as did changes to the model. Their gains ... are therefore largely better training, with label smoothing on hERG $K_i$ the exception."
16. Ch7 "Which noise-handling method works best depends on the" (C): "works best depends" -> "works best thus depends"
17. Ch8 "Systematic noise, shared within a scaffold family or" (C): ", so a study that injects only Gaussian noise misses it." -> ". A study that injects only Gaussian noise therefore misses it."
18. Ch8 "The method carried over from small molecules to" (C): "to antibodies, where the gain depended" -> "to antibodies, although there the gain depended"
19. Ch8 "On antibodies and reactions distance also tracked the" (C): "distance also tracked the label better." -> "distance tracked the label better too."
20. Ch8 "A hybrid can also be the input to" (C): "A hybrid can also be the input" -> "Beyond new domains, a hybrid can be the input"
21. Ch8 "Label noise in molecular data can be handled," (L, C): "can be handled, but which method handles it depends on the prediction task, the data, the model and the kind of noise." -> "can be handled. Which method handles it, however, depends on the prediction task and the data, as well as on the model and the kind of noise."
22. Ch8 "No single model, representation or noise-handling method was" (L): "No single model, representation or noise-handling method was best on every dataset we tested." -> "No single model or representation was best on every dataset we tested, and neither was any noise-handling method."
23. Ch8 "A dataset's noise should be understood before a" (C): "should be understood" -> "should therefore be understood"
24. Ch8 "The features also matter: when the noise methods" (C): "The features also matter" -> "The features matter as well"
25. Ch8 "On small molecules it predicted better than any" (C): "..., and it stayed ahead through every noise treatment." -> ". It stayed ahead, moreover, through every noise treatment."
26. Ch8 "A new foundation model appears every few months" (L): "in molecular, protein and antibody modeling." -> "in molecular and protein modeling, antibodies included."

Not edited: lists of dataset names, the Contributions enumerate items, the Ch5/Ch7 limitations paragraphs (lists there name distinct results), Ch3 conclusions (author's text; only the table is marked).
