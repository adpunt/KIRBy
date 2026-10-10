# Ch4 supervisor comments mapped to current thesis.tex

Chapter is now `\chapter{Limitations of the similar property principle}\label{chap:flaws}`, thesis.tex lines 925-1260 (read 2026-10-10).
Every paragraph in the chapter is hers except the last row of Table `tab:flaws:requirements` (line 1254, AI marker at 1253).
Edits to her text add sentences only where he asked. Added sentences are marked `[added]` here; strip the tag when applying.
All old_strings were checked: each occurs exactly once in thesis.tex.

Edits that touch the same sentence are combined in one block, and the other comments point to it:
- line 973: one edit, under TH-p90d (covers TH-p90b, TH-p90c, TH-p92a).

---

### TH-p87a
Comment: This introduction paragraph is not easy to read (at least for a french scientist ;-) ) and should clearly say why we look a features/similarity in the context of noisy data.
Current location: the chapter abstract he marked (old p87 y=341-478) no longer exists. The introduction is now lines 929-939, `\section{Introduction}` under chap:flaws.
Owner: hers
Status: PARTLY
Reasoning: The abstract is gone. Line 929 now links similarity to noise in one clause ("This comes into play when identifying noisy molecules that in theory stick out from their neighbors."), but it never says why features matter for noisy data. One added sentence after that clause states it.
Edit:
```
old_string:
This comes into play when identifying noisy molecules that in theory stick out from their neighbors.
```
```
new_string:
This comes into play when identifying noisy molecules that in theory stick out from their neighbors. [added] A label is judged against the labels of its nearest molecules, so the features that decide which molecules are nearest also decide which labels look noisy.
```
Optional (her call): start a new paragraph at "There are some important exceptions to the SPP". Line 929 is 14 sentences long, and that is where it moves from the principle to its exceptions.

### TH-p87b
Comment: Nice paragraph
Current location: line 929 (Introduction, chap:flaws). The praised paragraph started "One of the fundamental assumptions in cheminformatics..." (old p87 y=543-761).
Owner: hers
Status: PRAISE
Reasoning: The paragraph survives in line 929, but it was changed. Three sentences about earlier chapters and noise were added at its start. The Bajusz sentence now names the eight coefficients. The closing "This chapter questions the underlying representation" became "In this chapter we ask if the standard methods of comparing molecules satisfy the SPP...".
Edit: none.

### TH-p88a
Comment: Similarity is in the eyes of the beholder ;-)
Current location: both highlighted sentences survive word for word. "Different representations also disagree about which molecules are close..." is at line 931. "None of these measures ask specifically about whether the specific features..." is at line 933.
Owner: hers
Status: PRAISE
Reasoning: This is an agreeing remark, not a request. The text it marks is unchanged.
Edit: none.

### TH-p88b
Comment: General comment: It is worth distinguishing between static task agnostic representations (e.g. ECFP) and representations learned against the target task (e.g trained NN embedding layer). We don't expect the former to necessarily define a smooth property landscape since they were not fine tuned for that. The latter however should create a smoother landscape with respect to the task and activity cliff would be real provided the global performance of the model is reasonable.
Current location: line 935 (Introduction, the paragraph on how much the representation changes similarity). The thesis never makes this distinction.
Owner: hers
Status: APPLIES
Reasoning: Every learned embedding in this thesis is frozen. Table `tab:reps_smallmol` (lines 1730-1737) lists GROVER, MHG-GNN, GraphMVP, MoLFormer, SMI-TED, ChemBERT, ChemBERTa-2 and SELFormer as frozen (verified in thesis.tex). So all of them are task-agnostic in his sense. The added sentences state that and give his expectation as an expectation, not a result. No citation backs the second sentence. Add one if she has it, or keep "we would expect".
Edit:
```
old_string:
only CLAMP was significantly better \citep{Praski2025}.
```
```
new_string:
only CLAMP was significantly better \citep{Praski2025}. [added] These comparisons, like this thesis, use representations fixed before the task is known, such as ECFP or a frozen pretrained embedding (Table~\ref{tab:reps_smallmol}). [added] Such a representation was never tuned to the property, so it need not give a smooth property landscape. [added] We would expect a representation learned on the task's labels, such as a network's last hidden layer, to give a smoother one, in which the remaining cliffs are more likely real.
```

### TH-p89a
Comment: 'is negatively impacted' ? (the similarity does not inherit the features but their flaw)
Current location: the sentence "If the features within these representations are flawed, the similarity measure inherits them." is gone. Its place is line 958, `\subsection{Flaws in standard representations}\label{sec:flaws:representations}`.
Owner: hers
Status: ALREADY ADDRESSED
Reasoning: Line 958 now reads: "Pairwise comparisons of these representations rely on these irrelevant features." It no longer says similarity inherits the features. The other two uses of "inherits" have correct objects. Line 937 has "inherits this choice", and line 1080 has "inherits the dilution described here".
Edit: none.

### TH-p89b
Comment: 'For instance, each...'
Current location: the sentence "Each PDV feature is a named physicochemical property, so its meaning is known, whereas the features of a frozen embedding ... do not say what chemical information, if any, they carry." is no longer in thesis.tex. A search for "named physicochemical" returns nothing.
Owner: hers
Status: GONE
Reasoning: The passage was cut, so his wording fix has nothing to apply to. One effect for her to know about: the last row of Table `tab:flaws:requirements` (line 1254), "Features that are hard to read chemically", now has no "(where shown)" pointer, and no section of the chapter shows this flaw any more. If she wants that row to point at something, she could restore the sentence at the end of line 958 with his wording: "For instance, each PDV feature is a named physicochemical property, ...".
Edit: none (optional restore above is her decision).

### TH-p90a
Comment: What is the proportion of 0 (absence) vs 1 (presence) of these values ?
Current location: line 964, `\subsubsection{Constant and near-constant features}\label{sec:flaws:constant}`. "On hERG $K_i$ (1,412 molecules) 1,815 of the 2,048 features are near-constant."
Owner: hers
Status: APPLIES
Reasoning: I counted this in this session from `/Volumes/seagate/kirby_cache/rep_cache/herg_ki/ecfp4.npy`, the same file `scratchpad/thesis_code_todo/constant/count_constant.py` reads. That file gives 1,815 near-constant bits, which matches the thesis. 1,812 of the 1,815 are off in at least 95% of the 1,412 molecules, and 3 are on in at least 95%. All 105 constant bits are off (`constant_off: 105` in `constant/constant_counts.json`). On QM9, all 1,997 near-constant bits are mostly off (same count, `rep_cache/qm9/ecfp4.npy`). No script in the repo saves the 1,812/3 split. She may want it added to count_constant.py before citing it.
Edit:
```
old_string:
That jumps to 1,997 out of 2,048 near-constant features for the HOMO--LUMO gap on QM9.
```
```
new_string:
That jumps to 1,997 out of 2,048 near-constant features for the HOMO--LUMO gap on QM9. [added] Almost all of these bits are near-constant because they are absent: on hERG $K_i$, 1,812 of the 1,815 are off in at least 95\% of molecules, and 3 are on.
```

### TH-p90b
Comment: '..depends on the distance metric' ?
Current location: line 973 (end of sec:flaws:constant). The old sentence "Which count matters depends on the distance." is gone. The paragraph now opens "Some distances ignore constant features."
Owner: hers
Status: APPLIES
Reasoning: His wording works as the opening of the paragraph. It is included in the combined line-973 edit under TH-p90d.
Edit: see TH-p90d.

### TH-p90c
Comment: Which standardization here? Is the standardization changing the value of the feature ? How?
Current location: line 952 (`\subsection{Distance metrics for continuous molecular descriptors}\label{sec:flaws:methods}`): "All features are standardized to zero mean and unit variance prior to distance computation." Line 973 uses the result without giving the formula.
Owner: hers
Status: APPLIES
Reasoning: This is z-scoring. Each value of a feature is replaced by (value minus feature mean) divided by feature standard deviation, so yes, the values change. Verified for this chapter's four measurements in `scratchpad/thesis_code_todo/distance/compute.py:63-66`. There the mean and the population standard deviation (`A.std(0)`) are taken over all molecules of the dataset, and features with standard deviation 0 are dropped. That script does not standardize binary features. They go to Tanimoto (`compute.py:45-61`, Tanimoto branch). Unverified: the standardization code of the exploratory sweep of noise-handling methods, which line 952 also covers. Its code is not on this machine (see the TODO at thesis.tex:1189).
Edit:
```
old_string:
All features are standardized to zero mean and unit variance prior to distance computation.
```
```
new_string:
All features are standardized to zero mean and unit variance prior to distance computation. [added] Each value $x$ of a feature is replaced by $(x-\mu)/s$, where $\mu$ and $s$ are that feature's mean and standard deviation over the molecules.
```

### TH-p90d
Comment: This paragraph is difficult to read. What is the trasformation from 1 -> 4.36 ?
Current location: line 973 (end of sec:flaws:constant). The two worked numbers (4.36 and 37.56) were cut, so the paragraph now asserts "near-constant features are amplified" without showing how.
Owner: hers
Status: APPLIES
Reasoning: For a 0/1 bit that is on in a share p of molecules, the mean is p and the population standard deviation is sqrt(p(1-p)). An "on" value of 1 therefore becomes (1-p)/sqrt(p(1-p)) = sqrt((1-p)/p). For p = 0.05 that is sqrt(19) = 4.36. For one molecule in 1,412 it is sqrt(1411) = 37.56. Both numbers are from the old build and check exactly against the formula in `compute.py:66`. A bit on in half the molecules becomes ±1. This edit also covers TH-p90b (opening sentence) and TH-p92a (closing take-home sentence). It keeps all of her sentences and adds the arithmetic.
Edit:
```
old_string:
Some distances ignore constant features. Tanimoto similarity (Equation~\ref{eq:tanimoto}; the distance is $1-T$) ignores bits off in both molecules, but a bit on in every molecule raises every molecule's measured similarity. It does not use the 105 constant ECFP4 bits on hERG $K_i$. Constant features add zero to a Euclidean or Manhattan distance; however, near-constant features are amplified. That is useful if those features define an AC. If not, they are contributing disproportionately to the measure of similarity.
```
```
new_string:
Which count matters depends on the distance metric, and some metrics ignore constant features. Tanimoto similarity (Equation~\ref{eq:tanimoto}; the distance is $1-T$) ignores bits off in both molecules, but a bit on in every molecule raises every molecule's measured similarity. It does not use the 105 constant ECFP4 bits on hERG $K_i$, which are all off. Constant features add zero to a Euclidean or Manhattan distance; however, near-constant features are amplified. [added] Standardization (Section~\ref{sec:flaws:methods}) turns a bit that is on in a share $p$ of the molecules into $\sqrt{(1-p)/p}$ where it is on. [added] A bit on in half of the molecules becomes 1, a bit on in 5\% becomes 4.36, and a bit on in one of the 1,412 hERG $K_i$ molecules becomes 37.56. [added] Two molecules that differ only in that rare bit are therefore far apart, although they differ in one substructure. That is useful if those features define an AC. If not, they are contributing disproportionately to the measure of similarity. [added] In short, the features that vary least between molecules carry the most weight in a standardized distance.
```
Note for her: in this chapter's own measurements (line 1060), binary features go to Tanimoto and are never standardized. The arithmetic applies to near-constant continuous features (Mordred, PDV), and to any sweep configuration that put a Euclidean or Manhattan distance on ECFP4 (unverified whether the sweep standardized those bits).

### TH-p92a
Comment: This section 4.2.1 needs to be simplified so that the reader understand easily the impact of conste/near-constant feature on the similarity metric
Current location: sec:flaws:constant, lines 960-973. The section is shorter than the build he read. The two count tables (old Tables 4.1 and 4.2, with their TODO) have moved to Additional file `app:add:constant_features`. What is left is line 962 (definitions), line 964 (counts) and line 973 (effect on distance).
Owner: hers
Status: APPLIES
Reasoning: The cut of the tables already simplified the section. What it still lacked was a plain statement of the effect on the distance. The combined edit under TH-p90d adds the worked numbers and closes with a one-sentence take-home: "In short, the features that vary least between molecules carry the most weight in a standardized distance." The highlight he placed here (old p93, "Any representation read from the 2D graph merges stereoisomers") belongs to TH-p93a.
Edit: see TH-p90d.

### TH-p93a
Comment: The 2D graph express stereoisomers via stereobonds.
Current location: line 979, `\subsubsection{Molecules with identical feature vectors}\label{sec:flaws:ties}`. "Any representation that reads from these subsequently merges stereoisomers."
Owner: hers
Status: APPLIES
Reasoning: He is right. A 2D molecular graph can carry chiral tags on atoms and E/Z labels on double bonds. Stereoisomers are merged because these representations do not read those labels, not because the graph lacks them. Verified for ECFP4: `src/kirby/representations/molecular.py:365` calls `GetMorganGenerator(radius=radius, fpSize=n_bits)` with no `includeChirality`, so RDKit's default (off) applies. That PDV and Mordred ignore stereo is the thesis's existing claim; I did not check it in code.
Edit:
```
old_string:
Any representation that reads from these subsequently merges stereoisomers.
```
```
new_string:
A 2D graph can store stereochemistry as labels on atoms and bonds, but these representations do not read those labels, so they merge stereoisomers.
```

### TH-p93b
Comment: Reader will be frustrate to not know what is 'more', different sources, embedding collision, ...? You go in more details next but this sentence finishes a bit brutally :-) )
Current location: line 979 (sec:flaws:ties). "...56 of which differ only in stereochemistry and 49 in more than that." The sentence is unchanged from the build he read.
Owner: hers
Status: APPLIES
Reasoning: "More" means the 49 have different InChIKey connectivity blocks. They are different 2D structures, not repeats (line 983 rules repeats out). The WL features label each atom by its element only, ignore bond orders, and drop labels found in fewer than 5% of molecules. Verified in `src/kirby/representations/molecular.py:4405-4484`: node label `atom.GetSymbol()`, edges unlabeled, `sparsity_threshold=0.95`. I split the 49 in this session from `scratchpad/thesis_code_todo/ties/ties_results.json`. 37 share a heavy-atom graph once bond orders, aromaticity, charges and hydrogens are removed. 12 have different heavy-atom graphs, for example a spiro ring joined at a different atom. No repo script saves this split. The second added sentence needs her sign-off on these two counts, or it can be dropped and the first sentence kept alone. Which of the 12 fail because of the 1-WL limit and which because of the 5% label filter was not checked.
Edit:
```
old_string:
56 of which differ only in stereochemistry and 49 in more than that.
```
```
new_string:
56 of which differ only in stereochemistry and 49 of which are different 2D structures. [added] These WL features label each atom by its element only, ignore bond orders, and keep only labels found in at least 5\% of molecules (Table~\ref{tab:reps_smallmol}). [added] Of the 49, 37 have the same heavy-atom graph and differ only in bond orders, charges or hydrogens, and 12 have different heavy-atom graphs.
```

### TH-p94a
Comment: This is the lower bound for model RMSE based on the original labels not the reassigned group mean label I assume ? How do you differentiate the training/test data in this lower bound assumption (the RMSE only plies to the test/validation data)
Current location: lines 987-991 (sec:flaws:ties), the RMSE_min equation and "The matching best $R^2$ is ...".
Owner: hers
Status: APPLIES
Reasoning: His first guess is right. The bound uses the measured labels against each group's mean. Verified in `scratchpad/thesis_code_todo/ties/ties.py:37` (`sse += ((yg - yg.mean())**2).sum()`) and `:54` (`sqrt(sse / n)`). It is computed once over all molecules of the dataset, with no training/test split. So it is an in-sample floor, and the text should say so. On a held-out fold, the corresponding floor is the same sum over that fold's tied molecules around their own mean. Possible caveat (unverified): `tests/find_best_hybrid_molecular.py:66-67` writes the Murcko scaffold with RDKit's default isomeric SMILES. If that function makes the folds of the later chapters, stereoisomers with a stereocentre inside the scaffold can fall in different folds. That would contradict "chirality-blind" at thesis.tex:418 for those chapters. I did not check which scaffold function makes the folds.
Edit:
```
old_string:
The matching best $R^2$ is $1 - \mathrm{RMSE}_{\min}^2 / \mathrm{Var}(y)$.
```
```
new_string:
The matching best $R^2$ is $1 - \mathrm{RMSE}_{\min}^2 / \mathrm{Var}(y)$. [added] Both use the measured labels of all molecules in the dataset, with no split into training and test molecules. [added] They are a floor on error over the whole dataset; on one held-out fold, the floor is the same sum over that fold's tied molecules.
```

### TH-p94b
Comment: You mean a repeat is based on a InChI key with stereo-chemistry on worth specifiying) and the repeat implies the InChI key was computed with all stereo-information. In this paragraph it could be inferred that there are only 3 types of ties: 'repeat' 'failure' and 'distinct', isn't there a 'merge' (stereo-chemistry merge) ?
Current location: line 983 (sec:flaws:ties). The three-category paragraph he read was cut. What remains: "No tie came from one compound listed twice, as confirmed by InChIKey, which differs between stereoisomers, or from a row of zeros."
Owner: hers
Status: PARTLY
Reasoning: The current sentence already says the InChIKey differs between stereoisomers, which answers his first point. It still does not name the stereo merge as its own kind. Verified in `ties.py:20-21,43-51`. A repeat is the same full standard InChIKey (`Chem.MolToInchiKey`, which carries the stereo layer). A failure is an all-zero vector. A stereo-only merge is the same first InChIKey block, which encodes connectivity only. Everything else is a merge of different structures. Table `tab:flaws:ties` already reports the stereo-only count.
Edit:
```
old_string:
No tie came from one compound listed twice, as confirmed by InChIKey, which differs between stereoisomers, or from a row of zeros.
```
```
new_string:
A tie can be a repeat of one compound (same full InChIKey, which includes stereochemistry), a failed featurization (all zeros), a merge of stereoisomers (same first InChIKey block, which encodes connectivity only), or a merge of different structures. No tie was a repeat or a row of zeros.
```

### TH-p95a
Comment: Does it make sense to measure the SALI index inside a tie group ? If we measure the index prior to group normalisation of the label we would measure a mix real activity-cliff (e.g. stere-effect) and measurement variability.
Current location: lines 1002-1006 (sec:flaws:ties), the SALI equation and "...is infinite when their similarity is 1 and their labels differ, and here the cliff is made by the representation rather than the chemistry."
Owner: hers
Status: APPLIES
Reasoning: The thesis computes no SALI value. It only uses the formula to show that a tied pair gives an infinite index. His point holds against the phrase "the cliff is made by the representation rather than the chemistry". Inside a tied group, the label gap can be a real stereo effect, a gap between repeat measurements, or both. Only the infinite index comes from the representation. Line 993 already gives the repeat-measurement scale (0.54, Kramer 2012), so the edit can point to it.
Edit:
```
old_string:
and here the cliff is made by the representation rather than the chemistry.
```
```
new_string:
and here the infinite index is made by the representation. [added] The label gap itself may be a real effect of stereochemistry, a gap between repeat measurements of the size given above, or both, and the index cannot separate them.
```

### TH-p97a
Comment: Slightly different for representation learned against the target task.
Current location: line 1049 (`\subsection{Flaws in similarity and other pairwise computations}\label{sec:flaws:similarity}`). "A similarity coefficient cannot tell you which features matter to the prediction task at hand."
Owner: hers
Status: APPLIES
Reasoning: Same point as TH-p88b, applied to this sentence. A short qualifier is enough here, since the full statement goes into line 935 (TH-p88b).
Edit:
```
old_string:
A similarity coefficient cannot tell you which features matter to the prediction task at hand.
```
```
new_string:
A similarity coefficient cannot tell you which features matter to the prediction task at hand, unless the representation itself was trained on that task's labels.
```

### TH-p98a
Comment: It would be nice to have a figure that summarizes the experiment so that the reader can easily understand where the 70 combinations come from.
Current location: lines 1060 and 1068-1070 (`\subsubsection{Distance and label difference}` in sec:flaws:similarity). "Across the 70 combinations of dataset and representation, ..." The text never says that 70 = 7 × 10.
Owner: hers
Status: NEEDS HER
Reasoning: The figure is her call. A simple schematic would work: seven dataset boxes, each with its ten base representations, feeding two measurements. Pending the figure, one phrase fixes the arithmetic. Verified in `scratchpad/thesis_code_todo/distance/results_all.json`: each of the seven datasets has exactly 10 base representations, 70 in all. On HLM clearance, ECFP4 is an eleventh, extra entry, and it is the dagger row in Table `tab:flaws:distance_summary`.
Edit (interim, without figure):
```
old_string:
Across the 70 combinations of dataset and representation,
```
```
new_string:
Across the 70 combinations of dataset and representation, seven datasets with ten base representations each,
```

### TH-p99a
Comment: It would be good to finish this resulting listing with a take home message for each representation otherwise the reader is loosing track of what is the goal. e.g. closer compound have closer labels => similarity principle etc. The reader is flooded with number for different type of measures and it needs simple (one sentence) recap/summary from time to time ;-) For instance do these numbers mean ECFP4 is following better the similariy principle than Mordred?
Current location: line 1072 (end of the hERG pair paragraph, `\subsubsection{Distance and label difference}`), plus the subsubsections at lines 1062-1065 (activity cliffs), 1074-1075 (neighbors) and 1077-1078 (MCS). None of them ends with a take-home sentence.
Owner: hers
Status: APPLIES
Reasoning: Four short added sentences, one at the end of each subsubsection, all restating what the numbers already in the text show. His Mordred question has an answer from Table `tab:flaws:distance_summary`. Neither representation follows the principle better everywhere. On hERG $K_i$ both reach 0.72 with five neighbors. The descriptor set is higher on LogD, Caco-2 and the three Biogen datasets, and ECFP4 is higher on QM9 (0.88 against 0.84). The all-pairs values are near zero for both.
Edits (four):
```
old_string:
The all-pairs correlation is set by the other 95\%, whereas the five-neighbor prediction uses only the closest pairs.
```
```
new_string:
The all-pairs correlation is set by the other 95\%, whereas the five-neighbor prediction uses only the closest pairs. [added] So the SPP holds for each molecule's closest neighbors but not across all pairs. [added] Neither ECFP4 nor the descriptor set follows it better everywhere: both reach 0.72 on hERG $K_i$, the descriptors are higher on four datasets and ECFP4 on QM9 (Table~\ref{tab:flaws:distance_summary}).
```
(Check: "four datasets" is LogD, Caco-2, RLM clearance, MDR1--MDCK efflux. HLM clearance is a fifth if the dagger ECFP4 counts. If she prefers no count, name them instead.)
```
old_string:
and on QM9 only 1\% to 24\% (Section~\ref{app:add:flaws} of the Additional files).
```
```
new_string:
and on QM9 only 1\% to 24\% (Section~\ref{app:add:flaws} of the Additional files). [added] So which molecules count as cliffs depends on the representation as much as on the labels.
```
```
old_string:
(Table~\ref{tab:flaws:neighbor_overlap}).
```
```
new_string:
(Table~\ref{tab:flaws:neighbor_overlap}). [added] So a neighbor-based method that switches representation compares each label with a largely different set of neighbors.
```
```
old_string:
to 0.51 (Caco-2 and LogD).
```
```
new_string:
to 0.51 (Caco-2 and LogD). [added] ECFP4 distance therefore agrees only partly with how much structure two molecules share.
```

### TH-p99b
Comment: What is the intermediate conclusion/ take home message of this section?
Current location: line 1080, the closing sentence of sec:flaws:similarity: "Every method in Section~\ref{sec:flaws:consequences} that takes a similarity as input inherits the dilution described here."
Owner: hers
Status: APPLIES
Reasoning: The section ends on a pointer forward, not on a conclusion. Two added sentences before it state the conclusion in plain words. They use no new numbers.
Edit:
```
old_string:
Every method in Section~\ref{sec:flaws:consequences} that takes a similarity as input inherits the dilution described here.
```
```
new_string:
[added] In short, close molecules do tend to have close labels, but over all pairs distance barely follows label difference. [added] Representations also disagree on which molecules are close and which are cliffs. Every method in Section~\ref{sec:flaws:consequences} that takes a similarity as input inherits the dilution described here.
```

### TH-p101a
Comment: At this point you have presented a lot of metrics/comparisons. It would be good to have a short recap/summary of the type of flaws and their origin to prepare the reader for the next section.
Current location: line 1142, first paragraph of `\subsection{Consequences for similarity-based methods}\label{sec:flaws:consequences}`. It opens "In this chapter, we have identified some issues within the representations and similarities computed over them."
Owner: hers
Status: APPLIES
Reasoning: The opening sentence names "some issues" without listing them. Replacing it with a three-flaw recap answers him directly. Her next sentence ("While this calls into question ...") still follows on. Every claim is already shown in sections 960-1080, and no new numbers are added.
Edit:
```
old_string:
In this chapter, we have identified some issues within the representations and similarities computed over them.
```
```
new_string:
[added] So far this chapter has shown three flaws, each with its own origin. [added] Most features of a general-purpose representation barely vary on one dataset, because the representation was built for all of chemistry. [added] Several representations give different molecules, mostly stereoisomers, one feature vector, mostly because they ignore stereochemistry. [added] A distance sums over every feature, so over all pairs it barely follows label difference, and representations disagree about which molecules are neighbors.
```

### TH-p102a
Comment: It would be nice to have a visual but only if you have time (not critical)
Current location: the sweep of 12 noise-handling methods, lines 1153-1187 (`\subsubsection{A small sweep of published noise-handling methods}\label{sec:lnl_motivation}`). Table `tab:flaws:lnl_methods` (lines 1156-1181) is new since his build, but there is no figure.
Owner: hers
Status: NEEDS HER
Reasoning: A figure needs the raw sweep results, for example retention against σ per method, coloured by whether a distance is used. The TODO at thesis.tex:1189 says those results are not on this machine. A search of the repo for DivideMix result files found none. He marked it not critical.
Edit: none.

### TH-p102b
Comment: Just out of curiosity, did you observe cases of retention ratio > 1 (better performance with noisy data ?, I expect not but was curious)
Current location: lines 1184-1187 (sec:lnl_motivation, retention results).
Owner: hers
Status: NEEDS HER
Reasoning: He asks a question for her to answer, not a change to the text. The answer needs the per-configuration retentions of the sweep, which are not on this machine (thesis.tex:1189 TODO).
Edit: none.

### TH-p102c
Comment: Retained from ..to ...? Does it mean between sigma =0 and sigma =1 (or 0.6) ? What is the % measuring?
Current location: lines 1184-1187 (sec:lnl_motivation). "We measured noise robustness by retention, $R^2$ under noise divided by $R^2$ with clean labels, as a percentage." Then "DivideMix ... retained the most, 86.7\% ...".
Owner: hers
Status: PARTLY
Reasoning: The definition is there. What it does not say is which σ the noisy $R^2$ comes from for DivideMix, small-loss selection and the uncertainty curriculum. Line 1187 gives σ = 0.6 for the heteroscedastic GP only, and says the study did not record the level for the others. The added sentence says what the percentage measures. NEEDS HER: whether the 86.7%, 84.2% and 82.7% are at one σ or averaged over σ = 0.1 to 1.0. If she does not know, the text should say "noise level not recorded" next to those three numbers too.
Edit:
```
old_string:
We measured noise robustness by retention, $R^2$ under noise divided by $R^2$ with clean labels, as a percentage.
```
```
new_string:
We measured noise robustness by retention, $R^2$ under noise divided by $R^2$ with clean labels, as a percentage. [added] A retention of 86.7\% means the model kept 86.7\% of the $R^2$ it reached at $\sigma = 0$.
```

### TH-p104a
Comment: Nice: The story is clear here , more noise less recovery and the amplitude of the effect is representation dependant
Current location: Table `tab:code_input_neighbor_recovery`, lines 1206-1223 (`\subsubsection{Noise detection by neighbors}\label{sec:flaws:neighbors}`).
Owner: hers
Status: PRAISE
Reasoning: The table and its caption survive word for word. The six recovery values and the six shares of positive configurations are unchanged.
Edit: none.

### TH-p104b
Comment: We haven't yet explained hybrid representaiotns at this point. Maybe I misunderstood this sentence, it felt confusing.
Current location: the paragraph "The study also found neighbors in an early hybrid of 100 PDV and MHG-GNN features, built by greedy forward selection..." is gone. A search for "early hybrid" in thesis.tex finds nothing.
Owner: hers
Status: ALREADY ADDRESSED
Reasoning: The sentence he found confusing was removed. One trace remains at line 1202: "Table~\ref{tab:code_input_neighbor_recovery} holds only the configurations built on a single representation." It hints that other configurations exist without saying what they were. Optional: delete that sentence, since nothing else refers to them.
Edit: none (optional deletion above).

### TH-p106a
Comment: Nice recap ! We could have investigated how many structure are merged across representation. This give an idea of the potential of breaking ties by using hybrid representations
Current location: the praised recap was the old Conclusions plus Table 4.7. The Conclusions are now lines 1229-1238. The paragraph on five properties of a better representation was cut. Table `tab:flaws:requirements` survives at lines 1241-1260, with "Combining representations and the alternatives" shortened to "Combining representations" and a new AI-written last row (line 1254).
Owner: hers (table row 5: AI)
Status: PRAISE
Reasoning: The table he praised survives. The text that introduced it ("A representation that avoids these flaws would need five properties") is gone, so line 1238 now leads into the table with no sentence pointing to it. His suggestion was already done, in line 985: "On hERG $K_i$, ECFP4, atom pairs, PDV, Mordred and SMI-TED tie the same 56 molecules, so joining any of these five representations leaves all 56 tied ... and joining all of a dataset's representations leaves no molecule tied." If she wants to tell him so, that is the sentence to point to. Unchanged from TH-p89b: row 5's flaw has no "where shown" section any more.
Edit: none.

---

## Counts
- APPLIES: 14 (p88b, p90a, p90b, p90c, p90d, p92a, p93a, p93b, p94a, p95a, p97a, p99a, p99b, p101a)
- PARTLY: 3 (p87a, p94b, p102c)
- ALREADY ADDRESSED: 2 (p89a, p104b)
- GONE: 1 (p89b)
- NEEDS HER: 3 (p98a, p102a, p102b)
- PRAISE: 4 (p87b, p88a, p104a, p106a)
Total 27.

## Numbers I introduced that are not in thesis.tex (each needs her sign-off)
- 1,812 off and 3 on, of the 1,815 near-constant ECFP4 bits on hERG $K_i$. Counted this session from `/Volumes/seagate/kirby_cache/rep_cache/herg_ki/ecfp4.npy`; no saved script.
- 4.36 and 37.56. From the old build; checked against sqrt((1-p)/p) with p = 0.05 and p = 1/1412.
- 37 and 12 of the 49 WL ties. Computed this session from `scratchpad/thesis_code_todo/ties/ties_results.json` by comparing heavy-atom graphs with bond orders, charges and hydrogens removed; no saved script.
