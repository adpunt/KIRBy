# Plan: Chapter 7 (label chap:noise), style and chemistry audit

Audited against THESIS_STYLE_GUIDE.md and CHEMISTRY_LANGUAGE_GUIDE.md on 2026-10-10. Every OLD block was checked with python3 -I (str.count == 1) against thesis.tex as it stood then; no two OLD blocks overlap. Every paragraph in this chapter carries "% TODO: manual review" above it (AI-written), so most edits are ai-text and keep that existing line. No author text is edited. Numbers were only moved or removed, never added: every number in a NEW block already appears in thesis.tex. [N7-xx] SOURCE comments are outside every OLD block and stay where they are.

Lead-in edits (E-new-paragraph) are anchored on a `\label{...}` that sits at the end of a heading line, so they apply whatever PLAN_titles.md does to the heading text.

## Edits

### E1 — Opening looks back to Chapter 6
- file: thesis.tex
- kind: ai-text
- why: Style guide item 1: B and V open each chapter by naming the previous chapter's finding; this chapter opened with a question and never named Chapter 6.
- todo_line: keep existing
OLD:
```latex
Can we find the noisy labels in a QSAR dataset, and does doing something about them improve a model's held-out score? We ask this on two kinds of labels. The first has damage injected on purpose, so we know which labels were damaged and by how much. The second is the labels as delivered, which already carry assay error that nobody has marked.
```
NEW:
```latex
Chapter~\ref{chap:similarity} showed that, on the five small-molecule datasets, distance in a hybrid tracks the label more closely than distance in a single representation, and that prediction from nearest neighbors improves. This chapter asks whether that helps with label noise. Can we find the noisy labels in a QSAR dataset, and does doing something about them improve a model's held-out score? We ask this on two kinds of labels. The first has damage injected on purpose, so we know which labels were damaged and by how much. The second is the labels as delivered, which already carry assay error that nobody has marked.
```

### E2 — Introduction summary: below chance, split
- file: thesis.tex
- kind: ai-text
- why: Results show the residual detector below chance on Buchwald--Hartwig and LogD, not at chance; one 35-word sentence split (task item 5).
- todo_line: keep existing
OLD:
```latex
Noise detection finds outliers, shifted labels and flipped classes well above chance, and falls to chance on labels clipped at an assay limit. Noise mitigation works, best of all when a known flag says which labels are wrong, and on several datasets it passes random and shuffled controls. Noise-robust training, seed averaging and stacking most of all, helps about as much as the best mitigation, though without a clean-label check we cannot yet say whether that is noise handling or a better model.
```
NEW:
```latex
Noise detection finds outliers, shifted labels and flipped classes well above chance, and falls to or below chance on clipped labels. Noise mitigation works, best of all when a known flag says which labels are wrong, and on several datasets it passes random and shuffled controls. Noise-robust training, seed averaging and stacking most of all, helps about as much as the best mitigation. Without a clean-label check, though, we cannot yet say whether that is noise handling or a better model.
```

### E3 — Define hERG IC50 at first use
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 16 and section 3 (name the thing and gloss it at first use): hERG IC50 is used only in this chapter and was never defined; the label is pIC50 from ChEMBL (checked in src/kirby/datasets/herg.py, load_herg_ic50_measurements: 9 - log10 of the nM value, relation '=').
- todo_line: keep existing
OLD:
```latex
This chapter uses the datasets of Chapters~\ref{chap:method} and~\ref{chap:similarity} (Table~\ref{tab:datasets}), plus one more, hERG IC$_{50}$, which is used only here. QM9 is the clean-label control (Section~\ref{sec:noise:tests}), and hERG IC$_{50}$ is the one dataset with real repeat measurements. Folds and metrics are those of Chapter~\ref{chap:method}.
```
NEW:
```latex
This chapter uses the datasets of Chapters~\ref{chap:method} and~\ref{chap:similarity} (Table~\ref{tab:datasets}), plus one more, hERG IC$_{50}$, which is used only here. Its labels are pIC$_{50}$ values for the hERG channel from ChEMBL. The IC$_{50}$ is the concentration of a molecule that gives half of the maximal inhibition in an assay, and pIC$_{50}$ is its negative base-10 logarithm in molar units. QM9 is the clean-label control (Section~\ref{sec:noise:tests}), and hERG IC$_{50}$ is the one dataset with real repeat measurements. Folds and metrics are those of Chapter~\ref{chap:method}.
```

### E4 — Split sentence on clean-label search
- file: thesis.tex
- kind: ai-text
- why: 31-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Each dataset has two downstream models. In each outer fold, a search on that fold's training molecules ranks each model's two best single representations and two best hybrids. Every method is tested on all four. In most experiments that search sees the clean training labels. Only the hybrid's features are then re-ranked, and its budget re-split between base representations, on the damaged labels. So the hybrid's features move with the damage, but the base representations and the way the hybrid is built are chosen with clean labels, which a real user would not have. There are two exceptions. The expanded published sweep repeats the search for every damage kind, level and draw, trained and scored on the damaged labels alone. The experiment on cleaning before the feature search (Section~\ref{sec:noise:mitigation}) compares cleaning before the search with cleaning after it.
```
NEW:
```latex
Each dataset has two downstream models. In each outer fold, a search on that fold's training molecules ranks each model's two best single representations and two best hybrids. Every method is tested on all four. In most experiments that search sees the clean training labels. Only the hybrid's features are then re-ranked, and its budget re-split between base representations, on the damaged labels. So the hybrid's features move with the damage. The base representations and the way the hybrid is built, however, are chosen with clean labels, which a real user would not have. There are two exceptions. The expanded published sweep repeats the search for every damage kind, level and draw, trained and scored on the damaged labels alone. The experiment on cleaning before the feature search (Section~\ref{sec:noise:mitigation}) compares cleaning before the search with cleaning after it.
```

### E5 — Censoring: say what the limit is and which end is clipped
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide items 12 and 13: "assay limit" does not say which limit, and clipping the top of a potency scale imitates the rarer case; the later sentence "without them a model never sees a high label" depends on this.
- todo_line: keep existing
OLD:
```latex
We damage a regression dataset in six ways (Table~\ref{tab:noise:kinds}). Four are noise conditions from Section~\ref{sec:noise_injection}: Gaussian, outlier, grouped-shifted and censoring at an assay limit. Under outlier noise, a random 5\% of molecules get an error three times wider, where Chapter~\ref{chap:robustness} used 10\%. Every other label still gets the ordinary error, so the 5\% carry about a third of the squared error, not all of it. The last two kinds move only some labels, and move them up. Each raises a random 10\% of the training labels by one fixed amount. In the first, the amount is a multiple of the standard deviation of the clean training labels. In the second, it is a multiple of the standard deviation of a LightGBM model's out-of-fold residual on the clean training labels. So the shift is sized by how far the model already misses. The grouped-wider condition and the Student-$t$ and Laplace shapes of Chapter~\ref{chap:robustness} are not used here.
```
NEW:
```latex
We damage a regression dataset in six ways (Table~\ref{tab:noise:kinds}). Four are noise conditions from Section~\ref{sec:noise_injection}: Gaussian, outlier, grouped-shifted and censoring. Censoring clips labels at a fixed value that imitates an assay's reporting limit, the edge of its measurable range. The clip is at the top of the label range. For potency labels such as hERG $K_i$, real censoring falls mostly at the low end, so the clip imitates the rarer case. Under outlier noise, a random 5\% of molecules get an error three times wider, where Chapter~\ref{chap:robustness} used 10\%. Every other label still gets the ordinary error, so the 5\% carry about a third of the squared error, not all of it. The last two kinds move only some labels, and move them up. Each raises a random 10\% of the training labels by one fixed amount. In the first, the amount is a multiple of the standard deviation of the clean training labels. In the second, it is a multiple of the standard deviation of a LightGBM model's out-of-fold residual on the clean training labels. So the shift is sized by how far the model already misses. The grouped-wider condition and the Student-$t$ and Laplace shapes of Chapter~\ref{chap:robustness} are not used here.
```

### E6 — Split flip-rule list
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
A classification dataset has its labels flipped instead, and the level is the share of training labels flipped. Six rules decide which labels flip. A label can flip uniformly, more often in the rarer classes, more often in one direction, at a rate that varies by molecule, in proportion to class frequency, or along a confusion matrix. With no matrix given, the last rule is the same as the first, so on a two-class dataset five of the six are distinct.
```
NEW:
```latex
A classification dataset has its labels flipped instead, and the level is the share of training labels flipped. Six rules decide which labels flip. A label can flip uniformly, more often in the rarer classes or more often in one direction. It can also flip at a rate that varies by molecule, in proportion to class frequency, or along a confusion matrix. With no matrix given, the last rule is the same as the first, so on a two-class dataset five of the six are distinct.
```

### E7 — Split screen-correlation sentence
- file: thesis.tex
- kind: ai-text
- why: 31-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Most experiments first run a screen of two levels: none and 1.0, or none and 25\% of labels clipped under censoring. The screen leaves out the shift sized by model error. The held-out predictions it produces correlate 0.93 to 0.97 with those of the plain shift in 10\%, about as closely as two draws of the plain shift correlate with each other. An experiment that is taken further runs all six kinds at levels 0, 0.5, 1.0 and 1.5 on datasets of 5,000 molecules or fewer. Under censoring, those levels are 0, 10\%, 25\% and 40\% clipped. On a classification dataset the screen flips 10\% of labels and the full run flips up to 40\%. Each non-zero level is drawn three times, except censoring, which involves no random draw and runs once. The random seed depends on the damage kind and the draw, not on the level or the fold. So the levels nest: on every molecule, the error at level 1.0 is exactly twice the error at 0.5. A molecule also carries the same damage in every fold it trains in.
```
NEW:
```latex
Most experiments first run a screen of two levels: none and 1.0, or none and 25\% of labels clipped under censoring. The screen leaves out the shift sized by model error. The held-out predictions it produces correlate 0.93 to 0.97 with those of the plain shift in 10\%. Two draws of the plain shift correlate about as closely with each other. An experiment that is taken further runs all six kinds at levels 0, 0.5, 1.0 and 1.5 on datasets of 5,000 molecules or fewer. Under censoring, those levels are 0, 10\%, 25\% and 40\% clipped. On a classification dataset the screen flips 10\% of labels and the full run flips up to 40\%. Each non-zero level is drawn three times, except censoring, which involves no random draw and runs once. The random seed depends on the damage kind and the draw, not on the level or the fold. So the levels nest: on every molecule, the error at level 1.0 is exactly twice the error at 0.5. A molecule also carries the same damage in every fold it trains in.
```

### E8 — Split further-detectors sentence
- file: thesis.tex
- kind: ai-text
- why: 39-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
One experiment compares several further detectors with the out-of-fold residual. One is the value of each molecule to a nearest-neighbor predictor, as a Shapley value \citep{ghorbani2019data, jia2019efficient}. The others are the cleanlab regression score \citep{zhou2024detecting}, and the residual divided by a spread for that molecule, from NGBoost, a quantile forest or a conformal interval. cleanlab draws its internal folds at random rather than by scaffold group.
```
NEW:
```latex
One experiment compares several further detectors with the out-of-fold residual. One is the value of each molecule to a nearest-neighbor predictor, as a Shapley value \citep{ghorbani2019data, jia2019efficient}. Another is the cleanlab regression score \citep{zhou2024detecting}. The rest divide the residual by a spread for that molecule, from NGBoost, a quantile forest or a conformal interval. cleanlab draws its internal folds at random rather than by scaffold group.
```

### E9 — Split conformal sentence
- file: thesis.tex
- kind: ai-text
- why: 37-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Two experiments test the out-of-fold LightGBM residual under the shift in 10\% sized by model error. The first ranks molecules by the residual itself. The second divides each residual by a second LightGBM model's out-of-fold estimate of how large that molecule's residual usually is. It then turns the result into conformal $p$-values against a calibration set of training molecules \citep{bates2023testing}, and flags molecules with the Benjamini--Hochberg rule \citep{benjamini1995controlling}, which bounds the expected share of false flags at 5\%, 10\% or 20\%. The calibration set is a random 20\% or 50\% of the training molecules or, where a dataset has repeat measurements, the molecules with three or more measurements that agree. Both experiments grade the ranking by the area under the ROC curve.
```
NEW:
```latex
Two experiments test the out-of-fold LightGBM residual under the shift in 10\% sized by model error. The first ranks molecules by the residual itself. The second divides each residual by a second LightGBM model's out-of-fold estimate of how large that molecule's residual usually is. It then turns the result into conformal $p$-values against a calibration set of training molecules \citep{bates2023testing}. Molecules are flagged with the Benjamini--Hochberg rule \citep{benjamini1995controlling}, which bounds the expected share of false flags at 5\%, 10\% or 20\%. The calibration set is a random 20\% or 50\% of the training molecules or, where a dataset has repeat measurements, the molecules with three or more measurements that agree. Both experiments grade the ranking by the area under the ROC curve.
```

### E10 — Mitigation framing: split, no code reference
- file: thesis.tex
- kind: ai-text
- why: Two sentences of 31 and 37 words (task item 5); "the many methods in the code" refers to something the reader cannot see.
- todo_line: keep existing
OLD:
```latex
A noise mitigation method has two halves (Table~\ref{tab:noise_terms}). Something selects the molecules: a detector, a known flag or every molecule. Then one of three actions follows: delete the molecule, make it count less, or move its label. We describe each method by four things: what selects the molecules, how many it acts on, the action, and for a move, the target and how far the label moves (Table~\ref{tab:noise:mitigation}). Written this way, the many methods in the code reduce to a few selectors crossed with three actions, and the many ways of moving a label differ only in where the label moves to and how far.
```
NEW:
```latex
A noise mitigation method has two halves (Table~\ref{tab:noise_terms}). Something selects the molecules: a detector, a known flag or every molecule. Then one of three actions follows: delete the molecule, make it count less, or move its label. We describe each method by four things (Table~\ref{tab:noise:mitigation}). They are what selects the molecules, how many it acts on, the action, and for a move, the target and how far the label moves. Written this way, the many methods we tested reduce to a few selectors crossed with three actions. The many ways of moving a label differ only in where the label moves to and how far.
```

### E11 — Cliff put-back: cliffs in a property that is not an activity
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 18: the cliff test is applied to LogD and Caco-2, whose labels are not biological activities, so say the term is borrowed.
- todo_line: keep existing
OLD:
```latex
Cliff put-back runs after a cleanup and undoes it on activity-cliff molecules: deleted molecules return, lowered weights go back to 1 and moved labels go back to their training value. A training molecule is a cliff molecule if it has a partner with ECFP4 Tanimoto similarity of at least 0.55 whose label is far from its own \citep{Maggiora2006, Stumpfe2012}. Far means more than twice the median absolute deviation of the labels. The cliff set is rebuilt in every fold from the damaged training labels. Its control puts back the same number of molecules drawn at random from the non-cliff molecules.
```
NEW:
```latex
Cliff put-back runs after a cleanup and undoes it on activity-cliff molecules: deleted molecules return, lowered weights go back to 1 and moved labels go back to their training value. A training molecule is a cliff molecule if it has a partner with ECFP4 Tanimoto similarity of at least 0.55 whose label is far from its own \citep{Maggiora2006, Stumpfe2012}. Far means more than twice the median absolute deviation of the labels. On datasets whose label is not a biological activity, such as LogD, the same test finds pairs that differ sharply in that property, and we keep the name. The cliff set is rebuilt in every fold from the damaged training labels. Its control puts back the same number of molecules drawn at random from the non-cliff molecules.
```

### E12 — No second person (delivered labels)
- file: thesis.tex
- kind: ai-text
- why: Style guide item 2: no "you" in the body.
- todo_line: keep existing
OLD:
```latex
On labels as delivered, nothing tells you which molecules are wrong, so a gain can come from removing hard molecules rather than wrong ones. One experiment runs seven cleanups on delivered labels with the neighbor gap and the forest residual as detectors. Its control shuffles each detector's scores across the training molecules, at three seeds, and reruns the cleanup with the same share and strength. The same number of molecules is touched, but not the ones the detector picked.
```
NEW:
```latex
On labels as delivered, nothing says which molecules are wrong, so a gain can come from removing hard molecules rather than wrong ones. One experiment runs seven cleanups on delivered labels with the neighbor gap and the forest residual as detectors. Its control shuffles each detector's scores across the training molecules, at three seeds, and reruns the cleanup with the same share and strength. The same number of molecules is touched, but not the ones the detector picked.
```

### E13 — No second person (known flag)
- file: thesis.tex
- kind: ai-text
- why: Style guide item 2: no "you" in the body.
- todo_line: keep existing
OLD:
```latex
When a label is clipped at an assay limit, you know two things a detector has to guess: which labels are wrong, and in which direction. The limit $L$ is read off the clipped training labels as their largest value, if at least two molecules and 1\% of the training molecules sit on it. An out-of-fold LightGBM fit to the clipped labels gives each molecule a prediction $p_i$. Let $\sigma$ be the standard deviation of $y_i - p_i$ over the molecules below the limit, and $m_i$ the inverse Mills ratio,
```
NEW:
```latex
When a label is clipped at an assay limit, we know two things that a detector has to guess: which labels are wrong, and in which direction. The limit $L$ is read off the clipped training labels as their largest value, if at least two molecules and 1\% of the training molecules sit on it. An out-of-fold LightGBM fit to the clipped labels gives each molecule a prediction $p_i$. Let $\sigma$ be the standard deviation of $y_i - p_i$ over the molecules below the limit, and $m_i$ the inverse Mills ratio,
```

### E14 — Split source-model definition
- file: thesis.tex
- kind: ai-text
- why: 40-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
where $b_s$ is the source's offset, $u_i$ is an error shared by every measurement of the molecule, $v_{is}$ is an error of its own, and $\rho$ of 0, 0.3 or 0.7 sets how much of the error the measurements share. The offsets have standard deviation 0 or $0.5\,\sigma_e$. The training label is the mean of the molecule's measurements. The methods remove each source's offset, estimated by a mixed model with the source as a fixed effect and the molecule as a random intercept. They also weight each molecule by the inverse of its estimated error variance \citep{hedges1985meta}. Two further arms use the true $\rho$ and $\sigma_e$, which a user never has, so they are bounds and not methods.
```
NEW:
```latex
where $b_s$ is the source's offset, $u_i$ is an error shared by every measurement of the molecule and $v_{is}$ is an error of its own. The value of $\rho$, 0, 0.3 or 0.7, sets how much of the error the measurements share. The offsets have standard deviation 0 or $0.5\,\sigma_e$. The training label is the mean of the molecule's measurements. The methods remove each source's offset, estimated by a mixed model with the source as a fixed effect and the molecule as a random intercept. They also weight each molecule by the inverse of its estimated error variance \citep{hedges1985meta}. Two further arms use the true $\rho$ and $\sigma_e$, which a user never has, so they are bounds and not methods.
```

### E15 — hERG IC50 molecule identity and spread
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 24: two SMILES strings can encode one molecule, so say the string is ChEMBL's canonical SMILES (the loader groups on ChEMBL's canonical_smiles); the spread is of pIC50 values, not of raw IC50 records; 32-word sentence split.
- todo_line: keep existing
OLD:
```latex
hERG IC$_{50}$ carries its own known flag: how many times each molecule was measured, and how far apart those measurements are. A molecule is one SMILES string, its count is its number of ChEMBL IC$_{50}$ records, and its spread is the variance of those records. Three methods use the flag. The first weights each molecule by its count. The second weights it by the inverse of its estimated error variance, $1/(\sigma_u^2 + \tilde s_i^2/n_i)$ \citep{hedges1985meta}. Here $n_i$ is the number of measurements behind the label, and $\sigma_u^2$ is the error that remains however often a molecule is measured. Its spread $\tilde s_i^2$ is pulled toward the spread pooled over all molecules, as if the pool were three more measurements, because a variance from two or three records is unreliable \citep{smyth2004limma}. The third moves each label toward an out-of-fold LightGBM prediction, further for a less precise label. The share moved is $v_i/(v_i + \tau^2)$, where $v_i$ is the molecule's spread over its count and $\tau^2$ is the variance of the out-of-fold residual. Held-out molecules are graded only if they have three or more measurements.
```
NEW:
```latex
hERG IC$_{50}$ carries its own known flag: how many times each molecule was measured, and how far apart those measurements are. A molecule is one canonical SMILES string as given by ChEMBL. Its count is its number of ChEMBL IC$_{50}$ records, and its spread is the variance of their pIC$_{50}$ values. Three methods use the flag. The first weights each molecule by its count. The second weights it by the inverse of its estimated error variance, $1/(\sigma_u^2 + \tilde s_i^2/n_i)$ \citep{hedges1985meta}. Here $n_i$ is the number of measurements behind the label, and $\sigma_u^2$ is the error that remains however often a molecule is measured. Its spread $\tilde s_i^2$ is pulled toward the spread pooled over all molecules, as if the pool were three more measurements. We do this because a variance from two or three records is unreliable \citep{smyth2004limma}. The third moves each label toward an out-of-fold LightGBM prediction, further for a less precise label. The share moved is $v_i/(v_i + \tau^2)$, where $v_i$ is the molecule's spread over its count and $\tau^2$ is the variance of the out-of-fold residual. Held-out molecules are graded only if they have three or more measurements.
```

### E16 — Split network description
- file: thesis.tex
- kind: ai-text
- why: 38-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
The twelve published methods of Chapter~\ref{chap:flaws} are run again (Table~\ref{tab:noise:robust}). Every one uses the same network as the plain model: three hidden layers of 256, 128 and 64 units, dropout 0.2, the Adam optimizer at a learning rate of $10^{-3}$, and up to 200 epochs with early stopping. The code is a new implementation of the same twelve, not a rerun of the earlier study. It is wider in four ways. It runs on many datasets instead of QM9 alone, under every damage kind instead of Gaussian noise alone, and on hybrids as well as single representations. It also scores each method against the plain network on the same damaged labels in the same feature space. Four of the twelve select molecules with a detector and delete them, so under Table~\ref{tab:noise_terms} they are noise mitigation, and the table marks them so. Most were built for image classification, and several of our versions differ from their source, as the table states.
```
NEW:
```latex
The twelve published methods of Chapter~\ref{chap:flaws} are run again (Table~\ref{tab:noise:robust}). Every one uses the same network as the plain model. It has three hidden layers of 256, 128 and 64 units and dropout 0.2. It is trained with the Adam optimizer at a learning rate of $10^{-3}$ for up to 200 epochs, with early stopping. The code is a new implementation of the same twelve, not a rerun of the earlier study. It is wider in four ways. It runs on many datasets instead of QM9 alone, under every damage kind instead of Gaussian noise alone, and on hybrids as well as single representations. It also scores each method against the plain network on the same damaged labels in the same feature space. Four of the twelve select molecules with a detector and delete them, so under Table~\ref{tab:noise_terms} they are noise mitigation, and the table marks them so. Most were built for image classification, and several of our versions differ from their source, as the table states.
```

### E17 — Methods caveat paragraph duplicates the Results one
- file: thesis.tex
- kind: ai-text
- why: The same three caveats, with the same QM9 numbers, are stated again as the first paragraph of the Results (the [N7-01] paragraph, which ch7_numbers.py tracks); keep them there and point to it here.
- todo_line: keep existing
OLD:
```latex
A method usually has several settings, such as a share to delete or a strength. Where we report whether a method worked, its best setting is picked on the held-out score, and so are the single representation and the hybrid it is compared on. That makes every such count optimistic. The network's score on the same labels also differs between runs, so a small network gain can be the spread between fits rather than the method. On QM9 the network's second hybrid scores 0.3618, 0.3568 or 0.3484 RMSE on the same fold and labels, depending on the experiment that fitted it (Section~\ref{sec:noise:res:overall}). Most results are on the first outer fold, and the results give the fold count wherever more folds exist.
```
NEW:
```latex
Three caveats apply to every result. Most results are on the first outer fold, each method's best setting is picked on the held-out score, and the network's score varies between fits. Section~\ref{sec:noise:res:overall} states them with their numbers.
```

### E18 — Split standing-caveat sentence
- file: thesis.tex
- kind: ai-text
- why: 34-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Three things hold for every result in this chapter, so we state them once. Most results are on the first outer fold only, and we give the fold count wherever more exist. Where a method has several settings, such as a share to delete or a strength, we report its best setting on the held-out score. The single representation and the hybrid it is compared on are picked the same way. So every ``it helped'' leans optimistic. Last, the network does not give the same score twice. On QM9 the network's second hybrid scores 0.3618, 0.3568 or 0.3484 RMSE on the same fold and the same labels, depending on which experiment fitted it, while LightGBM gives one score in every experiment. A network gain of about 0.01 RMSE or less is inside that spread.
```
NEW:
```latex
Three things hold for every result in this chapter, so we state them once. Most results are on the first outer fold only, and we give the fold count wherever more exist. Where a method has several settings, such as a share to delete or a strength, we report its best setting on the held-out score. The single representation and the hybrid it is compared on are picked the same way. So every ``it helped'' leans optimistic. Last, the network does not give the same score twice. On QM9 the network's second hybrid scores 0.3618, 0.3568 or 0.3484 RMSE on the same fold and the same labels, depending on which experiment fitted it. LightGBM gives one score in every experiment. A network gain of about 0.01 RMSE or less is inside that spread.
```

### E19 — Split three-fold sentence (hERG deletion)
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Under hERG's label flips every action helped, and deleting the flagged molecules helped most. With 10\% of labels flipped along a confusion matrix, gradient-boosted trees on the hybrid score 0.8028 AUROC. Deleting the 10\% of molecules the forest residual flags raises this to 0.8356, moving their labels toward the neighbors' mean to 0.8338, and down-weighting them to 0.8140. On the best single representation the same three give 0.8253, 0.8250 and 0.8108, from 0.8060. Deletion is the one action here with a control, and it passes it. With the neighbor-label gap as the detector, deleting takes the hybrid to 0.8375 against 0.8179 for the best of five random deletions of the same size. The gain holds on three folds: with the forest residual, the hybrid goes 0.8028 to 0.8356, 0.8164 to 0.8258 and 0.8223 to 0.8381, against 0.8127, 0.8122 and 0.8347 for the best random deletion.
```
NEW:
```latex
Under hERG's label flips every action helped, and deleting the flagged molecules helped most. With 10\% of labels flipped along a confusion matrix, gradient-boosted trees on the hybrid score 0.8028 AUROC. Deleting the 10\% of molecules the forest residual flags raises this to 0.8356, moving their labels toward the neighbors' mean to 0.8338, and down-weighting them to 0.8140. On the best single representation the same three give 0.8253, 0.8250 and 0.8108, from 0.8060. Deletion is the one action here with a control, and it passes it. With the neighbor-label gap as the detector, deleting takes the hybrid to 0.8375 against 0.8179 for the best of five random deletions of the same size. The gain holds on three folds. With the forest residual, the hybrid goes 0.8028 to 0.8356, 0.8164 to 0.8258 and 0.8223 to 0.8381. The best random deletion gives 0.8127, 0.8122 and 0.8347.
```

### E20 — Split LogD action list
- file: thesis.tex
- kind: ai-text
- why: 40-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
On a regression dataset the actions come closer together. On LogD with 5\% outliers, the network on the first single representation goes from 0.6273 RMSE to 0.6030 by deleting the flagged 5\%, 0.6149 by down-weighting, 0.6124 by label smoothing and 0.6005 by the score-scaled shrink toward an out-of-fold prediction. On the first hybrid, down-weighting is best, 0.5881 to 0.5634, and the shrink makes it slightly worse, 0.5900. Against random deletion, deleting is less convincing here than on hERG. On the second hybrid it passes, 0.5847 to 0.5460 against 0.5584, but on the first hybrid it loses to random deletion, 0.5690 against 0.5647. On the second fold the same deletion makes the second hybrid worse, 0.6242 to 0.6332.
```
NEW:
```latex
On a regression dataset the actions come closer together. On LogD with 5\% outliers, the network on the first single representation scores 0.6273 RMSE. Deleting the flagged 5\% brings it to 0.6030, down-weighting to 0.6149, label smoothing to 0.6124 and the score-scaled shrink toward an out-of-fold prediction to 0.6005. On the first hybrid, down-weighting is best, 0.5881 to 0.5634, and the shrink makes it slightly worse, 0.5900. Against random deletion, deleting is less convincing here than on hERG. On the second hybrid it passes, 0.5847 to 0.5460 against 0.5584, but on the first hybrid it loses to random deletion, 0.5690 against 0.5647. On the second fold the same deletion makes the second hybrid worse, 0.6242 to 0.6332.
```

### E21 — Split five-detector sentences
- file: thesis.tex
- kind: ai-text
- why: 43- and 35-word sentences (task item 5).
- todo_line: keep existing
OLD:
```latex
Which detector selects the molecules to delete matters more than which action follows. Deleting the top 10\% by each of five detectors under the same hERG flips takes the hybrid from 0.8028 to 0.8375 with the cliff-aware gap, 0.8270 with the GP, 0.8128 with the variance-head network, 0.8067 with NGBoost and 0.8064 with the heteroscedastic GP. On the single representation the last three make things worse. The three that predict a spread from the features alone never read a label (Section~\ref{sec:noise:detection}), so they cannot see a flipped one. On LogD with outliers the GP was best, taking the network's hybrid from 0.5879 to 0.5581. On a second fold of the hERG flips, none of the five improved the single representation (0.8155 untreated, 0.7946 to 0.8147 after deletion), and only the cliff-aware gap improved the hybrid, from 0.8314 to 0.8332.
```
NEW:
```latex
Which detector selects the molecules to delete matters more than which action follows. Under the same hERG flips, the hybrid scores 0.8028 AUROC untreated. Deleting the top 10\% by each of five detectors gives 0.8375 with the cliff-aware gap and 0.8270 with the GP. The variance-head network gives 0.8128, NGBoost 0.8067 and the heteroscedastic GP 0.8064. On the single representation the last three make things worse. The three that predict a spread from the features alone never read a label (Section~\ref{sec:noise:detection}), so they cannot see a flipped one. On LogD with outliers the GP was best, taking the network's hybrid from 0.5879 to 0.5581. On a second fold of the hERG flips, none of the five improved the single representation (0.8155 untreated, 0.7946 to 0.8147 after deletion). Only the cliff-aware gap improved the hybrid, from 0.8314 to 0.8332.
```

### E22 — Split flags-of-known-quality sentence
- file: thesis.tex
- kind: ai-text
- why: 34-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
The clearest evidence that the detector, and not the action, carries the gain comes from flags of known quality. On hERG $K_i$ with outliers, down-weighting the 10\% of molecules the residual flags takes the network from 0.6887 to 0.6419 RMSE on the first single representation and from 0.6917 to 0.6388 on the hybrid. Down-weighting the same number of molecules, none of them damaged, gives 0.6921, 0.7094 and 0.7171 on the single representation. The same action on the wrong molecules makes the model worse.
```
NEW:
```latex
The clearest evidence that the detector, and not the action, carries the gain comes from flags of known quality. On hERG $K_i$ with outliers, down-weighting the 10\% of molecules the residual flags takes the network from 0.6887 to 0.6419 RMSE on the first single representation. On the hybrid it goes from 0.6917 to 0.6388. Down-weighting the same number of molecules, none of them damaged, gives 0.6921, 0.7094 and 0.7171 on the single representation. The same action on the wrong molecules makes the model worse.
```

### E23 — Down-weighting paragraph: split, Buchwald--Hartwig unit
- file: thesis.tex
- kind: ai-text
- why: 34-word sentence (task item 5); Buchwald--Hartwig RMSE given without its unit (chemistry guide item 29; Chapter 5 uses "percentage points of yield").
- todo_line: keep existing
OLD:
```latex
Making flagged molecules count less is the gentlest action, and it helped on most datasets. Under the hERG flips, weighting by the gap over neighbor spread took the hybrid from 0.8028 to 0.8155 and barely moved the single representation, 0.8060 to 0.8063. Weighting by the neighbor gap in the best of four distances took the network on LogD with outliers from 0.6273 to 0.5915 on the single representation and from 0.5881 to 0.5560 on the hybrid. That is the best of many settings, picked on the held-out score. Down-weighting near-identical pairs whose labels disagree took LightGBM on Buchwald--Hartwig under Gaussian noise from 22.0472 to 20.6614 RMSE on the hybrid. QM9 sets a limit on how these gains can be read. With no noise added, neighbor weights took the network's second hybrid from 0.3618 to 0.3476 at their best setting, and detector weights to 0.3488. LightGBM barely moved, 0.3719 to 0.3707 and to 0.3724. So a network gain of about 0.01 RMSE from weighting is not evidence that noise was removed, while a gain with the tree models is not affected by this.
```
NEW:
```latex
Making flagged molecules count less is the gentlest action, and it helped on most datasets. Under the hERG flips, weighting by the gap over neighbor spread took the hybrid from 0.8028 to 0.8155 and barely moved the single representation, 0.8060 to 0.8063. Weighting by the neighbor gap in the best of four distances took the network on LogD with outliers from 0.6273 to 0.5915 on the single representation. On the hybrid it went from 0.5881 to 0.5560. That is the best of many settings, picked on the held-out score. Down-weighting near-identical pairs whose labels disagree took LightGBM on Buchwald--Hartwig under Gaussian noise from 22.0472 to 20.6614 RMSE, in percentage points of yield, on the hybrid. QM9 sets a limit on how these gains can be read. With no noise added, neighbor weights took the network's second hybrid from 0.3618 to 0.3476 at their best setting, and detector weights to 0.3488. LightGBM barely moved, 0.3719 to 0.3707 and to 0.3724. So a network gain of about 0.01 RMSE from weighting is not evidence that noise was removed, while a gain with the tree models is not affected by this.
```

### E24 — Split QM9 deletion sentence
- file: thesis.tex
- kind: ai-text
- why: 32-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Deleting never helped on clean labels. On QM9 with no noise added, the best deletion setting took the network's hybrid from 0.3550 to 0.3931 RMSE and LightGBM's from 0.3719 to 0.4106, and every other configuration also got worse. A deleted QM9 molecule takes a correct label with it. So when deleting helps on a damaged dataset and beats random deletion, as it does under the hERG flips, the gain is noise removed.
```
NEW:
```latex
Deleting never helped on clean labels. On QM9 with no noise added, the best deletion setting took the network's hybrid from 0.3550 to 0.3931 RMSE and LightGBM's from 0.3719 to 0.4106. Every other configuration also got worse. A deleted QM9 molecule takes a correct label with it. So when deleting helps on a damaged dataset and beats random deletion, as it does under the hERG flips, the gain is noise removed.
```

### E25 — Split hERG smoothing sentence (delivered labels)
- file: thesis.tex
- kind: ai-text
- why: 35-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
On the labels as delivered there is no ground truth, so the control is the same cleanup driven by shuffled detector scores. hERG $K_i$ gives the clearest small-molecule result. Label smoothing with the neighbor gap takes the network on the first single representation from 0.6380 to 0.5775 RMSE, against 0.6420, 0.6419 and 0.6455 with shuffled scores. With the forest residual it takes the hybrid from 0.5989 to 0.5783, against 0.5998, 0.5877 and 0.5960. On hERG, label smoothing takes gradient-boosted trees from 0.8369 to 0.8455 AUROC on the first single representation, against at most 0.8350 shuffled, and from 0.8545 to 0.8600 on the second hybrid, against at most 0.8550. Caco-2 and LogD pass the control too, but by small amounts. Down-weighting takes the network on Caco-2's second single representation from 0.2657 to 0.2548, where one shuffle reaches 0.2581, and on LogD's from 0.4212 to 0.4143, where one shuffle reaches 0.4172.
```
NEW:
```latex
On the labels as delivered there is no ground truth, so the control is the same cleanup driven by shuffled detector scores. hERG $K_i$ gives the clearest small-molecule result. Label smoothing with the neighbor gap takes the network on the first single representation from 0.6380 to 0.5775 RMSE, against 0.6420, 0.6419 and 0.6455 with shuffled scores. With the forest residual it takes the hybrid from 0.5989 to 0.5783, against 0.5998, 0.5877 and 0.5960. On hERG, label smoothing takes gradient-boosted trees from 0.8369 to 0.8455 AUROC on the first single representation, against at most 0.8350 shuffled. On the second hybrid it goes from 0.8545 to 0.8600, against at most 0.8550. Caco-2 and LogD pass the control too, but by small amounts. Down-weighting takes the network on Caco-2's second single representation from 0.2657 to 0.2548, where one shuffle reaches 0.2581, and on LogD's from 0.4212 to 0.4143, where one shuffle reaches 0.4172.
```

### E26 — Split three-fold smoothing sentence
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
The hERG $K_i$ smoothing gain on the hybrid holds on three folds. In a second experiment, label smoothing took the network's hybrid from 0.5989 to 0.5671 RMSE on fold 0, from 0.5500 to 0.5467 on fold 1 and from 0.5495 to 0.5370 on fold 2. The second single representation got worse on fold 0, 0.5892 to 0.5965, and better on the other two, 0.6102 to 0.6003 and 0.6452 to 0.6291.
```
NEW:
```latex
The hERG $K_i$ smoothing gain on the hybrid holds on three folds. In a second experiment, label smoothing took the network's hybrid from 0.5989 to 0.5671 RMSE on fold 0. On folds 1 and 2 it went from 0.5500 to 0.5467 and from 0.5495 to 0.5370. The second single representation got worse on fold 0, 0.5892 to 0.5965, and better on the other two, 0.6102 to 0.6003 and 0.6452 to 0.6291.
```

### E27 — Buchwald--Hartwig deletion: split, unit
- file: thesis.tex
- kind: ai-text
- why: 41-word sentence (task item 5); "RMSE in yield" made the same as Chapter 5's "percentage points of yield" (chemistry guide item 29).
- todo_line: keep existing
OLD:
```latex
Deleting does not pass its control on the small-molecule labels as delivered. On Caco-2, hERG $K_i$ and LogD no deletion setting beat both no cleaning and every random deletion. Buchwald--Hartwig is different. Deleting the 20\% of reactions with the largest neighbor-label gap takes LightGBM on the hybrid from 20.0823 to 18.0417 RMSE in yield, against 19.8986 for the best random deletion, and on the second single representation from 20.2242 to 18.3284, against 19.9173. Random deletion alone also helped the first single representation, 19.4875 to 18.8663, so this score moves between fits as well.
```
NEW:
```latex
Deleting does not pass its control on the small-molecule labels as delivered. On Caco-2, hERG $K_i$ and LogD no deletion setting beat both no cleaning and every random deletion. Buchwald--Hartwig is different. Deleting the 20\% of reactions with the largest neighbor-label gap takes LightGBM on the hybrid from 20.0823 to 18.0417 RMSE, in percentage points of yield. The best random deletion gives 19.8986. On the second single representation the same deletion goes from 20.2242 to 18.3284, against 19.9173. Random deletion alone also helped the first single representation, 19.4875 to 18.8663, so this score moves between fits as well.
```

### E28 — Split cliff-share sentence
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Putting activity-cliff molecules back after a cleanup can be tested on hERG $K_i$ only. On Caco-2, hERG and LogD between 54\% and 72\% of the training molecules count as cliff molecules, so there are not enough other molecules to draw a random set of the same size. On hERG $K_i$, the network's hybrid scores 0.5972 untreated, 0.5960 after deleting flagged molecules, and 0.5786 with the cliff molecules put back, against 0.5949 with random molecules put back. Fold 2 agrees, 0.5370 against 0.5453, and fold 1 does not, 0.5684 against 0.5572.
```
NEW:
```latex
Putting activity-cliff molecules back after a cleanup can be tested on hERG $K_i$ only. On Caco-2, hERG and LogD between 54\% and 72\% of the training molecules count as cliff molecules. That leaves too few other molecules to draw a random set of the same size. On hERG $K_i$, the network's hybrid scores 0.5972 untreated, 0.5960 after deleting flagged molecules, and 0.5786 with the cliff molecules put back, against 0.5949 with random molecules put back. Fold 2 agrees, 0.5370 against 0.5453, and fold 1 does not, 0.5684 against 0.5572.
```

### E29 — Split censoring-detection sentence
- file: thesis.tex
- kind: ai-text
- why: 34-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
A detector can only be graded where we know which labels were damaged, so this section uses injected damage alone. The out-of-fold forest residual finds outliers, shifted labels and flipped classes well above chance on every dataset it ran on. On Caco-2 under outlier noise its average precision is 0.3381, against a chance level of 0.0512, and on hERG under uniform flips at 10\% it is 0.3970 against 0.0929. Censoring is the exception. At a clipped share of 0.25 the residual falls below chance on Buchwald--Hartwig (0.1929 against 0.2499) and on LogD (0.1957 against 0.2369), and is only a little above it on Caco-2 (0.3065 against 0.2503). A model fitted to clipped labels predicts values near the limit for the clipped molecules, so their residuals are small. This is the behavior Chapter~\ref{chap:robustness} found in the models' own uncertainty, which was lowest on exactly the labels that had been clipped.
```
NEW:
```latex
A detector can only be graded where we know which labels were damaged, so this section uses injected damage alone. The out-of-fold forest residual finds outliers, shifted labels and flipped classes well above chance on every dataset it ran on. On Caco-2 under outlier noise its average precision is 0.3381, against a chance level of 0.0512, and on hERG under uniform flips at 10\% it is 0.3970 against 0.0929. Censoring is the exception. At a clipped share of 0.25 the residual falls below chance on Buchwald--Hartwig (0.1929 against 0.2499) and on LogD (0.1957 against 0.2369). On Caco-2 it is only a little above chance (0.3065 against 0.2503). A model fitted to clipped labels predicts values near the limit for the clipped molecules, so their residuals are small. This is the behavior Chapter~\ref{chap:robustness} found in the models' own uncertainty, which was lowest on exactly the labels that had been clipped.
```

### E30 — Split GP-detector sentence
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
A Gaussian process's own predicted spread plus its residual is the one model-uncertainty detector that could be graded. The three that predict a spread from the features alone never see a label, and the files mark them as undefined under every damage kind that can be graded. The GP was slightly ahead of the cliff-aware gap under outliers, 0.3449 against 0.3350 on Caco-2 and 0.4084 against 0.3853 on LogD, and behind it under every shift and every hERG flip kind. Detectors from outside cheminformatics did better than the residual where the residual is weakest. Under censoring the nearest-neighbor Shapley value reached 0.3906 on Buchwald--Hartwig against 0.1949 for the residual, and 0.6402 against 0.4905 on SMAC colloidal stability. Under the shift in 10\% the cleanlab score reached 0.8288 on Buchwald--Hartwig against 0.2955. The Shapley value's best number is taken over many more settings than the residual's, so these gaps lean in its favor.
```
NEW:
```latex
A Gaussian process's own predicted spread plus its residual is the one model-uncertainty detector that could be graded. The three that predict a spread from the features alone never see a label, and the files mark them as undefined under every damage kind that can be graded. The GP was slightly ahead of the cliff-aware gap under outliers, 0.3449 against 0.3350 on Caco-2 and 0.4084 against 0.3853 on LogD. It was behind it under every shift and every hERG flip kind. Detectors from outside cheminformatics did better than the residual where the residual is weakest. Under censoring the nearest-neighbor Shapley value reached 0.3906 on Buchwald--Hartwig against 0.1949 for the residual, and 0.6402 against 0.4905 on SMAC colloidal stability. Under the shift in 10\% the cleanlab score reached 0.8288 on Buchwald--Hartwig against 0.2955. The Shapley value's best number is taken over many more settings than the residual's, so these gaps lean in its favor.
```

### E31 — Split conformal-result sentence
- file: thesis.tex
- kind: ai-text
- why: 32-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Two more experiments graded the plain residual under damage sized by the model's own error. It ranked the shifted molecules above chance on every dataset there, for example 0.20 to 0.25 against 0.106 on expression titer and 0.131 to 0.155 against 0.0997 on Caco-2. Turning such a ranking into a decision is harder. Conformal $p$-values with a false-discovery-rate rule flagged no molecule at all at $q = 0.05$, and none at $q = 0.1$ when the calibration set was drawn from the noisy training labels. Only with a clean calibration set, which a user would not have, did it flag 10 to 37 molecules in a few passes. Acting on those flags left the held-out score where it was in almost every pass.
```
NEW:
```latex
Two more experiments graded the plain residual under damage sized by the model's own error. It ranked the shifted molecules above chance on every dataset there, for example 0.20 to 0.25 against 0.106 on expression titer and 0.131 to 0.155 against 0.0997 on Caco-2. Turning such a ranking into a decision is harder. Conformal $p$-values with a false-discovery-rate rule flagged no molecule at all at $q = 0.05$. At $q = 0.1$ it flagged none when the calibration set was drawn from the noisy training labels. Only with a clean calibration set, which a user would not have, did it flag 10 to 37 molecules in a few passes. Acting on those flags left the held-out score where it was in almost every pass.
```

### E32 — Split break-even sentence
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5); the two break-even ranges now name their datasets.
- todo_line: keep existing
OLD:
```latex
The break-even precision explains when acting on a detector can work (Section~\ref{sec:noise:mitigation}). Under outliers the residual's top 5\% are mostly damaged molecules. Its precision is 0.62 to 0.64 on Caco-2 and 0.60 to 0.62 on hERG $K_i$, and acting on its flags pays off once the precision passes 0.13 to 0.27 and 0.16 to 0.42. Under the shift sized by model error the picture reverses. On Caco-2 the residual's top 5\% has a precision of 0.17, and deleting them only pays off above 0.62. On hERG $K_i$ under the plain shift in 10\%, the top 10\% has a precision of 0.26 against a break-even of 0.68. Finding the damaged labels above chance is not enough. The detector has to clear a bar that depends on the damage, the action and the model, and under the shifts it mostly does not.
```
NEW:
```latex
The break-even precision explains when acting on a detector can work (Section~\ref{sec:noise:mitigation}). Under outliers the residual's top 5\% are mostly damaged molecules. Its precision is 0.62 to 0.64 on Caco-2 and 0.60 to 0.62 on hERG $K_i$. Acting on its flags pays off once the precision passes 0.13 to 0.27 on Caco-2 and 0.16 to 0.42 on hERG $K_i$. Under the shift sized by model error the picture reverses. On Caco-2 the residual's top 5\% has a precision of 0.17, and deleting them only pays off above 0.62. On hERG $K_i$ under the plain shift in 10\%, the top 10\% has a precision of 0.26 against a break-even of 0.68. Finding the damaged labels above chance is not enough. The detector has to clear a bar that depends on the damage, the action and the model, and under the shifts it mostly does not.
```

### E33 — Split QM9 censoring sentence
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5). The score list stays because ch7_numbers.py re-checks these numbers under [N7-20] (see For the author).
- todo_line: keep existing
OLD:
```latex
Clipping at an assay limit is the damage we could most often undo, and undoing it needs no detector. The flag says which labels sit at the limit, and the correction only has to decide how far to raise them. Figure~\ref{fig:noise:censoring_ladder} shows the four steps for every model and configuration on Caco-2, hERG $K_i$, LogD and QM9, with a quarter of the training labels clipped. On LogD, the network on the best single representation goes from 0.4314 RMSE on the labels before clipping to 0.6354 after clipping, and back to 0.4795 after censored EM. That is 76\% of the lost RMSE won back (Equation~\ref{eq:recovery}). On the hybrid the same three numbers are 0.4059, 0.6049 and 0.4610. On QM9, where the labels before clipping are computed and carry no assay error, the network on the hybrid goes 0.3513, 0.5196 and 0.4406, and LightGBM on the hybrid 0.3694, 0.5240 and 0.4301. Caco-2 behaves the same way at a smaller scale, 0.2459, 0.3042 and 0.2740 for the network on the hybrid.
```
NEW:
```latex
Clipping at an assay limit is the damage we could most often undo, and undoing it needs no detector. The flag says which labels sit at the limit, and the correction only has to decide how far to raise them. Figure~\ref{fig:noise:censoring_ladder} shows the four steps for every model and configuration on Caco-2, hERG $K_i$, LogD and QM9, with a quarter of the training labels clipped. On LogD, the network on the best single representation goes from 0.4314 RMSE on the labels before clipping to 0.6354 after clipping, and back to 0.4795 after censored EM. That is 76\% of the lost RMSE won back (Equation~\ref{eq:recovery}). On the hybrid the same three numbers are 0.4059, 0.6049 and 0.4610. QM9's labels before clipping are computed and carry no assay error. There the network on the hybrid goes 0.3513, 0.5196 and 0.4406, and LightGBM on the hybrid 0.3694, 0.5240 and 0.4301. Caco-2 behaves the same way at a smaller scale, 0.2459, 0.3042 and 0.2740 for the network on the hybrid.
```

### E34 — Split fold-list sentence (censored EM)
- file: thesis.tex
- kind: ai-text
- why: 36-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
The gain holds on the further folds where they exist. On LogD, the network on the hybrid after censored EM scores 0.4798, 0.4719, 0.4702 and 0.4749 on folds 0 to 3, against 0.6128, 0.5961, 0.5720 and 0.5697 clipped and 0.4040, 0.4180, 0.4114 and 0.4129 before clipping. On Caco-2 folds 1 and 2 it scores 0.2906 and 0.3484 against 0.3216 and 0.4004 clipped. hERG $K_i$ and QM9 have only fold 0.
```
NEW:
```latex
The gain holds on the further folds where they exist. On LogD, the network on the hybrid after censored EM scores 0.4798, 0.4719, 0.4702 and 0.4749 on folds 0 to 3. Clipped, it scores 0.6128, 0.5961, 0.5720 and 0.5697, and before clipping 0.4040, 0.4180, 0.4114 and 0.4129. On Caco-2 folds 1 and 2 it scores 0.2906 and 0.3484 against 0.3216 and 0.4004 clipped. hERG $K_i$ and QM9 have only fold 0.
```

### E35 — Three moves: split, Buchwald--Hartwig unit
- file: thesis.tex
- kind: ai-text
- why: 32-word sentence (task item 5); Buchwald--Hartwig RMSE without unit (chemistry guide item 29).
- todo_line: keep existing
OLD:
```latex
Which of the three moves works best depends on the dataset. Censored EM is best on LogD and QM9. The constant lift is best on hERG $K_i$, where the network on the second single representation goes 0.5891 before clipping, 0.7485 clipped, 0.6662 after the constant lift and 0.7037 after censored EM. The truncated-normal mean is behind both on LogD, hERG $K_i$ and Caco-2. On Buchwald--Hartwig only the random forest was hurt by clipping, and censored EM brought it back to 17.585 RMSE from 18.691, against 17.594 before clipping. On the antibody thermostability set, scored by Spearman, the network on the hybrid goes 0.7487, 0.6097 and 0.7164 after the constant lift.
```
NEW:
```latex
Which of the three moves works best depends on the dataset. Censored EM is best on LogD and QM9. The constant lift is best on hERG $K_i$. There the network on the second single representation goes 0.5891 before clipping, 0.7485 clipped, 0.6662 after the constant lift and 0.7037 after censored EM. The truncated-normal mean is behind both on LogD, hERG $K_i$ and Caco-2. On Buchwald--Hartwig only the random forest was hurt by clipping, and censored EM brought it back to 17.585 RMSE from 18.691, in percentage points of yield, against 17.594 before clipping. On the antibody thermostability set, scored by Spearman, the network on the hybrid goes 0.7487, 0.6097 and 0.7164 after the constant lift.
```

### E36 — Split deletion-of-clipped sentence
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Deleting the clipped molecules uses the same flag and throws the molecules away. It is the worst choice. It is worse than leaving the labels clipped on Caco-2 (0.3403 against 0.3042, network on the hybrid), hERG $K_i$ (0.8962 against 0.7485, network on the second single representation) and QM9 (0.5412 against 0.5196). On LogD it beats the clipped labels with the network on the hybrid, 0.5547 against 0.6049, but stays far behind the lift. The clipped molecules are the high end of the label range, and without them a model never sees a high label.
```
NEW:
```latex
Deleting the clipped molecules uses the same flag and throws the molecules away. It is the worst choice. It is worse than leaving the labels clipped on Caco-2 (0.3403 against 0.3042, network on the hybrid) and QM9 (0.5412 against 0.5196). The same holds on hERG $K_i$ (0.8962 against 0.7485, network on the second single representation). On LogD it beats the clipped labels with the network on the hybrid, 0.5547 against 0.6049, but stays far behind the lift. The clipped molecules are the high end of the label range, and without them a model never sees a high label.
```

### E37 — Split scaffold-offset sentence
- file: thesis.tex
- kind: ai-text
- why: 35-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Under grouped-shifted noise every scaffold family carries its own offset, and the family is a known flag. Subtracting each family's estimated offset worked on all three datasets it ran on, Caco-2, hERG $K_i$ and LogD. On Caco-2, the network on the hybrid scored 0.3214, 0.3443 and 0.3612 RMSE on the three damaged draws, and 0.2796, 0.2936 and 0.2668 after the random-intercept model, against 0.2385, 0.2423 and 0.2314 on clean labels. On LogD the two draws went 0.6837 and 0.6899 to 0.5457 and 0.5725, with clean labels at 0.3981 and 0.4017. The random-intercept model, which shrinks the offset of a small family, was best on Caco-2 and LogD. On hERG $K_i$ the plain family mean did as well, 0.8042 to 0.6914 on the first draw against 0.6934 for the random intercept.
```
NEW:
```latex
Under grouped-shifted noise every scaffold family carries its own offset, and the family is a known flag. Subtracting each family's estimated offset worked on all three datasets it ran on, Caco-2, hERG $K_i$ and LogD. On Caco-2, the network on the hybrid scored 0.3214, 0.3443 and 0.3612 RMSE on the three damaged draws. After the random-intercept model it scored 0.2796, 0.2936 and 0.2668, against 0.2385, 0.2423 and 0.2314 on clean labels. On LogD the two draws went 0.6837 and 0.6899 to 0.5457 and 0.5725, with clean labels at 0.3981 and 0.4017. The random-intercept model, which shrinks the offset of a small family, was best on Caco-2 and LogD. On hERG $K_i$ the plain family mean did as well, 0.8042 to 0.6914 on the first draw against 0.6934 for the random intercept.
```

### E38 — Offsets on delivered labels are not activity
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide item 17: Caco-2 and LogD labels are not biological activities; "scaffolds" made the same as "scaffold families" in the rest of the paragraph.
- todo_line: keep existing
OLD:
```latex
On the labels as delivered the same methods made every dataset worse. On Caco-2 the network on the hybrid went from 0.2333 to 0.2669 after subtracting family means, and on LogD from 0.3977 to 0.4872. Nothing had shifted a family, so the estimated offsets were real differences in activity between scaffolds, and subtracting them removed signal. That is the right answer for a method that assumes damage that is not there. It is also a warning against using it without reason to think the damage exists.
```
NEW:
```latex
On the labels as delivered the same methods made every dataset worse. On Caco-2 the network on the hybrid went from 0.2333 to 0.2669 after subtracting family means, and on LogD from 0.3977 to 0.4872. Nothing had shifted a family, so the estimated offsets were real differences in the label between scaffold families, and subtracting them removed signal. That is the right answer for a method that assumes damage that is not there. It is also a warning against using it without reason to think the damage exists.
```

### E39 — Buchwald--Hartwig unit (sources)
- file: thesis.tex
- kind: ai-text
- why: Buchwald--Hartwig RMSE without unit (chemistry guide item 29).
- todo_line: keep existing
OLD:
```latex
With simulated sources, removing each source's offset made little difference on small molecules. On hERG $K_i$ the network on the hybrid scored 0.7423 with the plain mean of the measurements and 0.7485 after removing the offsets, against 0.5989 on clean labels. The averaged label was likely already close to the truth there, since averaging over several sources cancels part of their offsets. Buchwald--Hartwig, where the sources' offsets were large against the label spread, is the one dataset where removing them helped in most settings. Even there the gain can be small: the random forest on the hybrid moved from 21.315 to 21.080 RMSE, against 17.797 on clean labels.
```
NEW:
```latex
With simulated sources, removing each source's offset made little difference on small molecules. On hERG $K_i$ the network on the hybrid scored 0.7423 with the plain mean of the measurements and 0.7485 after removing the offsets, against 0.5989 on clean labels. The averaged label was likely already close to the truth there, since averaging over several sources cancels part of their offsets. Buchwald--Hartwig, where the sources' offsets were large against the label spread, is the one dataset where removing them helped in most settings. Even there the gain can be small: the random forest on the hybrid moved from 21.315 to 21.080 RMSE, in percentage points of yield, against 17.797 on clean labels.
```

### E40 — Split flag comparison sentence
- file: thesis.tex
- kind: ai-text
- why: 35-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Set side by side, the damage kinds sort by how much the treatment knows. Clipping and scaffold-family offsets come with a flag that says which labels are affected, and on every small-molecule dataset they ran on the flag-driven moves won back about half or more of the lost RMSE. Outliers, Gaussian noise and shifts in a few labels have no flag, and the treatment depends on a detector. Those were helped too, as Section~\ref{sec:noise:res:action} shows, but the gains were smaller.
```
NEW:
```latex
Set side by side, the damage kinds sort by how much the treatment knows. Clipping and scaffold-family offsets come with a flag that says which labels are affected. On every small-molecule dataset they ran on, the flag-driven moves won back about half or more of the lost RMSE. Outliers, Gaussian noise and shifts in a few labels have no flag, and the treatment depends on a detector. Those were helped too, as Section~\ref{sec:noise:res:action} shows, but the gains were smaller.
```

### E41 — Split SMAC move sentence
- file: thesis.tex
- kind: ai-text
- why: 35-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
SMAC colloidal stability gives the strongest controlled result on any labels as delivered. Deleting the 20\% of antibodies the forest residual flags takes the support vector regressor on the first single representation from 0.2740 to 0.4034 Spearman. The same deletion with shuffled detector scores gives 0.2207, 0.2345 and 0.2782. On the second hybrid it goes from 0.3108 to 0.3782, against 0.3281, 0.2784 and 0.3002 shuffled. Moving the flagged labels toward an out-of-fold prediction takes the network from 0.0651 to 0.3185 on the single representation and from 0.2267 to 0.4423 on the hybrid, where no shuffle gets above 0.1327 and 0.2228. These are large gains, and fold 0 holds out only 53 antibodies, so one fold's Spearman can move a long way between fits. The shuffled control answers part of that, because the shuffles were fitted on the same fold and stayed low.
```
NEW:
```latex
SMAC colloidal stability gives the strongest controlled result on any labels as delivered. Deleting the 20\% of antibodies the forest residual flags takes the support vector regressor on the first single representation from 0.2740 to 0.4034 Spearman. The same deletion with shuffled detector scores gives 0.2207, 0.2345 and 0.2782. On the second hybrid it goes from 0.3108 to 0.3782, against 0.3281, 0.2784 and 0.3002 shuffled. Moving the flagged labels toward an out-of-fold prediction takes the network from 0.0651 to 0.3185 on the single representation, where no shuffle gets above 0.1327. On the hybrid it goes from 0.2267 to 0.4423, where no shuffle gets above 0.2228. These are large gains, and fold 0 holds out only 53 antibodies, so one fold's Spearman can move a long way between fits. The shuffled control answers part of that, because the shuffles were fitted on the same fold and stayed low.
```

### E42 — Split ddG and titer sentences
- file: thesis.tex
- kind: ai-text
- why: 34- and 31-word sentences (task item 5).
- todo_line: keep existing
OLD:
```latex
Binding free energy and expression titer also pass the shuffled control on their labels as delivered. On binding free energy, label smoothing takes the network's first single representation from 0.3414 to 0.4840 Spearman, against 0.2772, 0.3520 and 0.4595 shuffled, and its hybrid from 0.2589 to 0.2866, against at most 0.2759. On expression titer, deletion takes the random forest's second single representation from 0.3445 to 0.4068, against at most 0.4007, and its second hybrid from 0.2244 to 0.3168, against at most 0.2945. On both datasets the cleaned single representation still scores above the cleaned hybrid.
```
NEW:
```latex
Binding free energy and expression titer also pass the shuffled control on their labels as delivered. On binding free energy, label smoothing takes the network's first single representation from 0.3414 to 0.4840 Spearman, against 0.2772, 0.3520 and 0.4595 shuffled. Its hybrid goes from 0.2589 to 0.2866, against at most 0.2759. On expression titer, deletion takes the random forest's second single representation from 0.3445 to 0.4068, against at most 0.4007. Its second hybrid goes from 0.2244 to 0.3168, against at most 0.2945. On both datasets the cleaned single representation still scores above the cleaned hybrid.
```

### E43 — Antibody fold size: remove an unmeasured comparison
- file: thesis.tex
- kind: ai-text
- why: The sentence compared a Spearman gain of 0.05 with 5\% of an RMSE on "a thousand held-out molecules", a comparison that was never computed and mixes two metrics; 36 words.
- todo_line: keep existing
OLD:
```latex
Two things about these datasets make large gains easier to see. Their held-out folds are small, 44 to 135 antibodies, and several start from weak scores, so there is room to move. And a Spearman on 53 antibodies moves a long way between fits, so a gain of 0.05 there is a smaller step than 5\% of the RMSE on a small-molecule set with a thousand held-out molecules. The controls above are why we still count SMAC, binding free energy and expression titer as real results. After cleaning, the hybrid is ahead of the single representation on these datasets in about half the conditions, much as it was before cleaning.
```
NEW:
```latex
Two things about these datasets make large gains easier to see. Their held-out folds are small, 44 to 135 antibodies, and several start from weak scores, so there is room to move. And a Spearman on 53 antibodies moves a long way between fits, so a gain there is weaker evidence than the same gain on a larger held-out set. The controls above are why we still count SMAC, binding free energy and expression titer as real results. After cleaning, the hybrid is ahead of the single representation on these datasets in about half the conditions, much as it was before cleaning.
```

### E44 — Small-molecule summary: split, reactions not rows
- file: thesis.tex
- kind: ai-text
- why: 37-word sentence (task item 5); "rows" for Buchwald--Hartwig reactions (author's terms).
- todo_line: keep existing
OLD:
```latex
The small-molecule datasets show the same split as the methods. Under injected damage they are helped often and by useful amounts, most of all when a flag says which labels are wrong: on LogD, censored EM won back 76\% of the RMSE that clipping cost. On the labels as delivered the controlled gains are small, with one exception. hERG $K_i$'s label smoothing gain, 0.6380 to 0.5775 RMSE on the single representation, is larger than the network's own spread and passes its control. No other gain on small-molecule labels as delivered does both. hERG under flips is the clearest controlled result under damage, on three folds. Buchwald--Hartwig is the one dataset where deleting flagged rows from the labels as delivered beats random deletion.
```
NEW:
```latex
The small-molecule datasets show the same split as the methods. Under injected damage they are helped often and by useful amounts, most of all when a flag says which labels are wrong. On LogD, censored EM won back 76\% of the RMSE that clipping cost. On the labels as delivered the controlled gains are small, with one exception. hERG $K_i$'s label smoothing gain, 0.6380 to 0.5775 RMSE on the single representation, is larger than the network's own spread and passes its control. No other gain on small-molecule labels as delivered does both. hERG under flips is the clearest controlled result under damage, on three folds. Buchwald--Hartwig is the one dataset where deleting flagged reactions from the labels as delivered beats random deletion.
```

### E45 — hERG IC50 results: split
- file: thesis.tex
- kind: ai-text
- why: 45-, 34- and 32-word sentences (task item 5).
- todo_line: keep existing
OLD:
```latex
hERG IC$_{50}$ is the one dataset whose noise was measured rather than injected. Weighting each molecule by its number of measurements took the support vector regressor on the hybrid from 0.7598, 0.7663 and 0.7615 RMSE on three draws to 0.7283, 0.7218 and 0.7068, and weighting by the inverse of the estimated error variance to 0.7351, 0.7377 and 0.7382. The same count weights made the network worse, 0.7119 to 0.7647 on the first draw. For the network, adding the log of the measurement count as a feature helped instead, 0.7119, 0.7010 and 0.6994 to 0.6556, 0.6649 and 0.6320. Shrinking each label toward a prediction by its spread also helped, to 0.6733, 0.6677 and 0.6589. NGBoost gained a little from the same extra feature, and the interval model, which fits each molecule's range of measurements, did slightly worse than the point model, 0.7360 against 0.7244 on the first draw. Training on one row per measurement instead of the mean barely changed the score: 0.7668 to 0.7652 for the support vector regressor on the hybrid and 0.7256 to 0.7343 for the network. These scores are on held-out molecules with three or more measurements, where the mean label is closest to the truth. The detection of molecules flagged by ChEMBL curators has not run.
```
NEW:
```latex
hERG IC$_{50}$ is the one dataset whose noise was measured rather than injected. Weighting each molecule by its number of measurements took the support vector regressor on the hybrid from 0.7598, 0.7663 and 0.7615 RMSE on three draws to 0.7283, 0.7218 and 0.7068. Weighting by the inverse of the estimated error variance gave 0.7351, 0.7377 and 0.7382. The same count weights made the network worse, 0.7119 to 0.7647 on the first draw. For the network, adding the log of the measurement count as a feature helped instead, 0.7119, 0.7010 and 0.6994 to 0.6556, 0.6649 and 0.6320. Shrinking each label toward a prediction by its spread also helped, to 0.6733, 0.6677 and 0.6589. NGBoost gained a little from the same extra feature. The interval model, which fits each molecule's range of measurements, did slightly worse than the point model, 0.7360 against 0.7244 on the first draw. Training on one row per measurement instead of the mean barely changed the score. The support vector regressor on the hybrid went from 0.7668 to 0.7652 and the network from 0.7256 to 0.7343. These scores are on held-out molecules with three or more measurements, where the mean label is closest to the truth. The detection of molecules flagged by ChEMBL curators has not run.
```

### E46 — Split clean-before-search sentence
- file: thesis.tex
- kind: ai-text
- why: 40-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
The hybrid's features are chosen against the labels, so noisy labels can mislead the search itself. We tested cleaning before the search against cleaning after it (Section~\ref{sec:noise:mitigation}). We found no earlier work that cleans labels before choosing the features. On the labels as delivered of expression titer, the network on the hybrid scores 0.1550 Spearman with no cleaning. Deleting the 20\% of antibodies the forest residual flags raises it to 0.3899 when the cleaning only reaches the fit, to 0.5619 when it reaches both the search and the fit, and to 0.4231 when it reaches the search alone. On SMAC colloidal stability with a quarter of the labels clipped, deleting the flagged 5\% gives 0.2369, 0.4165, 0.4889 and 0.2214 in the same order. In both, the best result comes from cleaning before the search and fitting on the cleaned labels.
```
NEW:
```latex
The hybrid's features are chosen against the labels, so noisy labels can mislead the search itself. We tested cleaning before the search against cleaning after it (Section~\ref{sec:noise:mitigation}). We found no earlier work that cleans labels before choosing the features. On the labels as delivered of expression titer, the network on the hybrid scores 0.1550 Spearman with no cleaning. Deleting the 20\% of antibodies the forest residual flags raises it to 0.3899 when the cleaning only reaches the fit. It reaches 0.5619 when the cleaning reaches both the search and the fit, and 0.4231 when it reaches the search alone. On SMAC colloidal stability with a quarter of the labels clipped, deleting the flagged 5\% gives 0.2369, 0.4165, 0.4889 and 0.2214 in the same order. In both, the best result comes from cleaning before the search and fitting on the cleaned labels.
```

### E47 — Split largest-gain sentences
- file: thesis.tex
- kind: ai-text
- why: 37- and 32-word sentences (task item 5).
- todo_line: keep existing
OLD:
```latex
The largest single gain is on SMAC colloidal stability's labels as delivered, where the network on the second hybrid goes from $-0.0143$ to 0.5596 when the labels are shrunk toward an out-of-fold prediction for the search alone. That starts from a score near zero, on one fold of 53 antibodies. The same treatment can also go the other way. Deleting flagged antibodies before the search took the network's first hybrid on SMAC from 0.2267 to 0.0853. On hERG $K_i$ under Gaussian noise, deleting before the search took the second hybrid from 0.7181 to 0.6488 RMSE on the first noise draw, and from 0.7583 to 0.7763 on the second. On hERG $K_i$'s labels as delivered, the one dataset with a second fold here, label smoothing before the search gave 0.5294 on fold 1 against 0.5506 with no cleaning. No random-flag control has run for this experiment, and there is no QM9 result.
```
NEW:
```latex
The largest single gain is on SMAC colloidal stability's labels as delivered. There the network on the second hybrid goes from $-0.0143$ to 0.5596 when the labels are shrunk toward an out-of-fold prediction for the search alone. That starts from a score near zero, on one fold of 53 antibodies. The same treatment can also go the other way. Deleting flagged antibodies before the search took the network's first hybrid on SMAC from 0.2267 to 0.0853. On hERG $K_i$ under Gaussian noise, deleting before the search took the second hybrid from 0.7181 to 0.6488 RMSE on the first noise draw. On the second draw it went from 0.7583 to 0.7763. On hERG $K_i$'s labels as delivered, the one dataset with a second fold here, label smoothing before the search gave 0.5294 on fold 1 against 0.5506 with no cleaning. No random-flag control has run for this experiment, and there is no QM9 result.
```

### E48 — Split search-on-damaged sentence
- file: thesis.tex
- kind: ai-text
- why: 35-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
A separate experiment asks how much the search itself loses when it sees damaged labels. Both arms are fitted on labels with Gaussian noise at level 1.0 and scored on the held-out labels as delivered. One arm's hybrid was searched on the clean labels, the other's on the damaged ones. On Caco-2 the network's second hybrid scored 0.298, 0.328, 0.328, 0.368 and 0.346 RMSE on the five folds when searched on clean labels, and 0.303, 0.329, 0.334, 0.365 and 0.340 when searched on damaged labels. On LogD with the support vector regressor the two arms agree to the third decimal on two of three folds. The largest loss was on hERG $K_i$ with the network, 0.669 against 0.779 on the second fold, where one noise draw reached 0.979. On SMAC colloidal stability the two arms swap places from fold to fold. For most datasets, the search itself is not where Gaussian noise does its damage.
```
NEW:
```latex
A separate experiment asks how much the search itself loses when it sees damaged labels. Both arms are fitted on labels with Gaussian noise at level 1.0 and scored on the held-out labels as delivered. One arm's hybrid was searched on the clean labels, the other's on the damaged ones. On Caco-2 the network's second hybrid scored 0.298, 0.328, 0.328, 0.368 and 0.346 RMSE on the five folds when searched on clean labels. Searched on damaged labels, it scored 0.303, 0.329, 0.334, 0.365 and 0.340. On LogD with the support vector regressor the two arms agree to the third decimal on two of three folds. The largest loss was on hERG $K_i$ with the network, 0.669 against 0.779 on the second fold, where one noise draw reached 0.979. On SMAC colloidal stability the two arms swap places from fold to fold. For most datasets, the search itself is not where Gaussian noise does its damage.
```

### E49 — Two-learner result: split, unit
- file: thesis.tex
- kind: ai-text
- why: 41-word sentence (task item 5); Buchwald--Hartwig RMSE without unit (chemistry guide item 29).
- todo_line: keep existing
OLD:
```latex
The other noise-robust training methods are in Table~\ref{tab:noise:res:robust_broad}. A Huber loss for the network helped under outliers, 0.7413 to 0.7269 on hERG $K_i$, and made Gaussian noise worse at every threshold. Each model's own robust loss helped under the shift sized by model error, for example 0.3134 to 0.2978 for the network on Caco-2, against 0.2759 on clean labels. The two-learner selection beat its random-deletion control with tree models: on Buchwald--Hartwig under the shift in 10\%, the random forest on the first single representation went 20.3381 to 18.6007 RMSE against 20.2822, 20.5441 and 20.7828 for random deletion at three seeds. The tree models and the network ran on different datasets, though, so we cannot say whether the tree models or the datasets are the reason.
```
NEW:
```latex
The other noise-robust training methods are in Table~\ref{tab:noise:res:robust_broad}. A Huber loss for the network helped under outliers, 0.7413 to 0.7269 on hERG $K_i$, and made Gaussian noise worse at every threshold. Each model's own robust loss helped under the shift sized by model error, for example 0.3134 to 0.2978 for the network on Caco-2, against 0.2759 on clean labels. The two-learner selection beat its random-deletion control with tree models. On Buchwald--Hartwig under the shift in 10\%, the random forest on the first single representation went from 20.3381 to 18.6007 RMSE, in percentage points of yield. Random deletion at three seeds gave 20.2822, 20.5441 and 20.7828. The tree models and the network ran on different datasets, though, so we cannot say whether the tree models or the datasets are the reason.
```

### E50 — Published sweep: claim matches the table, fewer table numbers
- file: thesis.tex
- kind: ai-text
- why: "Never works" contradicts the table (small-loss selection 1 of 31; confident learning, which also deletes high-error molecules, 16 of 31); the antibody and multistage counts are already in Table tab:noise:res:sweep, so only the numbers that carry the point stay (task item 6).
- todo_line: keep existing
OLD:
```latex
Table~\ref{tab:noise:res:sweep} expands the sweep of Section~\ref{sec:lnl_motivation} to the datasets of this chapter, every damage kind and the hybrid. Each cell counts the conditions where the method beat the plain network on both the best single representation and the best hybrid. Deleting the molecules the network fits worst never works on small molecules. Small-loss selection helped in 1 of 31 small-molecule conditions, against 5 of 5 on Buchwald--Hartwig and 11 of 25 on the antibodies. Multistage cleaning helped in 0, 2 and 4. The reaction side is one dataset, so we read it as one dataset's result and no more.
```
NEW:
```latex
Table~\ref{tab:noise:res:sweep} expands the sweep of Section~\ref{sec:lnl_motivation} to the datasets of this chapter, every damage kind and the hybrid. Each cell counts the conditions where the method beat the plain network on both the best single representation and the best hybrid. Deleting the molecules with the largest training loss, as small-loss selection and multistage cleaning do, almost never helps on small molecules. Small-loss selection helped in 1 of 31 small-molecule conditions and multistage cleaning in none. On Buchwald--Hartwig, small-loss selection helped in all five. The reaction side is one dataset, so we read it as one dataset's result and no more.
```

### E51 — Split sweep-reversal sentences
- file: thesis.tex
- kind: ai-text
- why: 41- and 39-word sentences (task item 5).
- todo_line: keep existing
OLD:
```latex
The sweep agrees with the earlier study in one place and reverses it in another. In Chapter~\ref{chap:flaws}, the heteroscedastic GP held up best of the twelve under Gaussian noise on QM9. Here it is the only method that helps on binding free energy's labels as delivered, 0.3414 to 0.4475 Spearman on the best single representation and 0.2498 to 0.3663 on the second hybrid, though the first hybrid drops from 0.3048 to 0.2880. Distance-based selection was the worst of the twelve in the earlier study. On hERG with 10\% of labels flipped one way only, it is the only one of the twelve that helps both sides, 0.8151 to 0.8166 AUROC on the first single representation and 0.8214 to 0.8245 on the first hybrid. The same method gives the opposite verdict on a different task and different data.
```
NEW:
```latex
The sweep agrees with the earlier study in one place and reverses it in another. In Chapter~\ref{chap:flaws}, the heteroscedastic GP held up best of the twelve under Gaussian noise on QM9. Here it is the only method that helps on binding free energy's labels as delivered. It takes the best single representation from 0.3414 to 0.4475 Spearman and the second hybrid from 0.2498 to 0.3663. The first hybrid drops from 0.3048 to 0.2880. Distance-based selection was the worst of the twelve in the earlier study. On hERG with 10\% of labels flipped one way only, it is the only one of the twelve that helps both sides. It goes from 0.8151 to 0.8166 AUROC on the first single representation and from 0.8214 to 0.8245 on the first hybrid. The same method gives the opposite verdict on a different task and different data.
```

### E52 — Limitations: split fold list
- file: thesis.tex
- kind: ai-text
- why: 33-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Most results rest on the first outer fold. Further folds exist for the clipped-label corrections on LogD and Caco-2, deletion under hERG's flips, label smoothing and cliff put-back on hERG $K_i$, and the comparison of searches on clean and damaged labels. They mostly agree with fold 0, and the text gives the exceptions. The results to confirm on all folds first are, in order:
```
NEW:
```latex
Most results rest on the first outer fold. Further folds exist for the clipped-label corrections on LogD and Caco-2 and for deletion under hERG's flips. They also exist for label smoothing and cliff put-back on hERG $K_i$, and for the comparison of searches on clean and damaged labels. They mostly agree with fold 0, and the text gives the exceptions. The results to confirm on all folds first are, in order:
```

### E53 — Limitations: five controls, not six
- file: thesis.tex
- kind: ai-text
- why: Section sec:noise:tests says "Five controls ran" and lists five; this paragraph said six.
- todo_line: keep existing
OLD:
```latex
Most methods have no control. Six controls ran, one per experiment (Section~\ref{sec:noise:tests}), and none on an antibody dataset under injected damage. The random put-back control was usable only on hERG $K_i$. Three of the methods with large gains have no QM9 result with no noise added: changing the model, cleaning before the search, and moving every label toward a Huber-fitted prediction. Until those exist, their gains cannot be split into noise removed and a better model.
```
NEW:
```latex
Most methods have no control. Five controls ran, one per experiment (Section~\ref{sec:noise:tests}), and none on an antibody dataset under injected damage. The random put-back control was usable only on hERG $K_i$. Three of the methods with large gains have no QM9 result with no noise added: changing the model, cleaning before the search, and moving every label toward a Huber-fitted prediction. Until those exist, their gains cannot be split into noise removed and a better model.
```

### E54 — Limitations: split ground-truth sentence
- file: thesis.tex
- kind: ai-text
- why: 34-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
QM9 is our only clean control, and we tested one of its properties, the HOMO--LUMO gap. A control on more computed properties would show whether the QM9 results hold beyond that one target. There is also no way to correct a label without ground truth and know it was corrected, and no dataset here has a measured floor of assay error below which no model can go. Seven of the datasets in Chapters~\ref{chap:method} and~\ref{chap:similarity} have no noise results yet: adme\_mdr1, adme\_rlm, flab\_expression\_koenig, flab\_binding\_koenig, abagym\_EGFR, abagym\_Ang2 and ginkgo\_hic.
```
NEW:
```latex
QM9 is our only clean control, and we tested one of its properties, the HOMO--LUMO gap. A control on more computed properties would show whether the QM9 results hold beyond that one target. There is also no way to correct a label without ground truth and know it was corrected. Nor does any dataset here have a measured floor of assay error below which no model can go. Seven of the datasets in Chapters~\ref{chap:method} and~\ref{chap:similarity} have no noise results yet: adme\_mdr1, adme\_rlm, flab\_expression\_koenig, flab\_binding\_koenig, abagym\_EGFR, abagym\_Ang2 and ginkgo\_hic.
```

### E55 — Conclusion: split
- file: thesis.tex
- kind: ai-text
- why: 31-word sentence (task item 5).
- todo_line: keep existing
OLD:
```latex
Noisy labels in QSAR data can be found and, under the right conditions, fixed. Fixing works best with a flag: raising labels clipped at an assay limit won back half to three quarters of the lost RMSE on the small-molecule datasets, and needed no detector. With a detector, deleting or relabeling flagged molecules helped under label flips and outliers, and passed random and shuffled controls on hERG, hERG $K_i$, Buchwald--Hartwig and three antibody sets. On clean QM9 labels deletion never helped, so those gains are noise removed. The hybrid stays ahead of the single representation on small molecules through every treatment, and the ranking of noise-handling methods depends on the feature space they run in. Chapter~\ref{chap:discussion} brings these results together with the rest of the thesis.
```
NEW:
```latex
Noisy labels in QSAR data can be found and, under the right conditions, fixed. Fixing works best with a flag. Raising labels clipped at an assay limit won back half to three quarters of the lost RMSE on the small-molecule datasets, with no detector. With a detector, deleting or relabeling flagged molecules helped under label flips and outliers, and passed random and shuffled controls on hERG, hERG $K_i$, Buchwald--Hartwig and three antibody sets. On clean QM9 labels deletion never helped, so those gains are noise removed. The hybrid stays ahead of the single representation on small molecules through every treatment, and the ranking of noise-handling methods depends on the feature space they run in. Chapter~\ref{chap:discussion} brings these results together with the rest of the thesis.
```

### E56 — Lead-in for the results by noise type
- file: thesis.tex
- kind: new-paragraph
- why: Style guide item 4: B and V put one or two sentences between a section heading and its first subsection; this section went straight into its first subsection. Anchored on the \label so it survives the title edits.
- todo_line: % TODO: manual review
OLD:
```latex
\label{sec:noise:res:kind}
```
NEW:
```latex
\label{sec:noise:res:kind}

% TODO: manual review
This section takes the kinds of damage one at a time. It starts with the three that come with a known flag: clipped labels, offsets shared within a scaffold family, and several sources per molecule. It ends with the kinds that need a detector.
```

### E57 — Lead-in for the results by dataset
- file: thesis.tex
- kind: new-paragraph
- why: Style guide item 4, as above. Anchored on the \label so it survives the title edits.
- todo_line: % TODO: manual review
OLD:
```latex
\label{sec:noise:res:dataset}
```
NEW:
```latex
\label{sec:noise:res:dataset}

% TODO: manual review
This section reads the results by domain: the antibody datasets first, then the small-molecule and reaction datasets, then hERG IC$_{50}$, the one dataset with real repeat measurements.
```

## For the author (not changed)

- K_i typography. Chapter 7 uses `$K_i$` and `p$K_i$` consistently, matching the rest of the thesis. IUPAC sets the subscript roman (`$K_\mathrm{i}$`); that is one macro and a thesis-wide replacement (chemistry guide item 10), so it is not planned here. The censoring figure's panel title prints "hERG Ki" in roman type.
- "Scaffold family" and "scaffold group" are both used in this chapter for the same set of molecules. Example: the mitigation table row "Scaffold-group offset & Scaffold family & All & Move & Minus the family's mean residual". The Chapter 2 plan defines "scaffold family"; the chemistry guide (item 14) prefers "scaffold group". This is a thesis-wide choice.
- "Binding free energy" names a dataset of ΔΔG, the change in binding free energy on mutation (chemistry guide item 8). Used about ten times here, e.g. "On binding free energy, label smoothing takes the network's first single representation". Thesis-wide rename, not planned here.
- The hERG blocking dataset is called "hERG" throughout this chapter ("Under hERG's label flips"), beside "hERG $K_i$" and "hERG IC$_{50}$". Chapter 5 calls it "hERG blocking". Three hERG datasets in one chapter make the short name easy to misread.
- Cliff put-back: "a partner with ECFP4 Tanimoto similarity of at least 0.55". MoleculeACE (Chapter 2) uses a similarity above 90%. With 0.55, between 54% and 72% of the training molecules on Caco-2, hERG and LogD count as cliff molecules, which is why no random control fits there. Check that 0.55 is the threshold you want, and say why in the text.
- SMAC is never expanded (first use in Chapter 5, "SMAC colloidal stability"). Add it to the abbreviations or expand it once.
- Three `Table~\ref` targets do not exist yet and will print "??": tab:noise:res:bygroup ("Table~\ref{tab:noise:res:bygroup} shows how often this happens"), tab:noise:res:robust_broad ("The other noise-robust training methods are in Table~\ref{tab:noise:res:robust_broad}") and the planned mitigation table. All three wait for your approval of the table form.
- Censoring direction for LogD. E5 says real potency censoring falls mostly at the low end. For LogD either end can be censored; no sentence says which end a LogD assay limits. Not verified against the OpenADMET source.
- Task item 6 (numbers already in a float): the clipped-label paragraph ("On LogD, the network on the best single representation goes from 0.4314 RMSE ...") repeats values that Figure fig:noise:censoring_ladder plots. They were kept because ch7_numbers.py re-checks them by the [N7-20] ID. If you cut them, cut them there too.
- Headline claims carry no fold count (style guide item 35): "Noise mitigation works." opens Section sec:noise:res:overall and the Discussion. Most of its support is fold 0; the hERG deletion result holds on three folds.
- Units of RMSE for hERG IC$_{50}$ are pIC$_{50}$ units; the text gives none ("from 0.7598, 0.7663 and 0.7615 RMSE on three draws"). Left, since the other small-molecule RMSEs in the chapter also go without units.
- The many score-pair sentences under 30 words but dense with numbers (e.g. the LogD outlier paragraph) were split, not cut. The style guide (item 28) suggests per-fold tables for them, which is outside this plan (no new tables).
- Censoring-ladder caption (author text, "Held-out RMSE (lower is better) at four steps"): it does not say that each panel is one dataset (Caco-2, hERG $K_i$, LogD, QM9) or give the RMSE unit. Nothing in it is wrong, so it is not edited here (style guide item 16).
- The chapter's conclusion already points forward ("Chapter~\ref{chap:discussion} brings these results together with the rest of the thesis."), so no edit there.

## Checker notes

Checked 2026-10-10 against thesis.tex. All 58 OLD blocks were exact, unique substrings (python3 -I, str.count == 1). E1 to E55 each sit in a paragraph with "% TODO: manual review" above it, so ai-text is correct for all of them. No label changed, no existing comment dropped, no banned phrasing found. Facts spot-checked in source: hERG IC50 labels are 9 - log10(nM IC50) grouped on ChEMBL canonical_smiles (src/kirby/datasets/herg.py, tests/experiments/noise/_probe_chembl_herg.py); censoring clips the top of the label range (tab:noise:kinds row); "Five controls ran" in sec:noise:tests; sweep counts 1/5/11 and 0/2/4 in tab:noise:res:sweep; Chapter 6 claims in E1 match its results text; section order in E56 and E57 matches the headings.

- E5 fixed: NEW wrote "hERG p$K_i$", a notation used nowhere in thesis.tex; changed to "hERG $K_i$".
- E19 fixed: NEW added "on folds 0 to 2", a fold assignment the original sentence did not state; removed.
- E58 deleted: the censoring-ladder caption is author text, and adding a panel description and a unit fixes nothing wrong (taste, not error); its NEW caption also had a 46-word sentence. Moved to For the author.
- Header line updated: no author text is edited now.
