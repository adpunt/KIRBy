# Lens proposals: chap:noise and chap:discussion

Read 2026-10-10: thesis.tex lines 3391-5076 in full (chap:noise has no \input; every table is inline).
These are NEW items only. The comments in comments_ch6.md and comments_ch7.md are left to the other agent.
Line numbers are from my read today.
Ownership: every paragraph has `% [AI text, revised 2026-10-10]` directly above it unless marked "hers".
The paragraphs at 4597 ("ChEMBL curators flagged ...") and 4989 and 4991 (Conclusions, paragraphs 2 and 3) have no marker directly above them, so I treat them as hers.

Counts: 15 high, 12 low.

---

## HIGH

### X-noise-1
Location: line 3410, sec:noise:methods (and the captions of fig:noise:score_vs_level, line 3822, and fig:noise:detection_vs_cleaning, line 4316); Owner: AI
Quote: "Within each outer fold, a search on the training molecules picks the two best single representations and two best hybrids for each of the dataset's two downstream models."
His comment: What are "first single", "second hybrid"? Is the "first single" in the tables the same as the "best single representation" in the text? On hERG K_i the second single representation scores better than the first.
Lens item: 4 (also 17)
Why it matters: the tables and the text use "first single", "best single representation", "first hybrid" and "top hybrid" for the configurations, and none of these names is defined. In tab:noise:res:change_model_delivered the network's "Second single" on hERG K_i (0.589) beats its "First single" (0.638). So "first" cannot mean best on the held-out molecules.
Verified: tests/experiments/_shared.py lines 1820-1940. 'hybrid' is the model's top hybrid from the per-fold general search, and 'hybrid__general2' is the next one with distinct features. I did not check how the two single representations are ordered. NEEDS HER: confirm they are ordered by the same score on the training molecules.
This rewrite also covers X-noise-3 (same paragraph).
Proposed change:
old_string:
```
In most experiments, the representations are chosen on clean labels. Within each outer fold, a search on the training molecules picks the two best single representations and two best hybrids for each of the dataset's two downstream models. Every method is tested on all four. Only the hybrid's feature ranking and budget split are redone on the damaged labels. The rest of the search uses clean labels, which a real user would not have. Two experiments differ.
```
new_string:
```
In most experiments, the representations are chosen on clean labels. Within each outer fold, a search on the training molecules picks the two best single representations and two best hybrids for each of the dataset's two downstream models. We call them the first and second single representation and the first and second hybrid, in the order of the search's score on the training molecules. The first therefore need not score best on the held-out molecules. Every method is tested on all four. Only the hybrid's feature ranking and budget split are redone on the damaged labels. The rest of the search uses clean labels, which a real user would not have. Section~\ref{sec:noise:res:search} measures what this costs. Searching on labels with Gaussian noise barely changed the held-out RMSE of the network on Caco-2 or of the support vector regressor on LogD. The largest loss was on hERG $K_i$ with the network, 0.669 against 0.779 RMSE on one fold. Two experiments differ.
```
Captions: in the fig:noise:score_vs_level caption, replace "Solid lines are the top hybrid" with "Solid lines are the first hybrid". In the fig:noise:detection_vs_cleaning caption, replace "Network on the top hybrid" with "Network on the first hybrid". The figure legends come from tests/analysis/thesis_figure_style.py CONFIG_LABELS ('hybrid': 'Top hybrid'). Change it there to 'First hybrid' and redraw, if the drawn legends say "Top hybrid".
Priority: high

### X-noise-2
Location: line 3719, Results opening; Owner: AI
Quote: "Second, where a method has several settings, we report its best on the held-out score, which makes every gain optimistic for the single representation and the hybrid alike."
His comment: Choosing the setting on the held-out fold and then reporting the score on that same fold is data snooping. How large can this make a gain? Is there an inner validation?
Lens item: 8
Why it matters: the reader needs to know that the QM9 test bounds this effect. The same best-setting choice is made there: tab:noise:res:qm9_deletion says "the best of 5, 10 and 20% deleted", tab:noise:res:weighting says "at its best setting", and the source comment at line 3731 says "Largest gain over every cleanup setting".
Proposed change:
old_string:
```
Second, where a method has several settings, we report its best on the held-out score, which makes every gain optimistic for the single representation and the hybrid alike.
```
new_string:
```
Second, where a method has several settings, we report its best on the held-out score. No inner folds choose the setting, so every gain is optimistic, for the single representation and the hybrid alike. The same choice is made on QM9 with no noise added. Section~\ref{sec:noise:res:qm9test} therefore shows how large a gain the choice of setting alone can give.
```
NEEDS HER: confirm "No inner folds choose the setting". The original sentence implies it but does not say it.
Priority: high

### X-noise-3
Location: line 3410, sec:noise:methods; Owner: AI
Quote: "The rest of the search uses clean labels, which a real user would not have."
His comment: So the representations were chosen with information a user does not have. Does this favor the reported results? Say here how much it matters.
Lens item: 8
Proposed change: included in the X-noise-1 rewrite. It adds the pointer to sec:noise:res:search and the hERG K_i pair 0.669 against 0.779 from tab:noise:res:search_clean_vs_damaged.
Priority: high

### X-noise-4
Location: tables throughout chap:noise; Owner: AI (the captions have the marker above their table)
Quote, for example: tab:noise:res:offsets_delivered bolds 0.233 (as delivered, the best) and 0.267 (family mean, the worst) in the Caco-2 row. tab:noise:res:detect_residual bolds 0.193 on Buchwald--Hartwig, which is below chance. tab:noise:res:qm9_deletion bolds 0.355 and 0.393, a before-and-after pair.
His comment: Bold usually means the best result. Here bold marks the values quoted in the text, and some of them are the worst in the row. The reader will be misled.
Lens item: 16
Tables affected: herg_flip_actions, delete_vs_random, logd_outlier_actions, known_flags, weighting, qm9_deletion, hergki_smoothing_folds, detect_residual, detect_families, importance_scaling, breakeven, censoring_moves_main, censoring_lift, family_offsets, offsets_delivered, smac_weights, search_clean_vs_damaged, change_model, change_model_delivered, sweep_examples, qm9_sweep, neighbor_space, herg_ic50. These are all hand-written in thesis.tex, not generated.
Proposed change: option A is to add one sentence to each of these captions, "Bold: the values quoted in the text." Option B is to re-bold the best value per block, at most four bold cells per table. Example for A:
old_string:
```
Removing a shared offset per scaffold family from the labels as delivered; QM9 with no noise added. Network on the first hybrid. One row is one dataset. Held-out RMSE, lower is better. Outer fold 0.
```
new_string:
```
Removing a shared offset per scaffold family from the labels as delivered; QM9 with no noise added. Network on the first hybrid. One row is one dataset. Held-out RMSE, lower is better. Bold: the values quoted in the text. Outer fold 0.
```
Priority: high

### X-noise-5
Location: line 4976, sec:noise:res:repr (also 3739, 3893, 4595); Owner: AI
Quote: "The network gains from cleaning more often than the support vector regressor on Caco-2, hERG $K_i$ and LogD. ... On $\Delta\Delta G$ the support vector regressor gains more often, and on expression titer the random forest gains more often than the network."
His comment: Again the effect depends on the model: the network gains from smoothing and weighting on clean QM9 and LightGBM does not; count weights help the SVR and hurt the network. Is this discussed anywhere? Why?
Lens item: 18
Why it matters: the model-dependent results are scattered over four sections, and no paragraph brings them together. This paragraph is the right place.
Proposed change:
old_string:
```
The network gains from cleaning more often than the support vector regressor on Caco-2, hERG $K_i$ and LogD. Part of that may be the network's own spread between fits, since a deterministic model whose score is unchanged never counts as helped. On $\Delta\Delta G$ the support vector regressor gains more often, and on expression titer the random forest gains more often than the network.
```
new_string:
```
The same cleaning helps some models and not others. The network gains from cleaning more often than the support vector regressor on Caco-2, hERG $K_i$ and LogD. Part of that may be the network's own spread between fits, since a deterministic model whose score is unchanged never counts as helped. The network also gains from label smoothing and weighting on QM9 with no noise added, where LightGBM barely moves, as seen in Section~\ref{sec:noise:res:qm9test}. On hERG IC$_{50}$, weighting by the measurement count helped the support vector regressor and made the network worse, as seen in Table~\ref{tab:noise:res:herg_ic50}. On $\Delta\Delta G$ the support vector regressor gains more often, and on expression titer the random forest gains more often than the network.
```
NEEDS HER: one more sentence, either the reason she believes or "We did not test why." I have not written a reason. It would be my verdict.
Priority: high

### X-noise-6
Location: line 3426, sec:noise:conditions; Owner: AI
Quote: "We clip at the top of the label range, the rarer case, as real censoring of potency labels such as hERG $K_i$ falls mostly at the low end."
His comment: Why simulate the rarer case? Will the conclusions on censoring carry over to real assays, where weak binders are reported as "> 10 µM"?
Lens item: 8 (also 7)
Proposed change: NEEDS HER: one sentence giving the reason for clipping at the top. Put it right after the quoted sentence. If there is a known consequence, say it too: for example, whether clipping at the bottom was run anywhere, or whether the methods are symmetric in the limit. I found no such run in the chapter.
Priority: high

### X-noise-7
Location: line 3815, sec:noise:res:action; Owner: AI
Quote: "This control is optimistic, since the best random deletion is picked on the held-out score."
His comment: Optimistic for whom? Picking the best random deletion on the held-out score makes the control harder to beat. The sentence reads as if the method's gain were overstated.
Lens item: 19
Proposed change:
old_string:
```
This control is optimistic, since the best random deletion is picked on the held-out score.
```
new_string:
```
This control is strict. The best of the random deletions is picked on the held-out score, which favors random deletion.
```
Priority: high

### X-noise-8
Location: lines 4979-4981, \label{sec:noise:discussion} (no heading); Owner: AI
Quote: "On small molecules, if you build a hybrid, you lose nothing by cleaning the labels afterwards. Whether a noise-handling method works depends on the feature space it runs in, as Chapter~\ref{chap:flaws} found for distance."
His comment: "Lose nothing" is too strong. Deletion made QM9 worse. What exactly is the intermediate conclusion?
Lens item: 19 (also 2)
Note: the label has no \section or \subsection command, so a \ref to it would print the Results section number. The same holds for \label{sec:noise:limitations} at line 4994 (see X-noise-15).
Proposed change:
old_string:
```
On small molecules, if you build a hybrid, you lose nothing by cleaning the labels afterwards. Whether a noise-handling method works depends on the feature space it runs in, as Chapter~\ref{chap:flaws} found for distance.
```
new_string:
```
On small molecules, cleaning the labels did not cost the hybrid its lead over the best single representation, as seen in Section~\ref{sec:noise:res:repr}. Cleaning can still make a model worse, as deletion did on QM9 with no noise added. The method that did best also changed between the hybrid and the single representations, as seen in Section~\ref{sec:noise:res:sweep}.
```
Priority: high

### X-noise-9
Location: line 4183, sec:noise:res:detectors; Owner: AI
Quote: "The space in which the neighbors are found changes detection little. ... Averaged over the detector settings instead, the hybrid's features come out ahead more often, so this ranking depends on how a detector's setting is chosen."
His comment: What is the take-home? This is one of the chapter's questions: are noisy labels easier to find in a hybrid? Answer it in one sentence.
Lens item: 2
Proposed change:
old_string:
```
The space in which the neighbors are found changes detection little. We group the spaces into four families,
```
new_string:
```
Noisy labels were not easier to find with neighbors in the hybrid, at least with each detector at its best setting. The space in which the neighbors are found changes detection little. We group the spaces into four families,
```
Priority: high

### X-noise-10
Location: line 4882, sec:noise:res:sweep; Owner: AI
Quote: "In Chapter~\ref{chap:flaws}, it was also one of the methods that held up under Gaussian noise on QM9, with a retention of 79.0\%."
His comment: Retention of what, from what to what? (He asked this exact question before.)
Lens item: 4
Verified: retention is defined at line 1184 as $R^2$ under noise divided by $R^2$ with clean labels, as a percentage.
Proposed change:
old_string:
```
with a retention of 79.0\%.
```
new_string:
```
keeping 79.0\% of its clean-label $R^2$.
```
Priority: high

### X-noise-11
Location: lines 3757 and 3795 (tables tab:noise:res:herg_flip_actions and tab:noise:res:delete_vs_random); Owner: AI
Quote: the text at 3757 reads "With the neighbor-label gap as the detector, it took the hybrid to 0.838 AUROC". tab:noise:res:herg_flip_actions gives 0.838 to the "Cliff-aware gap" row (fold 0, first hybrid). tab:noise:res:delete_vs_random gives 0.838 to the "Neighbor-label gap" row.
His comment: The same value appears for two different detectors in two tables. Is one row mislabelled?
Lens item: 17
The source comments name two different experiments. Line 3759 names n23, activity_cliff_aware. Line 3786 names n24, neighbor_residual. A coincidence is possible.
Proposed change: NEEDS HER. Check both values in tests/analysis/ch7_numbers.py (N7-03, N7-05). If both are right, add one sentence to the tab:noise:res:delete_vs_random caption: "Its first row and the cliff-aware gap row of Table~\ref{tab:noise:res:herg_flip_actions} come from separate experiments that both reach 0.838."
Priority: high

### X-noise-12
Location: line 4987, sec:noise:conclusion, paragraph 1; Owner: AI
Quote: "Label smoothing, weighting and changes to the model improved QM9 as well, so their gains on measured labels are largely better training, with label smoothing on hERG $K_i$ the exception."
His comment: Link this back to drug safety. hERG is a cardiac liability, and two of the results that pass the clean-label test are on hERG.
Lens item: 14
Verified: line 3751 names label smoothing on hERG $K_i$ and on hERG blocking among the three results that pass. Line 416 describes hERG and cites Sanguinetti2006.
Proposed change:
old_string:
```
Moving every label toward a Huber-fitted prediction helped hERG $K_i$ under damage and made QM9 worse, so its gains are noise removed.
```
new_string:
```
Moving every label toward a Huber-fitted prediction helped hERG $K_i$ under damage and made QM9 worse, so its gains are noise removed. Two of the results that pass this test are on hERG, namely label smoothing on hERG $K_i$ and on hERG blocking. hERG binding is screened early in drug discovery because blocking the channel can cause arrhythmia \citep{Sanguinetti2006}.
```
Priority: high

### X-noise-13
Location: line 4991, sec:noise:conclusion, paragraph 3; Owner: hers
Quote: "The hybrid stayed ahead of the single representation on small molecules in almost every condition, before and after treatment."
His comment: The introduction asked whether each method improves the hybrid more than the best single representation. The conclusion should answer that question directly.
Lens item: 2 (also 12)
Verified: line 4948 says "Cleaning rarely changes which of the two wins, and when it does, it more often moves the win to the hybrid."
Proposed change (minimal, one added sentence):
old_string:
```
The hybrid stayed ahead of the single representation on small molecules in almost every condition, before and after treatment.
```
new_string:
```
The hybrid stayed ahead of the single representation on small molecules in almost every condition, before and after treatment. Cleaning rarely changed which of the two won, and when it did, it more often moved the win to the hybrid.
```
Priority: high

### X-disc-1
Location: line 5010, sec:discussion:summary; Owner: AI
Quote: "The worst configurations in its QM9 sweep relied on a pairwise distance measure, though some had a distance added that they were not designed with. ... Any method that relies on such a distance inherits the problem, and the adaptive hybrid representation was built to repair it."
His comment: "inherits the problem" → "is negatively impacted by it". "A distance added that they were not designed with" is hard to read.
Lens item: 19 (also 1)
Verified: line 1185 says all 20 lowest-retention configurations used a distance. Line 1190 says four methods add a distance only in some configurations.
Proposed change:
old_string:
```
The worst configurations in its QM9 sweep relied on a pairwise distance measure, though some had a distance added that they were not designed with. We then looked at the features the distance is computed over. Within one dataset, most features of a standard representation barely vary. On hERG $K_i$, 1,815 of the 2,048 ECFP4 features take one value in at least 95\% of molecules. Every feature that varies adds a term to the distance, so when only a few matter, the rest dominate it. Any method that relies on such a distance inherits the problem, and the adaptive hybrid representation was built to repair it.
```
new_string:
```
In its QM9 sweep, every one of the configurations that kept the least predictive performance used a pairwise distance between molecules. Some methods use a distance only in some of their configurations. We then looked at the features the distance is computed over. Within one dataset, most features of a standard representation barely vary. On hERG $K_i$, 1,815 of the 2,048 ECFP4 features take one value in at least 95\% of molecules. Every feature that varies adds a term to the distance. So when only a few features matter, the others can dominate it. Any method that relies on such a distance can be negatively impacted by this. The adaptive hybrid representation was built to repair it.
```
Priority: high

### X-disc-2
Location: line 5017, sec:discussion:summary; Owner: AI
Quote: "Cleaning guided by a detector beat random deletion and shuffled detector scores on hERG, hERG $K_i$, Buchwald--Hartwig and five antibody datasets."
His comment: The summary leaves out the chapter's main caveat: on clean QM9, several of these gains also appear, so they are better training, not noise removed. A reader of this chapter alone would overstate the result.
Lens item: 2 (also 8)
Verified: line 4987.
Proposed change:
old_string:
```
Cleaning guided by a detector beat random deletion and shuffled detector scores on hERG, hERG $K_i$, Buchwald--Hartwig and five antibody datasets.
```
new_string:
```
Cleaning guided by a detector beat random deletion and shuffled detector scores on hERG blocking, hERG $K_i$, Buchwald--Hartwig and five antibody datasets. QM9, whose labels are computed, showed that not every gain is noise removed. Label smoothing, weighting and changes to the model also improved QM9, so their gains on measured labels are largely better training, except label smoothing on hERG $K_i$.
```
Priority: high

---

## LOW

### X-disc-3
Location: line 5028, sec:discussion:noise; Owner: AI
Quote: "In both cases the data carry a flag that a correction can use, while a method that assumes absent noise can do harm."
His comment: "assumes absent noise"? You probably mean a method built for noise the data do not have. Also tie this to drug safety, where censored labels are common.
Lens item: 19 (also 14)
Verified: line 3426 says real censoring of hERG $K_i$ potency labels falls mostly at the low end. Line 416 cites Sanguinetti2006 for hERG screening.
Proposed change:
old_string:
```
In both cases the data carry a flag that a correction can use, while a method that assumes absent noise can do harm.
```
new_string:
```
In both cases the data carry a flag that a correction can use. A method built for a kind of noise the data do not have can do harm. Safety data are no exception. hERG binding is screened early in drug discovery \citep{Sanguinetti2006}, and real censoring of hERG $K_i$ labels falls mostly at the low end of potency.
```
Priority: low. It is high if the author wants every chapter end to carry the drug-safety link. In that case use either this item or X-disc-4, not both, to avoid repeating the citation.

### X-disc-4
Location: line 5065, sec:discussion:conclusions; Owner: AI
Quote: "Label noise in molecular data can be handled, but the right method depends on the task, the data, the model and the kind of noise."
His comment: The thesis ends without returning to drug safety and toxicity prediction.
Lens item: 14
Verified: tab:noise:res:censoring_moves_main gives the network's first hybrid on hERG $K_i$ as 0.608 before clipping, 0.731 clipped and 0.664 after censored EM.
Proposed change:
old_string:
```
No single model, representation or noise-handling method was best on every dataset we tested.
```
new_string:
```
No single model, representation or noise-handling method was best on every dataset we tested. This matters in drug safety assessment, where hERG binding is screened early \citep{Sanguinetti2006}. On hERG $K_i$, raising clipped labels won back part of the score that clipping cost.
```
Priority: low. It goes with X-disc-3; the author picks one or both.

### X-disc-5
Location: line 5049, sec:discussion:limitations; Owner: AI
Quote: "Most of the noise results come from the first outer fold. QM9 with no noise added is the only clean control, and most noise-handling methods have no control."
His comment: The limitations should also say that the settings were chosen on the held-out score.
Lens item: 8
Proposed change:
old_string:
```
Most of the noise results come from the first outer fold. QM9
```
new_string:
```
Most of the noise results come from the first outer fold. Where a method has several settings, we report its best on the held-out score, so every gain is optimistic. QM9
```
Priority: low. It repeats X-noise-2 in the limitations; it becomes high if X-noise-2 is not taken.

### X-noise-14
Location: line 4268, sec:noise:res:detectors; Owner: AI
Quote: "With a calibration set drawn from the noisy training labels, it flagged nothing at $q = 0.05$ or $q = 0.1$. A clean calibration set, which a user would not have, flagged 10 molecules at $q = 0.1$ in one of 120 passes on the measured datasets. On QM9 with outliers or a shift injected, it flagged 23 to 485 molecules in 21 of 28 passes at $q = 0.1$."
His comment: The reader is flooded with counts. What is the message? And what is q? It was never defined.
Lens item: 2 (also 4). In the methods (line 3527) the rates are given as 5, 10, 20% and q is never named.
Proposed change:
old_string:
```
Turning such a ranking into a decision is harder. The second experiment turned the residuals into conformal $p$-values and flagged molecules with a false-discovery-rate rule. With a calibration set drawn from the noisy training labels, it flagged nothing at $q = 0.05$ or $q = 0.1$. A clean calibration set, which a user would not have, flagged 10 molecules at $q = 0.1$ in one of 120 passes on the measured datasets. On QM9 with outliers or a shift injected, it flagged 23 to 485 molecules in 21 of 28 passes at $q = 0.1$. At $q = 0.05$ it flagged molecules in 9 passes.
```
new_string:
```
Turning such a ranking into flags gave almost no flags on measured labels. The second experiment turned the residuals into conformal $p$-values. It flagged molecules by the Benjamini--Hochberg rule at a bound $q$ on the expected share of false flags. With a calibration set drawn from the noisy training labels, it flagged nothing at $q = 0.05$ or $q = 0.1$. A clean calibration set, which a user would not have, flagged 10 molecules at $q = 0.1$ in one of 120 passes on the measured datasets. Only on QM9 with outliers or a shift injected did it flag often. There it flagged 23 to 485 molecules in 21 of 28 passes at $q = 0.1$, and flagged molecules in 9 passes at $q = 0.05$.
```
Priority: low

### X-noise-15
Location: line 3751, sec:noise:res:qm9test; Owner: AI
Quote: "The test rests on one computed property of one dataset, and a few methods have no QM9 result, as discussed in Section~\ref{sec:noise:limitations}."
His comment: The reader follows the reference and finds nothing. The limitations paragraph does not name the methods without a QM9 result. Name them here.
Lens item: 5
Note: \label{sec:noise:limitations} (line 4994) has no heading, so the reference prints the Conclusions section number. The paragraph there mentions only that the search was never cleaned on QM9.
Proposed change: NEEDS HER: the list of methods with no QM9 result. The hERG IC$_{50}$ repeat-measurement methods are one case by construction. Then:
old_string:
```
and a few methods have no QM9 result, as discussed in Section~\ref{sec:noise:limitations}.
```
new_string:
```
and a few methods have no QM9 result, namely [NEEDS HER: list].
```
Priority: low

### X-noise-16
Location: line 3748, sec:noise:res:qm9test; Owner: AI
Quote: "The twelve published methods pass the test in the other direction."
His comment: Which direction? Deletion also "passed" by making QM9 worse, so this phrase is hard to follow.
Lens item: 1
Proposed change:
old_string:
```
The twelve published methods pass the test in the other direction. On QM9
```
new_string:
```
The twelve published methods mostly behave as a method that removes noise should. On QM9
```
Priority: low

### X-noise-17
Location: line 4444, sec:noise:res:censoring; Owner: AI
Quote: "On these, clipping a quarter of the training labels raised the held-out Spearman in most configurations"
His comment: Why would clipping a quarter of the labels improve the model? The reader will wonder.
Lens item: 21
Proposed change: NEEDS HER: one sentence on a possible reason, or "We did not test why." Put it after the quoted sentence. (Spearman is rank-based. Clipping changes the ranks only among the clipped top quarter, which may matter. This is unverified and is not a proposal for text.)
Priority: low

### X-noise-18
Location: line 3725, sec:noise:res:overall; Owner: AI
Quote: "Mitigation helps most often under injected damage and least often on the labels as delivered, as counted for each group in Additional file~\ref{app:add:noise_bygroup}."
His comment: A one-sentence subsection that sends the reader to an appendix. Restate the key result here.
Lens item: 5 (also 11)
Proposed change: NEEDS HER. The appendix holds counts (tab_noise_bygroup), and her rule keeps tallies out of results. So the options are (a) one score pair per group here, or (b) merge this sentence into the opening of sec:noise:res:qm9test and drop the one-line subsection.
Priority: low

### X-noise-19
Location: line 3616, sec:noise:censoring; Owner: AI
Quote: "Censored EM repeats this for three rounds, refitting the downstream model out of fold each round, after the iterated estimator of \citet{buckley1979linear}."
His comment: What does EM stand for? It is spelled out only later, for the random-intercept model.
Lens item: 4
Proposed change:
old_string:
```
Censored EM repeats this for three rounds,
```
new_string:
```
Censored EM, a form of expectation maximization (EM), repeats this for three rounds,
```
Priority: low

### X-noise-20
Location: line 3629, sec:noise:censoring; Owner: AI
Quote: "A molecule is one canonical SMILES string from ChEMBL, its count is its number of IC$_{50}$ records"
His comment: With or without stereochemistry? Are salt forms merged with the parent?
Lens item: 7
Verified: src/kirby/datasets/herg.py groups by ChEMBL's `canonical_smiles` (lines 92-105, 545-565). ChEMBL's canonical SMILES keep stereochemistry. I did not check whether salts are stripped.
Proposed change: NEEDS HER on salts. Then:
old_string:
```
A molecule is one canonical SMILES string from ChEMBL,
```
new_string:
```
A molecule is one canonical SMILES string from ChEMBL, with its stereochemistry, so stereoisomers count as separate molecules [NEEDS HER: and salts are/are not merged],
```
Priority: low

### X-disc-6
Location: line 5007, sec:discussion:summary; Owner: AI
Quote: "model family explained 47.2\% of the variance in \aucnorm{}, and the representation far less."
His comment: "Far less" — how much? The reader will want the number.
Lens item: 6
Verified: line 534 gives 8.1%.
Proposed change:
old_string:
```
and the representation far less.
```
new_string:
```
and the representation 8.1\%.
```
Priority: low

### X-disc-7
Location: line 5020, sec:discussion:summary; Owner: AI
Quote: "Fourth, feature importance helped in building the hybrid, but reusing the importance scores during training added little."
His comment: Reusing them how, and where is this shown? The paragraph gives no result for "during training".
Lens item: 4 (also 5)
Proposed change: NEEDS HER: name the experiment (importance-weighted features for the model or for the distance?) and its section. Or cut the second half of the sentence if it refers to a result no longer in the thesis.
Priority: low

---

Not proposed (the lens says he ignores these, or the other agent covers them): the foundation-model sentence at 5071 (comment TH-p192a/b); the model dependence of hybrid gains in Table 6.8 (TH-p175a); the bold and clipping of old Table 6.10 (TH-p177a); the redundancy of the similarity section with Ch5 (TH-p164a).
