# Review: Chapter 7 (chap:noise), after the style pass

Compared with thesis.tex.before_style_apply_2026-10-10 by diff, and read in full as it stands on 2026-10-10. Every paragraph in this chapter carries "% TODO: manual review" (AI text), in the backup and now. The author's text in this unit is the section and subsection titles only. Each changed title has an "edited your text" note.

her_text_check.py hits in this unit: #1 (chapter title, has its note), #2 to #4 (figure blocks and a paragraph written by the figures session, not hers), #5 (AI-marked lead-in paragraph, false positive), #6 (a redraw comment rewritten by the figures session, not hers).

Numbers: every number in the chapter's prose is in the backup or in a SOURCE/CUT comment, except 0.3653 and 0.3171 (Caco-2, 40% clipped). Those were set by the figures session and are in figures_ch5_7/f11_score_vs_damage_level.csv. No NEW block below adds a number. V30 reuses 79.0%, which is already in Chapter 4.

Refs: every \ref in the chapter resolves. tab:noise:res:bygroup and tab:noise:res:robust_broad resolve through the \input'd files in figures_ch5_7/.

Passed: the opening links to Chapter 6, and the last line points to Chapter 8. Titles are in sentence case. There are no "Results:" prefixes and no Discussion section. Limitations sit inside Conclusions. Style guide items 23, 24, 34, 36 and 77 are done.

Every OLD block below was checked with python3 -I str.count == 1 against thesis.tex as it stood at the time of writing. No two OLD blocks overlap.

## Fixes

### V1: Introduction, the feature-space paragraph is vague
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "that is where the hybrid comes in" and "how badly standard representations can serve a distance" do not say what was found. The paragraph also ends on a two-question rhetorical frame. The Chapter 4 finding is quoted as it appears in Chapter 1 (line 25).
OLD:
```latex
Every method in this chapter works in a feature space, and that is where the hybrid comes in. A detector that compares a molecule's label with its neighbors' labels, or with a model's prediction, is only as good as the features it reads. Chapter~\ref{chap:flaws} showed how badly standard representations can serve a distance. We built the adaptive hybrid representation of Chapter~\ref{chap:method} on the hypothesis that the feature set is the root of that problem. So for every method we ask two questions, in this order. Does the method work at all? And if it does, does it work better on the hybrid than on the best single representation?
```
NEW:
```latex
Every method in this chapter reads a feature space. A detector compares a molecule's label with its neighbors' labels, or with a model's prediction, and both depend on the features. Chapter~\ref{chap:flaws} showed that in a general-purpose representation, most features do not vary or do not relate to the label, yet each adds a term to the distance. We built the adaptive hybrid representation of Chapter~\ref{chap:method} to leave those features out. For every method we therefore ask whether it improves the held-out score. Where it does, we ask whether it improves the hybrid more than the best single representation.
```

### V2: Introduction, dramatic closing sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "The published methods have no place for these." is a short sentence used for effect. State what is missing.
OLD:
```latex
The published methods have no place for these.
```
NEW:
```latex
None of the twelve published methods has a step that uses such a flag.
```

### V3: Introduction summary, headline claims with no fold count
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Style guide gap 35: give the fold count with each headline claim. "Noise-robust training ... helps about as much as the best mitigation" is shown only on hERG $K_i$ (sec:noise:res:model). "Clean-label check" is QM9 in this chapter, so name it.
OLD:
```latex
Noise detection finds outliers, shifted labels and flipped classes well above chance, and falls to or below chance on labels clipped at an assay limit. Noise mitigation works, best of all when a known flag says which labels are wrong, and on several datasets it passes random and shuffled controls. Noise-robust training, seed averaging and stacking most of all, helps about as much as the best mitigation. Without a clean-label check, though, we cannot yet say whether that is noise handling or a better model.
```
NEW:
```latex
Most results are on the first outer fold. On it, noise detection finds outliers, shifted labels and flipped classes well above chance. It falls to or below chance on labels clipped at an assay limit. Noise mitigation improves the held-out score most when a known flag says which labels are wrong. On several datasets it also beats random deletion or shuffled detector scores. On hERG $K_i$, seed averaging and stacking help about as much as the best mitigation. These two have no QM9 result yet, so we cannot say whether that is noise handling or a better model.
```

### V4: Methods, "hERG" alone is ambiguous between three hERG datasets
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The chapter uses hERG $K_i$, hERG IC$_{50}$ and, about 20 times, a bare "hERG" for the hERG blocking dataset (Table tab:datasets: "hERG blocking", blocker or not). A reader cannot tell which one is meant. Define the short form once.
OLD:
```latex
hERG IC$_{50}$ is the one dataset with real repeat measurements. Folds and metrics are those of Chapter~\ref{chap:method}.
```
NEW:
```latex
hERG IC$_{50}$ is the one dataset with real repeat measurements. In this chapter, ``hERG'' on its own means the hERG blocking dataset, whose labels are blocker or non-blocker. Folds and metrics are those of Chapter~\ref{chap:method}.
```

### V5: Detectors, one-line dramatic sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "That matters." is a dramatic one-line sentence.
OLD:
```latex
predict a spread from the features alone and never read it. That matters. A detector that never sees a label can only find a wrong one if the damage is tied to structure.
```
NEW:
```latex
predict a spread from the features alone and never read it. Such a detector can only find a wrong label if the damage is tied to structure.
```

### V6: Mitigation, repeated "the many"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "the many methods ... The many ways" repeats itself and adds nothing.
OLD:
```latex
Written this way, the many methods we tested reduce to a few selectors crossed with three actions. The many ways of moving a label differ only in where the label moves to and how far.
```
NEW:
```latex
Written this way, the methods we tested reduce to a few selectors crossed with three actions. Methods that move a label differ only in the target and in how far the label moves.
```

### V7: Mitigation, "The code cites" in thesis prose
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The thesis refers to the source code as if it were an author. Name the rule as Table tab:noise:mitigation does ("Score-scaled shrink") and say what the formula does.
OLD:
```latex
One further rule moves a share $0.5\tilde s_i/(1 + 0.5\tilde s_i)$ of the way to the out-of-fold prediction. The code cites James--Stein shrinkage \citep{james1961estimation} for it, but the variance it estimates cancels, so it is a fixed function of the score.
```
NEW:
```latex
One further rule, the score-scaled shrink, moves a share $0.5\tilde s_i/(1 + 0.5\tilde s_i)$ of the way to the out-of-fold prediction. It was written as James--Stein shrinkage \citep{james1961estimation}. The variance it estimates cancels out of the share, so the share depends only on the score.
```

### V8: Weights, colon used as a reveal
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "That weight is gentle:" is an appraisal followed by a colon reveal.
OLD:
```latex
That weight is gentle: the most suspect molecule counts about 0.9 times as much as the least suspect.
```
NEW:
```latex
With that weight, the most suspect molecule counts about 0.9 times as much as the least suspect.
```

### V9: Cleaning before the search, rhetorical question
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "raises a question no published method has to answer" performs. The rhetorical question that follows adds nothing.
OLD:
```latex
The hybrid's features are chosen against the labels, which raises a question no published method has to answer. Should the labels be cleaned before the hybrid chooses its features, or after? One experiment tests the order.
```
NEW:
```latex
The hybrid's features are chosen against the labels, so the labels can be cleaned before the features are chosen or after. One experiment compares the two orders.
```

### V10: Standing caveats, informal wording
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "So every ``it helped'' leans optimistic" and "Last, the network does not give the same score twice" are informal. The second does not say what varies.
OLD:
```latex
The single representation and the hybrid it is compared on are picked the same way. So every ``it helped'' leans optimistic. Last, the network does not give the same score twice.
```
NEW:
```latex
The single representation and the hybrid it is compared on are picked the same way. Both picks make every reported gain optimistic. Third, the network's score changes from one fit to the next.
```

### V11: Overall effect, "Noise mitigation works." on fold-0 evidence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Style guide gaps 35 and 9: a flat one-line claim resting on outer fold 0. Say how the three examples were measured.
OLD:
```latex
Noise mitigation works. With a quarter of LogD's training labels clipped,
```
NEW:
```latex
Three results from the first outer fold show that noise mitigation can improve the held-out score. With a quarter of LogD's training labels clipped,
```

### V12: Changing a label against deleting, "less convincing"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "come closer together" is vague, and "less convincing" is an appraisal. Say what the control showed.
OLD:
```latex
On a regression dataset the actions come closer together. On LogD
```
NEW:
```latex
On a regression dataset the three actions differ less. On LogD
```

### V13: Same paragraph, second half
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: As V12.
OLD:
```latex
Against random deletion, deleting is less convincing here than on hERG. It passes on the second hybrid but loses to random deletion on the first hybrid.
```
NEW:
```latex
Deletion beats random deletion on the second hybrid but loses to it on the first hybrid.
```

### V14: Flags of known quality, appraisal and a repeated sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "The clearest evidence" is an appraisal. "the detector, and not the action" is a not-X contrast. The last sentence repeats the one before it.
OLD:
```latex
The clearest evidence that the detector, and not the action, carries the gain comes from flags of known quality.
```
NEW:
```latex
A second test of whether the gain comes from the detector uses flag sets whose content we control.
```

### V15: Same paragraph, repeated last sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: As V14.
OLD:
```latex
none of them damaged, makes the single representation worse. The same action on the wrong molecules makes the model worse.
```
NEW:
```latex
none of them damaged, makes the single representation worse. So the gain depends on which molecules the action reaches.
```

### V16: Weighting, "gentlest" and "most datasets"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "gentlest" is an appraisal. "most datasets" is not what the paragraph shows: it gives three damaged datasets, then QM9.
OLD:
```latex
Making flagged molecules count less is the gentlest action, and it helped on most datasets.
```
NEW:
```latex
Weighting keeps every molecule and every label. It helped on hERG, LogD and Buchwald--Hartwig.
```

### V17: Labels as delivered, "clearest"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: An appraisal word. Say what passed: both detectors pass on hERG $K_i$ in the sentences that follow.
OLD:
```latex
hERG $K_i$ gives the clearest small-molecule result.
```
NEW:
```latex
On hERG $K_i$, label smoothing passes this control with both detectors.
```

### V18: Paragraph moved from Discussion, "in its own right ... paid off"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The last sentence is appraisal ("a use case in its own right", "paid off most").
OLD:
```latex
Cleaning an experimental dataset with nothing injected is a use case in its own right, and on those three datasets it is the one that paid off most.
```
NEW:
```latex
Cleaning labels as delivered needs no injected damage, so it is the use closest to how a chemist would apply these methods. It helped most on those three antibody datasets.
```

### V19: Neighbor spaces, "less than we expected"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "than we expected" performs surprise. No expectation was stated earlier.
OLD:
```latex
Where the neighbors are found matters less than we expected.
```
NEW:
```latex
The space in which the neighbors are found changes detection little.
```

### V20: Break-even, "the picture", "not enough", "clear a bar"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "the picture reverses" is a banned phrase. "is not enough" is a dramatic one-liner. "clear a bar" is a metaphor.
OLD:
```latex
Under the shift sized by model error the picture reverses.
```
NEW:
```latex
Under the shift sized by model error, the residual's precision falls below the break-even.
```

### V21: Same paragraph, closing sentences
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: As V20.
OLD:
```latex
Finding the damaged labels above chance is not enough. The detector has to clear a bar that depends on the damage, the action and the model, and under the shifts it mostly does not.
```
NEW:
```latex
A detector that ranks the damaged labels above chance can still make the model worse. Acting on its flags helps only when its precision is above the break-even, which depends on the damage, the action and the model. Under the shifts, the residual's precision was mostly below it.
```

### V22: Paragraph moved from Discussion, vague and dramatic, and "chance" contradicts Results
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "agrees from the other side" is vague, and "And the damage that was most fixable ..." is a dramatic closer. "fell to chance" contradicts the detection results two paragraphs up: below chance on Buchwald--Hartwig and LogD.
OLD:
```latex
This chapter agrees from the other side. The residual detector fell to chance under censoring, because a model fitted to clipped labels predicts the clipped values. The three detectors that predict a spread from the features alone could not be graded at all, because they never read a label. And the damage that was most fixable is the damage that needs no detector.
```
NEW:
```latex
This chapter finds the same for detectors built on a model's predictions. The residual detector fell to or below chance under censoring, because a model fitted to clipped labels predicts the clipped values. The three detectors that predict a spread from the features alone could not be graded, because they never read a label. The damage that was easiest to correct, clipping and scaffold-family offsets, is the damage that comes with a flag and needs no detector.
```

### V23: Clipped labels, wrong quantity
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "lowers the RMSE of the clipped labels" is wrong: censored EM lowers the held-out RMSE of a model trained on the corrected labels. The LogD figure also needs its fold (gap 35).
OLD:
```latex
For every one of them censored EM lowers the RMSE of the clipped labels, but not back to the score before clipping. On LogD, the network on the best single representation wins back about three quarters of the lost RMSE (Equation~\ref{eq:recovery}).
```
NEW:
```latex
For every one of them, censored EM gives a lower held-out RMSE than training on the clipped labels, but not as low as training on the labels before clipping. On LogD, the network on the best single representation wins back about three quarters of the lost RMSE on outer fold 0 (Equation~\ref{eq:recovery}).
```

### V24: Deleting clipped molecules, one-line verdict
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "It is the worst choice." is a dramatic one-liner, and the next sentence already says it.
OLD:
```latex
throws the molecules away. It is the worst choice. It is worse than leaving
```
NEW:
```latex
throws the molecules away. It is worse than leaving
```

### V25: Per-molecule lift, rhetorical contrast
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "Knowing which labels are clipped is what helps. Knowing how far each one should go is not." is a rhetorical contrast pair.
OLD:
```latex
Knowing which labels are clipped is what helps. Knowing how far each one should go is not. The per-molecule lift raises each clipped label
```
NEW:
```latex
The per-molecule lift tests whether the size of each lift matters. It raises each clipped label
```

### V26: Same paragraph, closing sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The finding V25 removed from the opening goes here, as a result.
OLD:
```latex
So the gain comes from raising the flagged labels by about the right average amount, and the constant lift does exactly that.
```
NEW:
```latex
So the gain comes from knowing which labels are clipped and raising them by about the right average amount, which is what the constant lift does.
```

### V27: Scaffold-family offsets on delivered labels, "the right answer"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "That is the right answer ..." and "It is also a warning ..." perform. State the condition for use.
OLD:
```latex
That is the right answer for a method that assumes damage that is not there. It is also a warning against using it without reason to think the damage exists.
```
NEW:
```latex
The method assumes a shared offset per family, so it should be used only where there is reason to expect one.
```

### V28: Comparison across damage kinds, first of two overlapping paragraphs
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing; add "% TODO: manual review -- merged into the next paragraph, which said the same" below it
- why: This paragraph and the next say the same thing. The second one's own TODO asks for the merge. "sort by how much the treatment knows" is vague.
OLD:
```latex
Set side by side, the damage kinds sort by how much the treatment knows. Clipping and scaffold-family offsets come with a flag that says which labels are affected. On every small-molecule dataset they ran on, the flag-driven moves won back about half or more of the lost RMSE. Outliers, Gaussian noise and shifts in a few labels have no flag, and the treatment depends on a detector. Those were helped too, as Section~\ref{sec:noise:res:action} shows, but the gains were smaller.
```
NEW:
```latex
% TODO: manual review -- merged into the next paragraph, which said the same
```

### V29: Comparison across damage kinds, the merged paragraph
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Merge with V28 and give the fold. "Noise mitigation works in QSAR data" is a flat claim (gap 35). One sentence ran close to 30 words.
OLD:
```latex
Noise mitigation works in QSAR data, and it works best when something other than a model tells us which labels are wrong. A clipped label is flagged by the assay limit, and a shifted scaffold family by its scaffold. In both cases moving the flagged labels won back about half or more of the lost RMSE on every small-molecule dataset where the damage hurt. When a detector has to infer which labels are wrong, mitigation still helped. Under hERG's label flips it beat random deletion on three folds. But the gains were smaller, and they depended on whether the detector's precision cleared the break-even point for that damage.
```
NEW:
```latex
Set side by side, the damage kinds sort by whether a flag says which labels are wrong. A clipped label is flagged by the assay limit, and a shifted scaffold family by its scaffold. In both cases, on the first outer fold, moving the flagged labels won back about half or more of the lost RMSE. That held on every small-molecule dataset where the damage hurt. Outliers, Gaussian noise and shifts in a few labels have no flag, so the treatment depends on a detector. Mitigation still helped there, and under hERG's label flips deletion beat random deletion on three folds. The gains were smaller, and they depended on whether the detector's precision cleared the break-even for that damage.
```

### V30: Small molecules by dataset, vague opening and appraisal
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "show the same split as the methods" is vague, and "by useful amounts" is an appraisal.
OLD:
```latex
The small-molecule datasets show the same split as the methods. Under injected damage they are helped often and by useful amounts, most of all when a flag says which labels are wrong.
```
NEW:
```latex
On the small-molecule datasets, injected damage was often corrected, most of all when a flag says which labels are wrong.
```

### V31: Cleaning inside the search, repeated from Methods
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "We found no earlier work that cleans labels before choosing the features" repeats the Methods sentence "No source we found cleans labels before choosing the features."
OLD:
```latex
We found no earlier work that cleans labels before choosing the features. On expression
```
NEW:
```latex
On expression
```

### V32: Changing the model, "and it does well"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: An appraisal clause. The numbers that follow make the point.
OLD:
```latex
is the comparison point for every mitigation result, and it does well. On hERG
```
NEW:
```latex
is the comparison point for every mitigation result. On hERG
```

### V33: Retuning, a general claim from one example
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The claim rests on one pair (0.6911 against 0.7028). "and not from tuning as such" is a not-X contrast.
OLD:
```latex
So the retuning gain comes from fitting the model's settings to the noise, and not from tuning as such.
```
NEW:
```latex
In this example, the gain from retuning therefore comes from tuning on the damaged labels, since the same search on clean labels gains less.
```

### V34: Published sweep, contradicts Chapter 4
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The claim is wrong. Chapter 4 (sec:lnl_motivation, line 1738) says DivideMix retained the most (86.7%). The heteroscedastic GP retained 79.0% and was not the best. The opening sentence frames a not-X/Y reveal.
OLD:
```latex
The sweep agrees with the earlier study in one place and reverses it in another. In Chapter~\ref{chap:flaws}, the heteroscedastic GP held up best of the twelve under Gaussian noise on QM9. Here it is the only method
```
NEW:
```latex
In Chapter~\ref{chap:flaws}, the heteroscedastic GP was one of the methods that held up under Gaussian noise on QM9, with a retention of 79.0\%. Here it is the only method
```

### V35: Published sweep, unsupported "worst" and dramatic closer
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Chapter 4's printed text never ranks distance-based selection last. Only a comment there says so. Chapter 4 does say the methods that held up used the model's own loss or uncertainty. "The same method gives the opposite verdict ..." is a dramatic closing line.
OLD:
```latex
Distance-based selection was the worst of the twelve in the earlier study. On hERG with 10\% of labels flipped one way only, it is the only one of the twelve that helps both sides, with AUROC gains in the third decimal. The same method gives the opposite verdict on a different task and different data.
```
NEW:
```latex
The methods that held up in the earlier study estimated noise from the model's own loss or uncertainty. Distance-based selection flags molecules by their neighbor-label gap. Here it is the only one of the twelve that helps both sides on hERG with 10\% of labels flipped one way only. Its AUROC gains are in the third decimal.
```

### V36: Paragraph moved from Discussion, rhetorical question and contrast
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: It opens with a rhetorical question and uses "told a more mixed story" and "turned out to be". It closes with a not-X-but-Y pair: "is not good or bad on its own. It is good or bad in a feature space."
OLD:
```latex
Was fixing the feature set the right place to start? On small molecules the hybrid started ahead, and it stays ahead through every kind of cleaning, so a user who builds a hybrid loses nothing by cleaning afterwards. The detectors told a more mixed story. Neighbors found in the hybrid led to better cleaning slightly more often than neighbors in a single representation. For detection alone, though, the importance-weighted features of a single representation did better than the hybrid's. The hybrid search itself turned out to be robust to Gaussian noise: searching on damaged labels instead of clean ones changed Caco-2's score in the third decimal. And the ranking of noise-handling methods changed between the hybrid and the single representations in almost every condition. That is the result that ties this chapter to Chapter~\ref{chap:flaws}. A noise-handling method is not good or bad on its own. It is good or bad in a feature space.
```
NEW:
```latex
These results also show what the hybrid adds to noise handling. On small molecules the hybrid started ahead of the best single representation, and it stayed ahead through every kind of cleaning. A user who builds a hybrid therefore loses nothing by cleaning afterwards. Neighbors found in the hybrid led to better cleaning slightly more often than neighbors in a single representation. For detection alone, the importance-weighted features of a single representation did better than the hybrid's. Searching the hybrid on damaged labels instead of clean ones changed Caco-2's score in the third decimal. The ranking of noise-handling methods changed between the hybrid and the single representations in almost every condition. Whether a noise-handling method works therefore depends on the feature space it runs in, as Chapter~\ref{chap:flaws} found for distance.
```

### V37: Conclusions, unqualified claims
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "under the right conditions" is vague, and the fold count is missing (gap 35). "so those gains are noise removed" is too broad. The QM9 check covers deletion only, but the sentence before it also lists relabeling and shuffled-score controls.
OLD:
```latex
Noisy labels in QSAR data can be found and, under the right conditions, fixed. Fixing works best with a flag.
```
NEW:
```latex
On the first outer fold, noisy labels in QSAR data could be found, and in several settings corrected. Corrections worked best with a flag.
```

### V38: Conclusions, same paragraph
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: As V37. The long sentence is also split.
OLD:
```latex
With a detector, deleting or relabeling flagged molecules helped under label flips and outliers, and passed random and shuffled controls on hERG, hERG $K_i$, Buchwald--Hartwig and three antibody sets. On clean QM9 labels deletion never helped, so those gains are noise removed.
```
NEW:
```latex
With a detector, deleting or relabeling flagged molecules helped under label flips and outliers. It passed random and shuffled controls on hERG, hERG $K_i$, Buchwald--Hartwig and three antibody datasets. On clean QM9 labels deletion never helped, so the deletion gains are noise removed.
```

### V39: Deletion on QM9, fold count
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Gap 35. "never" rests on outer fold 0 of one experiment (SOURCE N7-08).
OLD:
```latex
Deleting never helped on clean labels.
```
NEW:
```latex
On the first outer fold, deleting never helped on clean labels.
```

### V40: Limitations, code names, and MDR1--MDCK listed as having no results
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The dataset names are code identifiers (adme\_mdr1 and so on), not the names in Table tab:datasets. The list also contradicts the chapter. Section sec:noise:res:overall says MDR1--MDCK efflux has 24 conditions on fold 0 that the table leaves out. The count "Seven" is dropped, so no new number is introduced.
OLD:
```latex
Seven of the datasets in Chapters~\ref{chap:method} and~\ref{chap:similarity} have no noise results yet: adme\_mdr1, adme\_rlm, flab\_expression\_koenig, flab\_binding\_koenig, abagym\_EGFR, abagym\_Ang2 and ginkgo\_hic.
```
NEW:
```latex
These datasets of Chapters~\ref{chap:method} and~\ref{chap:similarity} have no noise results yet: rat microsomal clearance, anti-VEGF scan expression, anti-VEGF scan binding, EGFR binding, angiopoietin-2 binding and HIC retention. MDR1--MDCK efflux has results on outer fold 0, but the tables in this chapter leave them out.
```

### V41: Real censoring slot (Methods), code name
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: ginkgo\_pr\_cho is a code name. The dataset is "CHO polyreactivity" in Table tab:datasets (197 antibodies, which matches).
OLD:
```latex
The same moves run on real censoring in ginkgo\_pr\_cho, where
```
NEW:
```latex
The same moves run on real censoring in CHO polyreactivity, where
```

### V42: Real censoring slot (Results), code name
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: As V41. The bold RESULT NEEDED placeholders are left alone.
OLD:
```latex
ginkgo\_pr\_cho is the one dataset in this thesis with a visible floor
```
NEW:
```latex
CHO polyreactivity is the one dataset in this thesis with a visible floor
```

### V43: Antibody datasets, code names
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: As V41. ginkgo\_acsins is "AC-SINS self-association" in Table tab:datasets.
OLD:
```latex
ginkgo\_pr\_cho and ginkgo\_acsins, ran only part of the methods
```
NEW:
```latex
CHO polyreactivity and AC-SINS self-association, ran only part of the methods
```

### V44: Censoring moves, "antibody thermostability" is a nanobody dataset
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Chemistry guide, naming rows by what they are. The dataset (flab\_thermo\_tresanco) is "Nanobody thermostability" in Table tab:datasets.
OLD:
```latex
On the antibody thermostability set, scored by Spearman, the constant lift
```
NEW:
```latex
On nanobody thermostability, scored by Spearman, the constant lift
```

### V45: Neighbors in the hybrid, same naming
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: As V44.
OLD:
```latex
On antibody thermostability with outliers,
```
NEW:
```latex
On nanobody thermostability with outliers,
```

### V46: Results subsection title duplicates a Methods subsection title
- file: thesis.tex
- kind: author-title (changed in this pass, note exists; the note is updated)
- todo_line: replace the existing note with the one in NEW
- why: The pass renamed her "Results: Which Detectors Find the Suspect Labels" to "Noise detection". That is the same title as Methods Section sec:noise:detection, so the table of contents shows two "Noise detection" entries in one chapter. Keep her words and make them a topic.
OLD:
```latex
% TODO: manual review -- title was: Results: Which Detectors Find the Suspect Labels -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; question made a topic, named like the Methods subsection it reports on (sec:noise:detection)
\subsection{Noise detection}\label{sec:noise:res:detectors}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Which Detectors Find the Suspect Labels -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; question made a topic in your words ("Noise detection" would repeat the Methods subsection title)
\subsection{Detection of the suspect labels}\label{sec:noise:res:detectors}
```

## For the author

- The text uses "scaffold family" and your subsection title uses "scaffold group" (sec:noise:res:offsets). Chapter 2 defines "scaffold family". The chemistry guide (item 14) suggests "scaffold group" throughout. Your choice.
- "hERG" on its own means hERG blocking about 20 times. V4 defines the short form once. The alternative is to write "hERG blocking" every time.
- In V29, "about half or more of the lost RMSE" is not true for every draw. On LogD's second scaffold-family draw, the random-intercept model moves 0.6899 to 0.5725 against 0.4017 clean, which is less than half. Check the wording.
- V40: please confirm that MDR1--MDCK efflux has noise results and that the other six do not. The Methods comment (TODO missing result, dated 9 Oct) still lists seven.
- Figure F11's caption opens with "Left and middle:" instead of a noun phrase. It is not edited here, because the figures session owns it.
- The flow figure's caption opens with a clause ("How a noise method is run and scored ..."). Not edited, same reason.
- Table tab:noise:robust's caption says "``New'' means the code cites a paper that does not describe what it does". It has no AI marker, so it is treated as your text and left alone. Table tab:noise:detectors uses "``New'' means no published source was found". Consider the same wording.
- Chapter 4's comment (item 7, line about 1767) says distance-based selection was the worst of the twelve in the earlier study. Chapter 4's printed text does not say so, and V35 no longer relies on it.
- Style guide gaps 28 and 29 (numbers in prose, one float per 2,000 words) remain open. Fourteen "TABLE NEEDED" slots wait on your approval of the table form. No tables are created in this pass.
- \label{sec:noise:discussion} and \label{sec:noise:limitations} now sit without a heading, so they point to the enclosing subsection. Nothing references them.

## Applied

- V1: applied as written.
- V2: applied, "twelve" dropped ("None of the published methods has a step that uses such a flag."), since this chapter's sweep is wider than the twelve.
- V3: applied as written (checked: N7-38 has no QM9 row).
- V4: applied as written.
- V5: applied as written.
- V6: applied as written.
- V7: applied as written.
- V8: applied as written.
- V9: applied as written.
- V10: applied as written.
- V11: applied as written (paragraph gives three fold-0 examples).
- V12: applied as written.
- V13: applied as written.
- V14: applied as written.
- V15: applied as written.
- V16: applied as written (paragraph names hERG, LogD, Buchwald--Hartwig).
- V17: applied as written.
- V18: applied, adapted: "Cleaning labels as delivered needs no injected damage, so it is a separate use of these methods. On those three antibody datasets, it is the use that gave the largest gains." (NEW's "closest to how a chemist would apply" was a new claim.)
- V19: applied as written.
- V20: applied as written.
- V21: applied as written.
- V22: applied as written.
- V23: applied as written.
- V24: applied as written.
- V25: applied as written.
- V26: applied as written.
- V27: applied as written.
- V28: applied; paragraph replaced by the merge note, its existing TODO line kept.
- V29: applied; kept the Section~\ref{sec:noise:res:action} reference from V28's paragraph ("Mitigation still helped there (Section~\ref{sec:noise:res:action}), ...").
- V30: applied as written.
- V31: applied as written.
- V32: applied as written.
- V33: applied as written.
- V34: applied as written (79.0\% checked in Chapter 4).
- V35: applied as written.
- V36: applied as written.
- V37: applied as written.
- V38: applied as written.
- V39: applied as written.
- V40: applied, opening changed from "These datasets" to "The following datasets" ("These" had no referent); no count added.
- V41: applied as written.
- V42: applied as written.
- V43: applied as written.
- V44: applied as written.
- V45: applied as written.
- V46: applied as written.
