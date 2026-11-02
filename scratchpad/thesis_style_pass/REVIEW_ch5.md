# Review: Chapter 5 (chap:method), after the style pass

Read in full on 2026-10-10 against thesis.tex.before_style_apply_2026-10-10 (diff of the chapter) and the live thesis.tex. Every OLD block below was checked with `python3 -I` (`str.count == 1`) against the live thesis.tex on 2026-10-10. Another session edits captions and results; re-check each OLD before applying.

## What was checked, and the state found

- her_text_check.py on the chapter: 10 hits. Six are false positives (AI-marked paragraphs, or notes already present). Two are real: the concatenation paragraph and the internal-evaluator paragraph changed with no note (V1, V2). Two notes cover taste-only word changes, planned as reverts (V3, V4).
- Structure: Introduction, Methods, Results, Conclusions; no Discussion section; limitations are paragraphs inside Conclusions with sec:method:limitations kept; no "Results:" prefixes; chapter and section titles in sentence case. The chapter opens from Chapter 4 and its last paragraph points to Chapter 6.
- Labels: every label of the backup chapter is still present. Every \ref in the chapter resolves (Additional-file labels through the xr build of additional_files).
- Numbers: every number in the chapter's prose that is not in the backup comes with a SOURCE comment from the results session, except 2.3, 28.2 and 11.1 hours (see For the author). No edit below changes or rounds a number.
- Chemistry guide items in this chapter: 5, 23, 25, 26, 28, 30 are done. Item 29 (yield RMSE unit) is planned (V35, V36). Item 19 (antibodies called molecules) is planned where it occurs (V15, V18, V42, V45, V47, V55). Items 5 (units of the activity labels) and 8 (ddG name) need the author.
- Remaining failures: about 45 AI-written sentences over 30 words, several rhetorical contrasts and one-line sentences in the Results and Conclusions, an out-of-date provisional note on PCA, an index clash in the stability equation, and a caption with no units.

## Edits

### V1 — Unnoted edits to your text in the concatenation paragraph
- file: thesis.tex
- kind: her-text-note
- todo_line: add the line shown in NEW, directly above the paragraph and below her own TODO
- why: Two changes to her paragraph had no note: "limitation combinations" became "limited combinations" (wrong word form) and the hyphen in "conceptually-straightforward" was removed (no hyphen after an -ly adverb). The strict rule requires a note for word substitutions and typography. Both fix real errors, so keep them and add the note.

OLD:
```latex
The most straightforward method of combining representations is concatenation. This however degrades
```
NEW:
```latex
% TODO: manual review -- edited your text: "limitation combinations" became "limited combinations" (wrong word form); the hyphen in "conceptually-straightforward" was removed (no hyphen after an -ly adverb).
The most straightforward method of combining representations is concatenation. This however degrades
```

### V2 — Unnoted punctuation change in the internal-evaluator paragraph
- file: thesis.tex
- kind: her-text-note
- todo_line: add the line shown in NEW
- why: Her comma splice "feature sets, it explicitly" became a semicolon with no note. It fixes a real error, but punctuation that restructures a sentence needs a note.

OLD:
```latex
The internal evaluator's scores determine how the budget is split and which representations are kept. The internal evaluator ranks candidate feature sets; it explicitly
```
NEW:
```latex
% TODO: manual review -- edited your text: comma splice "feature sets, it explicitly" replaced by a semicolon.
The internal evaluator's scores determine how the budget is split and which representations are kept. The internal evaluator ranks candidate feature sets; it explicitly
```

### V3 — Revert a taste-only word change: "methodology"
- file: thesis.tex
- kind: revert
- todo_line: keep existing
- why: Removing "you" needed a passive, but "feature selection methodology" also became "a feature selection method". That word change fixes nothing and is not in the note. Back to her word.

OLD:
```latex
a dimensionality reduction technique or a feature selection method can be used.
```
NEW:
```latex
a dimensionality reduction technique or feature selection methodology can be used.
```

### V4 — Revert a taste-only word change: "discovered"
- file: thesis.tex
- kind: revert
- todo_line: keep existing
- why: "as I discovered" needed only I -> we. "discovered" also became "found", which fixes nothing and is not in the note. Back to her word.

OLD:
```latex
Unfortunately we live in reality, and as we found, the answer
```
NEW:
```latex
Unfortunately we live in reality, and as we discovered, the answer
```

### V5 — Results lead-in: 35-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
On each outer fold, the search ran on the fold's training samples only, and the hybrid and the single representation it is compared with were both picked there and scored once on the held-out fold.
```
NEW:
```latex
On each outer fold, the search ran on the fold's training samples only. The hybrid and the single representation it is compared with were both picked there and scored once on the held-out fold.
```

### V6 — "On small molecules the hybrid works."
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Dramatic one-line sentence; the next sentence already says what was found.

OLD:
```latex
On small molecules the hybrid works. It scores better than the best single representation on every dataset,
```
NEW:
```latex
On small molecules the hybrid scores better than the best single representation on every dataset,
```

### V7 — Hindsight paragraph: rhetorical contrast
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "The gain is not a matter of ..." is the banned not-X opening; "So we know" is chatty.

OLD:
```latex
The gain is not a matter of the hybrid beating a poorly chosen representation. Every representation in each pool was also scored on its own on the same folds. So we know which single representation was best on the held-out molecules, a choice no user could make in advance.
```
NEW:
```latex
Every representation in each pool was also scored on its own on the same folds. This gives the single representation that was best on the held-out molecules, a choice no user could make in advance.
```

### V8 — Hindsight paragraph: vague lead-in to the shuffling test
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "The reason lies in what the hybrid draws on." announces instead of saying what was done.

OLD:
```latex
The reason lies in what the hybrid draws on. We shuffled each representation's features in turn on the held-out molecules.
```
NEW:
```latex
To see what the hybrid draws on, we shuffled each representation's features in turn on the held-out molecules.
```

### V9 — Hindsight paragraph: colon reveal and idiom
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Colon-as-reveal and the idiom "leaves information on the table".

OLD:
```latex
This is the premise of the thesis: a single representation leaves information on the table, and selecting from several recovers it.
```
NEW:
```latex
This is the premise of the thesis, that a single representation misses information that selecting from several can recover.
```

### V10 — Antibody lead-in: "the picture"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "the picture" is on the banned list.

OLD:
```latex
On antibodies the picture differs from one dataset to the next (Table
```
NEW:
```latex
On antibodies the result differs from one dataset to the next (Table
```

### V11 — "Two antibody datasets are used for one point each."
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Says nothing; the sentences after it carry the content.

OLD:
```latex
Two antibody datasets are used for one point each. On angiopoietin-2 binding
```
NEW:
```latex
On angiopoietin-2 binding
```

### V12 — Rare scaffolds: rhetorical opening and unsupported "helps most"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "is not confined to common chemistry" is a rhetorical negation. "helps most" generalizes from one dataset and model (LogD with the DNN); the sentence is also 30 words.

OLD:
```latex
The gain on small molecules is not confined to common chemistry. We split
```
NEW:
```latex
On small molecules the hybrid also gains on rare scaffolds. We split
```

### V13 — Rare scaffolds: closing sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: See previous item.

OLD:
```latex
Rare scaffolds are where a single representation has least to go on, because few training molecules resemble the held-out ones, and that is where drawing on several representations helps most.
```
NEW:
```latex
On rare scaffolds few training molecules resemble the held-out ones, so a single representation has less to go on.
```

### V14 — PCA paragraph: "the two comparisons" is ambiguous
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The paragraph names three experiments; "the two comparisons" does not say which two. The three differences are between the first (main comparison) and the third.

OLD:
```latex
So the two comparisons differ in three ways at once
```
NEW:
```latex
So the first and third experiments differ in three ways at once
```

### V15 — Ambroise paragraph: antibodies called molecules
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The small datasets meant here are antibody sets (chemistry guide item 19).

OLD:
```latex
where many configurations are compared on few molecules \citep{Tibshirani2015}
```
NEW:
```latex
where many configurations are compared on few samples \citep{Tibshirani2015}
```

### V16 — Lead-in to "Sources of the gain": garbled last sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "it tests what using importance during training, tuning the model and adding a new embedding change" does not parse, and calibration (last paragraph of the section) is missing.

OLD:
```latex
Last, it tests what using importance during training, tuning the model and adding a new embedding change.
```
NEW:
```latex
Last, it reports what changes when importance is used during training, when the model is tuned and when a new embedding is added. It ends with the calibration of the hybrid's prediction intervals.
```

### V17 — Recursive elimination: rhetorical opening
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "The hybrid's gain is not only the gain of having fewer features." is the banned not-only contrast; the comparison that follows says it.

OLD:
```latex
The hybrid's gain is not only the gain of having fewer features. At the same number of features, drawn
```
NEW:
```latex
At the same number of features, drawn
```

### V18 — Importance in training: "weighting molecules"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sample weighting also runs on antibody datasets (chemistry guide item 19).

OLD:
```latex
by scaling features, weighting molecules, or keeping
```
NEW:
```latex
by scaling features, weighting samples, or keeping
```

### V19 — Calibration: not-only contrast
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "A better input gives a more useful statement of uncertainty, not only a lower error." is a rhetorical contrast and a general claim from small molecules.

OLD:
```latex
A better input gives a more useful statement of uncertainty, not only a lower error. 
```
NEW:
```latex
On small molecules the hybrid therefore narrows the intervals as well as lowering the error. 
```

### V20 — Calibration: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
The calibration set is part of a training fold of a few hundred antibodies, and the hybrid chosen on those antibodies is optimistic about them, the same selection bias as above.
```
NEW:
```latex
The calibration set is part of a training fold of a few hundred antibodies. The hybrid chosen on those antibodies is optimistic about them, the same selection bias as above.
```

### V21 — Shared rows: "not X. It is Y" pattern
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Banned rhetorical pattern ("The claim is not ... It is ...").

OLD:
```latex
So the gain does not come from breaking the ties. It comes from the information the other representations add to every molecule.
```
NEW:
```latex
So most of the gain comes from the information the added representations give every molecule, and little from breaking the ties.
```

### V22 — Transfer: 33-word general claim from one pair
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Over 30 words, and "a reasonable start on a related small one" generalizes from one pair of datasets.

OLD:
```latex
Settings found once on a large dataset are a reasonable start on a related small one, at a fraction of the cost, but the search on the dataset itself is still the best.
```
NEW:
```latex
Reusing settings saves the search on the smaller dataset at some cost in error, and we tested this on one pair of datasets only.
```

### V23 — Separate sites: 36-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
When each site holds one family of representations for the same molecules, one feature selection over all families beats both
```
NEW:
```latex
In the first experiment, each site holds one family of representations for the same molecules. One feature selection over all families beats both
```

### V24 — Conclusions: "signal" and performance sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "holds more than 29% of the signal" is vague (29% is a share of the shuffling drop). "That was the hypothesis ... and on small molecules it holds" performs the conclusion. The number is kept.

OLD:
```latex
and no one holds more than 29\% of the signal. That was the hypothesis behind building the hybrid, that the feature set is part of the problem, and on small molecules it holds.
```
NEW:
```latex
and no one of them carries more than 29\% of the drop in score when its features are shuffled. On small molecules, this supports the hypothesis behind the hybrid, that the feature set is part of the problem.
```

### V25 — Conclusions: not-only contrast
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "The gain comes from selection, not only from having fewer features." is the banned contrast; the next sentence says it with the comparison.

OLD:
```latex
The gain comes from selection, not only from having fewer features. At the same number of features the hybrid beats
```
NEW:
```latex
At the same number of features the hybrid beats
```

### V26 — Conclusions: "The DNN wants more features"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Anthropomorphic, and stated in general when the Results show it on Caco-2 and LogD only.

OLD:
```latex
The DNN wants more features than the SVM, and the number
```
NEW:
```latex
On Caco-2 and LogD the DNN picked more features than the SVM, and the number
```

### V27 — Conclusions: PCA described as still waiting
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: The Results now report the PCA comparison (it disagrees between experiments with the DNN), so "waits for the DNN re-runs" is out of date. The sentence is also 38 words.

OLD:
```latex
Most DNN results were computed before the correction of 8 October 2026, the Gaussian-process results will change when that model is replaced, and the comparison with PCA at the same number of features waits for the DNN re-runs.
```
NEW:
```latex
Most DNN results were computed before the correction of 8 October 2026. The Gaussian-process results will change when that model is replaced. With the DNN, the comparison with PCA at the same number of features disagrees between experiments.
```

### V28 — Conclusions: 32-word sentence on separate sites
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
so the differences between real sites in curation and chemistry are not tested, and only Caco-2 has finished the experiment with flipped labels.
```
NEW:
```latex
so the differences between real sites in curation and chemistry are not tested. Only Caco-2 has finished the experiment with flipped labels.
```

### V29 — Conclusions: 34-word sentence on dataset size
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
Most of the antibody datasets in this chapter are of that size, so the antibody results say as much about dataset size as about antibodies, and the two cannot be separated with these datasets.
```
NEW:
```latex
Most of the antibody datasets in this chapter are of that size. The antibody results therefore say as much about dataset size as about antibodies, and these datasets cannot separate the two.
```

### V30 — Conclusions: 32-word sentence on antibody pools
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
several of them variants of ESM-2, so the antibody hybrids had few representations of different kinds to combine.
```
NEW:
```latex
several of them variants of ESM-2. The antibody hybrids therefore had few representations of different kinds to combine.
```

### V31 — Conclusions: 35-word sentence on recursive elimination
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
Recursive feature elimination over all representations joined beat the hybrid on binding free energy with the SVM and on CHO polyreactivity at every budget, and on hERG $K_i$ with the DNN at the larger budgets.
```
NEW:
```latex
Recursive feature elimination over all representations joined beat the hybrid at every budget on binding free energy with the SVM and on CHO polyreactivity. It also beat it on hERG $K_i$ with the DNN at the larger budgets.
```

### V32 — Dependence on the model: 32-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
The number of representations differs in the same way: most small-molecule folds
```
NEW:
```latex
The number of representations differs in the same way. Most small-molecule folds
```

### V33 — Run times: which dataset is which bound
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: "2.3 to 28.2 hours ... (hERG $K_i$ and MDR1--MDCK efflux)" leaves the reader to pair names with bounds, and the low bound is the hERG total the next sentence calls too low. Numbers unchanged.

OLD:
```latex
The searches of all 5 outer folds of one dataset took 2.3 to 28.2 hours on the small molecules, summed over folds (hERG $K_i$ and MDR1--MDCK efflux). The hERG $K_i$ total is likely too low, because one of its folds recorded only 90 seconds. They took 11.1 hours on Buchwald--Hartwig yield, and 0.4 to 38.6 hours on the antibodies (CHO polyreactivity and the anti-VEGF expression scan).
```
NEW:
```latex
Summed over its 5 outer folds, the search of one small-molecule dataset took from 2.3 hours (hERG $K_i$) to 28.2 hours (MDR1--MDCK efflux). The hERG $K_i$ total is likely too low, because one of its folds recorded only 90 seconds. The search took 11.1 hours on Buchwald--Hartwig yield. On the antibodies it took from 0.4 hours (CHO polyreactivity) to 38.6 hours (the anti-VEGF expression scan).
```

### V34 — Tuning test: "pays"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Informal verb.

OLD:
```latex
We tested whether tuning pays on one outer fold of
```
NEW:
```latex
We tested whether tuning helps on one outer fold of
```

### V35 — Tuning test: Buchwald--Hartwig RMSE without unit
- file: thesis.tex
- kind: chemistry
- todo_line: keep existing
- why: Reaction-yield RMSE needs its unit at first mention in the section (chemistry guide item 29).

OLD:
```latex
from 18.04 to 20.14 with the CGR fingerprint.
```
NEW:
```latex
from 18.04 to 20.14 percentage points of yield with the CGR fingerprint.
```

### V36 — Tuning table caption: units and short title
- file: thesis.tex
- kind: caption
- todo_line: keep existing
- why: Units are given only for Buchwald--Hartwig, and "in yield (%)" is not the unit of an RMSE. No short title for the List of Tables, unlike the other tables. Numbers untouched.

OLD:
```latex
\caption{Held-out RMSE of single representations at default and tuned model settings, outer fold 0 only. Lower is better. Buchwald--Hartwig in yield (\%).}
```
NEW:
```latex
\caption[Default and tuned model settings on single representations.]{Held-out RMSE of single representations at default and tuned model settings, outer fold 0 only. One row is one dataset, model and representation. Lower is better. RMSE is in eV for QM9, p$K_i$ for hERG $K_i$, log$_{10}$ efflux ratio for Caco-2 and percentage points of yield for Buchwald--Hartwig; LogD has no unit.}
```

### V37 — Alternatives table caption: 35-word sentence
- file: thesis.tex
- kind: caption
- todo_line: keep existing
- why: Caption sentence over 30 words; split into a noun-phrase title and a column key.

OLD:
```latex
{Methods for combining or reducing representations, compared by whether they use the labels, keep the original features, let one model see several representations, and must refit to add a representation.}
```
NEW:
```latex
{Methods for combining or reducing representations. The columns give whether each method uses the labels, keeps the original features, lets one model see several representations, and must refit to add a representation.}
```

### V38 — Number-of-features figure caption: British spelling
- file: thesis.tex
- kind: spelling
- todo_line: none (trivial spelling)
- why: "colour" -> "color" (American spelling). The figures session owns this caption: apply only if the OLD text is still there.

OLD:
```latex
one colour per downstream model
```
NEW:
```latex
one color per downstream model
```

### V39 — Nanobody budgets: 36-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
On nanobody thermostability, where budgets from 50 to 2,000 features were searched, the search on each outer fold's training antibodies picked
```
NEW:
```latex
On nanobody thermostability, budgets from 50 to 2,000 features were searched. The search on each outer fold's training antibodies picked
```

### V40 — Representation counts on LogD: 33-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
On LogD, where counts of 2, 3, 5 and all were searched, the hybrid used all 10 representations on nine of the ten folds (five folds for each of the two downstream models).
```
NEW:
```latex
On LogD, counts of 2, 3, 5 and all were searched. The hybrid used all 10 representations on nine of the ten folds, five for each of the two downstream models.
```

### V41 — Global pool: 35-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
as shares of a pool of $2B$ candidate features, each offer capped at the representation's width and holding only features scoring above 0.
```
NEW:
```latex
as shares of a pool of $2B$ candidate features. Each offer is capped at the representation's width and holds only features scoring above 0.
```

### V42 — DNN evaluator: 33-word sentence; antibodies called molecules
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words; the evaluator also runs on antibody datasets.

OLD:
```latex
Only the DNN evaluator shares its downstream model's architecture and training settings, and even it draws its 20\% early-stopping molecules at random,
```
NEW:
```latex
Only the DNN evaluator shares its downstream model's architecture and training settings. Even it draws its 20\% early-stopping samples at random,
```

### V43 — Representation count: 42-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
on that fold contribute features, so the internal evaluator decides which representations are kept as well as how the budget is split among them
```
NEW:
```latex
on that fold contribute features. The internal evaluator therefore decides which representations are kept, as well as how the budget is split among them
```

### V44 — Augmentation donors: 33-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
Coulomb matrix eigenvalues, and include features that describe only differences between variants,
```
NEW:
```latex
Coulomb matrix eigenvalues. They also include features that describe only differences between variants,
```

### V45 — Residual ranking: antibodies called molecules
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Augmentation also runs on antibody datasets (chemistry guide item 19).

OLD:
```latex
for the internal-training molecules. For binary tasks
```
NEW:
```latex
for the internal-training samples. For binary tasks
```

### V46 — Residual ranking: split the y/r sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
would favor features that repeat what the base captures; ranking against $\mathbf{r}$ favors
```
NEW:
```latex
would favor features that repeat what the base captures. Ranking against $\mathbf{r}$ favors
```

### V47 — Augmentation blocks: 35-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words; antibodies called molecules.

OLD:
```latex
are each added to the base and scored with the downstream model on the internal-validation molecules.
```
NEW:
```latex
are each added to the base. Each is scored with the downstream model on the internal-validation samples.
```

### V48 — Capped mode: 34-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
The base features are ranked once by random forest importance (50 trees, depth at most 10) and the $k^*$ weakest are dropped, after any unspent budget is filled, keeping at least one base feature.
```
NEW:
```latex
The base features are ranked once by random forest importance (50 trees, depth at most 10). Any unspent budget is filled first, and then the $k^*$ weakest base features are dropped, keeping at least one.
```

### V49 — Second use of augmentation: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
In a second use, a user keeps one strong representation at full width as the base and adds a few extras, with $B$ its width, so only uncapped mode makes sense.
```
NEW:
```latex
In a second use, a user keeps one strong representation at full width as the base, with $B$ its width, and adds a few extras. Only uncapped mode makes sense there.
```

### V50 — Pruning: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
that share every other setting, and a value is removed if it is among the best in too few groups.
```
NEW:
```latex
that share every other setting. A value is removed if it is among the best in too few groups.
```

### V51 — Comparator: 33-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
that of the best single representation, signed so that a positive value means the hybrid scored better (for RMSE the sign is flipped).
```
NEW:
```latex
that of the best single representation. It is signed so that a positive value means the hybrid scored better, so for RMSE the sign is flipped.
```

### V52 — Binding free energy grouping: 32-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
from SKEMPI 2.0 and AB-Bind and is grouped by protein complex,
```
NEW:
```latex
from SKEMPI 2.0 and AB-Bind. It is grouped by protein complex,
```

### V53 — Best representation per model: 35-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words; dangling participle ("Scoring ..., the best one is").

OLD:
```latex
Scoring each representation on its own, the best one is the same
```
NEW:
```latex
We scored each representation on its own. The best one is the same
```

### V54 — Downstream models: 35-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
\citep{Rasmussen2005}, and each dataset uses the two that scored best
```
NEW:
```latex
\citep{Rasmussen2005}. Each dataset uses the two that scored best
```

### V55 — kNN: 35-word sentence; molecules
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words; kNN runs on every dataset, antibodies included.

OLD:
```latex
The kNN model, which is not a default downstream model, takes $k = 5$ neighbors by default, lowered to the square root of the number of training molecules when
```
NEW:
```latex
The kNN model is not a default downstream model. It takes $k = 5$ neighbors by default, lowered to the square root of the number of training samples when
```

### V56 — Reporting rule: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
Each hybrid is paired by fold with the best single representation, and each comparison is reported
```
NEW:
```latex
Each hybrid is paired by fold with the best single representation. Each comparison is reported
```

### V57 — Main comparison: 35-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
The main comparison takes, on each outer fold, the single representation and the hybrid with the best scores on the training samples, refits both on those samples and scores them once on the held-out samples.
```
NEW:
```latex
On each outer fold, the main comparison takes the single representation and the hybrid with the best scores on the training samples. It refits both on those samples and scores them once on the held-out samples.
```

### V58 — Ensembles: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
Each picks its members, the top 3 representations or all of them, and one of seven ways to pool predictions,
```
NEW:
```latex
Each picks its members, the top 3 representations or all of them. Each also picks one of seven ways to pool predictions,
```

### V59 — Many or few: 35-word sentence; "arms"
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words; "arms" is not her word.

OLD:
```latex
The search's hybrid differs from the hand-built arms in two ways at once: it ranks representations by their standalone scores rather than summed importance,
```
NEW:
```latex
The search's hybrid differs from the hand-built hybrids in two ways at once. It ranks representations by their standalone scores rather than by summed importance,
```

### V60 — Selection bias: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
The selection bias at a budget is the search's own score minus the held-out score, signed so that
```
NEW:
```latex
The selection bias at a budget is the search's own score minus the held-out score. It is signed so that
```

### V61 — Calibration method: 44-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
For the two best single representations and the two best hybrids of each fold's search, split-conformal 90\% prediction intervals are built on the training samples and scored on the held-out samples, by the fraction
```
NEW:
```latex
Split-conformal 90\% prediction intervals are built on the training samples for the two best single representations and the two best hybrids of each fold's search. They are scored on the held-out samples by the fraction
```

### V62 — Stability index: $f$ indexes both folds and features
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: $\mathbf{z}_f$ runs over the $F$ folds, but in the equation and the next paragraph $f$ runs over the $d$ features ($s_f^2$, $\hat p_f$). One letter for two indices. Use $m$ for the fold; the 36-word sentence is split as well.

OLD:
```latex
Selection on each of the $F$ folds gives a binary vector $\mathbf{z}_f \in \{0,1\}^d$, where $d$ is
```
NEW:
```latex
Selection on fold $m$, for $m = 1, \dots, F$, gives a binary vector $\mathbf{z}_m \in \{0,1\}^d$. Here $d$ is
```

### V63 — Blueprint reading: 51-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
To read a blueprint, each selected feature is traced to its base representation, and its importance in the fitted model is measured with SHAP \citep{Lundberg2020} on tree models, with the DeepLIFT-based SHAP explainer on the DNN, and with Kernel SHAP on 50 held-out samples for the SVM and the Gaussian process.
```
NEW:
```latex
To read a blueprint, each selected feature is traced to its base representation. Its importance in the fitted model is measured with SHAP. Tree models use TreeSHAP \citep{Lundberg2020}, the DNN the DeepLIFT-based SHAP explainer, and the SVM and the Gaussian process Kernel SHAP on 50 held-out samples.
```

### V64 — Transfer study: 33-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
The first study asks whether search settings (importance method, allocation method, budget $B$ and internal evaluator) found on a large dataset can be reused on a small one instead of a new search.
```
NEW:
```latex
The first study asks whether search settings found on a large dataset can be reused on a small one instead of a new search. The settings are the importance method, allocation method, budget $B$ and internal evaluator.
```

### V65 — Transfer comparisons: 34-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
They are the best single representation, a hybrid searched on the target's training molecules as the upper reference, the source settings
```
NEW:
```latex
They are the best single representation and, as the upper reference, a hybrid searched on the target's training molecules. The other two are the source settings
```

### V66 — Shared rows design: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
Two hybrids add features of two partner representations to it: ECFP4 and PDV, which are also blind to stereochemistry, or ECFP4 and MHG-GNN, which give no two molecules the same row.
```
NEW:
```latex
Two hybrids add features of two partner representations to it. One adds ECFP4 and PDV, which are also blind to stereochemistry. The other adds ECFP4 and MHG-GNN, which give no two molecules the same row.
```

### V67 — Federated learning: 38-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
without moving the data \citep{McMahan2017}, and it has been run across
```
NEW:
```latex
without moving the data \citep{McMahan2017}. It has been run across
```

### V68 — Distance weighting: 32-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
to that site's training molecules, so a site counts most on chemistry it has seen.
```
NEW:
```latex
to that site's training molecules. A site therefore counts most on chemistry it has seen.
```

### V69 — Families experiment: 31-word sentence
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentence over 30 words.

OLD:
```latex
with one feature selection over all families, and with one selection per family at an even share of the budget.
```
NEW:
```latex
with one feature selection over all families. It also compares both with one selection per family at an even share of the budget.
```

### V70 — Flipped-label experiment: three long sentences (33, 38, 32 words)
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentences over 30 words.

OLD:
```latex
In the first, labels are made two-class (an efflux ratio of at least 2 on Caco-2, a LogD of at least 3), and five sites are formed by $k$-means clusters of the molecules. Up to 30\% of each site's training labels are flipped, at different rates on different sites. Selection at the sites is compared, at each flip level, with the ensembles and the student model of the published process, with one model on the pooled molecules, and with the same selection made on all molecules pooled. In one version, each site first removes the molecules it flags as noisy. Some sites then hold too few molecules, groups or classes to score a representation, and their selection is refused, so this version
```
NEW:
```latex
In the first, labels are made two-class: an efflux ratio of at least 2 on Caco-2, and a LogD of at least 3. Five sites are formed by $k$-means clusters of the molecules. Up to 30\% of each site's training labels are flipped, at different rates on different sites. At each flip level, selection at the sites is compared with the ensembles and the student model of the published process. It is also compared with one model on the pooled molecules, and with the same selection made on all molecules pooled. In one version, each site first removes the molecules it flags as noisy. Some sites then hold too few molecules, groups or classes to score a representation, and their selection is refused. So this version
```

### V71 — Second and third site experiments: 34- and 32-word sentences
- file: thesis.tex
- kind: ai-text
- todo_line: keep existing
- why: Sentences over 30 words.

OLD:
```latex
In the second experiment, each of five sites holds the same molecules as a different representation, and one hybrid over all sites is compared with the best single site and with the published ensembles. In the third, the molecules are split into $k$-means clusters, and features are selected per cluster and used for a model on all molecules, or used for a separate model per cluster.
```
NEW:
```latex
In the second experiment, each of five sites holds the same molecules as a different representation. One hybrid over all sites is compared with the best single site and with the published ensembles. In the third, the molecules are split into $k$-means clusters and features are selected per cluster. These features are used either for one model on all molecules or for a separate model per cluster.
```

### V72 — LogD blueprint: list made ambiguous by an apposition
- file: thesis.tex
- kind: chemistry
- todo_line: keep existing
- why: "tertiary amine environments, a nitrogen bonded to three carbons (written CN(C)C), and aryl chloride environments" reads as a list of three. Make the gloss a relative clause.

OLD:
```latex
Among the bits are tertiary amine environments, a nitrogen bonded to three carbons (written \texttt{CN(C)C}), and aryl chloride environments.
```
NEW:
```latex
Among the bits are tertiary amine environments, in which a nitrogen is bonded to three carbons (\texttt{CN(C)C}), and aryl chloride environments.
```

## For the author

- Two result slots you said to keep open are now filled with text dated 2026-10-10: the dataset-property slot (c03 has not run) and the PCA slot (you wanted it after the DNN re-runs). Keep the text or go back to the slots.
- The text in the dataset-property slot repeats the budget picks of the subsection "Dependence on the downstream model" (LogD DNN 2,000 on every fold, Caco-2 DNN above the SVM, QM9 LightGBM three representations).
- The tuning test and Table tab:tuning_base_reps are a result placed inside Methods. They also tuned the DNN, while the sentence before them says the DNN relies on early stopping (your existing TODO). Decide where the test goes and which DNN statement stays.
- Run times: the lower bound for small molecules, 2.3 hours, is hERG K_i, which the next sentence calls too low. The values 2.3, 28.2 and 11.1 hours are not in the SOURCE line (it gives only 0.41 and 38.58). Check them against ledger.json.
- "Binding free energy" is a ddG dataset, the change in binding free energy on mutation (chemistry guide item 8). Renaming it touches Table tab:datasets and about 40 prose uses across chapters, and needs the sign convention.
- Dablander targets: "MAE in log units of activity" mixes log nM Ki (D2, factor Xa) with log uM IC50 (main protease) (chemistry guide item 5). The script does not say which units it read, so each target's unit needs checking before it can be stated.
- Results use fold counts throughout ("on 4 of 5 folds"). A dated note in your memory says no k-of-N tallies in results (style guide gap 33). Decide whether that rule covers fold counts.
- The bold rule in the caption of Table tab:hybrid_vs_best_single (the four largest gains as a fraction of the single score) is a derived metric. The figures session owns that caption.
- mRMR is described as having run on three antibody datasets, but the Results report no mRMR score.
- Your own sentences over 55 words (gap 27) were not split, because they are your text: "The input to the search is ..." (69 and 54 words) and "For example, while there are many high-performing graph-based embeddings ..." (55).
- Your phrasing "In a dream world ... Unfortunately we live in reality" (gap 25, informal) is kept as yours.
- Chapter 5 length (gap 45): nothing moved to Additional files, as decided.
- Your TODO "Figure 5.1: the search pipeline schematic" may be done, since Figure fig:hybrid_search_space exists.
- "Search grid per dataset" drops settings that "scored clearly worse"; the TODO above it asks whether to state the exact rules.
- Your chapter opening says this chapter measures performance under injected noise. Only the separate-sites experiment flips labels here.

## Applied

Applied 2026-10-10 with the Edit tool. Every OLD was found once in the live thesis.tex before editing.

- V1: applied (note added above her concatenation paragraph).
- V2: applied (note added above her internal-evaluator paragraph).
- V3: applied (reverted to "feature selection methodology").
- V4: applied (reverted to "discovered").
- V5: applied.
- V6: applied.
- V7: applied.
- V8: applied.
- V9: applied.
- V10: applied.
- V11: applied.
- V12: applied.
- V13: applied.
- V14: applied.
- V15: applied.
- V16: applied.
- V17: applied.
- V18: applied.
- V19: applied.
- V20: applied.
- V21: applied.
- V22: applied.
- V23: applied.
- V24: applied.
- V25: applied.
- V26: applied.
- V27: applied, adapted: kept "waits for the DNN re-runs" at the end of the new PCA sentence, because the author's slot note says the PCA conclusion waits for those re-runs.
- V28: applied.
- V29: applied.
- V30: applied.
- V31: applied.
- V32: applied.
- V33: applied.
- V34: applied.
- V35: applied.
- V36: applied, adapted: "p$K_i$ units" instead of "p$K_i$"; units checked against Table tab:datasets.
- V37: applied.
- V38: applied (OLD still present in the caption).
- V39: applied.
- V40: applied.
- V41: applied.
- V42: applied.
- V43: applied.
- V44: applied.
- V45: applied.
- V46: applied.
- V47: applied.
- V48: applied.
- V49: applied.
- V50: applied.
- V51: applied, adapted: "...scored better. For RMSE the sign is therefore flipped." instead of "scored better, so for RMSE...".
- V52: applied.
- V53: applied.
- V54: applied.
- V55: applied.
- V56: applied.
- V57: applied.
- V58: applied.
- V59: applied, extended: "Every arm uses" in the sentence before also became "Each of these uses", so "arms" no longer appears in the paragraph.
- V60: applied.
- V61: applied.
- V62: applied ($\mathbf{z}_f$ appeared nowhere else).
- V63: applied, adapted word order: "TreeSHAP is used for tree models \citep{Lundberg2020}, the DeepLIFT-based SHAP explainer for the DNN, and Kernel SHAP on 50 held-out samples for the SVM and the Gaussian process."
- V64: applied.
- V65: applied.
- V66: applied.
- V67: applied.
- V68: applied.
- V69: applied.
- V70: applied.
- V71: applied.
- V72: applied.
