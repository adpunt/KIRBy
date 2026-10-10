# Proposed supervisor comments, Chapters 1 and 2 (lens TH_LENS.md)

Source read: /Users/apunt/repos/KIRBy/thesis.tex lines 1-394 (Introduction, lines 2-37; QSAR
modelling and label noise, lines 39-394), read in full on 2026-10-10. Line numbers are from that read.
Only two paragraphs in these chapters carry `% [AI text, revised 2026-10-10]`: line 131 and the
Student-t row of Table tab:noise_conditions (line 236). Everything else is hers.
Supporting checks in other chapters are cited by line.

12 high, 20 low. High first.

---

## High

### X-1-1
Location: lines 10-14, chap:intro (sec:intro:questions); Owner: hers
Quote: "The first half of this thesis therefore asks four questions." ... "We extend these methodologies to antibody and chemical reaction prediction tasks."
His comment: The first half gets a clear list of questions, but the second half does not. Chapters 5 and 7 say their questions "are set out in Chapter 1", so the reader goes back and finds none.
Lens item: 5 (also 3, 12)
Evidence: line 1268 (Ch5) "The questions this chapter and Chapter~\ref{chap:similarity} address are set out in Chapter~\ref{chap:intro}"; line 4997 (Ch7) and line 5002 (Ch8) also point to "the research questions of Chapter 1". The questions below are taken from Ch5 conclusions (line 2646), Ch6 introduction (lines 2669-2672) and Ch7 introduction (lines 3396-3399).
Proposed change:
old_string:
```
We extend these methodologies to antibody and chemical reaction prediction tasks.
```
new_string:
```
We extend these methodologies to antibody and chemical reaction prediction tasks.

The second half of this thesis asks three questions. First, how do we select the best set of features for a given dataset and model, and does the method carry over from small molecules to antibodies and reactions? Second, does distance between molecules in a hybrid track their label differences better than distance in the base representations? Do the methods built on that distance, such as prediction from nearest neighbors, work better in a hybrid? Third, can we find the noisy labels in a QSAR dataset, and does acting on them improve a model's held-out score? Does it improve the hybrid more than the best single representation?
```
Priority: high

### X-1-2
Location: line 12, chap:intro; Owner: hers
Quote: "Some of the worst-performing methods within this sweep relied on molecular distance measures. We posited that this distance measure was at least partially to blame for the poor performance. ... so with enough irrelevant features, the distance becomes irrelevant as well"
His comment: Please say in one sentence why distance matters for noisy labels: these methods judge a label against the labels of its neighbors. Then the reader understands why the thesis turns to similarity.
Lens item: 1 (also 12)
Proposed change:
old_string:
```
Some of the worst-performing methods within this sweep relied on molecular distance measures. We posited
```
new_string:
```
Some of the worst-performing methods within this sweep relied on molecular distance measures. Several of these methods judge a label by comparing it with the labels of the molecule's nearest neighbors. They can only be as good as the distance that picks those neighbors. We posited
```
and
old_string:
```
so with enough irrelevant features, the distance becomes irrelevant as well,
```
new_string:
```
so with enough irrelevant features, the distance no longer reflects the property,
```
Priority: high

### X-1-3
Location: line 26, chap:intro (Outline); Owner: hers
Quote: "The hybrid demonstrated higher predictive performance across all small-molecule datasets when compared to the base representations on select downstream models. On antibodies the change in performance depended on the data, however, the larger datasets tended to benefit the most."
His comment: "On select downstream models": which ones? A reviewer will read this as cherry-picking. Also the ensemble baseline result should be said here.
Lens item: 6 (also 18, 8)
Evidence: Ch5 conclusions, line 2648: "With both models, it beat the best single representation on held-out molecules on every dataset"; line 2652: "On four antibody datasets the hybrid beat every single representation, but on most of them an ensemble of one model per representation did better."
Proposed change:
old_string:
```
The hybrid demonstrated higher predictive performance across all small-molecule datasets when compared to the base representations on select downstream models. On antibodies the change in performance depended on the data, however, the larger datasets tended to benefit the most.
```
new_string:
```
On held-out molecules, the hybrid predicted better than the best base representation on every small-molecule dataset, with both of each dataset's downstream models. On four antibody datasets the hybrid beat every single representation, but on most of them an ensemble of one model per representation did better. The larger antibody datasets tended to benefit the most.
```
Priority: high

### X-1-4
Location: line 36, Collaborations; Owner: hers
Quote: "Lhasa also supplied the hERG data used in Chapters~\ref{chap:flaws} to~\ref{chap:noise}; in Chapter~\ref{chap:method}, two of its sets and the public ChEMBL hERG $K_i$ set are treated as data held at separate sites"
His comment: Which hERG data is from Lhasa? The reader will think hERG Ki, used everywhere, is Lhasa data. Please say Lhasa's is the hERG blocking (classification) set and hERG Ki is public ChEMBL.
Lens item: 7 (also 4)
Evidence: line 414 (Ch3) "We also selected hERG $K_i$ data from ChEMBL"; line 1648 (Ch5 caption) "The hERG blocking data were supplied by Lhasa Limited"; line 3402 (Ch7) "hERG ... means the hERG blocking dataset, whose labels are blocker or non-blocker".
Proposed change:
old_string:
```
Lhasa also supplied the hERG data used in Chapters~\ref{chap:flaws} to~\ref{chap:noise}; in Chapter~\ref{chap:method}, two of its sets and the public ChEMBL hERG $K_i$ set
```
new_string:
```
Lhasa also supplied the hERG blocking data, whose labels are blocker or non-blocker, used in Chapters~\ref{chap:flaws} to~\ref{chap:noise}. The hERG $K_i$ data are public, from ChEMBL. In Chapter~\ref{chap:method}, two of Lhasa's sets and the public ChEMBL hERG $K_i$ set
```
Priority: high

### X-2-1
Location: line 63, chap:background, molecular graphs; Owner: hers
Quote: "2D graphs rely on adjacency matrices that encode topological atom- and bond-level features but leave out geometric properties such as bond angles or torsion, meaning they may fail to capture stereochemistry."
His comment: The 2D graph expresses stereoisomers via stereo bonds and chirality tags. Your next sentence even says the featurization encodes chirality and stereochemistry, so the two sentences contradict.
Lens item: 7
Note: he made the same point on old p93 (Ch4, TH-p93a); this is the place where the claim is first made, in Ch2.
Proposed change:
old_string:
```
but leave out geometric properties such as bond angles or torsion, meaning they may fail to capture stereochemistry.
```
new_string:
```
but leave out geometric properties such as bond angles or torsion. They can still record stereochemistry, as chirality tags on atoms and stereo labels on bonds, but not the conformation.
```
Priority: high

### X-2-2
Location: line 141, sec:mol_similarity; Owner: hers
Quote: "When the labels are corrupted or the important features are scarce, similarity scores end up dependent on compounded false or irrelevant information, and the assumption that similar molecules have similar properties breaks down."
His comment: Similarity is computed on the features only, so corrupted labels do not change it. The two problems are different: few relevant features spoil the similarity; noisy labels spoil what the neighbors tell us.
Lens item: 19 (also 1)
Proposed change:
old_string:
```
When the labels are corrupted or the important features are scarce, similarity scores end up dependent on compounded false or irrelevant information, and the assumption that similar molecules have similar properties breaks down.
```
new_string:
```
When the important features are scarce, similarity scores depend on irrelevant information. When the labels are corrupted, similarity itself is unchanged, since it uses only the features, but a molecule's nearest neighbors carry wrong labels. In either case, the labels we observe no longer follow the similar property principle.
```
Priority: high

### X-2-3
Location: line 153, Noise in molecular data; Owner: hers
Quote: "However, Bayesian methods such as Gaussian processes (GPs) do not make this assumption, and neither does conformal prediction \citep{Kolmar2021}."
His comment: Conformal prediction does not model the label distribution at all; it assumes exchangeability and calibrates intervals on labels that are noisy too. Please be more precise, a reviewer (and Lhasa) will pick this up.
Lens item: 19 (also 7)
Proposed change: NEEDS HER. Check what Kolmar2021 actually says about GPs and conformal prediction, and add a citation for the conformal sentence. Suggested wording if it matches the source:
old_string:
```
However, Bayesian methods such as Gaussian processes (GPs) do not make this assumption, and neither does conformal prediction \citep{Kolmar2021}.
```
new_string:
```
However, Gaussian processes (GPs) model each label as a noisy observation of an underlying value \citep{Kolmar2021}. Conformal prediction makes no assumption about how labels are distributed, but it calibrates its intervals on labels that carry the same noise [NEEDS HER: citation].
```
Priority: high

### X-2-4
Location: lines 335-336, sec:noiseinject_pkg, Regression noise; Owner: hers
Quote: "The Gaussian noise has variance the level times the square of the unit. Here the level scales the variance, not the standard deviation as in Equation~\ref{eq:dose_matching}."
His comment: The same word "level" means two different things in Chapter 3 and Chapter 7. The reader comparing levels will be confused. Say what a level of 1.0 means here in standard deviation.
Lens item: 4 (also 17)
Proposed change (the 0.71 is $\sqrt{0.5}$, computed from the definition in the quoted sentence; no new data):
old_string:
```
Here the level scales the variance, not the standard deviation as in Equation~\ref{eq:dose_matching}.
```
new_string:
```
Here the level scales the variance, not the standard deviation as in Equation~\ref{eq:dose_matching}. A level of 1.0 therefore adds noise with a standard deviation of one unit, and a level of 0.5 adds about 0.71 units ($\sqrt{0.5}$).
```
Priority: high

### X-2-5
Location: line 389, sec:evaluation; Owner: hers
Quote: "For small molecules the group is the Bemis--Murcko scaffold family (Section~\ref{sec:noise_injection})."
His comment: Is the scaffold with or without stereochemistry, and where do acyclic molecules go? (Same question as for the InChIKey.) Chapter 3 and Chapter 5 seem to answer differently.
Lens item: 7 (also 17, 21)
Evidence, two things that disagree: line 418 (Ch3) groups by "the chirality-blind Bemis--Murcko scaffold, implemented such that each distinct acyclic molecule forms its own group". Line 1684 (Ch5) groups by "Bemis--Murcko scaffold, maintaining stereochemistry. On QM9 the largest group is the empty scaffold of acyclic molecules, with 1,041 of the 9,978 molecules." So the two groupings differ in two ways: stereochemistry, and whether acyclic molecules share one group.
Proposed change: NEEDS HER to confirm the difference is intended. If it is:
old_string:
```
For small molecules the group is the Bemis--Murcko scaffold family (Section~\ref{sec:noise_injection}).
```
new_string:
```
For small molecules the group is the Bemis--Murcko scaffold family (Section~\ref{sec:noise_injection}). Chapter~\ref{chap:robustness} ignores chirality in the scaffold and gives each acyclic molecule its own group. The later chapters keep stereochemistry and put all acyclic molecules in one group, the empty scaffold.
```
Priority: high

### X-2-6
Location: line 57, Molecular representations in QSAR (neural fingerprints); Owner: hers
Quote: "They can also be pre-trained once on large unlabeled datasets and then fine-tuned for specific downstream tasks, opening the door to transfer learning"
His comment: Worth saying whether the learned representations in this thesis are fine-tuned on the target or used as fixed embeddings. A fixed, task-agnostic embedding is not expected to give a smooth property landscape, a task-trained one should.
Lens item: 9
Note: this is the definition point for his general comment on old p88 (TH-p88b); the Ch4 fix depends on it being stated here. Ch3 uses published checkpoints (line 422: ChemBERTa-77M-MTR mean-pooled, MHG-GNN pretrained checkpoint), and grep finds no fine-tuning anywhere in thesis.tex.
Proposed change: NEEDS HER to confirm no representation in Chapters 5-7 (antibody language models included) is fine-tuned.
old_string:
```
although attention visualization offers some insight \citep{Chithrananda2020}.
```
new_string:
```
although attention visualization offers some insight \citep{Chithrananda2020}. In this thesis, learned representations are taken from published pre-trained checkpoints and are not fine-tuned on the property being predicted. Like fingerprints, they are therefore task-agnostic, and distance in them need not follow the property.
```
Priority: high

### X-2-7
Location: lines 324-325, sec:noiseinject_pkg, Regression noise (and table row line 241); Owner: hers
Quote: "Censoring is parameterized by the fraction $f$ of labels clipped, with its limit at the $(1-f)$ quantile of the clean training labels. On the training split the censored set is the $k = \mathrm{round}(fn)$ largest labels"
His comment: In practice potency is mostly censored on the inactive side (IC50 above the highest tested concentration). Why clip the largest labels? And are censored values flagged in real data (qualifier ">")?
Lens item: 7 (also 21)
Evidence: Ch7 line 3426 already says "We clip at the top of the label range, the rarer case, as real censoring of potency labels such as hERG $K_i$ falls mostly at the low end." Ch2 and Ch3 do not say it.
Proposed change: the first sentence is copied from Ch7. NEEDS HER for the reason the top was chosen.
old_string:
```
of the clean training labels. On the training split the censored
```
new_string:
```
of the clean training labels. The largest labels are clipped, the rarer case, as real censoring of potency labels such as hERG $K_i$ falls mostly at the low end [NEEDS HER: why the top]. On the training split the censored
```
Priority: high

### X-2-8
Location: lines 200-207, sec:noise_injection (Equation eq:dose_matching); Owner: hers
Quote: "where $\tau$ is the target standard deviation of the noise, in the units of the label"
His comment: Here tau is in label units, but in Chapter 3 tau is the noise level (0 to 1.5). How does tau relate to the level? Please write it.
Lens item: 4 (also 17)
Evidence, two things that disagree: line 207 (Ch2) defines $\tau$ as "the target standard deviation of the noise, in the units of the label". Line 499 (Ch3) writes "$\tau \in \{0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5\}$, each a fraction of the standard deviation". The link between the two is given only at line 316: "The requested noise $\tau$ is the noise level times the standard deviation of the clean training labels."
Proposed change (Ch2):
old_string:
```
drawn from the condition's distribution at unit scale.
```
new_string:
```
drawn from the condition's distribution at unit scale. The target $\tau$ is the noise level times the standard deviation of the clean training labels.
```
Also, in Ch3 line 499, write the level as $x$ (as in the retention curve, line 351) instead of $\tau$.
Priority: high

---

## Low

### X-1-5
Location: line 6, chap:intro; Owner: hers
Quote: "Repeat measurements of the same compound-target pair have a standard deviation of about $0.68$ log units for pIC$_{50}$ and $0.54$ log units for p$K_i$ ... Restricting to measurements from a single assay lowers the mean disagreement from $0.50$ to $0.27$ log units \citep{landrum2024}, or from about a 3.2-fold to a 1.9-fold difference."
His comment: Many numbers; one sentence of take-home would help (what this means for a model). Also, is "mean disagreement" the same measure as the standard deviation before? And Table 2.1 gives 0.44 for Kramer, here 0.54.
Lens item: 2 (also 17)
Evidence: Table tab:noise_sources (line 173) cites Kramer2012 for "differing by 0.44 log units on average"; line 6 cites Kramer2012 for "a standard deviation of about 0.54 log units". The two are different statistics; neither place says so.
Proposed change: NEEDS HER to check which statistic landrum2024's 0.50 and 0.27 are.
old_string:
```
or from about a 3.2-fold to a 1.9-fold difference.
```
new_string:
```
or from about a 3.2-fold to a 1.9-fold difference. Repeat measurements therefore differ by roughly half a log unit, and by less within one assay. This sets a floor on the error a model can show when scored against such labels.
```
and in Table tab:noise_sources:
old_string:
```
differing by 0.44 log units on average \citep{Kramer2012}
```
new_string:
```
differing by 0.44 log units on average, a standard deviation of 0.54 \citep{Kramer2012}
```
Priority: low

### X-1-6
Location: line 20, chap:intro (Outline); Owner: hers
Quote: "The same experiments were also run on three assay datasets: LogD, Caco-2 and hERG $K_i$."
His comment: You could say what each endpoint is, and that hERG is a key cardiac safety endpoint. This links the thesis to drug safety from the start.
Lens item: 14 (also 4)
Proposed change:
old_string:
```
three assay datasets: LogD, Caco-2 and hERG $K_i$.
```
new_string:
```
three assay datasets: LogD (lipophilicity), Caco-2 (intestinal permeability) and hERG $K_i$ (binding to the hERG potassium channel, whose blockade is a cardiac safety risk).
```
Priority: low

### X-1-7
Location: line 22, chap:intro (Outline); Owner: hers
Quote: "Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular similarity."
His comment: "Another topic" suggests it is disconnected from noise, but the next sentences show it is not. Say the link directly.
Lens item: 1 (also 19)
Proposed change:
old_string:
```
Chapter~\ref{chap:flaws} takes a turn to look at another topic: molecular similarity.
```
new_string:
```
Chapter~\ref{chap:flaws} turns to molecular similarity, on which many methods for noisy labels rely.
```
Priority: low

### X-1-8
Location: line 24, chap:intro (Outline); Owner: hers
Quote: "It takes in several base representations and selects the most important features for that task by a fixed budget. ... The actual output of the hybrid search is not the set of features, but rather the settings that define the feature selection method used to generate that hybrid representation."
His comment: "Budget" and "settings" are not yet defined. What settings?
Lens item: 4
Proposed change:
old_string:
```
selects the most important features for that task by a fixed budget.
```
new_string:
```
selects the most important features for that task, up to a fixed budget, the number of features kept.
```
and
old_string:
```
but rather the settings that define the feature selection method used to generate that hybrid representation.
```
new_string:
```
but rather the settings that define the feature selection method used to generate that hybrid representation, such as how features are ranked and how the budget is split among the base representations.
```
Priority: low

### X-1-9
Location: line 28, chap:intro (Outline); Owner: hers
Quote: "ranking each candidate by how well each molecule's five nearest neighbors predict its label. On every small-molecule dataset, distance in the hybrid tracked label differences more closely, and the nearest neighbors' predictions improved."
His comment: Candidate what? And the hybrid is chosen by the neighbors' predictions, then judged by them: is this on molecules not used in the search? "More closely" than what?
Lens item: 8 (also 4, 6)
Evidence: Ch6 line 2678: "everything an experiment compares is chosen from searches on that fold's training molecules alone, and then scored on its held-out molecules."
Proposed change:
old_string:
```
ranking each candidate by how well each molecule's five nearest neighbors predict its label. On every small-molecule dataset, distance in the hybrid tracked label differences more closely, and the nearest neighbors' predictions improved.
```
new_string:
```
ranking each candidate hybrid by how well each molecule's five nearest neighbors predict its label. On held-out molecules of every small-molecule dataset, distance in the hybrid tracked label differences more closely than in the base representations, and the nearest neighbors' predictions improved.
```
Priority: low

### X-2-9
Location: line 43, Molecular representations in QSAR (first paragraph); Owner: hers
Quote: "The success of machine learning hinges on the quality of input data. ... For smaller datasets, the number of parameters is limited by the size of the training set, to prevent overfitting."
His comment: This opening does not say why representations matter for a thesis on noise. One sentence would do it.
Lens item: 12 (also 1)
Proposed change:
old_string:
```
For smaller datasets, the number of parameters is limited by the size of the training set, to prevent overfitting.
```
new_string:
```
For smaller datasets, the number of parameters is limited by the size of the training set, to prevent overfitting. For this thesis, the representation matters twice. It is the input to the model, and it decides which molecules count as neighbors when a method looks for noisy labels.
```
Priority: low

### X-2-10
Location: line 75, Models for molecular property prediction; Owner: hers
Quote: "Decision trees handle data in a ``contributory'' rather than ``consensus'' manner. In consensus methods such as linear regression or neural networks"
His comment: In QSAR "consensus model" usually means combining the predictions of several models. Using it for a single global model may confuse the reader. Where do these two terms come from?
Lens item: 19 (also 4)
Proposed change (no citation is given for the two quoted terms; if they come from a source, NEEDS HER to cite it instead):
old_string:
```
Decision trees handle data in a ``contributory'' rather than ``consensus'' manner. In consensus methods such as
```
new_string:
```
Decision trees make local rather than global predictions. In global methods such as
```
Priority: low

### X-2-11
Location: line 131, sec:mol_similarity (first paragraph); Owner: AI
Quote: "Nearest-neighbor prediction and GP kernels rely on this principle, as do many noise detection methods. However, similarity is only defined relative to a representation and a coefficient"
His comment: Similarity is in the eyes of the beholder ;-) Say here how a noise detector uses neighbors, so the reader sees why the choice of representation matters for noise.
Lens item: 1 (also 9)
Proposed change (full paragraph; marker line kept; all citations kept):
old_string:
```
Much of cheminformatics rests on the similar property principle, namely that structurally similar molecules tend to have similar properties, including biological activity \citep{Johnson1990, vanTilborg2022}. Nearest-neighbor prediction and GP kernels rely on this principle, as do many noise detection methods. However, similarity is only defined relative to a representation and a coefficient, the formula that compares two feature vectors \citep{Maggiora2014}. Two molecules that are near-identical under a substructure fingerprint may be far apart under a PDV. A computed similarity value is therefore easily over-interpreted \citep{Maggiora2014}.
```
new_string:
```
Much of cheminformatics rests on the similar property principle: structurally similar molecules tend to have similar properties, including biological activity \citep{Johnson1990, vanTilborg2022}. Nearest-neighbor prediction and GP kernels rely on this principle. So do many noise detection methods, which flag a label that differs from the labels of the molecule's neighbors. However, similarity is only defined relative to a representation and a coefficient, the formula that compares two feature vectors \citep{Maggiora2014}. Two molecules that are near-identical under a substructure fingerprint may be far apart under a PDV. A computed similarity value is therefore easily over-interpreted \citep{Maggiora2014}. A noise detector built on neighbors depends on both choices as well.
```
Priority: low

### X-2-12
Location: line 138, sec:mol_similarity; Owner: hers
Quote: "Continuous representations are instead compared with a distance, most often the Euclidean distance ... Both measures are sums over the features"
His comment: Chapter 4 then uses Manhattan, not Euclidean. Worth saying here. Also Tanimoto is a ratio of sums, not a sum.
Lens item: 17 (also 19)
Evidence: line 1060 (Ch4) "all others by Manhattan distance on standardized features"; line 947 defines Manhattan.
Proposed change:
old_string:
```
$\sqrt{\sum_{j=1}^{d}(x_j - x'_j)^2}$. Both measures are sums over the features,
```
new_string:
```
$\sqrt{\sum_{j=1}^{d}(x_j - x'_j)^2}$. Chapter~\ref{chap:flaws} uses the Manhattan distance, the sum of absolute differences, for these representations. Both measures are built from sums over the features,
```
Priority: low

### X-2-13
Location: line 145, sec:mol_similarity; Owner: hers
Quote: "Another barrier to distance-based reasoning is persistent noise across labels."
His comment: What is "persistent" noise? Noise in most labels, or noise shared by neighbors (same lab, same series)?
Lens item: 6 (also 19)
Proposed change (matches the next sentence, "when all of the neighbors have some form of noise as well"; NEEDS HER if she meant noise shared within a scaffold family, which would point to grouped-shifted instead):
old_string:
```
Another barrier to distance-based reasoning is persistent noise across labels.
```
new_string:
```
Another barrier to distance-based reasoning is noise in most labels, not only in a few.
```
Priority: low

### X-2-14
Location: line 147, sec:mol_similarity (last paragraph); Owner: hers
Quote: "When many of their features carry no information about the property, distances computed over them are dominated by those features."
His comment: A one-sentence recap of the section before going to noise would help the reader.
Lens item: 3
Proposed change:
old_string:
```
distances computed over them are dominated by those features.
```
new_string:
```
distances computed over them are dominated by those features. In short, every method that relies on similarity, noise detection included, is only as good as the representation the similarity is computed in.
```
Priority: low

### X-2-15
Location: line 196, Figure fig:noise_conditions caption, panel h; Owner: hers
Quote: "noise that ignores the groups gives about 0.14 with about 50 labels per group."
His comment: Where does 0.14 come from?
Lens item: 4
Proposed change (0.14 = $1/\sqrt{50}$; 2,000 labels in 40 groups is 50 per group, both in the caption):
old_string:
```
noise that ignores the groups gives about 0.14 with about 50 labels per group.
```
new_string:
```
noise that ignores the groups gives about 0.14, that is $1/\sqrt{50}$, with about 50 labels per group.
```
Priority: low

### X-2-16
Location: line 236, Table tab:noise_conditions, Student-t row; Owner: AI
Quote: "Differences between repeated measurements are non-normal \citep{Kruger2012}; no published estimate of $\nu$, so we chose $\nu = 5$"
His comment: Kruger studied differences between related targets, as the Laplace row says. Why 5 degrees of freedom?
Lens item: 19 (also 21)
Proposed change (row rewritten to match the Laplace row and line 155; marker line kept):
old_string:
```
Student-$t$ & Shape: heavier tails than Gaussian & $\nu = 5$ & Differences between repeated measurements are non-normal \citep{Kruger2012}; no published estimate of $\nu$, so we chose $\nu = 5$ \\
```
new_string:
```
Student-$t$ & Shape: heavier tails than Gaussian & $\nu = 5$ & Differences in binding affinity between related targets are non-normal, and differences between assays resemble them \citep{Kruger2012}; no published estimate of $\nu$, so we chose $\nu = 5$ \\
```
NEEDS HER: a few words on why 5 was chosen, if there is a reason.
Priority: low

### X-2-17
Location: line 225, sec:noise_injection; Owner: hers
Quote: "so under grouped-wider we chose to draw scaffold families until they covered $20\%$ of the training molecules."
His comment: This share has no source. Did you try other shares? A reviewer might challenge it.
Lens item: 8
Proposed change: NEEDS HER. One sentence saying whether other shares were run, and if so the result; if not, say the value was not varied.
Priority: low

### X-2-18
Location: line 301, Implementation of the noise conditions; Owner: hers
Quote: "The two implementations have been confirmed to deliver the same noise using uniform seeds."
His comment: How was this confirmed? A Rust and a Python random generator do not give the same draws from the same seed. Same draws, or same statistics?
Lens item: 8 (also 19)
Proposed change: NEEDS HER. Replace with what was compared (for example the delivered noise standard deviation and mean per condition and level) and where the record is.
Priority: low

### X-2-19
Location: lines 352-354, sec:noiseinject_pkg, Robustness metrics; Owner: hers
Quote: "and lies approximately within $[0, 1]$. It is also less directly correlated with clean-label $R^2$ than other robustness metrics we tested"
His comment: Why "approximately"? Can retention be above 1? And "less directly correlated": by how much, and against which metrics?
Lens item: 6 (also 21)
Note: he asked "retention > 1?" on old p102 (TH-p102b) as a result question; this is the definition, where the answer belongs.
Proposed change (first part follows from the definition $\mathrm{ret}(x) = m(x)/m(0)$ at line 351):
old_string:
```
and lies approximately within $[0, 1]$.
```
new_string:
```
and usually lies within $[0, 1]$. It exceeds 1 if noise raises the score, and falls below 0 if $R^2$ turns negative.
```
Second part: NEEDS HER. Ch3 line 499 also says only "less directly correlated" with no numbers. Give the correlations, or name the other metrics tested.
Priority: low

### X-2-20
Location: line 174, Table tab:noise_sources, federated learning row; Owner: hers
Quote: "Federated learning & Teacher models label a shared public set, and a student model learns from those labels"
His comment: This is one kind of federated learning, federated knowledge distillation. Better to name it.
Lens item: 19 (also 7)
Evidence: Ch5 line 1977 refers back to this row as "the federated knowledge-distillation process of Chapter~\ref{chap:background} (Table~\ref{tab:noise_sources})".
Proposed change:
old_string:
```
Federated learning & Teacher models label a shared public set,
```
new_string:
```
Federated learning (knowledge distillation) & Teacher models label a shared public set,
```
Priority: low

### X-2-21
Location: line 393, chap:background Conclusions; Owner: hers
Quote: "Measures of similarity depend on the representation as well, and a representation with uninformative features generates negative effects across the board. ... No method works on every dataset, all of them show promise conditionally."
His comment: "Across the board": which effects? Also link the chapter end to noise and to drug safety.
Lens item: 14 (also 6)
Proposed change:
old_string:
```
Measures of similarity depend on the representation as well, and a representation with uninformative features generates negative effects across the board.
```
new_string:
```
Measures of similarity depend on the representation as well. A representation with many uninformative features weakens both the model and every method that relies on similarity, including neighbor-based noise detection.
```
Optional, NEEDS HER approval since it is a claim: add after the last sentence of the chapter, "This matters most where labels support safety decisions, such as hERG data pooled from many sources."
Priority: low

### X-2-22
Location: line 45, Molecular representations in QSAR (SMILES); Owner: hers
Quote: "which consists of a sequence of ASCII characters generated by a fixed set of rules to specify a given chemical identity \citep{Weininger1988}."
His comment: One molecule has many valid SMILES. Worth saying, since Chapter 4 uses canonical and randomized SMILES.
Lens item: 7
Evidence: line 1153 (Ch4) lists "canonical SMILES and randomized SMILES" among the representations.
Proposed change (NEEDS HER: a citation for canonical SMILES):
old_string:
```
to specify a given chemical identity \citep{Weininger1988}.
```
new_string:
```
to specify a given chemical identity \citep{Weininger1988}. One molecule has many valid SMILES strings. A canonical SMILES fixes one of them, and randomized SMILES sample others, as in Chapter~\ref{chap:flaws}.
```
Priority: low

### X-2-23
Location: line 47, Molecular representations in QSAR (MACCS); Owner: hers
Quote: "the Molecular ACCess System (MACCS) key, a substructure key-based fingerprint that encodes predefined atom symbols and properties, as well as bond types \citep{Xie2020}."
His comment: More precisely, each MACCS bit is one predefined substructure pattern (SMARTS).
Lens item: 7
Evidence: Ch5 line 1393 "MACCS keys are named by their SMILES arbitrary target specification (SMARTS) pattern".
Proposed change:
old_string:
```
a substructure key-based fingerprint that encodes predefined atom symbols and properties, as well as bond types \citep{Xie2020}.
```
new_string:
```
a substructure key-based fingerprint in which each bit records whether one predefined substructure pattern is present, from a single atom type to a small ring or bond pattern \citep{Xie2020}.
```
Priority: low
