# Review: Chapter 3 (chap:robustness) after the style pass of 2026-10-10

Compared: thesis.tex.before_style_apply_2026-10-10 (chapter at lines 641-1271) against thesis.tex (lines 671-1351). Every OLD block below was checked with python3 -I str.count to occur exactly once in the current thesis.tex. Edits do not overlap. Edits are in file order.

Findings in short:
- 24 of her paragraphs, captions or table headers were changed with no "edited your text" note. Most are "logD" -> "LogD", "hERG" -> "hERG $K_i$", "K$_i$" -> "$K_i$", "Murcko" -> "Bemis--Murcko", a tense change and comma splices turned into semicolons. The changes fix real inconsistencies, so none is reverted; each gets its note.
- Chemistry guide item 7 is still open in this chapter. The text says repeat p$K_i$ measurements "disagree by about 0.54". In Chapter 4 (line 1527), 0.54 is the standard deviation and 0.44 the mean difference.
- An AI paragraph says all six representations are computed in Python. Her paragraph four lines above says Rust computes some of them.
- Her sentence "VBLL networks report only epistemic uncertainty" contradicts the chapter's own Equation (sampling_split) paragraph and Table tab:uncertainty_rise. Both give VBLL one aleatoric value per fit. Her TODO(cite) already asks for this check.
- In the AI table tab:condition_choice, the censoring row says the evidence is "The HOMO--LUMO gap only". Methods (line 755) says \aucnorm{} under censoring was also computed on the three assay datasets. Only the reporting is QM9-only.
- Gap 38 is not done. Five captions still open with the symbol \aucnorm{} instead of a noun phrase.
- The new sentence in her forests paragraph says RF ranks "on LogD first or second". The thesis gives "first on LogD at five of six" (line 1124). Second place for the sixth representation is not in the thesis.
- AI paragraph at 772 uses "Traditional approaches", "RAM bottlenecks at scale", "redundant deserialization overhead", and claims a "constant low memory footprint regardless of dataset size".

Checks that passed: the chapter opens by naming Chapter 2 and its last paragraph points to Chapter 4. The title and all section titles are in sentence case. There is no "Results:" prefix and no Discussion section. Limitations sit inside Conclusions. Every \ref in the chapter resolves: chapter, section, figure, table and equation labels are in thesis.tex/preamble, and the eleven app:noise:sec:* labels are in thesis_rework/stage1_takeaways_bullets/additional_files_body.tex. Every number in the current chapter is also in the backup chapter, except 7.4 (pH, already elsewhere in thesis.tex) and the TODO-only "20261008". The Wilcoxon, Tukey and bootstrap statements are unchanged, as decided.

## Edits

### V1 — Note on the chapter title change
- kind: author-text (note only)
- todo_line: % TODO: manual review -- title was: Noise Robustness Across Molecular Representations and Model Architectures -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
- why: The title was changed from title case with no note. Chapters 2 and 4 got the same change with a note; this one did not.
OLD:
```latex
\chapter{Noise robustness across molecular representations and model architectures}\label{chap:robustness}
```
NEW:
```latex
% TODO: manual review -- title was: Noise Robustness Across Molecular Representations and Model Architectures -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\chapter{Noise robustness across molecular representations and model architectures}\label{chap:robustness}
```

### V2 — Note on the first Introduction paragraph of hers
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "direction comparisons between how" -> "direct comparisons of how" ("direction" was a typo, and "between" needs two things); "responds", "remains" -> "respond", "remain" to agree with "representations" and "comparisons".
- why: "direction" -> "direct" and "between" -> "of" are word substitutions. The strict rule needs a note for them.
OLD:
```latex
While there have been many investigations into the effect of noise on molecular property prediction, direct comparisons of how
```
NEW:
```latex
% TODO: manual review -- edited your text: "direction comparisons between how" -> "direct comparisons of how" ("direction" was a typo, and "between" needs two things); "responds", "remains" -> "respond", "remain" to agree with "representations" and "comparisons".
While there have been many investigations into the effect of noise on molecular property prediction, direct comparisons of how
```

### V3 — VBLL sentence contradicts the chapter's own uncertainty definitions
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: "Because only its output layer keeps a learned observation-noise parameter, VBLL networks report only epistemic uncertainty" became a sentence saying VBLL learns one observation-noise variance per fit. The Uncertainty estimation subsection says VBLL's predicted noise variance "is one number for the whole fit", and Table tab:uncertainty_rise lists VBLL aleatoric as "one per fit", so "only epistemic" disagreed with both. Your TODO(cite) above still asks to check this against the implementation.
- why: Two parts of the chapter disagree. Line 739 says VBLL reports only epistemic uncertainty. Lines 822-823 and Table tab:uncertainty_rise give VBLL an aleatoric component, one value for the whole fit. The note goes below her TODO(cite) line, which stays.
OLD:
```latex
To get these networks producing uncertainty estimates, we applied four probabilistic transformations
```
NEW:
```latex
% TODO: manual review -- edited your text: "Because only its output layer keeps a learned observation-noise parameter, VBLL networks report only epistemic uncertainty" became a sentence saying VBLL learns one observation-noise variance per fit. The Uncertainty estimation subsection says VBLL's predicted noise variance "is one number for the whole fit", and Table tab:uncertainty_rise lists VBLL aleatoric as "one per fit", so "only epistemic" disagreed with both. Your TODO(cite) above still asks to check this against the implementation.
To get these networks producing uncertainty estimates, we applied four probabilistic transformations
```
OLD:
```latex
Because only its output layer keeps a learned observation-noise parameter, VBLL networks report only epistemic uncertainty.
```
NEW:
```latex
VBLL networks learn one observation-noise variance for the whole fit, so their aleatoric component is the same for every molecule and only the epistemic component varies between molecules.
```

### V4 — Split the 43-word sentence listing the variant models (AI paragraph)
- kind: ai-text
- todo_line: (existing "% TODO: manual review" stays)
- why: One 43-word sentence with three nested appositives. The limit is 30 words.
OLD:
```latex
The six variant models are the random forest with 300 trees, RF (300 trees); the heteroscedastic GP, GP (het.); the Bayesian networks with a variance head, BNN-$\alpha$ (var.\ head) and BNN-$\beta$ (var.\ head); and the heteroscedastic VBLL networks, VBLL-$\alpha$ (het.) and VBLL-$\beta$ (het.).
```
NEW:
```latex
The six variant models are RF (300 trees), a random forest with 300 trees, and GP (het.), the heteroscedastic GP. The other four are the Bayesian networks with a variance head, BNN-$\alpha$ (var.\ head) and BNN-$\beta$ (var.\ head), and the heteroscedastic VBLL networks, VBLL-$\alpha$ (het.) and VBLL-$\beta$ (het.).
```

### V5 — AI paragraph says Python computes all six representations; her paragraph says Rust computes some
- kind: ai-text
- todo_line: (existing "% TODO: manual review" stays)
- why: Her paragraph at the start of the subsection says Python "computes several representations" and Rust "computes the remaining representations". The AI paragraph says "The six representations described above are computed in Python." The two disagree, so the AI sentence goes. Which representations Rust computes is listed for the author.
OLD:
```latex
The six representations described above are computed in Python. The run halts if any molecule has an ECFP4 or Avalon fingerprint
that is entirely zero, so that a molecule that failed featurization cannot enter training.
```
NEW:
```latex
The run halts if any molecule has an ECFP4 or Avalon fingerprint that is entirely zero, so that a molecule that failed featurization cannot enter training.
```

### V6 — Rewrite the memory-mapping paragraph plainly (AI paragraph)
- kind: ai-text
- todo_line: (existing "% TODO: manual review" stays)
- why: "Traditional approaches", "RAM bottlenecks at scale", "redundant deserialization overhead" and "constant low memory footprint regardless of dataset size" are sales register, and the last is a claim nobody measured. The new text says what the pipeline does. No numbers added.
OLD:
```latex
Traditional approaches load entire datasets into memory from CSV or Parquet files, creating RAM bottlenecks at scale and redundant deserialization overhead across experimental replicates. Our pipeline instead serializes preprocessed data into custom memory-mapped binary files (\texttt{.mmap}) that enable on-demand access with constant low memory footprint regardless of dataset size (Figure~\ref{fig:fig_mmap}).
```
NEW:
```latex
A dataset loaded from a CSV or Parquet file is held in memory as a whole, and the file is parsed again for every replicate. Our pipeline instead writes the preprocessed data to memory-mapped binary files (\texttt{.mmap}) (Figure~\ref{fig:fig_mmap}). A memory-mapped file stays on disk, and only the records a program reads are loaded into memory.
```

### V7 — Remove the "It also ... It also" chain (AI paragraph)
- kind: ai-text
- todo_line: (existing "% TODO: manual review" stays)
- why: Three sentences in a row start "It also" or repeat "records". The new text groups what is recorded per molecule and what is written per run.
OLD:
```latex
For every molecule in every split, the Rust program records its canonical SMILES, clean label,
noisy label and the standardized value written to file. It also records the injected noise, its
scale and the noise pattern applied. It also writes a summary of each run that lists the parameters
of the noise condition and the amount of noise delivered.
```
NEW:
```latex
For every molecule in every split, the Rust program records its canonical SMILES, its clean and
noisy labels and the standardized value written to file. The record also holds the injected noise, its
scale and the noise pattern applied. A summary of each run lists the parameters
of the noise condition and the amount of noise delivered.
```

### V8 — Note on "Murcko" -> "Bemis--Murcko"
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "Murcko scaffolds" -> "Bemis--Murcko scaffolds", the name used everywhere else in the thesis (chemistry guide item 15).
- why: A term change in her text with no note.
OLD:
```latex
No held-out label carries injected noise, so every comparison against the injected noise was made on
```
NEW:
```latex
% TODO: manual review -- edited your text: "Murcko scaffolds" -> "Bemis--Murcko scaffolds", the name used everywhere else in the thesis (chemistry guide item 15).
No held-out label carries injected noise, so every comparison against the injected noise was made on
```

### V9 — Note on "who" -> "that"
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "models who report" -> "models that report"; "uncertainty rise" -> "uncertainty rises".
- why: A word substitution with no note ("rise" -> "rises" alone would be trivial).
OLD:
```latex
To assess the interaction between uncertainty estimation and label noise, we posed three questions.
```
NEW:
```latex
% TODO: manual review -- edited your text: "models who report" -> "models that report"; "uncertainty rise" -> "uncertainty rises".
To assess the interaction between uncertainty estimation and label noise, we posed three questions.
```

### V10 — Note on "logD" in the three-outcomes caption
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "logD" -> "LogD", the form used for the dataset everywhere else (chemistry guide item 11).
- why: A term change in her caption with no note.
OLD:
```latex
\caption[Variance shares of clean performance, noisy performance and robustness on all four datasets.]
```
NEW:
```latex
% TODO: manual review -- edited your text: "logD" -> "LogD", the form used for the dataset everywhere else (chemistry guide item 11).
\caption[Variance shares of clean performance, noisy performance and robustness on all four datasets.]
```

### V11 — Note on the grouped-shifted paragraph
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "lowered the \aucnorm{} of across all models and representations" (broken) -> "lowered the \aucnorm{} of every model on every representation"; the comma after "drop in predictive performance" became a semicolon, because two sentences were joined by a comma.
- why: A broken phrase was repaired and a comma splice restructured, with no note.
OLD:
```latex
While alternative shapes of noise may be accurately modeled by the Gaussian distribution, we do not find
```
NEW:
```latex
% TODO: manual review -- edited your text: "lowered the \aucnorm{} of across all models and representations" (broken) -> "lowered the \aucnorm{} of every model on every representation"; the comma after "drop in predictive performance" became a semicolon, because two sentences were joined by a comma.
While alternative shapes of noise may be accurately modeled by the Gaussian distribution, we do not find
```

### V12 — Note on the tense change "falls" -> "fell"
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "SVM on the other hand falls from first to fifth" -> "fell", to match "started" and "was ranked" in the sentence before.
- why: A tense change in her text with no note.
OLD:
```latex
Most of these observations are being made at the upper end of the noise range.
```
NEW:
```latex
% TODO: manual review -- edited your text: "SVM on the other hand falls from first to fifth" -> "fell", to match "started" and "was ranked" in the sentence before.
Most of these observations are being made at the upper end of the noise range.
```

### V13 — Measurement-error sentence uses the standard deviation as the typical disagreement (chemistry guide item 7); note on the opening phrase
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: "results coming on HOMO--LUMO gap from QM9" (broken) -> "results on the QM9 HOMO--LUMO gap"; "pK$_i$" -> "p$K_i$"; "disagree by about 0.54 log units" -> "differ by 0.44 log units on average, with a standard deviation of 0.54", as Chapter 4 states the same source, and "about one unit of that estimated laboratory error" -> "about one standard deviation of that laboratory error" (chemistry guide item 7). 0.6 x 0.915 = 0.55, so the noise-level comparison is unchanged. Not checked against Kramer 2012 in this session.
- why: The thesis quotes the same source with two meanings for "disagree by about". Chapter 4 (line 1527) gives 0.44 as the mean difference and 0.54 as the standard deviation. Both numbers are already in thesis.tex.
OLD:
```latex
So far, we have only discussed results on the QM9 HOMO--LUMO gap, a computed property whose labels carry no measurement error.
```
NEW:
```latex
% TODO: manual review -- edited your text: "results coming on HOMO--LUMO gap from QM9" (broken) -> "results on the QM9 HOMO--LUMO gap"; "pK$_i$" -> "p$K_i$"; "disagree by about 0.54 log units" -> "differ by 0.44 log units on average, with a standard deviation of 0.54", as Chapter 4 states the same source, and "about one unit of that estimated laboratory error" -> "about one standard deviation of that laboratory error" (chemistry guide item 7). 0.6 x 0.915 = 0.55, so the noise-level comparison is unchanged. Not checked against Kramer 2012 in this session.
So far, we have only discussed results on the QM9 HOMO--LUMO gap, a computed property whose labels carry no measurement error.
```
OLD:
```latex
Repeat measurements of the same compound disagree by about 0.54 log units for p$K_i$ \citep{Kalliokoski2013, Kramer2012}. The hERG $K_i$ labels have a spread of 0.915 log units, so a noise level of 0.6 of that spread is about one unit of that estimated laboratory error.
```
NEW:
```latex
Repeat p$K_i$ measurements of the same compound differ by 0.44 log units on average, with a standard deviation of 0.54 \citep{Kalliokoski2013, Kramer2012}. The hERG $K_i$ labels have a spread of 0.915 log units, so a noise level of 0.6 of that spread is about one standard deviation of that laboratory error.
```

### V14 — Caption of tab:assay_shape: noun-phrase opening (gap 38); note the table-row changes
- kind: author-text (caption)
- todo_line: replaces the existing note line above the caption (text in NEW)
- why: The caption opens with the symbol \aucnorm{}. The table rows were changed ("logD" -> "LogD", "hERG K$_i$" -> "hERG $K_i$") and the note does not say so.
OLD:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness under the noise distributions on the three assay datasets.]{\aucnorm{} (higher is more robust) under Gaussian noise,
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39); the caption now opens with the noun "Robustness" (gap 38); table rows "logD" -> "LogD" and "hERG K$_i$" -> "hERG $K_i$", the forms used everywhere else (chemistry guide items 10 and 11).
\caption[Robustness under the noise distributions on the three assay datasets.]{Robustness (\aucnorm{}, higher is more robust) under Gaussian noise,
```

### V15 — Caption of fig:grid: noun-phrase opening (gap 38)
- kind: author-text (caption)
- todo_line: replaces the existing note line above the caption (text in NEW)
- why: Gap 38: the references open every caption with a noun phrase.
OLD:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness of each base model on each representation on QM9.]{\aucnorm{} (higher is more robust) of the 13 base models
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39); the caption now opens with the noun "Robustness" (gap 38).
\caption[Robustness of each base model on each representation on QM9.]{Robustness (\aucnorm{}, higher is more robust) of the 13 base models
```

### V16 — Caption of fig:deep: noun-phrase opening (gap 38)
- kind: author-text (caption)
- todo_line: replaces the existing note line above the caption (text in NEW)
- why: Gap 38.
OLD:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness under every noise condition on a subset of models on QM9.]{\aucnorm{} (higher is more robust) under
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39); the caption now opens with the noun "Robustness" (gap 38).
\caption[Robustness under every noise condition on a subset of models on QM9.]{Robustness (\aucnorm{}, higher is more robust) under
```

### V17 — Caption of tab:censoring: noun-phrase opening (gap 38)
- kind: author-text (caption)
- todo_line: replaces the existing note line above the caption (text in NEW)
- why: Gap 38.
OLD:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness under censoring and under Gaussian noise on QM9.]{\aucnorm{} (higher is more robust) under censoring
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39); the caption now opens with the noun "Robustness" (gap 38).
\caption[Robustness under censoring and under Gaussian noise on QM9.]{Robustness (\aucnorm{}, higher is more robust) under censoring
```

### V18 — Note on the grouped-shifted assay paragraph: list the other changes
- kind: author-text (note only)
- todo_line: the existing note gains one sentence (text in NEW)
- why: The note covers the hERG $K_i$ rank change, but "group-shifted" -> "grouped-shifted" and "logD" -> "LogD" were also made and are not listed. "at under" -> "under" is a doubled word (trivial).
OLD:
```latex
Re-check once the fixed Chapter 3 results land.
Across all datasets, models struggle
```
NEW:
```latex
Re-check once the fixed Chapter 3 results land. Also "group-shifted" -> "grouped-shifted" (the condition's name elsewhere) and "logD" -> "LogD".
Across all datasets, models struggle
```

### V19 — Note on "hERG" -> "hERG $K_i$" (LightGBM paragraph)
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "hERG" -> "hERG $K_i$" twice, the dataset's name everywhere else.
- why: A word added to her text with no note.
OLD:
```latex
LightGBM has particularly varied results across the datasets.
```
NEW:
```latex
% TODO: manual review -- edited your text: "hERG" -> "hERG $K_i$" twice, the dataset's name everywhere else.
LightGBM has particularly varied results across the datasets.
```

### V20 — Note on "logD" -> "LogD" (rank transfer paragraph)
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "logD" -> "LogD" three times (chemistry guide item 11).
- why: A term change with no note.
OLD:
```latex
While the results are not identical between the datasets, there are some patterns that emerge
```
NEW:
```latex
% TODO: manual review -- edited your text: "logD" -> "LogD" three times (chemistry guide item 11).
While the results are not identical between the datasets, there are some patterns that emerge
```

### V21 — Forests paragraph: "on LogD first or second" is not in the thesis
- kind: author-text
- todo_line: the existing note gains one sentence (text in NEW)
- why: The new last sentence says RF ranks "on LogD first or second". The note says these ranks are "already given earlier in this section", but the earlier text (line 1124) says "first on LogD at five of six". The rank on the sixth representation is not stated anywhere. The NEW sentence uses only what line 1124 says.
OLD:
```latex
Re-check once the fixed Chapter 3 results land.
We consider the impact of label noise on each model family.
```
NEW:
```latex
Re-check once the fixed Chapter 3 results land. "On LogD first or second" became "first with five of the six representations on LogD", the rank given earlier in this section; the rank with the sixth was not checked.
We consider the impact of label noise on each model family.
```
OLD:
```latex
On QM9 it ranks second or third, and on LogD first or second.
```
NEW:
```latex
On QM9 it ranks second or third, and it ranks first with five of the six representations on LogD.
```

### V22 — Note on "logD" -> "LogD" (GP paragraph)
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
- why: A term change with no note.
OLD:
```latex
Other models perform conditionally well, and in some cases that condition is the representation they are paired with.
```
NEW:
```latex
% TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
Other models perform conditionally well, and in some cases that condition is the representation they are paired with.
```

### V23 — Note on "logD" -> "LogD" (feed-forward networks paragraph)
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
- why: A term change with no note.
OLD:
```latex
The feed-forward networks are the most variable,
```
NEW:
```latex
% TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
The feed-forward networks are the most variable,
```

### V24 — Note on "logD" -> "LogD" (noise tolerance paragraph)
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "logD" -> "LogD" twice (chemistry guide item 11).
- why: A term change with no note.
OLD:
```latex
While it is clear that QSAR model performance drops with label noise, different models drop faster than others.
```
NEW:
```latex
% TODO: manual review -- edited your text: "logD" -> "LogD" twice (chemistry guide item 11).
While it is clear that QSAR model performance drops with label noise, different models drop faster than others.
```

### V25 — Note on "logD" -> "LogD" (RF and QRF paragraph)
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
- why: A term change with no note.
OLD:
```latex
We compare RF and QRF directly with the same underlying architecture.
```
NEW:
```latex
% TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
We compare RF and QRF directly with the same underlying architecture.
```

### V26 — Note on "logD" in the header of tab:uncertainty_rise
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: column header "logD" -> "LogD" (chemistry guide item 11).
- why: A term change in her table with no note.
OLD:
```latex
\caption[Rise in mean predicted uncertainty under Gaussian noise.]
```
NEW:
```latex
% TODO: manual review -- edited your text: column header "logD" -> "LogD" (chemistry guide item 11).
\caption[Rise in mean predicted uncertainty under Gaussian noise.]
```

### V27 — Note on the comma splice (variance-head BNN paragraph)
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: the comma after "(Figure~\ref{fig:decomposition})" became a semicolon, because two sentences were joined by a comma.
- why: Punctuation that restructures a sentence, with no note.
OLD:
```latex
The full-BNNs with a variance head behave as expected
```
NEW:
```latex
% TODO: manual review -- edited your text: the comma after "(Figure~\ref{fig:decomposition})" became a semicolon, because two sentences were joined by a comma.
The full-BNNs with a variance head behave as expected
```

### V28 — Note on the noise-condition decomposition paragraph
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: the comma after "the pattern found in robustness" became a semicolon (comma splice); "group-shifted" -> "grouped-shifted noise", the condition's name elsewhere.
- why: Punctuation and a term change with no note.
OLD:
```latex
The noise conditions follow the pattern found in robustness;
```
NEW:
```latex
% TODO: manual review -- edited your text: the comma after "the pattern found in robustness" became a semicolon (comma splice); "group-shifted" -> "grouped-shifted noise", the condition's name elsewhere.
The noise conditions follow the pattern found in robustness;
```

### V29 — Note on "logD" in the aleatoric-against-injected caption
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
- why: A term change in her caption with no note.
OLD:
```latex
\caption[Aleatoric uncertainty against the added noise.]
```
NEW:
```latex
% TODO: manual review -- edited your text: "logD" -> "LogD" (chemistry guide item 11).
\caption[Aleatoric uncertainty against the added noise.]
```

### V30 — Note on the comma splice in the Conclusions
- kind: author-text (note only)
- todo_line: % TODO: manual review -- edited your text: the comma after "do not make much of an impact" became a semicolon, because two sentences were joined by a comma.
- why: Punctuation that restructures a sentence, with no note.
OLD:
```latex
Grouped-shifted and censored noise, the two conditions in which the error is systematic
```
NEW:
```latex
% TODO: manual review -- edited your text: the comma after "do not make much of an impact" became a semicolon, because two sentences were joined by a comma.
Grouped-shifted and censored noise, the two conditions in which the error is systematic
```

### V31 — tab:condition_choice, censoring row: the evidence cell understates what was run (AI table)
- kind: ai-text
- todo_line: (existing "% TODO: review -- table brought in ..." stays)
- why: Methods (line 755) says \aucnorm{} under censoring was also computed on the three assay datasets for the same five pairs. The chapter reports it only for QM9 (Table tab:censoring). "The HOMO--LUMO gap only" reads as if censoring was never run elsewhere.
OLD:
```latex
& The HOMO--LUMO gap only, on the five model and representation pairs run under censoring \\
```
NEW:
```latex
& Reported for the HOMO--LUMO gap only (Table~\ref{tab:censoring}), on the five model and representation pairs run under censoring \\
```

### V32 — tab:condition_choice caption: noun-phrase opening and metric with direction (AI caption)
- kind: ai-text
- todo_line: (existing "% TODO: review -- table brought in ..." stays)
- why: The caption opens with a question word, "Which model ...", and does not name the metric the rows rank by. Captions open with a noun phrase and state the metric and its direction.
OLD:
```latex
{Which model to prefer under each noise condition. One row is one noise condition. A dash means no model is named.}
```
NEW:
```latex
{Models to prefer and to avoid under each noise condition, judged by robustness (\aucnorm{}, higher is more robust). One row is one noise condition. A dash means no model is named.}
```

## For the author

- Which representations does the Rust part compute? Your QM9 pipeline paragraph says Python computes "several" and Rust "the remaining". V5 removes the AI sentence that said Python computes all six. Name them if you want the split stated.
- The Tanimoto-kernel GP (on ECFP4 alone) is in neither list of "thirteen base models" or "six variant models". The Methods notes comment says 20 configurations and seven variant models; the fig:transfer caption says 19 models. Say whether it is a seventh variant or was dropped.
- VBLL (V3): the replacement sentence follows the chapter's own table and equation. Your TODO(cite) asks for a check against the implementation, which is not in this repo.
- Caco-2 endpoint: the text now says "efflux ratio" (E3/E4 of the earlier pass). Confirm Chapter 3 used the log10 efflux ratio, as Chapter 5 does. Additional file app:datasets:ch3 may say.
- Censoring direction (chemistry guide item 13): the chapter clips the top of each label range. For p$K_i$, real censoring is mostly at the weak (low) end. One sentence saying which end is clipped and why would need your reason.
- "Scaffold family" (used throughout the chapter, defined in Chapter 2) against "scaffold group" elsewhere (chemistry guide item 14). One term for the whole thesis is your call.
- The "Exceeds spread ... of 6" column in tab:condition_cost is a tally (gap 33). Keep it, or replace it with per-replicate differences; tables are not changed in this pass.
- Statistical tests (gap 30) stay as decided. "Made no significant change" in the counterparts section rests on the QM9 Wilcoxon test, so it was left.
- Your sentences over 45 words, not split because splitting would be a style change to your text (gap 27): the two-architecture sentence in Models (50 words), "Up to a noise level of 0.3 ..." (67), the outlier/grouped-wider sentence in Per-sample uncertainty (51), "On QM9 with half the labels clipped ..." (49), the first Conclusions sentence after the semicolon fix (48), the GP/NGBoost/SVM summary sentence (46), and "When training on noisy experimental data ..." (51).
- The fig:aleatoric_injected caption has six sentences (the guide says 2-4). Shortening it would mean cutting your caption text.
- The forests paragraph's note asks whether "most reliably robust ... across all datasets" still holds, since QRF ranks first to fourth on Caco-2 and third to eighth on hERG $K_i$. This needs your judgment once the fixed Chapter 3 results land.
- "TODO: Ch3 update pending" at the chapter head and the two model-list TODOs in Noise conditions and levels are still open. They wait on the final result files.

## Applied

- V1: skipped, the title note was already in thesis.tex.
- V2: applied (note above her Introduction paragraph).
- V3: applied (note under her TODO(cite); VBLL sentence replaced; checked against table rows "one per fit" and the Uncertainty estimation paragraph).
- V4: applied (variant-model sentence split in two).
- V5: applied (sentence "computed in Python" removed; Rust paragraph at QM9 pipeline confirmed).
- V6: applied (memory-mapping sentences rewritten).
- V7: applied (per-molecule record sentences rewritten).
- V8: applied (note, Bemis--Murcko).
- V9: applied (note, "who" -> "that").
- V10: applied (note on variance-shares caption).
- V11: applied (note on grouped-shifted paragraph).
- V12: applied (note, "falls" -> "fell").
- V13: applied (note; 0.44 mean and 0.54 standard deviation, both already in Chapter 4).
- V14: applied (caption opens with "Robustness"; note extended).
- V15: applied (caption opens with "Robustness"; note extended).
- V16: applied (caption opens with "Robustness"; note extended).
- V17: applied (caption opens with "Robustness"; note extended).
- V18: applied (existing note extended).
- V19: applied (note, hERG $K_i$).
- V20: applied (note, LogD three times).
- V21: applied (note extended; RF LogD rank sentence matches the earlier "first on LogD at five of six").
- V22: applied (note, LogD).
- V23: applied (note, LogD).
- V24: applied (note, LogD twice).
- V25: applied (note, LogD).
- V26: applied (note on tab:uncertainty_rise header).
- V27: applied (note, semicolon).
- V28: applied (note, semicolon and grouped-shifted).
- V29: applied (note on aleatoric caption).
- V30: applied (note, semicolon in Conclusions).
- V31: applied (censoring row says "Reported for the HOMO--LUMO gap only").
- V32: applied (tab:condition_choice caption opens with a noun phrase and names the metric).
