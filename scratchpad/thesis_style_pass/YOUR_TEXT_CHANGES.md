# Changes to your text, 2026-10-10

Compared against `thesis.tex.before_style_apply_2026-10-10` and `main.tex.before_style_apply_2026-10-10`.
"Your text" means every paragraph, caption, table or list with no `% TODO: manual review` line above it in the backup. Titles are left out: the new titles stay, each with its "title was" line.

## 1. Kept changes to your text

Lines marked [note] carry one `% TODO: manual review -- fixed: ...` line above the paragraph in thesis.tex. The others are spelling, a missing or doubled word, or the same form of a name (LogD, $K_i$, p$K_i$, Bemis--Murcko); they have no comment.

- Ch 1, "QSAR models are of course, sensitive": [note] "disagree by" to standard deviation; Kramer2012 0.54 is an SD; also "pK$_i$" to "p$K_i$"
- Ch 1, "This thesis addresses the challenge of": "cheminformatcs" to "cheminformatics"
- Ch 1, "Chapter~\ref{chap:background} introduces molecular representations, model families,": "artifical" to "artificial"
- Ch 1, "Chapter~\ref{chap:robustness} looks into what makes a": [note] broken last sentence; "variational head" models named as in Chapter 3
- Ch 1, "Chapter~\ref{chap:flaws} takes a turn to look": [note] "any" to "consistently"; Chapter 4 has one neighbor-based configuration best
- Ch 1, "Chapter~\ref{chap:method} presents the adaptive hybrid representation.": "customises" to "customizes"; "genereate" to "generate"
- Ch 1, "The Oxford Protein Informatics Group (OPIG)": [note] "Chapters 5 and 6" to "5 to 7"; Chapter 7 uses antibody datasets
- Ch 1, "The adaptive hybrid representation, its software": [note] chapter range ends at Chapter 7, not 6
- Ch 2, "Another common representation is the molecular": [note] ECFPs leave chirality out by default, not "active chirality flags"
- Ch 2, "In the standard molecular graph, atoms": [note] 2D graphs do not hold atomic coordinates or bond angles
- Ch 2, "Neural networks (NNs), a very popular": "a" to "an"
- Ch 2, "We employed Gaussian process (GP) models": [note] "Standardized kernels" to "Standard kernels" (standardized means rescaled)
- Ch 2, "Studies involving noise, and specifically the": "Anderson-Darling" to "Anderson--Darling"
- Ch 2, "Our objective is to evaluate the": [note] validation labels do get noise (next sentence), so only test data stay clean
- Ch 2, "For a given label, every condition": "Here" to "where"
- Ch 2, caption "Terms for methods that": "neighbours'" to "neighbors'"
- Ch 3, "While there have been many investigations": "direction" to "direct"; "responds" to "respond"; "remains" to "remain"
- Ch 3, "We looked at six representations paired": "variants" to "variant"
- Ch 3, "To get these networks producing uncertainty": [note] VBLL has one aleatoric variance per fit, not epistemic only
- Ch 3, ". Each of these subsets is": [note] stray leading full stop and missing "by a" in a broken sentence
- Ch 3, "The other models follow the same": added missing "of"; added missing "in"; "are" to "is"
- Ch 3, "No held-out label carries injected noise,": "Murcko" to "Bemis--Murcko"
- Ch 3, "To compare the relative effect of": "its" to "their"; "counterpart" to "counterparts"
- Ch 3, "To assess the interaction between uncertainty": "rise" to "rises"; "who" to "that"
- Ch 3, "We compare the variance decomposition on": "noises" to "noise"
- Ch 3, caption "Variance shares of": "logD," to "LogD,"
- Ch 3, "The variance decomposition tells us what": "model's" to "models'"
- Ch 3, "NGBoost is a standout case in": "group-shifted" to "grouped-shifted"
- Ch 3, "Many studies involving experimental label noise": "to" to "into"
- Ch 3, "We compare Gaussian noise on the": doubled "from" removed; "relative" to "relatively"
- Ch 3, "While alternative shapes of noise may": stray "of" removed ("the \aucnorm{} of across")
- Ch 3, "So far, we have only discussed": "pK$_i$" to "p$K_i$"
- Ch 3, table tab:assay_shape (rows): "logD" to "LogD" (x7); "K$_i$" to "$K_i$" (x7)
- Ch 3, "Across all datasets, models struggle to": [note] on hERG K_i RF is first or second with four of six representations
- Ch 3, "While the results are not identical": "logD," to "LogD," (x2); "logD" to "LogD" (x2)
- Ch 3, "We consider the impact of label": [note] on Caco-2 only RF ranks third to seventh; QRF ranks first to fourth
- Ch 3, "Other models perform conditionally well, and": "logD" to "LogD"
- Ch 3, "The feed-forward networks are the most": "logD," to "LogD,"
- Ch 3, "While it is clear that QSAR": "logD," to "LogD," (x2)
- Ch 3, caption "Predictive performance against": "logD;" to "LogD;"
- Ch 3, caption "Robustness on the": "logD," to "LogD,"
- Ch 3, "The variance decomposition of predictive performance": added missing "in"; added missing "of"
- Ch 3, "We compare RF and QRF directly": "logD" to "LogD"
- Ch 3, "The Bayesian transformations that we used": added missing "ones"
- Ch 3, table tab:uncertainty_rise (header): "logD" to "LogD"
- Ch 3, "The noise conditions follow the pattern": "group-shifted" to "grouped-shifted"
- Ch 3, caption "Aleatoric uncertainty against": "logD," to "LogD,"
- Ch 3, "Since Gaussian, Laplace, Student-$t$ and grouped-shifted": "logD" to "LogD"; "logD." to "LogD."
- Ch 4, "The previous chapters have been focused": [note] abbreviation AC defined at first use
- Ch 4, "To investigate similarity estimation, we first": [note] source of the three Biogen datasets cited at first use
- Ch 4, caption "Two hERG $K_i$": "(R" to "(\textit{R}"; "S" to "\textit{S}"
- Ch 4, caption "Absolute p$K_i$ difference": [note] panel (a) shows similarity, not distance
- Ch 4, "The neighbor-based noise scores, sample removal,": [note] out-of-date pointer; these definitions are now in Chapter 7
- Ch 4, "The exploratory study also used neighbors": [note] out-of-date chapter reference; sample-removal study is now in Chapter 7
- Ch 5, "So far, Chapter~\ref{chap:flaws} explained how the": [note] sentence fragment joined to the next sentence with a comma
- Ch 5, "The most straightforward method of combining": "limitation" to "limited"; "conceptually-straightforward," to "conceptually straightforward,"
- Ch 5, "Beyond simply combining features, there are": [note, your late-fusion TODO] ensembling named as late fusion; "molecule" added where the sentence stopped
- Ch 5, "When you want to filter a": added missing "features"
- Ch 5, "Feature selection keeps a subset of": [note] mRMR is mutual information with the label minus redundancy
- Ch 5, "Formally, an adaptive hybrid representation is": [note] "fixed number of" removed; the hybrid can have fewer features than its budget
- Ch 5, "The hybrid search is a generalized": removed "set of"; "blueprint mappings" to "blueprints mapping"; "Chapter~\ref{chap:similarity}" to "(Chapter~\ref{chap:similarity})"
- Ch 5, "To run a hybrid search, we": [note] "input hybrids" to "input representations"; the inputs are base representations
- Ch 5, "Six settings in total are searched,": [note] table description updated; that table no longer has speed tiers
- Ch 5, "We select a set of up": doubled "in" removed ("within in the pool")
- Ch 5, "The four fixed-share methods compute importance": [note] "importance shares" to "budget shares"; shares come from standalone scores
- Ch 5, "The first method of feature allocation": [note] FcRn (neonatal Fc receptor) spelled out; not defined elsewhere
- Ch 5, "The most useful features for an": [note] twelve types are what the code supports; the thesis uses six
- Ch 5, "Antibody representations (Table~\ref{tab:reps_antibodies}) start from the": [note] CDR spelled out at first use
- Ch 5, caption "Representations used for antibodies.": [note] LM (language model) defined; it was not defined
- Ch 7, caption "Correcting clipped training labels.": one sentence added, "Dataset short names as in Table~\ref{tab:dataset_short_names}." The figure was redrawn today with short names (figures session); its note says so.
- Ch 7, table tab:noise:detectors, Source column: the gap-over-neighbor-spread row changed from "New" to Baserud2020 and Kordos2012, and the cliff-aware row gained "related to Chawla2006". This came from the literature session, not the style pass. Not changed here.
- Chapters 4 to 7: 43 `\paragraph{...}` headings became `\subsubsection{...}` (trailing full stop dropped). The text under them is unchanged.
- Front matter: Statement on the Use of Generative AI, 8 paragraphs shortened under your TODO "review this and make it way less complex", one note above each. List of Abbreviations rebuilt against the text (its own note above it). Abstract was AI-written; it moved to straight after the Table of Contents.

## 2. Suggestions not applied

These were made today and then reverted to your words. Apply them yourself if you want them.

- "scaffold groups" to "scaffold families", Ch 4 (2): "Distance and label difference" and "Agreement on neighbors".
- "Higher $R^2$ is better" or "Higher is better on both axes" added to a caption, Ch 3 (3): the $R^2$ curves on QM9; clean $R^2$ against \aucnorm{}; the noise-conditions figure.
- "(higher is more robust)" after \aucnorm{} in a caption, Ch 3 (4): the grid of 13 base models; the deep-model conditions figure; the censoring table; the assay-shape table (which also added "negative is a loss").
- "Rank 1 is ..." added to a caption, Ch 3 (2): the rank-against-level figure ("Rank 1 is the highest $R^2$"); the transfer figure ("Rank 1 is the most robust").
- "One point is one base model." added to a caption, Ch 3 (1): clean $R^2$ against \aucnorm{}.
- "higher is better" added to a caption, Ch 4 (1): all-pairs correlation, five-nearest-neighbor prediction and neighbor agreement.
- Not reverted, because the paragraph was AI-marked in the backup: Ch 5 "Here, the one-hot encoding is a base representation on three antibody datasets" now says "change in binding free energy ($\Delta\Delta G$)". The rest of the thesis says "binding free energy" for that dataset.

## 3. Counts per chapter

One row per chapter. "Your paragraphs" counts paragraphs, captions, tables and lists, not titles. "Unchanged" includes paragraphs where only a number from the results session changed, or a `\paragraph` heading was removed, or the paragraph moved.

| Chapter | Your paragraphs | Unchanged | Spelling or typography fixed | Real error fixed, with note | Under your own rewrite TODO |
|---|---|---|---|---|---|
| 1 | 19 | 7 | 3 | 5 | 4 |
| 2 | 53 | 45 | 4 | 4 | 0 |
| 3 | 103 | 71 | 28 | 4 | 0 |
| 4 | 60 | 54 | 1 | 5 | 0 |
| 5 | 80 | 65 | 4 | 10 | 1 (your late-fusion TODO) |
| 6 | 8 | 8 | 0 | 0 | 0 |
| 7 | 8 | 6 | 0 | 1 (caption, figures session) | 0; 1 table changed by the literature session |
| 8 | 1 | 1 | 0 | 0 | 0 |
| Front matter (main.tex) | 37 | 28 | 0 | 0 | 8 (AI statement); 1 abbreviation list rebuilt |

Ch 1 rewrite TODOs: "come back to this at the end", "definitely modify these", "waiting on results, may need another pass", "come back to all of this, it's all out of date".
