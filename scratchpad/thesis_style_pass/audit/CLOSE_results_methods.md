# Closing two benchmark rows: Results comparisons and Methods connectives (2026-10-10)

File: `thesis.tex`. Only paragraphs with a plain `% TODO: manual review` marker were edited. No number, table, caption, \ref, \label, heading or TODO line was changed.

Rows addressed:
- **Results, comparison of results (AI 30.1%, references 18.1-23.6%).** Where two sentences compared cell by cell, they were merged into one pattern sentence with the table pointer. Every dropped dataset name is recoverable from the table cited in the same sentence; each was checked against the table before editing.
- **Methods, distinct connectives / connectives used (AI 0.10, references 0.15-0.28).** Before: 231 connectives of 24 kinds in AI methods text, 55 of them "so" and 23 "also". Repeated "so" and "also" were removed or replaced, and some were replaced with kinds absent before (thus, hence, consequently, as a result, likewise). Ordinal "first/second/third" were left alone. The measure was not re-run (that needs `segment_full_author.py`, which writes shared files).

Total edits: 48 (4 Results, 44 Methods).

## Results (comparison of results)

| First eight words | Chapter | Old -> new |
|---|---|---|
| On the antibody datasets and Buchwald--Hartwig, the | 6 | "With the setting picked again for each fold, it wins only on EGFR binding, ΔΔG and SMAC colloidal stability (Table). It loses on nanobody thermostability, AC-SINS self-association, CHO polyreactivity, expression titer and Buchwald--Hartwig." -> "With the setting picked again for each fold, it wins only on EGFR binding, ΔΔG and SMAC colloidal stability, and loses on the other four and on Buchwald--Hartwig (Table)." Checked: the table has seven antibody rows plus Buchwald--Hartwig. |
| With the setting picked again, the hybrid retrieves | 6 | "Clustering is mixed (Table). On EGFR binding and SMAC colloidal stability the hybrid's clusters are less coherent at 3, 5 and 10 clusters, although on SMAC colloidal stability its retrieval still wins. On nanobody thermostability and CHO polyreactivity the hybrid's clusters are more coherent at every count." -> "Clustering is mixed: at every cluster count the hybrid's clusters are more coherent on nanobody thermostability and CHO polyreactivity, and less coherent on EGFR binding and SMAC colloidal stability (Table)." All 12 cells checked. The SMAC retrieval clause was dropped because the paragraph's first sentence already says that. |
| A molecule's distance to its nearest training molecules | 6 | "The distance hybrid tracks error better for both models on QM9, Caco-2, hERG K_i and LogD. [...] The hybrid tracks error worse for both models on hERG blocking." -> "The distance hybrid tracks error better for both models on QM9, Caco-2, hERG K_i and LogD, and worse for both on hERG blocking." The sentence about the largest change now follows it. |
| On the small-molecule regression datasets the hybrid also | 5 | "On hERG K_i the hybrid is ahead in both halves, by less on the rare half. On hERG blocking it is ahead on the rare half with LightGBM only." -> "On hERG K_i the hybrid is ahead in both halves, by less on the rare half, and on hERG blocking it is ahead on the rare half with LightGBM only." |

## Methods (distinct connectives)

| First eight words | Chapter | Old -> new |
|---|---|---|
| Before the full runs, we ran preliminary design | 5 | "..., so the rescaling is one fixed rule per metric." -> "... . Hence the rescaling is one fixed rule per metric." (also splits a 33-word sentence) |
| Some feature sets are too small or too | 5 | "They cannot contribute many features, so including them would distort the budget split." -> "Including them in the budget split would distort it, as they cannot contribute many features." |
| Blocks of the top k ∈ {0, 5, | 5 | "..., so keeping no extras is the default." -> "..., which makes keeping no extras the default." |
| In uncapped mode, extras are appended outside the | 5 | "..., so the width stays at most B." -> "..., and thus the width stays at most B."; "..., so their comparison mixes ..." -> "..., and a comparison between them mixes ..." |
| The residual ranking has four limitations. ρ captures | 5 | "It ignores redundancy, so two ..." -> "It ignores redundancy: two ..."; "..., so a 2,048-feature donor gets more chances" -> "..., which gives a 2,048-feature donor more chances" |
| Searching every value of Table in every fold | 5 | "We also kept the budgets ..." -> "Likewise, we kept the budgets ..." |
| The search also has an adaptive pruning option, | 5 | "The search also has an adaptive pruning option, which scores ..." -> "The search has an adaptive pruning option as well. It scores ..." (splits a 33-word sentence) |
| The hybrid is compared with the best single | 5 | "..., so it can differ between folds." -> "..., and it can differ between folds."; "..., so the comparator is often many times wider." -> "..., and the comparator is consequently often many times wider." |
| The reaction dataset is Buchwald--Hartwig C--N couplings from | 5 | "grouped by additive, so each held-out fold holds" -> "grouped by additive: each held-out fold holds" |
| In the Koenig deep mutational scan of the | 5 | "giving 225 groups, so all mutations at one position fall on the same side of a split and a model cannot look up" -> "giving 225 groups. With all mutations at one position on the same side of a split, a model cannot look up" (splits a 37-word sentence) |
| The two AbAgym deep mutational scans (EGFR and | 5 | "giving 30 groups, so no complex is on both sides" -> "giving 30 groups, and hence no complex is on both sides" |
| The two AbAgym deep mutational scans (EGFR and | 5 | "as the groups, so their outer folds are the competition's folds." -> "as the groups, which makes the outer folds the competition's own." |
| We tested whether tuning helps on one outer | 5 | "..., so the held-out fold never reached it." -> "..., which kept the held-out fold out of the tuning." |
| Summed over its 5 outer folds, the search | 5 | "..., so a fold resumed from a stopped run is undercounted." -> "..., which undercounts a fold resumed from a stopped run." |
| The hybrid is the best of many configurations, | 5 | "..., so its gain may only reward a large search." -> "...; hence its gain may only reward a large search." |
| As a second reference beside the best single | 5 | "..., so neither side's width is chosen on the held-out samples." -> "...; in this way, neither side's width is chosen on the held-out samples." |
| Datasets above 2,000 samples are first reduced to | 5 | "We also ran three other splits (...)" -> "Three other splits were run as well (...)" |
| A second check repeats each outer fold's search | 5 | "..., but each search also picks its own budget, so the two hybrids ..." -> "... . Each search, however, picks its own budget, so the two hybrids ..." (splits a 34-word sentence) |
| On regression datasets, every method receives B features | 5 | "..., so it has no result on any small-molecule dataset" -> "... . It consequently has no result on any small-molecule dataset"; "..., so only the features differ." -> "..., and only the features differ." |
| One search per outer fold is run for | 5 | "..., so the number of features can differ between settings." -> "..., and the number of features can thus differ between settings." |
| Importance is computed on the training samples by | 5 | "Sample weighting needs a model that accepts per-sample weights, so it is not run for the Gaussian process, and it is not run on multiclass tasks." -> "Sample weighting is not run for the Gaussian process, which takes no per-sample weights, nor on multiclass tasks." |
| Here p_f is the fraction of folds that | 5 | "Each fold picks its own budget, so the sets compared can differ in size, and the index allows for this through k̄." -> "Each fold picks its own budget, and the index allows for the resulting difference in set size through k̄." |
| Two further studies come from a collaboration with | 5 | "..., so the search draws on eight." -> "..., leaving eight for the search." |
| Our addition is that feature selection, as well | 5 | "..., so all features in the hybrid are distinct." -> "..., which keeps all features in the hybrid distinct." |
| All four separate-site experiments run on Caco-2 and | 5 | "..., so the even split can build fewer features ..." -> "... . The even split can, as a result, build fewer features ..." (splits a 31-word sentence) |
| In the first of three further experiments, labels | 5 | "It is also compared with one model ..." -> "It is compared as well with one model ..." |
| In the second experiment, each of five sites | 5 | "It also asks whether the clusters agree ..." -> "The same experiment asks whether the clusters agree ..." ("It" was ambiguous after the previous sentence) |
| Datasets, folds (grouped by scaffold family for small | 6 | "held-out values are also cut to the training range, so a single extreme molecule cannot dominate a distance." -> "held-out values are cut to the training range as well, which stops a single extreme molecule from dominating a distance." |
| The score uses the Manhattan distance of Chapter | 6 | "..., so a representation's weight in the distance, ..., is roughly ..." -> "... . A representation's weight in the distance, ..., is thus roughly ..." (splits a 33-word sentence) |
| We found no published source for the rank | 6 | "..., so we sweep both." -> "...; for this reason we sweep both." |
| We predict each held-out molecule's label from its | 6 | "The first was fixed before the run, so nothing in it is picked ..." -> "The first was fixed before the run, and nothing in it is picked ..." |
| Each dataset has two downstream models. In each | 7 | "..., so the hybrid's features move with the damage." -> "...; in this way the hybrid's features move with the damage." |
| We damage a regression dataset in six ways | 7 | "..., so the clip imitates the rarer case." -> "...; the clip imitates the rarer case." |
| A classification dataset has its labels flipped instead, | 7 | "..., so on a two-class dataset five of the six are distinct." -> "..., which leaves five distinct rules on a two-class dataset." |
| Only the training labels are damaged. Unlike Chapter | 7 | "..., so every score in this chapter is against the delivered labels." -> "...: every score in this chapter is against the delivered labels." |
| Three settings involve no injection: the labels as | 7 | "Three experiments also use damage of their own." -> "Three experiments add damage of their own." |
| The gap over neighbor spread is the buddy | 7 | "..., so a molecule among neighbors that disagree ..., scores lower." -> "... . A molecule among neighbors that disagree ..., scores lower as a result." (splits a 32-word sentence) |
| The main cleanup experiment takes the out-of-fold forest | 7 | "It also grades the cliff-aware gap as a detector beside it." -> "The cliff-aware gap is graded as a detector beside it." |
| Four experiments weight molecules by a neighbor or | 7 | "..., so the most suspect molecule counts a quarter ..." -> "..., under which the most suspect molecule counts a quarter ..." |
| The hybrid's features are chosen against the labels, | 7 | "..., so the labels can be cleaned before ..." -> "..., which means the labels can be cleaned before ..." |
| The hybrid's features are chosen against the labels, | 7 | "..., so its two cleaned-search arms do not exist." -> "... . Its two cleaned-search arms consequently do not exist." |
| On labels as delivered, nothing says which molecules | 7 | "..., so a gain can come from removing hard molecules ..." -> "... . A gain can come from removing hard molecules ..." |
| A scaffold family is also a known flag. | 7 | "..., so a family's mean out-of-fold residual estimates that offset, ..." -> "... . A family's mean out-of-fold residual estimates that offset, ..." |
| A source is a known flag when each | 7 | "..., so they are bounds and not methods." -> "...; hence they are bounds and not methods." |

## Left alone on purpose

- Results paragraphs where shortening the comparison would drop a bold table cell's mention, or a fold-level fact that no table shows. Examples: the search's split against an even split (Nb-Tm DNN at 50 features), the LogD actions under 5% outliers (Ch7, linked to N7-04), and the new-embedding paragraph.
- The regularization paragraph in Chapter 5. Its NOTE says EGFR binding, the anti-VEGF expression scan and antibody humanness were named on purpose.
- Ch7 detection-family paragraph: merging it would push a sentence over 30 words, or drop where ΔΔG is spelled out in this chapter.
