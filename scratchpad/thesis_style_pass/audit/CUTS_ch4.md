# Cuts, Chapter 4 (items 6, 12, 28, 29), 2026-10-10

## Item 6: Ch4 paragraph on distances used later -- Cut
- Cut from thesis.tex, Methods, end of "Distance metrics for continuous molecular descriptors": the paragraph
  "The later chapters use a different set, chosen after that study: ... because it breaks the triangle inequality."
  and its "% TODO: manual review" line. It was AI text (TODO directly above).
- No \label in it. No replacement sentence.
- Open, not edited (Chapter 6 belongs to another agent): the cut paragraph held the only Chapter 4 formula for
  Minkowski distance with p = 0.5. The Chapter 6 nearest-neighbor settings table row
  "Distance $d$ & Manhattan; cosine; Minkowski with $p = 0.5$ & Chapter~\ref{chap:flaws} & Reused" still names
  Chapter 4 as the source of that distance. Chapter 4 now defines Euclidean, Manhattan, cosine and Mahalanobis only.

## Item 12: Ch4 early greedy hybrid -- Cut
- Cut the paragraph "The study also found neighbors in an early hybrid of 100 PDV and MHG-GNN features, built by
  greedy forward selection ... with 44.4\% and 27.8\% of configurations positive." and its "% TODO: manual review" line
  (end of "Noise detection by neighbors"). AI text.
- One clause removed from the paragraph that pointed to it ("The exploratory study also used neighbors to clean labels..."):
  "; those built on an early hybrid are described below" -> deleted, sentence now ends "...built on a single representation."
  That paragraph is marked "-- fixed:" (her text); the clause was a pointer to the cut paragraph and would have dangled.
- No \label. No replacement sentence.

## Item 28: Tables 4.1, 4.2 (tab:constant_features, tab:constant_features_b) -- AF
- Both tables cut from thesis.tex ("Constant and near-constant features") and pasted unchanged into
  thesis_rework/stage1_takeaways_bullets/additional_files_body.tex, at the end of section app:add:flaws, under
  \addfile{Constant and near-constant features per representation} \label{app:add:constant_features}.
  Labels kept: tab:constant_features, tab:constant_features_b. Captions unchanged.
- Figure 4.1 (fig:constant_features) stays in the thesis.
- References repointed (her sentences; only the pointer changed):
  - "...those with counts in Table~\ref{tab:constant_features_b}." -> "Additional file~\ref{app:add:constant_features}."
  - "(Tables~\ref{tab:constant_features} and~\ref{tab:constant_features_b}, Figure~\ref{fig:constant_features})"
    -> "(Additional file~\ref{app:add:constant_features}, Figure~\ref{fig:constant_features})"
  - "(ChemBERTa-2 in Table~\ref{tab:constant_features}, called ChemBERTa ...)" -> "Additional file~\ref{app:add:constant_features}"
- No other \ref to either label in root thesis.tex. The caption of part 2 refers to part 1; both now in the AF.
- No replacement sentence: her paragraph above the figure already says what the tables show and now points to the AF.

## Item 29: Table 4.3 (tab:flaws:ties) -- Merge identical rows
- hERG K_i: ECFP4, Atom pairs, PDV, Mordred, SMI-TED (all 1,412 / 56 (4.0%) / 56 / 0.041 / 0.998 / 1.14) -> one row,
  representation cell \makecell[l]{ECFP4, Atom pairs, PDV,\\Mordred, SMI-TED}.
- LogD: Mordred, SMI-TED (5,028 / 633 (12.6%) / 633 / 0.043 / 0.999 / 1.00) -> one row.
- Caco-2: Atom pairs, PDV, SMI-TED (2,158 / 177 (8.2%) / 177 / 0.027 / 0.996 / 0.74) -> one row.
  ECFP4 (179 tied) and Avalon (181 tied) differ, so not merged.
- Numbers copied unchanged. 22 rows -> 16 rows.
- Caption: added "; representations with identical entries share a row" after "one row per representation that ties
  any molecules", because that phrase would otherwise be wrong. Marked "% TODO: manual review" above the caption.
- No label moved; references unchanged.
