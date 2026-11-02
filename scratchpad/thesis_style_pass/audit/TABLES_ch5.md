# Chapter 5 tables built for the cut numbers (2026-10-10)

Written by Claude. thesis.tex, Chapter 5 (`chap:method`), results sections. Snapshot before the edits:
`thesis.tex.before_build_tables_2026-10-10`.

All 20 "% TODO: table needed" comments in Chapter 5 are resolved. 20 tables were built. Each has a
`% TODO: manual review` line and a `% SOURCE:` comment above it. No existing SOURCE comment, label,
`\includegraphics` or `\input` was changed. The PDF built twice with no errors and no undefined
references. The only overfull box in the new tables is 0.5pt (the feature-reduction long table).

## Which results folder

Each table reads the file that the paragraph's SOURCE comment names: `/Volumes/seagate/kirby_results_2026-10-09`.
Two exceptions:
- The hybrid column of the two hindsight tables comes from `figures_ch5_7/tab_hybrid_vs_best_single.csv`. That table is the existing Table 5.x, generated from the 10-10 folder.
- The per-fold budget table reads the 10-10 c02 files, as its paragraph's SOURCE comment does.

The 10-10 folder has newer files that would add rows the tables now lack. They were not used, because no
paragraph names them and the results chat owns new results:
- c11: adme_rlm, abagym_Ang2, ginkgo_hic.
- c22: adme_rlm.
- c23: qm9, abagym_Ang2, ginkgo_hic.
- c24: qm9, adme_mdr1, abagym_Ang2, ginkgo_hic.
- c08: adme_rlm, flab_binding_koenig.
- c07: qm9, with 3 and 2 folds only.
- c17: abagym_Ang2, abagym_VEGF.

## Conventions used in every table

- Dataset short names from Table `tab:dataset_short_names`, with that pointer in the caption. Full names made 11 of the 20 tables 1 to 4 cm too wide. Caco-2 and LogD are their own short names, so the site tables carry no pointer.
- "DNN" for the network, as the chapter text writes it. The figure script's existing table writes "NN".
- Scores have three decimals and Buchwald--Hartwig RMSE has one. Two compared values that print the same in a row keep one more decimal (DECIMALS_RULE.md, point 4).
- Missing values are empty cells, never placeholders.
- No fold counts, differences or ranges in cells.
- At most 4 bold cells per table, on the rows the prose names.
- One table is landscape (the evaluator table). One uses `\tabcolsep` 4pt instead of 6pt (the search-optimism table). No font was shrunk.

## The 20 tables

| Label | Rows x numeric columns | Comment(s) it serves | Numbers from |
|---|---|---|---|
| tab:method:hindsight_smallmol | 16 x 4 (+ representation name) | small-molecule hindsight + c22 | c01 score_mean; main-table CSV; c22 perm_share_by_rep |
| tab:method:hindsight_antibodies | 24 x 4 (+ name) | antibody hindsight + c22 | c01; main-table CSV; c22 perm_drop_by_rep |
| tab:method:ensembles | 34 x 3 | c11 ensembles | c11 hybrid_top1, ensemble_s1_auto, ensemble_s2_auto |
| tab:method:top_rep_even_split (longtable) | 144 x 3 | top representation alone / even split | c18 depth_1, natural, uniform_all |
| tab:method:scaffold_halves | 16 x 3 | c07 scaffold halves | c07 scaffold_frequency easy/hard |
| tab:method:search_optimism | 24 x 8 | c33 search score vs held out | c33 per_fold training_rows_score, heldout; summary picked_budget_per_fold |
| tab:method:one_fold_fewer | 30 x 5 | c24 one fold fewer + stability comment | c24 own_score, mean_cross_score, nogueira_phi, n_features_selected_per_fold |
| tab:method:feature_reduction (longtable) | 108 x 4 | c17 RFE / random / mRMR | c17 per_budget downstream_score |
| tab:method:budget_per_fold | 38 x 10 | per-fold budget and representations | c02 (10-10) hybrid1 budget, features_by_rep |
| tab:method:variance_shares | 16 x 4 | c06 comment, variance part | c09_formal_analysis anova.eta2 |
| tab:method:budget_by_capacity | 24 x 5 | c06 comment, SVM budget per C | c06 optimal_budget by_fold |
| tab:method:evaluator (landscape) | 24 x 6 (+ 2 evaluator names) | c23 evaluator | c23 A_matched, B_slight_off, C_strong_off |
| tab:method:importance_training | 38 x 5 | c19 | c19 per_space.hybrid scores |
| tab:method:conformal | 34 x 4 | c08 | c08 picp, mpiw per fold, mean |
| tab:method:opig_qsar | 6 x 2 | OPIG qsar | pre-cut text values (4 decimals, none ending in 5) |
| tab:method:shared_rows | 3 x 3 | shared rows on hERG Ki | opig_shared_vectors_remedy.json |
| tab:method:settings_transfer | 2 x 4 | LogD settings on hERG Ki | logd_to_herg_ki.json |
| tab:method:family_sites | 4 x 4 | fed03 | fed03 aggregated rmse.mean |
| tab:method:flipped_sites | 12 x 6 | fed01 | fed01 caco2.partial.jsonl, mean over seeds |
| tab:method:cluster_sites | 2 x 5 | fed04 | fed04 aggregate |

## Paragraphs re-pointed

All are marked `% TODO: manual review`, so they are Claude's paragraphs. Each now points to its table next to
the claim. Score pairs that now sit in a table were taken out of the prose:
- Caco-2 PDV 0.298/0.277.
- The ΔΔG built share 0.83 and "at most 0.21".
- The VEGF-E ensemble 0.785/0.735.
- hERG Ki at 250 features 0.599/0.564.
- LogD rare scaffolds 0.670/0.573.
- LogD SVM at 1,000 features 0.430/0.445.
- LogD variance 48.5%/1.4%. The prose now says "about half" and "almost none".
- EGFR evaluator 0.353/0.299.
- Colloidal stability 0.261/0.357.
- LogD conformal width 1.652/1.403.
- Factor Xa 0.737/0.661.
- Graph kernel 0.611/0.571.
- Settings transfer 0.589/0.567.
- fed03 LogD 0.438/0.460.
- fed01 0.5085/0.390.
- fed04 0.529/0.532.

Three more paragraphs gained a pointer only: the Buchwald--Hartwig paragraph and the budget-figure paragraph
(both to `tab:method:budget_per_fold`), and the blueprint-stability paragraph (to `tab:method:one_fold_fewer`).

## Claims changed because the table contradicted them

Each one has a dated `% NOTE` or `% TODO: CHECK` comment in thesis.tex.

1. **Antibody hindsight paragraph.** It said "On three antibody datasets the hybrid beats even the single representation best in hindsight." The table shows four. The fourth is the anti-VEGF binding scan, whose hybrid score is in the 10-10 main table and whose best representation is SaProt or ProtT5, not one of ours. The paragraph now says "four". The Conclusions still say three datasets and were not touched.
2. **Search-optimism paragraph.** "By up to 0.157 Spearman on colloidal stability with the DNN" was c33 `summary.process_gap`. In `tests/experiments/complexity/c33*.py`, process_gap is two held-out scores subtracted: the per-fold picks minus the best fixed budget. It is not the search's own score minus the held-out score, which is what the sentence says. The table gives the search's own score and the held-out score. On those, the largest gap is EGFR binding with the DNN (0.371 against 0.152), not colloidal stability. The number was removed.
3. **Stability paragraph.** The cut ranges were 0.27 to 0.50 and 0.08 to 0.27. c24 gives 0.0748 for EGFR binding with the SVM. The paragraph now states the pattern in words.
4. **fed01 paragraph.** It said "It never beats the same selection made with all molecules pooled." With the SVM at 20% flips, selection at each site scores 0.5098 MCC and the pooled selection 0.5073. The sentence now names that exception.

## Claims not fully backed by a table

- **"Ahead on every fold" and "on all five folds".** This applies to the ensembles paragraph (VEGF-E DNN, QM9 DNN, hERG Ki SVM), factor Xa DNN, the graph kernel with SVM, and settings transfer with SVM. Fold-by-fold wins are counts, so the tables do not show them. These claims rest on the SOURCE files only.
- **"Angiopoietin-2: AntiBERTy was never ranked first on any fold."** No table lists the search's per-fold picks. The main table's CSV has them.
- **Rows and cells with no result.** QM9, rat microsomal clearance and MDR1--MDCK efflux are missing from c33. QM9, rat microsomal clearance and MDR1--MDCK efflux are missing from c24. The c22 columns are empty for LogD (4 of 5 folds in a partial file), the anti-VEGF expression scan (1 fold) and rat microsomal clearance (no 10-09 file). Rat microsomal clearance has 3 folds in c06. hERG blocking and antibody humanness are absent from c17 (skipped as classification) and from the width columns of c08.
- **c18 has no SOURCE comment in thesis.tex.** Its folder was taken from the Chapter 8 SOURCE line, which names c18 in the 10-09 folder. A new SOURCE line was added above the table.
- **c06 variance shares.** The paragraph's SOURCE names c06, but the numbers are in c09_formal_analysis anova.eta2. The table's SOURCE says so.
- **Two derived columns are kept because the prose's claims rest on them.** One is the effective number of representations, 1/sum(share^2). The other is the share of the score drop. Both are computed from per-representation shuffling shares. The author's rule against derived quantities may want them replaced.
- **c33 budgets disagree within one file.** The per-fold hybrid1 budget and `summary.picked_budget_per_fold` differ on hERG blocking LightGBM fold 5 (12,224 against 2,000) and antibody humanness XGBoost fold 2 (1,000 against 500). The table shows the summary value. LogD DNN picked 8,044 on one fold, above the "up to 8,000" in the paragraph.
- **The scaffold paragraph says "a 2,000-molecule subset".** hERG Ki has 1,412 molecules in all. The table caption says "at most 2,000".

## Check

Independent check of Chapter 5, 2026-10-10, by Claude. Every number below was re-read from the result files in this session with `python3 -I`; scripts are in the checking session's scratchpad, not in the repo.

### 1. "table needed" comments

None remain in Chapter 5.

### 2. Table references

Every `\ref{tab:...}` in the chapter resolves to a `\label` in thesis.tex or `figures_ch5_7/`. No duplicate labels. A test build (pdflatex twice, into the checking session's scratchpad, not the shared build folder) ran with no errors and no undefined `tab:method:` references. Seven non-table references in the chapter (`app:add:*`, `app:datasets`, `app:datasets:ch3`) point to the Additional files; they were there before the table build and resolve through xr-hyper.

### 3. Numbers traced

Every cell of the 20 built tables was recomputed from its SOURCE file and compared after rounding to the printed decimals:
c01/c22/CSV (both hindsight tables, all cells incl. effective number and shares), c11 (ensembles), c18 (432 cells), c07, c33 (scores, training samples, per-fold budgets), c24, c17 (424 cells), c02 10-10 (564 cells), c09, c06, c23 (incl. evaluator names), c19, c08, fed01 (raw partial file), opig shared rows, settings transfer, fed03.
The OPIG activity-cliff table and fed04 were traced to the pre-cut SOURCE comments (no value ends in 5 at four decimals).

No cell disagreed with its source. Nine rows printed two compared values the same, which the decimals rule (point 4) forbids. Raw values now printed with the extra decimals:
- Ensembles, Nb-Tm SVM: 0.687104 and 0.687096 (equal to five decimals).
- Top representation / even split: RLM SVM 1,000 (0.48346, 0.48347); Hum XGBoost 100 (0.95623, 0.95621); Hum RF 50 (0.96235, 0.96239).
- Evaluator, Caco-2 SVM near and far: 0.28527, 0.28531.
- Importance in training: LogD DNN both / above median (0.41677, 0.41685); Hum XGBoost features scaled / above median (0.95649, 0.95650); Hum RF unweighted / features scaled (0.96115, 0.96116).
- Family sites, LogD SVM: 0.43823, 0.43822.
Two printed ties are real ties and were left: c08 SMAC SVM coverage 0.938 (raw values identical), and equal pairs in the new embeddings table (the hybrid took no features from the embedding).

### 4. Claims checked against the tables

Claims the tables contradicted, now fixed in thesis.tex (each has a dated `% NOTE (Chapter 5 check, 2026-10-10)` or `% CUT FROM TEXT` comment):
1. Antibody hindsight paragraph: "two to four and a half representations" had no source. The table's effective numbers run from 1.41 (Hum, XGBoost) to 5.34 (VEGF-B, SVM). Now "about one and a half to five". "With the SVM" added for AntiBERTy on angiopoietin-2 (with the GP the best in hindsight is SaProt).
2. Small-molecule hindsight paragraph: the table includes the Buchwald--Hartwig rows, where the hybrid loses. Added "On the Buchwald--Hartwig reactions it does not."
3. Ensembles paragraph: the same-model ensemble beats the hybrid with both models on the Buchwald--Hartwig reactions too. Added.
4. Top representation / even split paragraph: "On small molecules every allocation method appears among the winning hybrids" had no table. True in the main table's CSV; cut from the text and kept as a comment.
5. Scaffold paragraph: "Caco-2 and hERG K_i show the same" (larger gain on rare scaffolds) was false for hERG K_i with both models and Caco-2 with the SVM. "A 2,000-molecule subset": hERG K_i has 1,412. hERG blocking is behind on the rare half with XGBoost. Rewritten.
6. Search-optimism paragraph and caption: "budgets up to 8,000" while LogD DNN picked 8,044 and hERG blocking 7,640. c33 also searches 0.5 to 2 times the training samples; now said in both.
7. One-fold-fewer paragraph: "on the regression datasets ... 0.009 to 0.037 RMSE" excluded Buchwald--Hartwig without saying so. Now names hERG K_i, Caco-2 and LogD.
8. Feature-reduction paragraph: hERG K_i with the DNN loses at 2,000 features as well as 1,000; the Buchwald--Hartwig reactions lose to recursive elimination at most budgets. Both added.
9. Variance-shares caption said "held-out score". c09 `anova.observations` says the search's own score on training rows. Caption corrected.
10. Capacity paragraph: "the DNN takes the top of the grid at every width on LogD". At 64 units fold 1 picked 1,000. Now "on all folds but one".

Paragraphs with claims and no table, now backed by new tables:
11. Regularization paragraph ("Choosing fewer features does not replace tuning", c16): no SOURCE, no table. New Table `tab:method:regularization` (11 rows, c16 `fold_reselection_result`). It shows EGFR binding, the anti-VEGF expression scan and antibody humanness favour the tuned wider hybrid, which the paragraph did not say; the last sentence ("with a few hundred the budget does") was changed to match. The paragraph said "the SVM's setting", but Buchwald--Hartwig and titer use the random forest's tree depth; now "the model's".
12. New-embedding paragraph (c13): no SOURCE, no table, and a tally ("seven or eight of eight"). New longtable `tab:method:new_embeddings` (57 embedding rows over hERG K_i, Caco-2, LogD, BH, SMAC, Titer). The tally is replaced by the one exception (Uni-Mol2 with the DNN on LogD). The c13 random-control caveat (author, 2026-10-07) is now in the text and caption. Eight embedding display names (ChemFM, Uni-Mol2, AbCDR-ESMC, CurrAb, IgBert, ISM, MINT, p-IgGen) are not in `thesis_figure_style.py` REP_LABELS and were written by hand.
13. fed02 paragraph (sites each holding one representation): no table, three scores in one sentence. New Table `tab:method:site_reps` (4 rows); the three scores moved into it.

Checked and backed (no change): ensembles "most antibody datasets"; c18 "worse at every budget on most" and the Nb-Tm/EGFR sentences; budget-per-fold sentences; variance "most datasets"; evaluator sentences; c19 sentences; conformal sentences; OPIG, shared rows, settings transfer, fed03, fed01, fed04 sentences.

### 5. Table rules

- One number per cell: yes in all 23 tables. The "128--64 units" cells in the capacity table are layer widths, not ranges.
- Bold: at most 4 per table in all 23.
- No `\small`, `\footnotesize`, `\adjustbox`, `\resizebox` in any new table. No overfull box from the three new tables in the test build.
- Captions: all name what one row is and the metric with direction (in the caption or the table's group lines). Five run past 75 words (hindsight small molecules 80, search optimism 87, one fold fewer 91, flipped sites 84, new embeddings about 85); they explain columns, not caveats, and were not cut.

### 6. The author's paragraphs

Every paragraph changed by the table build, and by this check, has a `% TODO: manual review` marker. No author paragraph was edited.

### Left for the author

- Conclusions still say three antibody datasets beat the best in hindsight; the table shows four (the TODO: CHECK above the antibody paragraph).
- Derived columns kept: effective number of representations and share of the score drop (both hindsight tables). Her rule against derived quantities may want them out.
- "Angiopoietin-2: AntiBERTy was never ranked first on any fold" is checked against the main table's CSV (`best_single_rep_by_fold`) but no table shows it.
- "On all five folds" / "on every fold" claims (ensembles, factor Xa, shared rows, settings transfer) rest on the SOURCE files only.
- c33's per-fold budget disagrees with `summary.picked_budget_per_fold` on two folds (noted in its SOURCE).
- The PCA paragraph (no table, many scores in prose) was outside the table build and was not changed.
