# Chapter 7 tables built for the "table needed" comments (2026-10-10)

Written by Claude. Chapter 7 is `\chapter{Detecting and handling label noise}\label{chap:noise}` in `thesis.tex`.
Snapshot before this pass: `thesis.tex.before_build_tables_2026-10-10` (made by the orchestrator).

All 20 `% TODO: table needed` comments in Chapter 7 are gone. 20 tables now back them. Each table is an
inline `table` float with a `% SOURCE:` line and `% TODO: manual review` above it. No `SOURCE [N7-xx]`
anchor, existing `\label`, figure or `\input` was touched.

## Where the numbers come from

Every cell was read from the result files, not typed from text:

- Results root `/Volumes/seagate/kirby_results_2026-10-10`. Noise view `/Volumes/seagate/kirby_results_2026-10-09_noise_view`
  (n15 and n21 only, as `ch7_numbers.py` reads them).
- Read with the key paths and helper functions of `tests/analysis/ch7_numbers.py`, imported read-only.
- Scripts: `scratchpad/thesis_style_pass/audit/tables_ch7_scripts/` (`vals.py` reads the files into one JSON; `gen.py` writes the LaTeX).
- `ch7_numbers.py` run on the same folders: every number in the pre-cut text (`thesis.tex.before_number_cut_2026-10-10`)
  still matches the raw value, except the four listed under "Differences from the pre-cut text" below.
- Scores are three decimals, Buchwald--Hartwig RMSE one decimal. Rounding is from the raw value, so no
  number was rounded twice. Where two compared cells print the same, both print one more decimal
  (DECIMALS_RULE.md, tie rule): 11 places, e.g. hERG first single 0.8253 vs 0.8250.

## Tables

Rows x numeric columns. "Comment" is the N7 paragraph whose `table needed` comment the table replaces.

| Label | Rows x cols | Comment(s) | Source |
|---|---|---|---|
| tab:noise:res:herg_flip_actions | 10 x 4 | N7-03, N7-05 (hERG part) | n07 rows (actions), n23 json + noise_full rows (detectors, fold 1 draw 0) |
| tab:noise:res:delete_vs_random | 9 x 3 | N7-03, N7-04; also backs N7-11 | n24 json and rows |
| tab:noise:res:logd_outlier_actions | 11 x 2 | N7-04, N7-05 (LogD part) | n07 rows, n23 json |
| tab:noise:res:weighting | 9 x 2 | N7-07 | values re-read by ch7_numbers N7-07 (n13, n21, n35) |
| tab:noise:res:shuffled | 14 x 5 | N7-09, N7-30, N7-31 | n18 json |
| tab:noise:res:hergki_smoothing_folds | 3 x 4 | N7-10 | n24 json (noise + noise_full) |
| tab:noise:res:detect_residual | 26 x 4 | N7-13; also backs N7-14 | n07 units.jsonl |
| tab:noise:res:detect_families | 25 x 5 | N7-15 | n21 json (noise view) |
| tab:noise:res:detect_other | 28 x 5 | N7-17 | n23 json, n32 json or checkpoint |
| tab:noise:res:breakeven | 18 x 5 | N7-19 | n31 json summary |
| tab:noise:res:censoring_folds | 7 x 3 | N7-21 | noise_full n30 rows |
| tab:noise:res:censoring_moves | 32 x 5 | N7-22 | n30 json |
| tab:noise:res:censoring_moves_other | 16 x 5 | N7-22 | n30 json |
| tab:noise:res:censoring_lift | 16 x 4 | N7-24 | n30 json |
| tab:noise:res:family_offsets | 9 x 4 | N7-26 | n34 rows |
| tab:noise:res:herg_ic50 | 15 x 3 | N7-34 | n19 rows, n29 json |
| tab:noise:res:clean_before_search | 10 x 4 | N7-35, N7-36 | n20 json (final, _partial, noise_full _partial) |
| tab:noise:res:search_clean_vs_damaged | 20 x 5 | N7-37 | n15 json (noise view) |
| tab:noise:res:change_model | 16 x 6 | N7-38, N7-40 | n40 json |
| tab:noise:res:change_model_delivered | 16 x 4 | N7-39 | n40 json, level 0 |

That is 20 labels; `tab:noise:res:censoring_moves` and `_other` are one comment split in two because 48
rows do not fit a page, and the n40 table is split the same way (damaged labels / labels as delivered).

Bold: 0 to 4 cells per table, only on what the paragraph describes (censoring_folds has none).
No `\small`, no adjustbox, no `\IfFileExists`.

## Paragraphs re-pointed

All are Claude paragraphs (marked `% TODO: manual review`); none is the author's.

- N7-03, N7-04, N7-05, N7-07, N7-09, N7-10, N7-11, N7-13, N7-14, N7-15, N7-17, N7-19, N7-21, N7-22,
  N7-24, N7-26, N7-30, N7-31, N7-34, N7-35, N7-36, N7-37, N7-38, N7-39, N7-40: `(Table~\ref{...})` added next to the claim.
- No number was put back into the prose.

## Claims I changed because the table contradicts them

Each is a short wording fix in a Claude paragraph. Check them.

1. N7-13. Was "falls below chance on Buchwald--Hartwig and LogD, and is only a little above it on Caco-2".
   The residual is also below chance on QM9 clipped 25% (0.157 against 0.250), a little above on hERG $K_i$
   (0.264 against 0.244), and well above on SMAC (0.506 against 0.243). Now: "Censoring is the exception on
   small molecules and reactions ... below chance on Buchwald--Hartwig, LogD and QM9, ... a little above it
   on Caco-2 and hERG $K_i$. On SMAC colloidal stability it stays well above chance."
2. N7-19. "the residual's precision falls below the break-even" now says "for deletion", twice. Under the
   model-error shift on hERG $K_i$ (top 5%) the precision 0.195 is above the down-weight break-even 0.026;
   under the plain shift (top 10%) 0.262 is above the relabel break-even 0.105.
3. N7-21. "LogD folds 0 to 3" is now "folds 0 to 4". Fold 4 exists at 25% clipped (0.398 / 0.515 / 0.437).
4. N7-22. Was "On Buchwald--Hartwig only the random forest was hurt by clipping". LightGBM is also hurt on
   the second hybrid (19.8 to 20.3) and the first single (19.49 to 19.51). Now: "clipping hurt the random
   forest in every configuration and LightGBM in only some".
5. N7-24. Was "On LogD, hERG $K_i$ and Caco-2, every shuffle beats the molecule's own lift". True on LogD and
   Caco-2 in all four configurations; on hERG $K_i$ only on the second single representation (the one the
   pre-cut text quoted). Now says so. QM9 sentence now names the first hybrid.
6. N7-26. Was "The random-intercept model was best on Caco-2 and LogD". LogD draw 2 is new: family mean
   0.570, random intercept 0.589. Now: "best on every Caco-2 draw and on the first two LogD draws".

## Differences from the pre-cut text (newer raw values used)

- hERG flips, n24 forest residual, fold 2: pre-cut 0.8223 / 0.8381 / 0.8347; raw now 0.8193 / 0.8346 / 0.8279
  (no cleaning / delete / best random). The prose claim (deletion beats random on all three folds) still holds.
- QM9 network second hybrid, untreated, n21: the file holds two values (0.3568, 0.3618). The table uses
  0.362, the value the N7-07 check reads for the weighting comparison.
- n41 conformal flags (N7-18, no table comment): the most molecules flagged in a pass is now 10, the text
  says "a few dozen at most". Not changed; no table was asked for there.

## Claims left unbacked or partly backed, and why

- tab:noise:res:weighting, QM9 detector-weight rows: the unweighted cell is empty. The n13 QM9 file stores
  only the weighted score (track_b.denoised), no unweighted score.
- tab:noise:res:herg_flip_actions, fold 1: only the five detectors ran on fold 1 (n23); the action rows
  (n07) have no fold 1, so those cells are empty. Fold 1 is draw 0 only, as the prose says; draws 1 and 2
  of fold 1 also exist, and on them the cliff-aware gap does improve the single representation
  (0.8026 to 0.8126, 0.7949 to 0.8029). The prose "none of the five improved the single representation"
  holds for draw 0 only.
- tab:noise:res:detect_other: n32 has no main-level grading for Caco-2 outliers and shifts, hERG flips,
  or QM9; those cells are empty. cleanlab is missing for Caco-2 clipped and all of SMAC. Five rows come
  from unfinished n32 runs (marked a).
- tab:noise:res:breakeven: only Caco-2 and hERG $K_i$ are shown, the two datasets the paragraph discusses.
  The n31 file now also has summaries for Buchwald--Hartwig and $\Delta\Delta G$; not added.
- tab:noise:res:censoring_folds: Caco-2 fold 0 is left out; it is a separate run (Figure 9's).
- tab:noise:res:hergki_smoothing_folds: the second single representation's label-smoothing score is the
  best detector setting on each fold (as the pre-cut text used), named in the caption.
- N7-26, hERG $K_i$: "the plain family mean did as well" is mixed across draws (family mean better on draw 1,
  random intercept much better on draw 2). Not changed.
- Detection tables include QM9 rows (n07 now has QM9 at damaged levels). The N7-13 TODO says QM9 detection
  does not exist; that note is out of date.

## Build

`pdflatex` twice on `main.tex`, no errors. No "Float too large" in Chapter 7. Two tables bleed slightly into
the right margin: detect_families by 5.8pt (2 mm) and censoring_moves_other by 10.5pt (3.7 mm). All others
fit the 441pt text width. PDF: `scratchpad/thesis_code_todo/build/main.pdf`; Chapter 7 tables are 7.7 to 7.27.

## Check

Independent check of Chapter 7 on 2026-10-10, by Claude. Values were re-read from the result files with
new scripts (`python3 -I`, read-only), not with `vals.py`. Scripts: session scratchpad `chk/`.

### 1. "table needed" comments
None left in Chapter 7.

### 2. Table references
Every `\ref` in the chapter resolves to a label. No label is duplicated. Build (pdflatex twice) has no
undefined reference, no "Float too large" and no overfull box in Chapter 7.

### 3. Numbers in the 20 tables built earlier
1,242 cells re-read from the result files. All of them trace, rounded to the printed decimals:
herg_flip_actions 30, delete_vs_random 27, logd_outlier_actions 22, weighting 17 (pre-cut text),
shuffled 70, hergki_smoothing_folds 12, detect_residual 104, detect_families 125, detect_other 140
(empty cells included), breakeven 90, censoring_folds 21, censoring_moves + _other 240,
censoring_lift 64, family_offsets 36, herg_ic50 37, clean_before_search 40, search_clean_vs_damaged 100,
change_model + _delivered 160.

Two notes, both fixed in captions:
- search_clean_vs_damaged, SMAC SVR fold 3: 0.186 / 0.116 is the mean of the two draws that have both
  arms. The third draw has no damaged-arm score. The caption said "three noise draws".
- herg_flip_actions, fold 1: each detector is at its best setting. The caption did not say so.

One reading choice, not changed: in detect_families the "Hybrid's features" column also takes the
similarity-search hybrid (`hybrid_similarity` family), as `ch7_numbers.py` N21_FAMILY does.

### 4. Claims checked against tables
Fixed (Claude paragraphs):
- N7-14: "every one of its settings is ahead" is not in the table (true on Caco-2: worst setting 0.230
  against 0.202). Removed. "Under outliers only its best setting is ahead" was wrong: on Caco-2 about
  ten settings beat the residual. Now "its best setting is ahead and its median setting behind".
- N7-17: "The Shapley value's best number" was wrong. The table holds means over settings. Rewritten.
- N7-19: "Under outliers" is the break-even experiment's own damage, wide errors in 5% of molecules.
  Renamed. 0.17 / 0.62 now 0.171 / 0.616.
- N7-22: "Censored EM is best on LogD and QM9" and "constant lift is best on hERG Ki" do not hold in
  every configuration. Now "in most configurations".
- N7-26: "On hERG Ki the plain family mean did as well" does not hold on draw 2. Now "neither of the
  two was best on every draw".
- N7-39: "0.475 to 0.408 RMSE on LogD" paired the SVR first single with stacking on the second single.
  Now 0.495 to 0.408 on the second single representation.
- N7-18: "a few dozen molecules at most" is now 10 per pass (n41, every pass that flags any).
- N7-44: "half of the twelve" was wrong. The second run has eleven methods (no distance-based
  selection), and six beat the plain network. Now "about half of them".
- N7-25: clipping raised the Spearman in most configurations, not "sometimes". Censored EM moved it
  back down in most configurations, not all.

Paragraphs whose numbers were cut with no table comment. Tables built from the result files:
- tab:noise:res:known_flags (N7-06). Prose now says "wide errors in 5% of molecules", the n31 damage.
- tab:noise:res:qm9_deletion (N7-08).
- tab:noise:res:cliff_putback (N7-12). Fold 1: delete 0.5681, put back 0.5684 (tie rule).
- tab:noise:res:importance_scaling (N7-16). Prose now says scaling raised detection under the shift
  "on every dataset but hERG Ki, where it was unchanged" (0.3259 to 0.3257).
- SMAC and ΔΔG sections added to tab:noise:res:censoring_moves_other (N7-25).
- tab:noise:res:offsets_delivered (N7-27).
- tab:noise:res:huber_blend (N7-29), per draw.
- tab:noise:res:smac_weights (N7-32).
- tab:noise:res:sweep_examples (N7-43).
- tab:noise:res:qm9_sweep (N7-44).
- tab:noise:res:neighbor_space (N7-47).
- N7-46 now points to tab:noise:res:weighting for its 0.556 / 0.591 pair.

Left, not fixed:
- N7-28: "removing each source's offset made little difference on small molecules". The n39 file
  does not support it in every case. At level 1.0, rho 0, tau 0.5, fold 0, mean of draws, offset
  removal moves SVR on LogD's first hybrid from 0.6570 to 0.6245 and LightGBM on QM9's first hybrid from
  0.599 to 0.572. The file has two condition prefixes for the same rho and tau on BH and the antibodies.
  This needs the author before a table is built. "Helped in most settings" on BH is also unbacked.
- N7-38: "Seed averaging won on each of the three noise draws" is in the n40 file (pass_gains all
  positive), but not in the table, which shows means.
- N7-41, N7-45, N7-46 (apart from the pointer), N7-48: count claims ("more often", "three
  exceptions", "almost every"). The author's rule bans count tallies in results tables, so no table.
- N7-47: "helped ... slightly more often" is a count (87 against 98 of 160 pairs, SOURCE only).

### 5. Table rules
Every Chapter 7 results table has one number per cell and at most 4 bold cells. No table shrinks its
font. Two tables overflowed the text width (detect_families by 5.8pt, censoring_moves_other by
10.5pt). Both now set `\tabcolsep` to 4pt; the font size is unchanged. The older sweep table
(tab:noise:res:sweep) had `\small`. It now uses two-line headers instead.
Left: the methods table tab:noise:robust (Section 7.2) still has `\small` and an adjustbox with
max width. It is not a results table, and this pass did not change it.

### 6. Author paragraphs
Every prose paragraph in Chapter 7 has a `% TODO: manual review` marker. No unmarked paragraph changed
against `thesis.tex.before_build_tables_2026-10-10`.
