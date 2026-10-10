# Thierry's comments on the 2026-10-01 PDF (received 2026-10-10)

68 margin comments, all on old pp. 87-192 (old Ch4-7 = now Ch4-8). Extracted from
`~/Downloads/thesis.pdf` into `comments_ch*.md`.

## 1. Applied to thesis.tex (43 notes)

Every change has a `% [TH edit 2026-10-10: ...]` line above its paragraph. `grep -n 'TH edit' thesis.tex`.
Per-comment detail, with the old and new text: `map_ch4.md`, `map_ch5.md`, `map_ch6_8.md`.
The PDF builds (271 pages, no undefined references).

Numbers that are new to the thesis and were counted in this session, not by a saved script:
- 1,812 off and 3 on, of the 1,815 near-constant ECFP4 bits on hERG Ki (`rep_cache/herg_ki/ecfp4.npy`).

## 2. His comments that need you (not applied)

Chapter 4
- Retention ratio above 1, and a figure of the 12-method sweep: both need the raw sweep results, which are not on this machine.
- Which sigma the 86.7%, 84.2% and 82.7% retentions come from.
- A figure showing where the 70 dataset-representation combinations come from (text now says 7 x 10).
- WL ties: of the 49 non-stereo ties, 37 share a heavy-atom graph and 12 do not (agent count, not applied).
- Row 5 of the requirements table ("hard to read chemically") lost its section when the PDV sentence was cut.

Chapter 5
- External test set (data snooping). Facts now in the text: each per-fold search sees only its training molecules; the hybrid and the best single representation are both picked on those. Each dataset's grid and two models came from a whole-dataset search that saw the held-out molecules (`scratchpad/precompute_cuts/DECISIONS.md`: "leaky, accepted").
- Which molecules the ten base representations per dataset were chosen on.
- Why topological torsions were a donor, not a base representation, on QM9.
- Spread in the main table: c02 records fold min and max, but `thesis_loader_f1.py` drops them. Two extra columns, or a per-fold table in the Additional files.
- Figure of the combination methods (his "if time").
- Search-space figure draws no outer fold and draws a Pruning box; the per-fold searches ran unpruned. The caption now says both; redrawing is yours.
- Variance: whether a hybrid changes the Chapter 3 variance split. No experiment measures this.

Chapter 6 to 8
- The random forest gains less from the hybrid than SVM and LightGBM. Nothing in the thesis discusses it. RF: Buchwald-Hartwig 14.4056 to 15.2383 RMSE (hybrid worse), humanness 0.9659 to 0.9614 AUROC (worse), anti-VEGF 0.4325 to 0.5112 Spearman (better), titer 0.2504 to 0.2706 (better).
- Similarity overlap: importance weighting in Ch6 may or may not be the Ch5 weighting (average-1 rescaling unverified), so no cross-reference was added.
- Praised results that did not survive: old Table 5.16 (allocation summary), old Table 5.26 (per-fold transfer), and the similarity hybrid row in the nearest-neighbor table.

## 3. Same lens on the rest of the thesis (proposals only, nothing applied)

146 proposals, 65 high priority, in `lens_ch1_2.md`, `lens_ch3_4.md`, `lens_ch5.md`, `lens_ch6.md`, `lens_ch7_8.md`.
Each has the place, his likely comment, and an edit. The old_strings in Ch4 and Ch5 were taken before
section 1 was applied; a few may need re-matching. The lens itself: `TH_LENS.md`.
