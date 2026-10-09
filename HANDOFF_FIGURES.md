# Handoff: the KIRBy figures (Chapters 5, 6 and 7)

Written 2026-10-10 by the Chapter 7 results chat, after the Chapter 7 results were written into
`thesis.tex`. Part 1 is what exists and what is wrong with it. Part 2 is what each figure should
show now that the content is written. Part 3 is the prompt for a fresh chat. Nothing here is
decided until the author says so; every "should" in part 2 is a proposal with a recommendation.

Everything below was read on 2026-10-10 from `thesis.tex`, `tests/analysis/thesis_figures.py`
and the PNGs in `figures_ch5_7/`.

---

## Part 1. What exists

### 1.1 Every slot, and whether it is in thesis.tex

The script is `tests/analysis/thesis_figures.py`. Its outputs go to `tests/results/thesis_figures/`
and are copied by hand into `figures_ch5_7/`, which is where `thesis.tex` reads them from.

| Slot | Chapter, section | Shape | In thesis.tex? | Last drawn | State |
|---|---|---|---|---|---|
| Search overview diagram | 5, methods | diagram, `figures_code_input/hybrid_search_space.png` | yes, line ~1872 | by hand | Needs the three changes in `THESIS_DIAGRAMS.md` Diagram 1: remove the Pruning box, add Representation count, rename Target metric. |
| F1 | 5.4 | table, `tab_hybrid_vs_best_single.tex` | yes | 9 Oct 20:48, 9 Oct results | Fine. |
| F1 variant (PCA, mRMR, recursive elimination) | 5.5 | table | no slot in the script | never | The outline asks for it (c17, c15, c09). Check whether Chapter 5 now has it as a hand table. |
| F2a | 5.5 | figure | yes | 9 Oct 20:48 | A TODO beside it: the figure marks the budget with the best held-out mean (c09), the text gives the budget each fold picked on its training molecules (c02). Those can differ. Author to say which. |
| F2b | 5.5 | figure, side by side with F2a | NO | never with current data | Removed from the script's registry. The outline still asks for it. Its data now sit in c02 (each fold's representation count). Needs a decision: draw it from c02, or drop it from the outline. |
| F3 | 5.6 | figure | yes | 9 Oct 20:56 | The author's favourite. Two things to fix and one to add, see 1.2. |
| F5 | 6.3 | table, `tab_distance_tracks_activity.tex` | yes | 9 Oct 22:03 | Fine. |
| F6 | 6.4 | table, `tab_nearest_neighbour_prediction.tex` | yes | 9 Oct 22:03 | Fine. |
| Chapter 7 methods flow | 7, methods | diagram | NO, `[FIGURE NEEDED]` at the end of the Methods preamble | never | Damaged training labels, to the per-fold single representations and hybrids, to the detector, the action and the held-out score. |
| Group table | 7, results start | table | NO, slot `tab:noise:res:bygroup` | numbers built 9 Oct | Waiting for the author to approve the printed form (shown in chat 9 Oct). Numbers: `tests/analysis/ch7_numbers.py` rebuilds them. |
| Broad tables, mitigation and noise-robust training | 7.5, 7.10 | tables | NO, slots `tab:noise:res:mitigation_broad`, `tab:noise:res:robust_broad` | never | Same form as the group table, one row per broad method. Waiting on the same approval. |
| F8 | 7.5 | figure | NO | 9 Oct 02:29, 8 Oct results | Needs work, see 1.3. |
| F9 | 7.7 | figure | yes, `fig:noise:censoring_ladder` | 9 Oct 22:20, 9 Oct results | Fine. Two panels per row since commit c3cf65b. |
| F10 | 7.6 | figure, one file per dataset | NO | 9 Oct 01:39, 8 Oct results | Useful as is (author, 10 Oct). Re-run when the n32 final files exist, see 1.4. |
| F11 | 7.7 | figure | NO | 9 Oct 01:39, 8 Oct results | The line-plot form is liked; the methods on it are the wrong ones, see 1.5. |
| D8.1 | 8 | diagram, the eight-step chain | NO, `[FIGURE NEEDED]` | never | Described in the comment above the slot. |

The script's default results folder is still `/Volumes/seagate/kirby_results_2026-10-08` unless
`tests/results/CURRENT_RUN` says otherwise. Every run must pass `--results-root` with the newest
folder (noise figures: the `_noise_view` folder).

### 1.2 F3, the blueprints

What it shows: for each dataset, one bar on outer fold 0 with the dataset's first downstream
model, split into how many features the hybrid took from each base representation.

- Binding free energy (`ddg_combined`) sits under its own heading, "Proteins". The signed-off
  dataset list puts it with the antibodies. The two disagree on one dataset's domain.
- Antibody humanness is on it. The signed-off list allows that dataset one named result in
  Chapter 5 (a task one representation already solves). A bar in the blueprint figure is a
  second appearance. Author to say whether it stays.
- The outline asks for a second, deeper panel for one small-molecule dataset, with segments
  broken down to named feature groups rather than whole representations. It is not drawn. The
  Chapter 5 TODO beside the figure says the per-fold feature indices are stored in c24
  (`by_model.<model>.stability.feature_sets_per_fold`) and can be named with the tool in
  `tests/tools/` that names blueprint features. `scratchpad/thesis_ch5_8/f3_logd_feature_names.csv`
  already holds LogD's names. So the panel can be built without compute.

### 1.3 F8, "did the cleaning work"

What it shows: a grid. One row is one method with one detector; one column is one dataset (or one
dataset and damage kind). The shade is how many of the 8 model-and-configuration pairs the method
improved over no cleaning, on outer fold 0. The author liked the redrawn version in
`figures_ch5_7/f08_redrawn_on_laptop/` (one file per experiment, grouped by detector, columns
split into labels as delivered and damaged).

What is wrong with it now, each stated as the figure against the chapter:

1. **It shows improvement and ignores the controls.** The chapter judges a method by whether it
   beat its control where one exists. F8 judges it by whether it beat no cleaning. The clearest
   case: raising each clipped label by its own amount is 8 of 8 dark purple on both damaged
   columns of the censored-labels file, but it loses to the same lifts dealt to the wrong
   molecules on Caco-2, hERG Ki and LogD (Chapter 7, clipped labels). So F8 and the text
   disagree on those cells.
2. **It shows methods that are cut.** "Replace with neighbours' mean" (knn_smooth), "Move toward
   GP prediction" (gp_smooth) and "Pull clipped labels toward the limit" (winsor) are on the
   signed-off CUT list. The chapter never mentions them.
3. **It is one file per experiment.** An experiment is a file, not a unit of the thesis (author,
   9 Oct). The chapter is organised by the three groups, with mitigation split by what selects
   the molecules: a detector, a known flag, every molecule. Ten F8 files against three groups.
4. **Rows are grouped by detector.** The chapter's first comparison is by action (delete, count
   less, move the label), and its second is the detector. Grouping by detector inside each
   experiment puts the same action in several places.
5. **It is drawn from the 8 October results.** The columns hold Caco-2, LogD, hERG Ki, hERG and
   SMAC only. On 9 October the noise results cover 13 datasets.
6. **QM9 with no noise added is missing.** The outline (7.4a) says every results table and figure
   carries each method's QM9 result with no noise added, because a gain there is not noise
   removal. F8 has no QM9 column.
7. **"Of 8" is not always 8.** The expanded published sweep runs the network only (4 pairs), and
   some experiments run fewer configurations. The colour bar says "of 8" for every cell.

### 1.4 F10, detection against cleaning

What it shows: one point per detector and setting. x is how well the detector ranks the damaged
molecules (AUROC); y is the held-out RMSE after acting on its flags, with dashed lines for no
cleaning and for clean labels. One file per dataset, from n32.

Why it is useful: it shows in one picture that finding the damaged labels and fixing the model
are different (Chapter 7, detectors). On hERG Ki under outliers, the detectors with the highest
AUROC give the worst RMSE after deleting.

What needs updating (the author asked to be told, not to have it redrawn without asking):
- It reads unfinished checkpoints for Caco-2, LogD and antibody thermostability. Re-run when
  n32's final files exist for those three.
- The y-axis says "After repair". The chapter's word is cleaning ("held-out RMSE after cleaning").
- The nearest-neighbour Shapley detector, which the text cites under censoring, cannot appear,
  because no action in n32 reads it. That is a fact about the experiment, not the figure.

### 1.5 F11, score against damage level

What it shows: held-out RMSE of the network on the first hybrid as damage rises, one line per
method, on hERG Ki under Gaussian noise and under clipping. The five lines are no cleaning and
four detector-driven mitigation methods from n07.

The form works. The content does not match what the chapter says matters:
- Under clipping it shows four detector methods, all on top of no cleaning. The chapter's result
  under clipping is the flag-driven moves (censored EM, constant lift) against deleting the
  clipped molecules. None of those is on it.
- Under Gaussian noise, the chapter's largest gains come from changing the model (seed
  averaging 0.7184 to 0.6557 on hERG Ki at level 1.0), which is noise-robust training. F11 shows
  mitigation only. The handoff asked for methods from more than one group.

---

## Part 2. What each Chapter 7 figure should show now (proposals)

The chapter's points, in the order it makes them (see `thesis.tex`, Chapter 7 results):
1. Mitigation works, best of all with a known flag.
2. Deleting against changing a label: under hERG flips every action helps and deletion beats
   random deletion on three folds; on clean QM9 deletion never helps.
3. Detectors: above chance except under clipping; finding is not fixing; the break-even precision
   decides.
4. By noise type: clipping and scaffold-family offsets are fixable with their flag.
5. Antibodies: three datasets pass the shuffled control on their labels as delivered.
6. Changing the model instead of the labels does as well as the best mitigation.
7. The hybrid stays ahead on small molecules through every treatment; the ranking of methods
   changes with the feature space.

### F8: keep the grid, change what fills it (recommended)

- One figure per group, three in all, or one figure with three blocks: mitigation selected by a
  detector; mitigation selected by a known flag (plus every molecule); noise-robust training.
- Rows: the signed-off DEEP and BROAD methods only, named in words, ordered by action (delete,
  count less, move the label) within each block. Detector named in the row label where it
  matters. CUT methods never appear.
- Columns: keep the author's split, labels as delivered on the left, injected damage on the right.
  One column per dataset (damage pooled, or the main damaged level of each kind). Add a QM9
  column, no noise added, set apart and greyed in its header, so a gain there reads as "not
  noise removal".
- Cell shade: as now, the share of model-and-configuration pairs improved, with the denominator
  written per block (or the shade as a share, 0 to 1, so 4-pair and 8-pair cells compare).
- Control marking: where the method has a control, mark the cell. Proposal: a dot when the method
  beat its control in at least half its pairs, a cross when it did not. Cells with no control
  stay unmarked. This is the one change that makes F8 agree with the text.
- Read from the 9 October results (or newer), Chapter 7 datasets only (the list at the top of
  `tests/analysis/ch7_numbers.py`, `CH7_DATASETS`).
- One alternative to weigh: F8 could be dropped in favour of the broad tables, since both show the
  same counts. Recommendation: keep F8 for the deep-dive methods and the tables for the broad
  ones, so the figure is not crowded.

### F9: keep as is

Caco-2, hERG Ki, LogD and QM9, censored EM, four steps. When the ginkgo_pr_cho real-censoring
result exists, it gets its own panel or its own small figure: its floor is at the bottom, its
metric is Spearman, and its truth is unknown, so it does not share an axis with the four above.

### F10: keep, update when n32 finishes

As in 1.4. Possible companion, not yet asked for: the break-even precision (n31), detector
precision against the precision needed to gain, per damage kind. It is the clearest picture of
why cleaning works under outliers and not under shifts. Proposal only.

### F11: keep the line plots, change the lines

Proposal, two or three panels, held-out RMSE against damage level, one dataset and model per
panel, lines from more than one group:
- Clipping: no cleaning, clean labels (flat reference), censored EM, constant lift, deleting the
  clipped molecules. Only Caco-2 has n30 at three clipped shares (0.1, 0.25, 0.4); the other
  datasets have 0 and 0.25 only. So this panel is Caco-2 unless more levels are run.
- Gaussian: no cleaning, seed averaging, stacking, retuning (n40), and the best detector-driven
  move (the score-scaled shrink, n07). hERG Ki, network, second hybrid, levels 0 to 1.5. Check
  that n40 has every level (it has 0, and the non-zero levels with 3 draws).
- Optional third, hERG flips: no cleaning, deleting the flagged molecules, best random deletion.
  Check which flip levels n24 has beyond 0.1 before proposing it.

### Tables

The group table and the two broad tables are waiting on the author's approval of the printed form
(shown 9 Oct). Once approved, build them as `.tex` files from `tests/analysis/ch7_numbers.py`'s
group-table step, the way F1, F5 and F6 are built, and `\input` them with `\IfFileExists`.

---

## Part 2a. Bold cells: at most 4 per table, every table in the thesis

Author, 2026-10-10: far too many cells are bolded. At most 4 bold cells per table. This applies
to every table in the thesis and its appendices, not just Chapters 5 to 7.

The rule in the outline ("SELECTIVE BOLD", THESIS_OUTLINE.md around line 459) still decides which
cells: (a) the best value in the row, for a comparison between named methods (Chapter 5); or (b)
the counter-example, for a table whose point is that the answer changes with the dataset
(Chapters 6 and 7). Settings, definition and roster tables get no bold at all. The new limit sits
on top: whichever rule, no more than 4 bold cells, and the caption says which rule.

Header rows are not cells for this count, but the outline's style is plain headers anyway.

Where the bold comes from, counted 2026-10-10 (bold commands per table, more than 4 only):

| Table | Bold commands | Where the bold is made |
|---|---|---|
| F1, `figures_ch5_7/tab_hybrid_vs_best_single.tex` | 42 | `tests/analysis/thesis_figures.py` lines ~266-279: bolds the better of the pair in every row, and bolds every header. The script has to change, not the `.tex`, or the next run puts it back. |
| F6, `figures_ch5_7/tab_nearest_neighbour_prediction.tex` | 6 | same function in the script |
| `thesis_appendix/noise/items/uncertainty_metrics_ecfp4.tex` | 12 | by hand |
| `thesis_appendix/noise/items/model_icc.tex` | 7 | by hand |
| `thesis_appendix/noise/items/bayesian_transformation_tests.tex` | 7 | by hand |
| `thesis_appendix/noise/items/excluded_configurations.tex` | 6 | by hand (a settings table: should have none) |
| `thesis_appendix/noise/items/rf_vs_qrf_assay.tex` | 5 | by hand |

No table inside `thesis.tex` itself had more than 4 on this count. Check again before starting:
the count matches `\textbf{`, `\bfseries` and `\mathbf{`, and `\mathbf` can be a vector rather
than bold. The appendix items are copies of the NoiseInject paper's additional files, seeded once
and edited separately (CLAUDE.md, thesis files), so edit the thesis copies only.

For each table over the limit: pick the at most 4 cells that make the table's point under rule (a)
or (b), unbold the rest, and say the rule in the caption. Show the author the chosen cells for any
table where the point is not obvious.

---

## Part 3. Handoff prompt

Paste everything below the line into a fresh Claude Code chat in `~/repos/KIRBy`.

---

> Read `~/repos/KIRBy/CLAUDE.md`, then `~/repos/KIRBy/HANDOFF_FIGURES.md` in full, then
> `~/repos/KIRBy/HANDOFF_CH5_8_COMMON.md` section 4 (the figure rules) and section 5.
>
> ## The job
>
> You are in charge of every KIRBy figure and results table in Chapters 5, 6 and 7 of `thesis.tex`:
> that each one exists, shows what its section says, reads the newest results, and is in
> `thesis.tex` with a short caption. The list of slots and what is wrong with each is in
> `HANDOFF_FIGURES.md` part 1. The proposals for what each Chapter 7 figure should show are in
> part 2.
>
> ## Order
>
> 1. Before changing any figure, bring the author the decisions in `HANDOFF_FIGURES.md`, one per
>    line, each with what changes either way and a recommendation. They are:
>    - F8: the proposed blocks, rows, QM9 column and control marking (part 2).
>    - F11: the proposed panels and lines (part 2).
>    - F2b: draw it from c02 or drop it from the outline.
>    - F2a: which budget the figure marks, c09's best held-out mean or c02's per-fold pick.
>    - F3: binding free energy under "Antibodies" or its own heading; humanness on it or not;
>      the deeper LogD panel.
>    - The printed form of the Chapter 7 group table, if the author has not approved it yet.
>    Show her one example image per proposal where you can draw it quickly with `--figure`.
> 2. Then make the changes she approves, one figure at a time.
> 3. Bold: at most 4 bold cells per table, in every table of the thesis and its appendices
>    (author, 2026-10-10). `HANDOFF_FIGURES.md` part 2a lists the tables over the limit and where
>    their bold is made. Fix the script for script-made tables (F1, F6), so a re-run does not put
>    the bold back. Settings and roster tables get no bold.
> 4. F10: do not redraw it on your own. Tell the author when n32's final files exist for Caco-2,
>    LogD and antibody thermostability, and that it needs re-running.
>
> ## Rules
>
> - Results folder: always pass `--results-root` with the newest `/Volumes/seagate/kirby_results_*`
>   folder (the `_noise_view` folder for noise figures). The script's default is 8 Oct.
> - Datasets: Chapters 5 and 6 use `THESIS_DATASETS` in the script. Chapter 7 uses `CH7_DATASETS`
>   in `tests/analysis/ch7_numbers.py`. The reaction set is never a single point in a by-domain
>   panel. No patient data in any form (CLAUDE.md).
> - Methods on Chapter 7 figures: only the signed-off DEEP and BROAD methods (outline, "CHAPTER 7
>   METHODS, SIGNED OFF"), named with the outline's NAMES FOR THE NOISE METHODS. Never a CUT method.
> - Every Chapter 7 figure carries QM9 with no noise added where the method has a QM9 result.
> - Script edits: change only the figure functions and loaders you are working on. Run
>   `git status tests/analysis/` before each edit; if the file has changes you did not make,
>   stop and tell the author. One commit per figure, pushed, the message naming the figure. Run
>   the script with `--figure <slot>` only.
> - Tell the author each change in one line: what the figure showed before, what it shows now.
> - After a figure is redrawn, copy the PNG and CSV into `figures_ch5_7/`, put it in `thesis.tex`
>   with `\IfFileExists` like the others, a `% TODO: manual review` line, a `% SOURCE:` line with
>   the exact command and results folder, and a caption that says only what is shown.
> - Chapter 7's numbers are tied to `[N7-xx]` IDs and `tests/analysis/ch7_numbers.py`. If a figure
>   change means the text should change, say so to the author; do not edit Chapter 7's text.
> - New results are expected (more QM9, extra folds, maybe new datasets). Put each figure's
>   command in a `% TODO: NEW RESULTS` comment beside it so it is redrawn when they land.
