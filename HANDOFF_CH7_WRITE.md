# Handoff: Chapter 7, label noise (two chats: methods, then results)

STATUS (2026-10-09): READY EXCEPT the Chapter 7 dataset list, which comes from the per-dataset
noise decisions in `HANDOFF_LEDGER_CUTS.md`. Do not paste until that list is in the outline.
Then paste everything below the line into a fresh Claude Code chat in `~/repos/KIRBy`, once for
the methods chat and once for the results chat, and say which one it is. Earlier versions:
`HANDOFF_CH7_WRITE.md.before_rewrite_2026-10-09`, `HANDOFF_CH7_WRITE.md.before_signoff_update_2026-10-09`.

---

> Read `~/repos/KIRBy/CLAUDE.md` and then `~/repos/KIRBy/HANDOFF_CH5_8_COMMON.md` in full. All of
> the common brief applies, including the scope lists in its section 2.
>
> ## The job
>
> `\chapter{Label Noise: Detecting It and Doing Something About It}` in `thesis.tex` has the OLD
> noise material of the former Chapter 6 under it, unchanged. Nothing new is written. Most of it
> will be replaced; keep what matches the outline. The outline's Chapter 7 is the plan: 7.1 to 7.15,
> with the block "CHAPTER 7 METHODS, SIGNED OFF" at its top.
>
> - **Methods chat:** outline 7.2 to 7.4a. The methods are the signed-off list, in the three groups
>   (noise detection, noise mitigation, noise-robust training), using the outline's NAMES FOR THE
>   NOISE METHODS and `Table~\ref{tab:noise_terms}` from Chapter 2. A mitigation method is
>   described by what selects the molecules (a detector, a known flag, or every molecule), how many
>   it acts on, its action (delete, count less, move the label), and for a move, its target and how
>   far. Methods come from the code (`tests/experiments/noise/<code>_*.py`; each method in the
>   signed-off block carries its experiment number). Provenance and citations come from
>   `METHOD_PROVENANCE.md`.
> - **Results chat:** outline 7.5 to 7.15, using the MEETING_METHODS approach (common brief,
>   section 4) for every method on the list: does it work at all, first; then is the hybrid
>   better.
>
> ## How much space each method gets
>
> The signed-off block sorts every method into DEEP DIVE, BROAD and CUT.
> - DEEP DIVE: its own discussion, with the before and after numbers and the conditional
>   conclusions.
> - BROAD: a row in a broad table, or one sentence. No discussion beyond that.
> - CUT: not in the thesis at all, not even as "we also tried".
> The published sweep (all twelve methods from Chapter 4 plus DivideMix) is ONE broad table with
> one sentence: the domain split in the deleting methods. Chapter 4 says Chapter 7 expands its
> sweep; this table is that expansion, and the text says so in one sentence.
>
> ## The emphasis (common brief, section 4)
>
> - First: does the method work at all. Noise mitigation was the author's main goal and she got it
>   working on 9 Oct. Say it with confidence where the results back it.
> - The flag finding: correction works far better when you KNOW which labels are damaged than when
>   a detector has to infer it. The censoring corrections are the strongest results and every one is
>   driven by a known flag. Say it here and in Chapter 8.
> - Then: was fixing the feature set right? Does the hybrid improve the situation?
> - The table "Overall: does it work, by group" goes in, with noise mitigation highlighted most.
>   The analysis page's table uses an OLD grouping (training with noisy labels / reweighting /
>   noise mitigation). Rebuild it in the three signed-off groups, with mitigation split by what
>   selects the molecules, for the signed-off methods and the Chapter 7 datasets only. Show the
>   author the printed form before it goes in (one decision).
> - Conditional conclusions: which methods work on which noise kind, model, dataset and hybrid
>   setting. The negative results live here.
> - Antibody results get their own named discussion; antibodies must not dominate by count.
>
> ## QM9, the clean-label control (outline 7.4a and the signed-off block)
>
> - QM9's labels are computed, so a method that improves QM9 with no noise added is changing the
>   model, not removing noise. Every kept method is read against it.
> - Deleting never helps clean labels, so deletion gains elsewhere are noise removal.
> - Down-weighting and label smoothing help clean QM9 with the network but not with lightgbm, by
>   about as much as random deletion moves the network. So a network gain of about 0.01 RMSE from
>   weighting is not readable as noise removal. This caveat goes on the n21, n13 and n12 deep dives.
> - n40 (stack, retuned) and n20 have no QM9 row yet. Check before writing those deep dives; if
>   still missing, say so in the text and to the author.
>
> ## Censoring (7.7)
>
> - n30 is the deep dive: Censored EM, truncated_normal_mean and the constant lift work; deleting
>   the clipped molecules hurts; a per-molecule lift does no better than shuffled lifts.
> - ginkgo_pr_cho real censoring: built as `tests/experiments/noise/n30b_real_censoring.py`
>   (commit 2c283f2), run as an srun outside the ledger. Its write-up is due in
>   `scratchpad/thesis_ch5_8/PRCHO_REAL_CENSORING.md`. It is the real-label half of n30. If the
>   result does not exist yet, leave a marked slot.
> - A QM9 arm (injected clipping plus the not-at-floor control) was approved but not built yet
>   (addendum in `HANDOFF_CENSORING_PRCHO.md`). Leave a marked slot.
>
> ## herg_ic50 (repeat measurements)
>
> One paragraph: per_record, ngboost_count_feature, count_feature, xgboost_interval_aft,
> count_weighted, spread_pooled_weighted, shrunk. The curator-flag detection (n22) has not run.
> n17, n26 and n36 are decided in the ledger chat; follow the outline.
>
> ## Figures and tables, and the script changes known to be needed
>
> | section | figure | change needed |
> |---|---|---|
> | 7.5 | F8, did the cleaning work | It is one file per experiment with every variant. Show only the signed-off methods, blocked by the three groups, mitigation split by selector. |
> | 7.6 | F10, detection against repair | It reads unfinished checkpoints (outline 7.6 note). Check the inputs are final before using it. |
> | 7.7 | F9, the censoring ladder | Add the ginkgo_pr_cho real-censoring result once it exists. |
> | 7.7 | F11, score against damage level | Its five default methods are all mitigation. Use signed-off methods from more than one group. |
> | 7.5 or 7.1 | NEW: the group table | Not in the script. Add it (see the emphasis above). |
> | broad tables | NEW: broad tables per group | One row per broad method: datasets with results, base helped, hybrid helped, both, in the dataset's metric. `scratchpad/thesis_ch5_8/CH7_METHODS_BY_GROUP.md` and its scripts in `scratchpad/thesis_ch5_8/ch7m/` show the counts; they read the analysis page, so the script must read the raw files instead. |
>
> Every count in the analysis pages is fold 0, with each method's best setting picked on the
> held-out score. Where more folds exist (outline, Chapter 7 standing caveats), use them and say
> the fold count.
>
> ## Her voice, nearest to this chapter
>
> Sentences 10 to 25 in `scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md` section B.4 (her noise
> background in Chapter 2 and the Chapter 3 discussion). Find each in `thesis.tex` with `grep -F`.
>
> ## Last stage: phrasing pass
>
> After the content is right, re-read every block written in this chapter and rewrite its wording
> in the author's voice (common brief, section 4, Writing). Wording only, no number or claim
> changes. Report which blocks changed.
