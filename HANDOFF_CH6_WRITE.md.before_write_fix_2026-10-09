# Handoff: Chapter 6, similarity, review and fix (two chats: methods, then results)

STATUS (2026-10-09): READY. Paste everything below the line into a fresh Claude Code chat in
`~/repos/KIRBy`, once for the methods chat and once for the results chat, and say which one it is.
Run the results chat after the methods chat. Earlier versions:
`HANDOFF_CH6_WRITE.md.before_rewrite_2026-10-09`, `HANDOFF_CH6_WRITE.md.before_signoff_update_2026-10-09`.

---

> Read `~/repos/KIRBy/CLAUDE.md` and then `~/repos/KIRBy/HANDOFF_CH5_8_COMMON.md` in full. All of
> the common brief applies, including the scope lists in its section 2.
>
> ## The job
>
> The first Chapter 6 chat rewrote `\chapter{Similarity in an Adaptive Hybrid Representation}` in
> `thesis.tex` as similarity only: 56 blocks marked `% TODO: manual review`. The author thinks it
> is semi-decent. Backup from before it: `thesis.tex.before_ch6_rewrite_2026-10-09`. Read the
> chapter in full, compare it with the backup, and give the author a short list: what is there,
> what is right, what does not match the outline and lists. She decides what is kept before
> anything changes.
>
> - **Methods chat:** outline 6.2. Check every methods sentence against the scripts
>   (`tests/experiments/similarity/<code>_*.py`, plus c04 in `tests/experiments/complexity/`), and
>   every citation against `METHOD_PROVENANCE.md`.
> - **Results chat:** outline 6.3 to 6.7. Check every result against the outline and the raw files.
>   Apply the MEETING_METHODS approach (common brief, section 4): what it does, single
>   representation and hybrid before and after, does it work at all, is the hybrid better. Then the
>   conditional conclusions: which datasets, distances and hybrid settings go with the hybrid
>   winning.
>
> ## Datasets
>
> The Chapters 5 and 6 block in common brief section 2. Chapter 6 uses the ones marked "Chapters 5
> and 6". adme_mdr1, adme_rlm, flab_expression_koenig, flab_binding_koenig, ginkgo_hic and
> abagym_Ang2 are Chapter 5 only, until their similarity results exist. buchwald_hartwig goes in
> the text, never as a single point in a by-domain panel.
>
> ## Decided with the author on 2026-10-09 (outline Chapter 6, "author agreed 2026-10-09")
>
> - s01, s05, s12 and s13 results are restated by domain.
> - s04 (activity cliffs) is combined into s12, which carries 6.5.
> - s06: the interval half is removed.
> - s03, s08 and s11 get one line each in the text, never their own section.
> - cross s09 is not used; the figure script makes the same summary.
>
> ## What goes in each section, and its figures
>
> | section | experiments | figures and tables |
> |---|---|---|
> | 6.2 methods | s02, s03, s05, s06, s08, s10, s11, s12, s13; c04 (n12 removed 2026-10-09: it is a noise experiment, Chapter 7) | — |
> | 6.3 does distance in a hybrid track activity | s01, s02; s08 as one line | F5 (a table, tab:distance_tracks_activity) |
> | 6.4 prediction from nearest neighbours | s05, s10, c04; s11 as one line | F6 (a table, tab:nearest_neighbour_prediction) |
> | 6.5 activity cliffs | s12, with s04 folded in | — |
> | 6.6 other uses of a distance | s06 (no interval half), s13; s03 as one line | — |
> | 6.7 conclusion | — | — |
>
> ## Figure script changes known to be needed
>
> - The outline's DECIDE note in 6.3 (which datasets F5 and F6 show) is settled: the signed-off
>   list. The script's default is now that list (`THESIS_DATASETS`, commit e31bf8d). Check F5 and
>   F6 draw it, and that a dataset with no similarity result simply has no row.
>
> ## Her voice, nearest to this chapter
>
> Sentences 26 to 34 in `scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md` section B.4 (the openings of
> her similarity chapter, Chapter 4). Find each in `thesis.tex` with `grep -F`.
>
> ## Last stage: phrasing pass
>
> After the content is right, re-read every block written in this chapter and rewrite its wording
> in the author's voice (common brief, section 4, Writing). Wording only, no number or claim
> changes. Report which blocks changed.
