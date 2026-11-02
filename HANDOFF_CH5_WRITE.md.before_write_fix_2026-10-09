# Handoff: Chapter 5, review and fix

STATUS (2026-10-09): READY. Paste everything below the line into a fresh Claude Code chat in
`~/repos/KIRBy`. Earlier versions: `HANDOFF_CH5_WRITE.md.before_rewrite_2026-10-09`,
`HANDOFF_CH5_WRITE.md.before_signoff_update_2026-10-09`.

---

> Read `~/repos/KIRBy/CLAUDE.md` and then `~/repos/KIRBy/HANDOFF_CH5_8_COMMON.md` in full. All of
> the common brief applies, including the scope lists in its section 2.
>
> ## The job
>
> Chapter 5 of `thesis.tex` (`\chapter{Building Adaptive Hybrid Representations}`) already has its
> experiment methods and results drafted: 59 blocks marked `% TODO: manual review`. They were
> written on 15 decisions the author approved without reviewing them (memory:
> `project_ch5_decisions_2026_10_09`). Review that text against the signed-off outline and lists.
> Keep what is right and fix the rest. Backup from before that chat:
> `thesis.tex.before_ch5_write_2026-10-09`.
>
> Start by reading the chapter in full and giving the author a short list: what is there, what is
> right, what does not match the outline and lists. Then re-check the 15 decisions with her, at
> most three at a time. She decides what is kept before anything changes.
>
> ## First edit: patient data
>
> PAM50 and TCGA are still in Chapter 5. Remove them everywhere, without opening any of their
> files: the dataset table (around line 2306), the TCGA representations table (around 2411), the
> grouping paragraph, and the `% SOURCE` comment that names a TCGA file (around 2048). Then
> recount any total that included them. `grep -n -i "pam50\|tcga"` must return nothing in the
> chapter.
>
> ## Decided with the author on 2026-10-09
>
> - The datasets: the block in common brief section 2. Antibody results are discussed properly, but
>   antibodies must not dominate by count. abagym_Ang2 and humanness_therapeutic get one result
>   each, briefly. ginkgo_hic is a usable negative. buchwald_hartwig goes in the text, never as a
>   single point in a by-domain figure.
> - c17 is not re-run. Its PCA arm is left out. `thesis.tex` methods and the experiment register
>   were updated. The old preliminary PCA tables (tab:prelim_reduction and the nanobody one) predate
>   this and need review.
> - The PCA comparison is computed PER FOLD: the hybrid picked on that fold's training molecules
>   against PCA at the same budget on the same fold, from c02 `pca_baseline_fold_scores`, with c11
>   `pca_matched` as a cross-check. Never PCA's best budget picked afterwards. Outline 5.4 has the
>   rule and the limit (PCA capped at the number of training molecules). It is a small check: a
>   sentence or a table row in 5.4, not its own section.
> - cross c03 (do the best hybrid settings follow dataset properties) is KEPT, as the
>   conditional-conclusions result for this chapter. It will start once the signed-off datasets
>   finish their complexity runs. Leave a marked slot until its result exists.
> - Out of the thesis: c05 (learning curves), c12 (joins whole data types; no result without
>   patient data), fed05 (cannot run since lhasa_herg was cut), opig_ginkgo, opig_graphinity,
>   opig_cgr_comparison, opig_trastuzumab, opig_humanness, opig_scaffold_hybrid,
>   opig_scaffold_similarity, cross c14, cross s09. Outline 5.3.5 still names opig_scaffold_hybrid
>   and opig_cdr_flexibility; leave them out.
> - Reported OPIG rows (5.7): opig_qsar; opig_shared_vectors_remedy and opig_blueprint_transfer, one
>   line each; opig_shared_vectors_census.
> - Caveats the author asked to state if the result is reported: c13 (LogD ran on 2,000 of 5,028
>   molecules) and c16 (outline 5.10).
>
> ## What goes in each section
>
> The outline sections are the shortlist. Space follows how much a result matters (common brief,
> WEIGHTING), not how many experiments feed it.
>
> | section | experiments (scripts are `tests/experiments/complexity/<code>_*.py`) | figures and tables |
> |---|---|---|
> | 5.3.4 methods | c02, c11, c16, c17, c19, c21, c23, c24, c33 | — |
> | 5.3.5 methods | opig_qsar (check which script the ledger row `opig:qsar` runs; `opig_activity_cliff_qsar.py` is the likely one, not verified), `opig_shared_vectors_census.py`, `opig_shared_vectors_remedy.py`, `opig_blueprint_transfer.py` (all in `tests/experiments/`); fed01 to fed04 (`tests/experiments/federated/`) | — |
> | 5.4 when the hybrid helps | c02 (core), c07, c11, c18, c22, c24, c33; PCA per fold from c02/c11 | F1 (table, c02); PCA per-fold table or sentence |
> | 5.5 why the hybrid helps | c02, c06, c08, c09, c11, c13, c15, c16, c17, c18, c19, c23, c33; c03 when it lands | F2a; F2b (see below) |
> | 5.5a extra features, decoy test | c21 | — |
> | 5.6 blueprints | c24 and the blueprints | F3 |
> | 5.7 OPIG | as above | — |
> | 5.8 separate sites | fed01 to fed04 | — |
>
> Outline 5.4 to 5.5 has "RESTATE BY DOMAIN (author agreed 2026-10-09)" notes for c11 and c19, and
> notes for c06, c08, c09, c13, c15, c16. Follow them.
>
> The antibodies need a proper look: why did the hybrid work so well there? The author's
> explanation is that antibodies are larger molecules, with more room for distinct featurisation,
> which she took full advantage of. Look at her blueprints (F3, outline 5.6). Where the numbers
> disagree with her explanation, show her both before writing.
>
> Conditional conclusions: which datasets, models and hybrid settings (budget, allocation method,
> concatenation) go with the hybrid winning. Start from `scratchpad/thesis_ch5_8/PATTERNS.md` and
> bring the patterns to the author before writing them.
>
> ## Figure script changes known to be needed
>
> - F1 is a table built from c02. Check it shows exactly the signed-off datasets.
> - F2b was removed from the script (commit 493526a), but outline 5.5 still asks for it beside F2a.
>   Ask the author whether it comes back.
> - The per-fold PCA comparison is not in the script. Add it as a table (one row per dataset and
>   model: hybrid score, PCA at the same budget, folds), from c02 `pca_baseline_fold_scores`.
> - c03's figure or table, once its result exists.
>
> ## Her voice, nearest to this chapter
>
> Sentences 35 to 42 and sentence 8 in `scratchpad/thesis_ch5_8/FIGURES_AND_STYLE.md` section B.4
> (the Chapter 5 introduction and the Chapter 1 summary of it). Find each in `thesis.tex` with
> `grep -F`; the line numbers there are from before later edits.
>
> ## Last stage: phrasing pass
>
> After the content is right, re-read every block written in this chapter and rewrite its wording
> in the author's voice (common brief, section 4, Writing). Wording only, no number or claim
> changes. Report which blocks changed.
