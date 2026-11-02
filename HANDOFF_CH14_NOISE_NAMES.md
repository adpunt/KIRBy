# Handoff: apply the noise-method terminology to Chapters 1 to 4

Written 2026-10-09. Paste the block below into a fresh Claude Code chat with `~/repos/KIRBy`.
The terminology it applies was settled by the author on 2026-10-09 and is recorded in
`THESIS_OUTLINE.md` under `NAMES FOR THE NOISE METHODS`.

---

> Apply a settled change of terminology to Chapters 1 to 4 of a DPhil thesis.
> Repository `~/repos/KIRBy`, file `thesis.tex`, plus one line in `main.tex`.
>
> ## Read these first, in this order
>
> 1. `~/repos/KIRBy/CLAUDE.md`. How to write to the author. Follow it in every reply.
> 2. `~/repos/KIRBy/THESIS_OUTLINE.md`, the block headed `NAMES FOR THE NOISE METHODS`. That
>    block is the specification. Do not redesign it.
> 3. `/Volumes/seagate/claude_analysis_2026-10-08/answers/_ref/METHOD_GROUPS.md`. All 85 noise
>    methods already sorted into the scheme, with what selects the molecules and what is done to
>    them. Use it whenever a sentence names a method and you need to know which group it is in.
>
> ## THE WARNING, AND IT IS THE WHOLE JOB
>
> **This is not find-and-replace.** The old vocabulary had categories that no longer exist, so
> many sentences have a subject that has to be rebuilt, not renamed. Three worked examples from
> the file:
>
> - `thesis.tex:464` says noise mitigation "changes, drops or reweights the training labels once".
>   Under the new scheme reweighting never touches the label, so that clause describes two
>   different things as one. The sentence is replaced, not edited.
> - `thesis.tex:142` and `:446` both state the old definition as a takeaway: "LNL covers every
>   method for training on wrong labels. Noise mitigation is the subset that changes the labels."
>   Both halves are now wrong. Mitigation is not a subset of a training-time field, and the
>   definition leaves out down-weighting and everything driven by a known flag.
> - `thesis.tex:108` says "a quick sweep of noise mitigation and various other LNL techniques".
>   That sweep was twelve methods spanning two of the three groups, so the phrase has to say
>   which.
>
> If a sentence only needs a word swapped, swap it. If its structure assumed the old categories,
> rewrite the sentence. Where you are unsure which it is, leave the text alone and list it for the
> author instead of guessing.
>
> ## The scheme, in brief
>
> A method that acts on the data has two halves, and the thesis always names both.
>
> What selects the molecules:
> - **a detector** -- a score inferred from the features and labels, saying how wrong each label
>   looks. The score is that molecule's **suspicion score**. 42 of the 85.
> - **a known flag** -- recorded in the data, nothing inferred: at the assay limit, how many times
>   measured, which source reported it, which scaffold group. 22 of the 85.
> - **every molecule** -- no selection at all. 6 of the 85.
>
> What is then done:
> - **delete** the molecule, **down-weight** it, or **move its label** toward a target by a
>   distance. Targets used in this thesis: a neighbours' mean, a predicted value, a group mean, a
>   quantile bound, an assay limit.
>
> The three groups:
> - **noise detection** -- a detector alone, nothing done afterwards.
> - **noise mitigation** -- one of the three selectors plus one of the three actions. 63 of 85.
> - **noise-robust training** -- nothing selected, the data untouched; the loss, the kernel, a
>   noise term, the training loop or the model changes. 22 of 85.
>
> Thirteen names in the repo are all the one action "move the label" and differ only in target and
> distance: `label_smooth`, `knn_smooth`, `oof_smooth`, `gp_smooth`, `prox_smooth`, `js_shrink`,
> `quantile_clip`, `constant_lift`, `move_confidence`, `move_proportional`, `winsor`,
> `censored_em`, `truncated_normal_mean`. Never present them as thirteen unrelated methods.
>
> **"LNL" names the problem, never a group.** Keep it in Chapters 2 and 4 only, where the
> literature is described, and cite \citet{Song2022} for noise-robust training alone, because all
> five of their categories act during training.
>
> ## The inventory you are working through
>
> Chapter line ranges in `thesis.tex`: Ch1 11-135, Ch2 136-643, Ch3 644-1280, Ch4 1281-1781.
>
> | chapter | lines carrying "LNL" or "learning with noisy labels" | of those, comment lines | lines carrying "noise mitigation" |
> |---|---|---|---|
> | Ch1 | 5 | 2 | 7 |
> | Ch2 | 11 | 7 | 10 |
> | Ch3 | 0 | 0 | 0 |
> | Ch4 | 14 | 7 | 4 |
>
> Chapter 3 is clean. Do not open it.
>
> Exact lines with "LNL" or "learning with noisy labels":
> - Ch1: 69, 90, 104, 108, 116
> - Ch2: 142, 444, 445, 446, 448, 452, 464, 478, 633, 634, 642
> - Ch4: 1286, 1290, 1316, 1610, 1617, 1620, 1621, 1629, 1632, 1634, 1638, 1653, 1691, 1748
>
> **Comment lines count and must be fixed.** Half of them are comment blocks, and those are what
> the author drafts from. A comment block that still states the old definition will put it back
> into the prose later.
>
> ## The edits, in order
>
> **1. Chapter 2, the subsection at line 444.** It is `\subsection{Learning with noisy
> labels}\label{sec:lnl_background}`. Rename the heading to `\subsection{Methods for noisy
> labels}`. **KEEP `\label{sec:lnl_background}` EXACTLY AS IT IS** -- five `\ref`s point at it,
> at lines 443, 1292, 1316, 1591 and 4884. Renaming the label means fixing all five and there is
> no reason to.
>
> **2. Replace the paragraph at line 464 entirely** with a paragraph built on the two halves and
> the three groups, in this order: the problem is called learning with noisy labels; a method
> that acts on the data has two halves; the three things that can select molecules; the three
> actions; the three groups that follow; then \citet{Song2022}'s five categories, cited for
> noise-robust training only, with the observation that all five act during training and so a
> detector that runs once and hands a smaller training set to any model has no place among them.
> Keep `Figure~\ref{fig:data_cleaning}` cited from it. Keep every existing citation: Han2018,
> shu2019metaweightnet, Song2022. Leave the `% TODO: manual review` and the
> `% [CLAUDE 2026-10-09] citation fix` lines above it in place.
>
> **3. Insert this table immediately after that paragraph.** It is the author's own layout. Do not
> restyle it, do not bold any cell, and do not add arrows -- it carries no metric. Verbatim:
>
> ```latex
> \begin{table}[htbp]
> \centering
> \caption[Terms for methods that handle noisy labels.]{Terms used throughout
> this thesis for methods that handle noisy labels. A noise mitigation method is
> named by both halves: what selects the molecules, and what is then done to
> them. The group follows from the two.}
> \label{tab:noise_terms}
> \begin{tabular}{lp{0.60\textwidth}}
> \toprule
> \multicolumn{2}{l}{\textit{What selects the molecules}} \\
> \midrule
> Detector & A score inferred from the features and the labels, saying how wrong
> each label looks. The score is that molecule's suspicion score. \\
> Known flag & Already recorded in the data, with nothing inferred: a label at
> the assay limit, how many times a molecule was measured, which source reported
> it, which scaffold group it belongs to. \\
> Every molecule & No selection. The method acts on all the labels. \\
> \midrule
> \multicolumn{2}{l}{\textit{What is done to them}} \\
> \midrule
> Delete & The molecule leaves the training set. \\
> Down-weight & The rows and labels are kept and each molecule carries a weight. \\
> Move the label & The label shifts toward a target by some distance. The target
> is a neighbours' mean, a predicted value, a group mean or an assay limit. \\
> \midrule
> \multicolumn{2}{l}{\textit{The groups}} \\
> \midrule
> Noise detection & A detector alone, with nothing done afterwards. \\
> Noise mitigation & A detector, a known flag or every molecule, and then one of
> the three actions. \\
> Noise-robust training & Nothing is selected and the data is untouched. The
> loss, the kernel, a noise term, the training loop or the model changes. \\
> \bottomrule
> \end{tabular}
> \end{table}
> ```
>
> Every later chapter that uses the terms cites `Table~\ref{tab:noise_terms}` once, on its first
> use, and never redefines them.
>
> **4. The notes block at lines 445 to 460.** Its two `% Takeaway:` lines and its Paragraph 1
> bullets all state the old definition. Rewrite them to the new scheme so a later drafting pass
> cannot reintroduce it.
>
> **5. The figure caption at line 471** says "Evaluation of a noise mitigation method", which is
> still correct under the new scheme -- the figure shows a detector-driven correction scored
> against an untreated model. Check that the body text around it does not call it a subset of LNL.
>
> **6. Chapter 1, lines 69, 90, 104, 108, 116.** These are the thesis outline paragraphs and the
> chapter summaries. 104 and 108 are body text; 69 and 90 are comment bullets; 116 is the Chapter 6
> summary. Each needs the group named rather than "LNL".
>
> **7. Chapter 4, the fourteen lines listed above.** Line 1617 is
> `\subsection{Exploratory study of learning with noisy labels}\label{sec:lnl_motivation}`. Rename
> the heading. The author's wording for what this section is: **a small sweep of published
> noise-handling methods**. "Noise-handling" because the twelve methods span two of the three
> groups, so neither "mitigation" nor "noise-robust training" covers them alone. Say in one
> sentence that Chapter 7 expands this sweep.
> **KEEP `\label{sec:lnl_motivation}`** -- thirteen `\ref`s point at it, at lines 1292, 1316, 1591,
> 1610, 1634, 1691, 1748, 3611, 4375, 4726, 4922, 4927, 5031. Eight of those are in Chapters 5 to
> 8 and are outside your range; leaving the label alone is what keeps them working.
>
> **8. `main.tex` line 191** has `\item[LNL] Learning with noisy labels` in the List of
> Abbreviations. Keep the entry, because Chapters 2 and 4 still use it. Gloss it so it reads as the
> name of the problem and not a family of methods.
>
> ## Two discrepancies to resolve, not paper over
>
> **The twelve split four and eight, or three and nine.** `thesis.tex:4338` (a comment, outside
> your range) says four of the twelve remove molecules and retrain. `src/kirby/utils/lnl_methods.py`
> has `DATA_LEVEL_METHODS` with three. The difference is `multistage_cleaning`, which removes ten
> per cent per round over three rounds with the removal coded inside the fit. By behaviour it
> selects and deletes, so four is right for the thesis and the code constant is drawn on a
> different line. Wherever Chapters 1 to 4 state the split, say **four remove molecules and retrain
> and eight change the training**. Flag the mismatch with the code constant to the author; do not
> edit the code.
>
> **"Noise mitigation" changes meaning.** It used to mean "changes the labels". It now means a
> selector plus any of the three actions, so down-weighting is inside it. Every existing use in
> Chapters 1 to 4 has to be read against the new meaning; some will be right already and some will
> now be too narrow.
>
> ## Rules
>
> - **Back up first**: `cp thesis.tex thesis.tex.before_noise_names_<date>`. Report the path.
> - **Another chat may be editing this file.** Run `git diff --stat thesis.tex` before you start
>   and again before you write. If it changed under you, stop and say so.
> - **Do not change a single number, result, dataset name or citation key.** This is a
>   terminology pass. If a renaming would change what a result claims, stop and ask.
> - **Do not edit `THESIS_OUTLINE.md`.** It is the specification, not a work product.
> - **No experiment codes anywhere in the thesis** -- no `n07`, `n18`, `c02`. Address methods by
>   what they do.
> - **Never write an AI verdict into the thesis.** If you infer something rather than reading it,
>   say so in the reply, not in the file.
>
> ## Deliverable
>
> 1. The edits applied to `thesis.tex` and `main.tex`, with the backup path named.
> 2. `git diff thesis.tex` summarised in the reply: for each change, the line, the old wording and
>    the new one. The author reviews this, so one line per change and no commentary.
> 3. A list of the sentences you did NOT change because the right rewrite was a judgement call,
>    with the question attached to each.
> 4. A count at the end: how many lines carried "LNL" or "learning with noisy labels" before and
>    after, per chapter, so the author can see the pass is complete.
