# Handoff: real censoring in ginkgo_pr_cho, a one-off srun experiment

Written 2026-10-09. Paste everything below the line into a fresh Claude Code chat in `~/repos/KIRBy`.

---

> Read `~/repos/KIRBy/CLAUDE.md` first and follow it in every reply. The author is the engineer
> and decides the design. Your first deliverable is a short design proposal (arms, feature
> spaces, models, folds, metric, run time). It is not code. Build nothing until she signs it
> off. Bring at most three decisions at once.
>
> ## Why this experiment exists
>
> QM9 is the thesis's control for labels that are known to be clean. ginkgo_pr_cho is meant to
> be the matching case for labels that are known to be censored, in the labels as delivered.
> It is polyreactivity (PR_CHO) for 197 antibodies from the Ginkgo GDPa1 set, and 58 of its 197
> labels sit exactly at the assay floor. This is a case study, run once on its own in an srun.
> It does not wait for the main runs.
>
> The question: the methods that fixed injected censoring, do they fix real censoring? Answer it
> with the best single representation and with the hybrid, side by side.
>
> ## Step 0: confirm the floor is real, before anything else
>
> - Count the labels at the minimum on the loaded dataset. Use the ginkgo loader, then grep
>   `src/kirby` and `tests/experiments/dataset_policy.py` for `ginkgo_pr_cho`.
> - Find what the GDPa1 source says about the PR_CHO assay range or detection limit. Quote it
>   with its URL, or say plainly that it was not found.
> - Report the value of the floor, how many labels sit on it, and the source evidence. Stop and
>   ask the author if the floor might not be a measurement limit.
>
> ## What is already known about censoring
>
> This is injected censoring, fold 0, one noise draw. Re-read every number from the raw files
> before repeating it.
>
> - Source of the summary: `scratchpad/thesis_ch5_8/CH7_RESULTS.md`, sections 2c and 4.
> - Raw files: `/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n30_censored_likelihood/<ds>.json`,
>   key `per_level['0.25'][config][arm][model]`.
> - Code: `tests/experiments/noise/n30_censored_likelihood.py`.
> - Earlier design notes:
>   - `tests/experiments/noise/HANDOFF_CENSORING_DEEPDIVE.md` (2026-09-28). It already names
>     adme_hlm and adme_rlm as real-floor candidates.
>   - The open item in `THESIS_OUTLINE.md`, Chapter 7 censoring results.
>
> What worked:
> - **Knowing which labels are clipped, and moving them by about the right average amount.**
>   Censored EM and the constant lift are strong with both models on caco2, herg_ki, logd, qm9
>   and flab_thermo_tresanco, and on buchwald_hartwig with rf. One example: logd, network on the
>   hybrid, RMSE goes 0.6049 clipped to 0.4604 lifted, against 0.4059 before clipping.
> - **Flag-driven moves.** On small molecules at a clipped share of 0.25 they helped both the
>   single representation and the hybrid in 38 of 38 rows. Detector-driven actions helped in
>   51 of 168, and noise-robust training in 25 of 87.
>
> What did not work:
> - **Giving each molecule its own lift.** It loses to its own control, which deals the same
>   lifts to the wrong clipped molecules: the correct assignment wins in 1 of 22 rows.
> - **Deleting the clipped molecules.** On herg_ki with the network, RMSE goes 0.742 to 0.890.
> - **Retuning the model under censoring.** It helped on only 1 or 2 of 8 pairs.
> - **Two datasets.** Censored EM failed on ginkgo_colloidal and ddg_combined.
>
> What was never tested:
> - **The flag itself.** No experiment applies the same move to the same number of labels that
>   are NOT at the limit.
> - **A lower limit.** n30 handles only a pile-up at the TOP label (`n30_censored_likelihood.py`
>   lines 287-300).
> - **A real-labels mode.** n30 runs injected clipping only; its lines 142-146 say "no real-floor
>   path exists".
>
> ## The arms to propose
>
> The author decides the final list.
>
> - **Baseline:** no treatment.
> - **Known flag, move the label DOWN off the floor:**
>   - constant lift;
>   - censored EM;
>   - truncated-normal mean;
>   - per-molecule lift, with its shuffled-lift control.
> - **Known flag, delete:** drop the floor labels.
> - **Censored likelihood in the model:** the AFT / Tobit arm, where n30 has one (around lines
>   440-455).
> - **New control:** the same move applied to the same number of labels NOT at the floor. This
>   tests whether the flag carries the gain.
> - **Feature spaces:**
>   - best and second-best single representation;
>   - first and second hybrid.
>
>   Take them from the existing per-fold searches, as n30 does. Labels as delivered are already
>   censored, so nothing is re-damaged.
> - **Models:** ginkgo_pr_cho's decided models are gp and dnn. The gp is due to be replaced
>   (fault A2 in `/Volumes/seagate/claude_analysis_2026-10-08/answers/FAULTS.md`). Ask the author
>   whether to use the replacement or the gp as it stands.
> - **Folds:** all 5 outer folds, since the dataset has only 197 antibodies.
>
> ## Scoring, which is the hard part
>
> The held-out labels are censored too, so a plain held-out score is partly corrupted by the same
> floor. Propose reporting all three of these, per fold:
> - Spearman on every held-out antibody, the dataset's metric;
> - Spearman on the held-out antibodies above the floor only;
> - a censoring-aware concordance (Harrell's C) that treats floor labels as censored. Check a
>   citation before using it.
>
> Report the per-fold scores and their spread. No significance tests.
>
> ## Code
>
> - Extend n30 with `side='lower'` and a real-labels mode that injects nothing. The alternative
>   is a small new script that imports n30's functions.
> - Do not change what n30 does for injected censoring.
> - Commit each change.
> - Optional, and the author decides: run the same path on adme_hlm and adme_rlm, the two
>   candidates from the 2026-09-28 handoff.
>
> ## Running it
>
> - Test locally first, on fold 0, time-boxed. Then the server.
> - Server format, which is a firm rule:
>   - ONE line to get an interactive shell, for example
>     `srun --clusters=htc --account=stat-cadd -p interactive --cpus-per-task=8 --mem=64G --time=03:00:00 --pty /bin/bash`;
>   - wait for the prompt;
>   - then plain commands typed inside it: `cd`, `export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh`,
>     one python command per line.
> - Never sbatch, arrays, shell variables holding commands, or `rm` in a pasted block.
>   Interactive srun runs for at most 4 hours, so give a run-time estimate first.
>
> ## The write-up
>
> Analyse it the way `/Volumes/seagate/claude_analysis_2026-10-08/answers/MEETING_METHODS.md`
> does. For each method:
> - what it does;
> - single representation before and after;
> - hybrid before and after;
> - does it work at all;
> - is the hybrid better after.
>
> Then compare with what injected censoring showed. Put it in one short markdown page in
> `scratchpad/`, not in `thesis.tex`, until the author decides where it goes. Every block that
> later goes into `thesis.tex` gets `% TODO: manual review` above it, then a phrasing pass in her
> voice.

---

## ADDED 2026-10-09, after n30b was built (commit 2c283f2): a QM9 arm, approved by the author

Paste this into the chat that built `n30b_real_censoring.py`.

> The author approved adding QM9 to this experiment. ginkgo_pr_cho tests the flag where the truth
> is unknown; QM9 tests it where the truth is known, because QM9's labels are computed and clean.
> - On QM9, clip the training labels at a share of 0.25 the way n30 does (top limit, n30's own
>   injection), then run the same treatments as n30b, INCLUDING the not-at-floor control: the
>   same move on the same number of training labels that are NOT at the limit.
> - Score against the unclipped held-out labels, which are known, so plain RMSE works here.
> - All 5 outer folds, QM9's two decided models, the same four configurations as n30b.
> - Do not change what n30 or n30b already do. Propose how (a mode in n30b, or a small script that
>   imports both), with a run-time estimate, before building. Same srun format rules as above.
