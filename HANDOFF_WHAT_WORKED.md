# Handoff — check the two lists of methods that worked

Written 2026-10-28 for a fresh chat. One job: find out whether the owner's two lists are
right, correct the audit where it is wrong, and leave her the two lists she will work from.

## The job

She keeps two lists of noise methods. List 1 is the methods that raised the top score under
at least one kind of damaged label other than censoring. List 2 is the methods that raised it
only under censoring. The lists are her headline for the third paper and they have never been
checked against the experiments that produced them.

Check every entry. Add anything missing. Move anything in the wrong list. Remove anything
that does not belong.

## What "worked" means

The best treated held-out score beats the best untreated score, across all models and all
representations, on the same damaged training labels. This is her definition. Do not replace
it, soften it, or substitute a mean across datasets.

Two checks support that reading and never sort or rank anything: the random comparison (the
same action applied to randomly chosen molecules) and QM9 (computed labels, so no lab error —
the clean control).

No significance tests, not across folds and not across datasets. Per-fold score pairs and
their spread, nothing else. A method that wins on one dataset and loses on another has done
exactly that; say so, do not average it away.

## Where the answers already are

`~/Documents/ch7_noise_methods_audit.html` holds 79 method blocks, `m1` to `m79`. Every block
opens with an overview line in `<span class="ov">` and then a heading **1. Data: raised the top
on / did not / not run**. That is the per-dataset verdict for that method, already written.

Start there: match each of her 40 listed names to its block and compare. Where a block's
verdict and her list disagree, go to the result files and settle it. Where the block itself is
wrong, fix the block.

The results sit in `tests/analysis/ch7_sources/` — one small loader per experiment (`n07.py`,
`n09_a.py`, `n12.py`, `n18q.py`, `n20.py`, `n21.py` and the rest) — and in
`tests/analysis/ch7_numbers.py`, which already reads the result rows and checks written claims
against them. Use that machinery rather than writing new readers.

The experiment scripts themselves are `tests/experiments/noise/nXX_*.py`.

## Three disagreements visible before you start

The audit's own overview says these raised the top score on no dataset, and all three are on
her lists:

| Method | Her list | What the audit block says |
|---|---|---|
| Multistage cleaning (`#m44`) | List 1, selection | With DNN only, raised the top on 0 of 3 small-molecule datasets with a recorded result, out of 5 tested; wins concentrate on shifted labels and on clean labels |
| Seed averaging (`#m56`) | List 1, training | Overview reports no dataset raised |
| XGBoost accelerated-failure-time interval model (`#m73`) | List 2, censoring only | Overview reports no dataset raised |

These three are where to begin. Settle each against the result files, then fix whichever side
is wrong — her list or the block.

Five more blocks carry the same kind of overview and are not on either list: heteroscedastic GP
spread (`#m7`), conformal p-values with Benjamini-Hochberg flags (`#m13`), NGBoost with a
log-count feature (`#m72`), random removal (`#m52`, a control) and ranking within each paper
(`#m79`, not a Chapter 7 method). Check them too, in case one belongs on a list.

## Six entries in List 1 whose code does not match the name

Confident learning, uncertainty curriculum, DivideMix, MentorNet, contrast-to-divide and the
heteroscedastic GP are all in List 1, and for all six the code does not compute the published
mechanism the name refers to. The write-up is
`thesis_provenance/NOISE_CODE_FIX_DECISIONS_2026-10-26.md`, headings C to H.

**Those six decisions are being handled in another chat. Do not build, fix or rename any of
them here.** What this handoff needs is the measurement question only: did the code that ran
raise the top score. If it did, the entry stays on the list, with a line in its block saying
the result belongs to the code as it ran and not to the published method. If the owner later
renames one, the result stays true and only the name moves.

## How to write into the audit

It is a single minified HTML file, dark theme, `body{display:flex}` with a `<nav>` and a
`<main>`. It ends in a `<script>`, not `</body></html>`, so anything appended to the end of the
file becomes a third flex child and lands outside the layout. Insert a new section with
`src.replace('</main>', new_section + '</main>', 1)`. `</main>` is the only safe anchor.

Each block is `<details id="mN" class="m">` with a `<summary>` carrying the title and the
overview span, then `<h4>` sections 1 to 5, each with a `<p class="pl">` in plain language and
a `<details class="ev">` holding the evidence. Keep that shape. Section headings `<h2 id="sN">`
run s0 to s5 and each has a matching nav entry; a new section needs its nav link too.

**This document is for her to read.** Correct a wrong sentence by rewriting it. Do not add
"checked on 2026-10-28", "verified against", "this was previously stated as" or any other note
about the checking having happened. No internal bookkeeping in her document. No AI verdict
written so that it later reads like a measured result — if something is a judgement rather than
a measurement, the sentence has to say which it is, in her words, or it does not go in.

Plain language throughout, and her vocabulary only: molecules, datasets, folds, features,
models, representations, cleaning, activity cliffs. Banned: "arms", "cells", "pools", "knobs",
"read-across", "SAC", "round" and "stage" for runs, and any acronym she has not used first.
Never a bare count or fraction — say what the top and the bottom count, in the same sentence,
before the number.

## The deliverable

A new section at the end of `</main>`, with its nav link, holding the two corrected lists. One
heading per list, the methods grouped as she groups them — List 1 by cleaning, selection and
training; List 2 as one group. Each entry links to its block (`#mN`). Where an entry moved
between lists or came off, one short sentence under it saying what the results show. These two
lists are what she will use going forward, so they are the point of the section, not an
appendix to it.

Also update any block whose section 1 was wrong, in place.

## Do not

- Do not open TCGA (PAM50, lung, survival), ANNITIA or NHANES — patient data, excluded from the
  thesis and from every count. Leave them out when counting datasets.
- Do not propose new experiments, new settings or new sweeps. The job is to report what the
  recorded results say.
- Do not cut a method from a list because you expect it to lose. Degenerate, broken or
  undefined are reasons; a theory is not.
- Do not run cluster jobs. Nothing here needs fitting anything.
- Do not report a count of methods as a result in the thesis sense. In the audit, "raised the
  top on 2 of 5 datasets" is how the document already speaks and is fine there.

## When you are done

Say which entries moved and which stayed, and name the ones you could not settle because no
recorded result exists for that method on that kind of damaged label. The second group matters
as much as the first — she needs to know where the lists rest on nothing.

## Outcome, 2026-10-29

Done in one chat. The two lists are section `#s6` of `~/Documents/ch7_noise_methods_audit.html`,
with a nav link. Backup of the file as it was: `ch7_noise_methods_audit.html.before_what_worked_2026-10-29`.

- List 1: all 35 entries stay. Multistage cleaning and Seed averaging were checked against the
  result files (n09 summaries, n40 summaries) and both raised the top under non-censoring noise.
- List 2: Constant lift, Truncated-normal mean, Censored EM and Own lift stay. Delete clipped is
  added (n30: flab_thermo_tresanco and ginkgo_titer under censoring). The XGBoost
  accelerated-failure-time interval model is taken off: its only raise is abcov_kd, not a Chapter 7
  dataset, and it raised there with no censoring too.
- Blocks corrected in place: m44, m61, m55, m57, m73 (section 1 or 4 was wrong against the result
  files), m28 (comparison against the untreated top added). m42, m60, m61, m63, m64, m65 carry one
  sentence: the result belongs to the code as it ran.
- Checker scripts, not kept in the repo: per-pass "raised the top" from `*.rows.jsonl` (generic),
  from n09 `per_sigma`, from n40 and n30 summaries.
