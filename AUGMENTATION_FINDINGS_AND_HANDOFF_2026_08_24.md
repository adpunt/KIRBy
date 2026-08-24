# Adding extra features to a strong representation — what we found, what we changed, what is left

**2026-08-24.** Written as a handoff. Assumes you know how the KIRBy pipeline works but not the
internals of any particular function. Everything below is measured unless it says otherwise.

Companion document: `AUGMENTATION_STAGE_DESIGN_2026_08_21.md` (the code map and the plan this
work followed). That document's §4b/§4c results are superseded by this one.

---

## 1. The question

You have one strong representation of a molecule — a few thousand numbers per molecule that
already predicts the property well. You also have a handful of small extra feature blocks:
chemical fingerprints, 3D shape descriptors. Each extra block on its own is much worse than the
strong representation.

**Can we pick a few useful columns out of those extra blocks and bolt them onto the strong one
to make it better?**

The pipeline already had code that claimed to do this. Nobody had ever checked whether it
worked. This document is that check, plus the repairs.

---

## 2. The short version

The code did not work, and it failed in a specific and diagnosable way: it chose extra features
by testing them against one held-back slice of data, and kept whatever made that slice look
better. That is marking your own homework. It reliably found features that flattered the slice
and nothing else.

**Measured on hERG (1,412 molecules), the mechanism claimed a +0.021 improvement on its own
held-back slice and delivered −0.004 on data it had never touched.** On one fold the sign
flipped outright. Fed columns of **pure random numbers**, it accepted them and reported a
positive gain for doing so.

After rebuilding it, on logD (5,028 molecules) it now **claims +0.0087 and delivers +0.0083** —
honest to three decimal places — and improves 4 folds out of 5.

It still accepts random columns roughly 2 times in 5, and it still fails outright on the
antibody dataset. Section 7 explains why the antibody case is probably unwinnable rather than
merely unfixed.

---

## 3. What was actually wrong

Four separate defects, found by instrumenting the real code and watching it run.

**3.1 It marked its own homework.** To decide whether a candidate column helped, the code held
back 20% of the training rows, trained with and without the column, and kept the column if the
held-back rows looked better. One test, one slice, no repeat, no margin. Run fifty times in a
row, that procedure is guaranteed to find columns that suit that particular slice.

Evidence — the same three folds, what the mechanism claimed versus what it delivered:

| fold | claimed on its own slice | delivered on untouched data |
|---|---|---|
| 0 | +0.0125 | **−0.0152** |
| 1 | +0.0362 | +0.0066 |
| 2 | +0.0155 | −0.0037 |
| **mean** | **+0.0214** | **−0.0041** |

**3.2 The acceptance bar only ever went up.** Each time a column was accepted, the score it had
to beat rose permanently. So the first two or three lucky columns raised the bar above what
anything else could reach, and the loop then rejected everything and stopped early. It examined
only 15 to 19 of the 25 candidates it was offered.

This is where the puzzling "it only ever keeps 3 to 6 columns" came from. That number was a
property of the loop's shape, not a judgement about the features. **Fed pure random numbers, the
same loop still admitted 1.7 columns per fold on average and still reported a positive gain.**

**3.3 It judged candidates without ever looking at the strong representation.** Both places
where candidates got ranked fitted a small model to the candidate block *on its own* and ranked
by that. So "useful" meant "predicts the property by itself" — never "adds something the strong
representation does not already have". For fingerprint bits sitting next to a 1,024-number
learned embedding, the marginally-best bits are exactly the ones most likely to be already
captured.

There was a second ranking method in the codebase that *did* look at the strong representation.
It had **zero callers**. It had never run. When we compared the two, they picked almost entirely
different columns — an overlap of **1, 0, 0, 0 and 1 out of 20** across five folds. So the choice
of ranker completely determines what gets offered, and only the blind one had ever been used.

**3.4 The honest scoring pass split molecules at random.** When the pipeline rebuilt a chosen
recipe to score it fairly, it split molecules randomly instead of keeping chemically-related
molecules on the same side. So the recipe was chosen under one set of conditions and scored under
another. This is the pipeline's standard entry point for experiments, so this affected real
results, not just tests.

---

## 4. What we changed

**4.1 The choosing rule.** The accept-one-at-a-time loop is gone. In its place: rank the
candidates, then try five fixed amounts — nothing, 5, 10, 20, 40 — and keep whichever scores
best. **Keeping nothing is a real outcome and wins ties.** This shrinks the number of decisions
from roughly fifty sequential accepts to five, which is why it stopped actively hurting.

**4.2 The ranking.** The method that looks at the strong representation is now the default
everywhere. It was also fixed: it had been measuring what the strong representation got wrong
using the same rows it had just learned from, which for a strong model gives near-zero, mostly
meaningless numbers. It now measures that on rows held out from its own fitting.

**4.3 Information that was being dropped.** Five places were rebuilding recipes without the
scaffold grouping, and one was also missing the real scoring metric and the real model. All now
receive them.

### One thing we tried that made it worse

We first tried judging each of the five amounts on **several** slices of data and averaging,
which sounds obviously better. An adversarial check proved it was worse, with a runnable demo:
the extra slices validated on rows that the *ranking step* had already learned from, so two of
the three votes were contaminated and biased toward adding more columns, and they outvoted the
one clean vote. That version still accepted 5 columns of pure noise. **It was reverted.** Do not
re-propose averaging over extra slices unless those slices respect the scaffold grouping.

---

## 5. Did it work

Same measurement each time: what the mechanism claims on its own held-back slice, versus what
those columns actually deliver on data neither the ranking nor the choosing ever saw. Pure
random columns are the control — a working mechanism must keep zero of them.

| dataset | molecules | claims | **delivers** | folds improved | random columns kept |
|---|---|---|---|---|---|
| logD | 5,028 | +0.0087 | **+0.0083** | 4 of 5 | 2.0 |
| hERG | 1,412 | +0.0183 | +0.0032 | 2 of 3 | **0.0** |
| antibody thermostability | 672 | +0.0056 | **−0.0045** | 0 of 2 acted | 1.0 |

hERG before and after the rebuild, same folds, same representation:

| | before | after |
|---|---|---|
| claims | +0.0214 | +0.0183 |
| **delivers** | **−0.0041** | **+0.0032** |
| random columns kept | 1.7 | **0.0** |

**What is genuinely fixed:** on the largest dataset the mechanism is now honest — it delivers
what it promises. It no longer hurts anywhere. On one hERG fold it correctly decided to add
nothing at all, which the old loop could never do because its first few accepts were forced.

**What is not fixed:** random columns still get in, on 2 of 5 logD folds and 1 of 5 antibody
folds. The antibody dataset still delivers negative on both folds where the mechanism chose to
act.

**Caveat that matters:** +0.0083 on logD sits inside that dataset's own fold-to-fold spread of
0.0299. It is a consistent direction across folds, not a proven win.

---

## 6. A negative result worth keeping

Before the repairs, we tested whether the extra features were any good *at all*, with the broken
choosing loop taken out of the picture entirely — we simply forced 20 columns on and measured,
across five base widths and five folds on hERG.

**Selection never beat picking columns at random.** Averaged over widths, the ranker the code
used beat matched random columns by +0.0022, and the base-aware ranker came in at **−0.0059**,
i.e. worse than random. The single best cell in the whole table was the random control.

So on hERG the problem was never only the machinery. That dataset's extra blocks carry nothing
useful for that representation, and no choosing rule can extract signal that is not there.
This is the main reason hERG should not be the dataset anything is judged on.

**Retracted:** mid-run we reported that narrowing the base representation made the extras matter
more, with a clean gradient. That was one fold. Across five folds it disappears entirely. Do not
cite a base-width effect.

---

## 7. Why the antibody dataset probably cannot be fixed

Your real sample size is the number of distinct scaffold families, not the number of molecules.
Molecules in the same family are near-duplicates.

The antibody thermostability set has 672 antibodies and its fold-to-fold spread is **0.1546** —
roughly twenty times the size of the effect we are trying to detect. Working from the number of
families rather than rows, the level that pure junk reaches by luck is above the level a real
effect would reach. The signal sits underneath its own noise floor.

No method recovers a signal beneath its noise floor. **"Cannot distinguish from zero" is the
correct answer for this dataset, and reporting it as such is more defensible than producing a
number.** Note also that this dataset offers only a single 48-column extra block, so it is a
weaker test of the idea than hERG was, not a stronger one.

---

## 8. Where the datasets stand

Targets established under each dataset's real metric, 5 folds, 3 model seeds:

| dataset | best representation | score | fold spread | verdict |
|---|---|---|---|---|
| logD | mordred | 0.5066 (error) | 0.0299 | flat top |
| hERG | mhg_gnn | 0.5856 (error) | 0.0452 | flat top |
| antibody thermostability | saprot | 0.5562 (correlation) | 0.1546 | flat top |

**All three are flat-topped** — no single representation clearly dominates. logD's top two are
0.0004 apart, which is arbitrary. So **there is still no local dataset that can test the "one
strong representation, improved slightly" case** from the original brief. That is a finding about
the dataset roster, not about the mechanism, and it means this work can only speak to the
"signal spread across several representations" half of the question.

Only four datasets have representations cached locally. Only one of them is an antibody dataset.

---

## 9. Open decisions

**9.1 How to stop junk features getting through.** Five options from the literature search,
cheapest first. Full citations in §11.

| | approach | cost | catches redundancy? |
|---|---|---|---|
| 1 | **Do nothing** — add everything, let the model ignore what is useless | **less than now** | n/a |
| 2 | **Fix how the existing numbers are read** — pair errors per molecule within scaffold families, require a margin over "add nothing", require 4 of 5 folds | **free** | no |
| 3 | **Screen whole blocks conditionally** — strip out what the strong representation explains from both sides, cut at a threshold derived from the number of blocks and families | ~1.2x | **yes** |
| 4 | **Prevent the harm instead of detecting it** — fit the strong representation first, let a small second model correct only its mistakes | ~2x | no |
| 5 | **Remove each block, refit, measure the damage** | 5–7x | yes |

Recommendation: **2 now because it is free, 3 next, and treat 1 as the null hypothesis that has
to be beaten.** Option 1 deserves to be taken seriously — it is cheaper than what you do now and
the benchmark evidence behind it is stronger than expected.

**9.2 The budget-limited way of adding features.** There are two routines for adding extra
features: one lets the total feature count grow, the other keeps it inside your budget by
removing weak features to make room. Only the first was rebuilt. The second still contains every
defect in §3.

Deleting it is **not** a cleanup, because it is still live in four places: the precompute
pipeline emits it as a standard condition, 8 cached result files on disk name it, one federated
experiment reads it off saved recipes, and about 14 rows of the per-dataset policy table specify
it. Deleting it orphans those files and breaks any saved recipe naming it.

**The two have never been compared.** The only stored results for the budget-limited one come
from a 50-molecule smoke test. There is no evidence either way, and a comparison today would be
unfair to it because it still has the broken loop.

---

## 10. What is left to do

1. **Re-run on the server.** Every number in this document came from purpose-built test scripts,
   not the production pipeline. No stored result reflects any of these changes. The cache has
   already been cleared, so this is a full regeneration.
2. **Implement whichever option is chosen from §9.1.**
3. **Decide the fate of the budget-limited routine (§9.2).**
4. **Commit** the two files still holding uncommitted changes.

### The cheapest experiment that settles §9.1

One run, one dataset (logD), four arms. Plant two decoys in the candidate pool: a block of pure
random numbers, and — this is the one currently missing — **a lightly-jittered copy of something
already inside the strong representation.** Only the second decoy can catch a rule that rewards
duplication; no amount of random noise will reveal it.

Arms: (a) no selection at all, (b) the current rule, (c) the current rule plus the free fixes
from option 2, (d) conditional block screening from option 3. Read three numbers per arm:
delivered improvement on untouched data, how often each decoy got accepted, and how often the arm
added nothing.

If (a) matches (d) on delivered improvement, the honest conclusion is that selection buys no
accuracy and is only worth its cost for the scientific claim — and writing that down is a
stronger result than a selector that accepts noise 2 folds in 5.

---

## 11. Sources

Behind option 1 (do nothing): Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 (45-dataset tabular
benchmark — boosted-tree accuracy barely moves when half the columns are removed);
Cherepanova et al., NeurIPS 2023 (feature-selection benchmark with deliberately injected junk
columns — no selection method reliably beat keeping everything).

Behind option 2: Breiman, Friedman, Olshen & Stone 1984 (the one-standard-error rule);
Nadeau & Bengio, *Machine Learning* 52:239, 2003 (any spread estimated from cross-validation
results alone is too small, because training sets overlap — with the correction);
Krstajic et al., *Journal of Cheminformatics*, 2014 (closest published match to this setting —
molecular property datasets at this size — found no single cross-validation run reliably picks
the best model); Diebold & Mariano 1995 (test paired per-item losses, not summary scores).

Behind option 3: the Frisch–Waugh–Lovell theorem (stripping only one side leaves bias);
Chernozhukov et al. 2018 (double machine learning — double stripping plus cross-fitting is what
removes the bias a strong flexible base model introduces); Fan & Lv, *JRSS-B*, 2008 (screening,
and its two failure modes); Donoho & Johnstone 1994 (the analytic noise-floor threshold).

Behind option 4: Klau et al., *BMC Bioinformatics*, 2018 (priority-Lasso); Hornung & Wright,
*BMC Bioinformatics*, 2019 (Block Forests); the sceptical companion is Herrmann et al.,
*Briefings in Bioinformatics*, 2021 (across many multi-omics datasets, block-aware methods rarely
beat the strong block alone by much — which matches the tiny effects seen here).

Behind option 5: Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, *JASA*, 2018 (leave-one-out
refitting with distribution-free intervals); Hooker, Mentch & Zhou, *Statistics and Computing*,
2021 (any trustworthy importance measure requires refitting).

**Deliberately excluded, with reasons.** *Boruta* (Kursa & Rudnicki, 2010) is built to keep every
feature related to the property including ones that duplicate what you already have — the
opposite of the question here — and Melikechi, Dunson & Miller, *Bioinformatics* 41(5), 2025,
measured its false-discovery rate above 0.75 when targeting 0.1. *Model-X knockoffs* (Candès,
Fan, Janson & Lv, *JRSS-B*, 2018) have a hard arithmetic floor: at a 10% target they return
either ten-or-more features or exactly zero, so "three columns help slightly" outputs zero
forever; and the guarantee requires knowing how the candidate columns are jointly distributed,
which for sparse binary fingerprints breaks silently. *Permutation importance and SHAP* hold the
model fixed, so they report whether a model uses a column, never whether its presence during
training degraded the fit (Strobl et al. 2008; Hooker et al. 2021; Kumar et al., ICML 2020).

**An honest disagreement in the search.** One reviewer rated a 2025 stability-selection variant
(Melikechi, Dunson & Miller) the best available fit, on the strength of the only published
head-to-head at this problem shape — tree-based importance, correlated features, a few hundred
rows — where it recovered 70% of true features at a 10% false-discovery rate. A second reviewer
simulated the older form at these effect sizes and found it underpowered. The reconciliation:
the published result is for choosing among hundreds of individual columns, whereas the decision
here is over four to eight blocks, and a "false discovery rate" over five things is not a
meaningful quantity. Recorded rather than resolved.

---

## 12. Where things live

Measurement scripts (purpose-built, local, not part of the pipeline):
`scratchpad/aug_pass1_target.py` (establishes the best representation per dataset),
`scratchpad/aug_mechanism_autopsy.py` (instruments the old choosing loop),
`scratchpad/aug_dilution_factorial.py` (the §6 negative result),
`scratchpad/aug_fix_verify.py` (the §5 before-and-after),
`scratchpad/run_c21_scoped.py` (runs the existing experiment pinned to one representation).

Local caveat: on this machine the first model fit after loading data crashes the interpreter
outright, with no error message, because two copies of a threading library get loaded. Every
script above works around it by touching the model library before loading data. This is a laptop
problem, not a server problem, which is why the workaround lives in the scripts and not in the
pipeline.
