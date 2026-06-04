# Decisions for you — overnight fix-fleet, 2026-06-04

12 agents went through every issue in `MASTER_ISSUES_2026_06_04.md`, reading the real code + audit docs (not memory) and smoke-testing locally. Full technical detail is in `SYNTHESIS_2026_06_04.md`. This file is the plain-English "what needs your eyes" list.

**Nothing has been committed.** Everything sits in the working tree. Nothing reaches the cluster until you commit → push → resubmit, so you are the sign-off gate. I did **not** change any speed gate or search-space knob — those are all written up in Section 3 for you to approve.

Big picture: a large fraction of the audited bugs were **already fixed** in committed history (the agents verified this against current code, not memory). The genuinely new work is: a handful of fresh bug fixes, three or four results-affecting fixes to confirm, several brand-new problems the agents caught, and a stack of cost levers waiting on your OK.

---

## 0. DO THIS FIRST — the cache fix isn't committed (deploy-blocker)

The fix that stops the endless cache-churn on neural-network representations (chembert, chemberta, graphmvp, grover, molformer, smited) **lives only in the working tree**. The committed code (`HEAD`) still has the old version that rejects those caches on every run. If the cluster deploy does a clean checkout, or any other session runs `git stash`/`git checkout`, **the churn comes back in full** — this is the mechanism that's cost you months.

Two things the agent also discovered and fixed (both were incomplete before):
- The staged fix only covered the *search* cache. The *baseline* cache (per representation+model) keyed on the same unstable hash, so neural reps still churned there. Now fixed too (`find_best_hybrid.py:186-200`).
- The "no-augmentation collision" fix (`a13e0f4`) only patched the 9 caller files, not the core. Any new no-aug caller could still collide. Now fixed in the core (`find_best_hybrid.py:3958-3978`).

**Action: commit `src/kirby/find_best_hybrid.py` before the re-run.** This is the single highest-value thing for the churn problem. I haven't committed it (you commit when you're ready).

---

## 1. Fixed automatically — no decision needed (FYI)

These are pure correctness fixes (crashes, wrong-task-type metrics, leakage filters, cache-key bugs, dead code). They don't change results for correct runs — they stop crashes and wrong numbers. Highlights:

- **A bug that would have silently poisoned the Graphinity antibody experiment** (`graphinity.py`): a recent "multi-letter chain" fix was greedy and stole digits from the mutation position — `YH102A` was being read as chain "H10", position 2. That corrupts *every* mutation-aware representation. Fixed to read chain/position correctly.
- **c24 (blueprint transfer) and federated condition C6 both already run now** — they used to produce nothing (a crash and a missing import); verified resolved in committed history.
- **c08 (calibration)** crashed on any classification model without probability output; **c07** could crash on a search failure; **c22 (explainability)** silently corrupted its SHAP importances for binary classification and scored on hard labels — all fixed.
- **OPIG**: InChIKey de-duplication added to `chembl_qsar`; loud warnings instead of silent fallbacks in humanness/ginkgo; per-fold errors now printed instead of swallowed (so a dropped fold that biases the mean is visible).
- **Federated**: added the antibody classification metrics (accuracy, weighted-F1), removed a memory-bomb fallback that concatenated all representations, fixed KMeans internal-state staleness, and made every condition print a real traceback on failure instead of a one-line message.
- **Similarity**: importance values are now clamped to ≥0 before distance weighting (signed importance was producing all-NaN distance matrices); s09 now flags a divide-by-zero instead of silently dropping a dataset; s10 now surfaces a failed hybrid arm instead of pretending it ran.
- **Shared infrastructure**: the silent random-split fallback is gone (now raises a clear error — see 2e); the mutual-information estimator is confirmed working (correlated data → MI 3.35, independent → 0.0).
- **Noise (n01–n13): zero changes needed** — every audited bug was already fixed; verified with 8 green smoke tests.

---

## 2. Results-affecting fixes — APPLIED, but confirm before you re-run

These are defensible bug fixes, but they **change numbers**, so per your "no methodology change without sign-off" rule I'm flagging each. They're in the working tree; tell me to revert any and I'll do it precisely. (To inspect: `git diff <file>`.)

**2a. c09 (formal analysis) — internal cross-validation changed from 3-fold to 5-fold.** `c09_formal_analysis.py`. Two of c09's sub-analyses used 3-fold while the rest of the suite uses 5-fold. That inconsistency *also* caused a guaranteed cache miss → ~120 full recomputes every run. The fix aligns them to 5-fold, which both removes the waste and makes the numbers consistent. **Changes c09's 9b/9c numbers.** Recommend: keep (it fixes a real waste and an inconsistency). Reject only if you deliberately wanted 3-fold there.

**2b. s11 (Bayesian optimization) — the rebuild step now matches its own blueprint.** `s11_bayesian_optimization.py`. When s11 periodically rebuilds its hybrid during the optimization, it was hardcoding the augmentation strategy to "uncapped" regardless of what the blueprint it's testing actually used. Now it uses the blueprint's own strategy. **Only changes results when you run s11 with `--augment`** (the molecular datasets carry augmentation reps). Recommend: keep (running the optimization on a differently-built hybrid than the blueprint was the bug).

**2c. Distance-tier evaluator list — Ridge removed.** `_shared.py:268`. The distance-based hybrid search used six internal scorers, one of which was Ridge (which you've banned as an internal evaluator). It's now five (knn, rf, mlp, lightgbm, svm). **This changes which hybrid gets selected in the distance tier AND invalidates the existing distance-tier cache** (the evaluator list is part of the cache key), so those get recomputed on the next run. Recommend: keep (matches your no-Ridge rule), but you should know about the cache rebuild. If you'd rather keep Ridge there for now to avoid the recompute, say so.

**2d. c18 (diversity vs depth) — budget sweep was stuck on [50, 100].** `c18_diversity_vs_depth.py`. A bug meant c18 ignored the configured budgets and always swept just 50 and 100. The fix makes it honor the configured budgets (currently [50, 500, 1000, 2000]). **Changes c18's results** (wider, correct budget sweep). Recommend: keep.

**2e. Random-split fallback now raises instead of silently random-splitting.** `_shared.py`. This enforces your non-negotiable no-random-splits rule. **Consequence to watch:** c05 (learning curves), c07 (data-complexity partitions), and s11 take subsamples/sub-populations that can drop the number of groups below the fold count. They'll now *raise* there instead of silently random-splitting — and because some of those experiments wrap their fold loop in a broad `except`, the error could be swallowed and show up as an empty result rather than a clear failure. **This needs a companion fix** (see 4d) so the re-run doesn't silently produce empty results for those.

**2f. Federated experiments now use a classifier on classification data.** `fed03`, `fed05` previously always used a regressor regardless of task. Now task-aware. Correct, but it changes the classification federated numbers.

**Proven-neutral (kept, no decision):** c17's mutual-information selection (mRMR) is now computed once and sliced per budget instead of recomputed from scratch every budget. The agent proved it produces *bit-for-bit identical* selections — so it's a free speed-up on c17's most expensive step, with no change to results.

---

## 3. Speed/cost levers — NOT applied, your call

I changed **none** of these (your rule). Each is a way to cut compute; all change results or behaviour, so they need your OK. **Important re-prioritisation:** the old "SVM is the 302-hour monster" picture is **stale** — SVM is already budget-capped and is now only 0.1–5% of the cost. The real cost driver today is **GP (Gaussian Process)**.

**3a. GP budget cap (the top lever now).** GP is 28–61% of the per-search wall time on small/medium datasets, because it's only capped above 1000 samples and runs unconstrained below that — exactly the small-antibody regime. Capping GP's max budget from 1000 to ~250 (the code's own earlier calibration says GP only wins at budget ≤250 anyway) is estimated to cut **20–40% of per-search wall time** on datasets under 1000 rows. Recommend: do it. (`find_best_hybrid.py:1720`)

**3b. SVM sample cap.** Lower `_max_samples` for SVM from 10000 to ~3000–5000 to kill its quadratic cost on the biggest datasets. Small win now (SVM already tamed); low priority.

**3c. Loop ordering — compute the expensive variants first.** Right now models run cheap→expensive and budgets run small→large, so a wall-clock timeout always kills the GP/SVM and the biggest-budget cells — biasing "best model" toward cheap models and "best budget" toward small. Reordering expensive-first means a timeout sacrifices the cheap, fast-to-redo cells instead. No compute saved; you just keep the results that matter when a job times out. (I *did* improve the precompute's existing reverse-order flag to be a proper cost-sort — it's opt-in and off by default, so it changes nothing unless you set `KIRBY_PRECOMPUTE_EXPENSIVE_FIRST=1`; this directly answers your earlier "can the reverse flag be improved?" question.)

**3d. Capped-augmentation search cost.** The "capped" augmentation strategy refits the model up to 50 times per configuration and re-runs a permutation-importance pass per candidate — it dominates the capped tier. Computing that importance pass once (instead of per candidate) plus lowering the candidate cap 50→20 is estimated at 3–5× faster on the capped tier. Changes which features get selected. (`hybrid.py:1979`)

**3e. c23 (evaluator–model matching) extra-model tier.** c23 runs CatBoost/DNN/CNN as both downstream models and evaluators, which caused a runtime explosion on the bigger datasets. Recommend gating those 3 expensive models behind an opt-in flag (default off) and keeping the 8 cheap ones — the matched-vs-mismatched question is the same, and the neural result is already known to be near-noise. (`c23`)

**3f. c17 mutual-information over all samples / large prefilter.** Two more c17 levers beyond the free one in Section 2: (i) the MI estimate uses every row though it converges by ~2000 — subsampling to 2000 rows makes cost flat instead of growing with dataset size; (ii) the feature prefilter keeps 2000 features when 500 barely changes the result. Both change the numbers slightly. With the free once-slice fix already applied, these are now lower-urgency.

**3g. s01 / s03 (similarity) large-n cost.** s01 builds a full N×N distance matrix per representation before subsampling (≈7 GB each at 30k rows → out-of-memory risk on nucleotide/TCGA). s03's mutual-information kNN is quadratic (~111 h at 30k rows — the biggest similarity cost). Recommend: a large-N skip+flag for s01, and capping s03's MI estimate at ~2000 rows. Both change behaviour.

**3h. c05 (learning curves) repeats.** c05 redraws each training-size point 3 times for error bars. Two options: drop to 1 repeat for a fast screening pass (loses error bars); and separately, the full-data point currently runs the *identical* search 3 times under 3 different cache names — de-duplicating that to 1 is a provably free 3× saving at that point (left unapplied because it touches the core sampling loop).

**3i. Screening vs full budgets.** The wave precomputes at the full 8 budgets (this is fine — the precompute is *not* screened, contrary to earlier worry). But 7 experiments (c06, c07, c09, c10, c15, c18, s03) sweep only the 4 "screening" budgets internally, so their "best budget" is coarse. Switching them to the full 8 is more accurate at a modest cost (extra downstream fits only, not a search rebuild). Your call.

---

## 4. New problems the agents found — need a decision

**4a. n10 (noise + hybrid neighborhood) will silently produce empty hybrid results.** This is the most important new finding for the re-run. n10 asks for hybrid blueprints at budgets [25, 100, 250], but the noise precompute only builds [50, 500, 1000, 2000]. n10 has a strict guard that *skips* any budget it can't match — so all three of its hybrid budgets get skipped, and n10 reports only the baseline (no hybrid-vs-baseline comparison, which is the entire point) while *appearing* to succeed. Fix: either add 25/100/250 to the noise precompute budgets, or change n10's budgets to ones that are precomputed (e.g. [50, 500, 1000]). Recommend the latter (cheaper). Not applied (it's a budget/result choice).

**4b. s11 will crash on molecular datasets that have an ECFP4 representation.** s11's custom Tanimoto kernel is incompatible with the installed scikit-learn (it raises `AttributeError` inside the GP fit). The skip logic only skips that condition when ECFP4 is *absent*; when it's present (the intended molecular runs) the condition runs and crashes the experiment. Fix: either make the kernel expose the hyperparameters scikit-learn expects, or turn that condition off entirely. Check the server's scikit-learn version first. Not applied (needs the server version).

**4c. OPIG precompute bypass — the "stuck in survey stage" root cause.** Several OPIG experiments search under a dataset name the precompute step never seeds, so every search is a cold recompute. The unregistered names are: `chembl_qsar`, `cdr_flexibility`, `humanness_species`, `ginkgo_tm2`, `ginkgo_aggregation`, `ginkgo_purity`, `graphinity_foldx`, `graphinity_rosetta`. Fix: register those 8 in `DATASET_REPS` (copying the rep list from a sibling) and add them to the precompute submission. (A few other cases — CGR, Kingsmith, blueprint transfer — are genuinely per-slice and can't share a precompute; those are cold by design.) Not applied (it edits `DATASET_REPS`, your call).

**4d. Companion fix for the no-random-splits enforcement (from 2e).** c05, c07, and s11 need to gracefully reduce the fold count to the number of available groups, or skip with an explicit flag, when a subsample/sub-population has too few groups — otherwise the new "raise instead of random-split" behaviour will surface as crashes or swallowed empty results on the re-run. This is a small, safe follow-up I can do next session (it does *not* reinstate random splitting).

**4e. s11 internal random split.** s11 uses a shuffled (non-group-aware) KFold to pick which representation seeds its optimization surrogate (`s11:552`). It's not the final evaluation split, but it's technically a random split. Leave it (it's internal to the optimization simulation) or make it group-aware? Your call — not changed.

**4f. Noise classification estimators.** Several noise-estimation methods in n08/n11 fit a *regressor* on integer class labels for classification datasets — they don't crash, but the noise score is meaningless there. Recommend dropping those methods on classification datasets (cost-neutral; just removes meaningless rows). Your call.

**4g. fed05 still uses a single 80/20 split.** fed01–fed04 are now proper 5-fold with error bars; fed05 (the antibody extension) still uses one split with one seed on tiny silos (~146 antibodies), so its numbers are the noisiest possible. Converting it to k-fold is a real restructure (the cross-silo conditions share one split). Recommend doing it before quoting any fed05 number; I flagged it as a larger task rather than rushing it tonight.

---

## 5. Estimate corrections — your time model was partly wrong (good news, mostly)

Three things in the estimate work turned out to be **false alarms**, which *shrinks* the wasted-compute picture:

- **"SCREENING vs FULL forces a full cache rebuild" — FALSE.** The precompute function ignores the screening setting entirely and always builds the full 8 budgets. There are *not* two cache families and there is *no* cold rebuild from this. (Its docstring lies about reading the budget config — that's the trap that fooled the earlier analysis.) The screening setting only coarsens the budget curve for 7 experiments (Section 3i).
- **"cx and sim each re-run the precompute (≈2×)" — FALSE for sequential runs.** They share the cache by filename, independent of which suite calls it. You only pay 2× if the cx and sim jobs for the same dataset run at the *same time* before either finishes writing. Fix = run the precompute job to completion first, or sequence cx before sim.
- **The distance tier IS precomputed** (an audit doc said it wasn't — stale). No on-demand latency problem.

And one correction that **re-prioritises** the cost work:
- **GP, not SVM, is the cost driver** (Section 3a). The 302-hour SVM figure predates the budget cap that already tamed it.

These belong in `ESTIMATES_INVESTIGATION_2026_06_04.md` (the CACHE-3 / COST-11 / TIME-1 / TIME-7 lines should be downgraded, and the COST-6 SVM line marked stale in favour of the GP lever). I've noted them in `SYNTHESIS_2026_06_04.md`; say the word and I'll edit the estimates doc directly.

---

## 6. How the working tree is organised

- **Safe to commit now:** everything in Section 1 (pure correctness) + the deploy-blocker in Section 0.
- **Review before committing:** Section 2 (results-affecting — confirm or tell me to revert).
- **Nothing from Section 3 is in the tree** (cost levers — all write-ups, awaiting your OK).
- The opt-in expensive-first precompute flag (Section 3c) is in the tree but off by default — changes nothing unless you set the env var.
- Backups / detail: `SYNTHESIS_2026_06_04.md` (every agent's full report), `MASTER_ISSUES_2026_06_04.md` (the source list).

If you want, next session I can: (1) apply whichever Section 3 levers you approve, (2) do the 4d companion fix, (3) register the 4c datasets, (4) fold the Section 5 corrections into the estimates doc, and (5) commit + push the approved set.

---

## 7. What I did NOT touch tonight (and why — so it's not mistaken for "missed")

Three slices of `MASTER_ISSUES` are deliberately out of scope for a pre-re-run bug pass, because they aren't bugs to fix — they're either the re-run itself or multi-day feature work:

- **Forward-compatibility models (FWD-1…24):** "integrate MINT / ESM3 / Boltz-2 / SaProt / MolE / TabPFN / …". Each is a multi-day model integration, not a bug. Separate roadmap; untouched.
- **Phase-5 / expansion items (ISS-66…85):** "re-run all surveys + rebuild the Bill of Materials", new TCGA tiers, new datasets. The survey re-run IS the thing you're about to do; rebuilding the BOM comes after. Not pre-run fixable.
- **Results-validity findings (RES-1/2/49 etc.):** "the deep experiments ran on the smallest datasets → re-run on powered datasets." These are conclusions that *require* the re-run on bigger data — they're an outcome of running, not a code fix. The underlying *code* bugs they reference (the BRK-* / CXA-* / SIMA-* crashes and wrong-metrics) were all handled by the agents.

Everything that was an actual code bug, leakage, crash, wrong-metric, cache-correctness issue, or cost lever **was** worked — fixed (Sections 1–2), or written up for your sign-off (Sections 3–4).
