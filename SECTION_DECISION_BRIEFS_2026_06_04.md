# Section decision briefs — prepped for your return (2026-06-04)

Plain-English decision briefs, built section-by-section by read-only investigators (code-verified, not memory). For each decision: **purpose** (what it's for) · **what's wrong** · **options** (with a recommendation). Nothing here was acted on autonomously — these await your call.

**Progress:** §1 A–D ✅ closed · §4 E–H ✅ closed · §2, §3, §5, §6, §7, §8 ✅ **all prepped below** (read-only; awaiting your decisions). See the CONSOLIDATED summary at the bottom.

---

# §3 — FWD + ISS (122 IDs)

Of 122, ~72 are already-done-before-this-session (no action). The live content is below.

## A. The 12 committed-without-your-sign-off (UNAUTH) items — keep or revert

**Only 4 of the 12 change anything. The other 8 are dead or inert.**

### The 4 that matter
1. **ISS-29 — antibody near-duplicate grouping** (`antibody_baseline_survey.py:314` → `sequence_identity_clusters`, threshold 0.95)
   - *Purpose:* stop near-identical antibodies (single-residue analogs) landing in both the train and test halves of a fold. It was leaking — 87% of near-duplicate pairs spanned the split on one dataset, which inflates scores.
   - *What's wrong:* committed without your OK, and it **changes every antibody number**.
   - *Options:* **keep** (it enforces your own no-leakage rule — right content, only the sign-off was skipped) or revert to old exact-match grouping. Clean revert either way.
   - *Recommend:* **keep.**
2. **ISS-84 — balanced folds for yes/no tasks** (`find_best_hybrid.py:_make_cv_splitter`, StratifiedGroupKFold for classification)
   - *Purpose:* keep the class mix even across folds for classification (TCGA, antibody clf) so the scores are less noisy.
   - *What's wrong:* committed without OK; **changes classification numbers**.
   - *Options:* **keep** (standard, defensible) or revert to plain grouped folds. Clean revert.
   - *Recommend:* **keep.**
3. **IgBlend (FWD-15) — the one with active harm** (`antibody.py:1307` stub raises at `:1331`, but registered in the default list at `:1708/1727`)
   - *Purpose:* a future antibody model (sequence + 3D structure).
   - *What's wrong:* it's an empty placeholder that **errors every time**, and it's in the default antibody run-list — so every default antibody survey tries it and fails on it (caught per-rep, but wastes a slot + logs an error).
   - *Options:* **remove it from the default list** (or delete the stub) / leave it erroring.
   - *Recommend:* **remove from the default list** (it's the only UNAUTH item doing active damage).
4. **ProSST (FWD-17) — the one that actually works** (`protein.py:2892`, registered `antibody.py:1706`)
   - *Purpose:* a structure-aware protein model.
   - *What's wrong:* committed without OK, AND it runs **sequence-only** (the structure input is fed blanks), so despite the name it is not structure-aware — could mislead if read as such.
   - *Options:* keep (a working sequence rep) / keep but clearly flag the sequence-only limitation / revert.
   - *Recommend:* **keep + flag the limitation** in the rep name/notes.

### The 8 inert ones (keep-or-delete, **zero results impact** either way)
- **ISS-71** Boltz-molecular stub (errors if called; nothing calls it)
- **ISS-85** Butina molecular split (an option, off by default, nothing uses it)
- **6 dead protein-model placeholders** — ESM-C 6B, AlphaFold3, Chai-1, Protenix, S3F, Boltz-2-protein — none is reachable by any survey (no dispatch entry); they're disconnected dead code.
- *Recommend:* **revert all 8** for a clean, signed-off tree. Keeping them is harmless (they never run) but leaves un-signed-off dead code. (Boltz-2-protein is the only one with real working code, but nothing calls it and DTA deliberately excluded it — keep its function only if you want the future capability.)

## B. 6 thesis-writing decisions (no code) — ISS-30, ISS-76–80
How to present known/expected results to a reviewer. No code change — these are write-up calls for when you draft the thesis:
- **ISS-30:** the ΔΔG (SKEMPI) split already isolates each protein complex to one fold; a deeper ~56% structural overlap can't be removed with so few complexes → document it as a known caveat.
- **ISS-76–80:** five "looks bad but is expected" cases — your cross-validation scores run slightly higher than published single-split numbers; sequence-only ΔΔG has a real low ceiling (~0.25–0.50 vs ~0.73 structure-based); many antibody tasks are genuinely near-unpredictable (matches recent benchmarks). Decide how to frame each.

## C. Feature-gaps — now-or-never
- **Already DONE (the "not run" label is stale):** MINT, SaProt, ESM3, IgBERT, GAUCHE GP — coded **and** have committed results. Cross off.
- **Coded + ready, but need the cluster rerun to produce numbers:** MolE, MiniMol, the cancer-imaging reps (TITAN/Virchow2/CONCH); plus a trivial one-line add of Boltz-molecular into the molecular run-list. These only matter once Phase-5 reruns on ARC.
- **Genuinely big new work → mark "not in this pass" (never, for now):** TabPFN, a per-residue CNN (the current CNN is the "wrong"/fixed-vector kind — architecturally blocked), PaDEL (needs a Java toolkit; RDKit already covers it), Yield-BERT (existing reaction reps cover it), MOJO joint-omics, AlphaFold reps, Kermut GP (GAUCHE already covers the chemistry-kernel-GP idea), S2ALM (also needs a weight-source decision — public weights unconfirmed). Each is a new dependency/loader, repeatedly deferred, rated optional in the research docs.

## D. The one master lever
**Phase-5 rerun on ARC (ISS-81/82/83)** — server-only, you drive it. This is what turns every "coded + ready" rep in bucket C into actual numbers. The single highest-value action for §3 is scheduling that rerun.

## §3 decisions waiting for you
1. The 4 live UNAUTH: keep ISS-29 / ISS-84 / ProSST(flagged)? remove IgBlend from default? (my rec: yes to all four)
2. The 8 inert UNAUTH: revert-all for a clean tree, or leave?
3. Confirm the big feature-gaps are "not in this pass" (never, for now).
4. Thesis-writing items (B) noted for drafting — no action now.

---

# §2 — T / ORD / LIN (57 IDs)

Already settled (not re-litigated): TIME-c23 = **reverted** this session (working tree shows the c23 gate removed); screening→FULL = **kept**.

**Exactly ONE genuine decision in this whole section:**

**The 7 "expensive-first" loop-ordering changes (ORD-1/2/3/4/NEST/KNOB/REVFLAG) — UNAUTH, but treat as ONE decision.**
- *Purpose:* today the pipeline always runs the most expensive variant last (slowest model, biggest budget, costliest precompute condition). If a job hits the 7-day cluster wall, the casualty is always the expensive thing you most wanted. The change adds an **optional** "do expensive work first" mode so a timeout drops the cheap, easy-to-redo work instead.
- *What's wrong:* committed without the discussion you asked for. **But:** verified opt-in / **default-OFF** (`find_best_hybrid.py:3115`, `precompute_hybrids.py:473` — only fires if an env var is set; no SLURM script sets it), and **result-neutral on any run that completes** (the winner is chosen by order-independent `max()`/`sorted()` over the full results set; runtime-asserted set-preserving + unit-tested). It only changes *which partial results survive a timeout* — which is the whole point.
- *Options:* **keep** (leave the flags OFF; gain a timeout-mitigation lever; discuss *when* to flip) / revert (delete the two sort blocks). 
- *Recommend:* **keep, leave OFF.** Pure process issue; the artifact is benign dead code until switched on.

**No code decision (informational):**
- TIME-c17 once-slice: a bit-identical speedup (mRMR computed once, sliced per budget — proven prefix-monotone). Keep, informational.
- 18 FLAGGED timing write-ups + 15 LIN estimate-caveats = **no code**; cost notes and "trust these hours as order-of-magnitude" caveats.

*(Caveat: `DECISIONS_FOR_USER.md` wrongly says loop-ordering is "not applied" — it IS in the commit, default-OFF. That doc is unreliable; this brief supersedes it.)*

---

# §8 — CXA / SIMA / NA (61 IDs)

Already settled: SIMA-20 (Ridge removed) = KEEP; SIMA-16 (similarity screening→FULL) = KEEP.

**Live items — all in the "rule-mandated / correctness, recommend KEEP" family:**
1. **CXA-12 — refuse to random-split groupless data** (`find_best_hybrid.py:_make_cv_splitter`). This is the *other half* of the EVAL-3 fix already done this session: it makes the multi-fold path raise instead of falling back to a random split when there are no groups. Mandated (no-random rule), only converts silent-wrong-numbers into a clear crash, never changes a properly-grouped result. **Recommend KEEP.**
2. **SIMA-5 — s11 honours the blueprint's recipe** (`s11:1337-1399`). The Bayesian-optimization experiment rebuilds its feature space mid-run; the old code silently used different defaults (step size 5, forced "uncapped" augmentation) instead of the blueprint's actual settings. Correctness fix. **Results-changing only with `--augment`** (those s11 augmented numbers are stale → need a rerun); non-augment unaffected. **Recommend KEEP.**
3. **SIMA-10 — s11 handles degenerate splits per seed** (`s11:979-1018`). On tiny sub-populations the group-aware split can fail; the fix skips-and-flags that seed (and labels an all-failed dataset honestly) instead of crashing or faking a curve. No healthy result moves. **Recommend KEEP.**
4. **★ n10/4a — NOT botched (verified, reversing the earlier tag).** Earlier this was tagged a botch ("applied the expensive option"). **Code-verified false:** n10 requests budgets `[25,100,250]` (`n10:74`), all of which are in `FULL_BUDGETS` (`_shared.py:259`), and `precompute_hybrid_searches` *always* builds FULL_BUDGETS regardless of config (`_shared.py:2680-2687`). So n10's budgets already match precompute, and the `SCREENING→FULL` config edit adds **zero** precompute cost. The "expensive/botched" tag rested on a false premise (that precompute reads `config['budgets']` — it doesn't). **Action: keep the code, correct the stale doc.** No code change, no decision really needed beyond acknowledging the correction.

---

# §5 — RES / MISS / OTHER / DEAD (149 IDs)

**Effectively ZERO code actions.** Verified: no unauthorized and no botched change hides here (the LEGIT-FIX items — graphinity regex, n08 format crash, c08 clf guard — are clean; the one nearby results-changing item, c09 3→5 fold, is booked under §7/CX, not here).
- **Feature-gaps → all "not in this pass":** MISS-1..18, 22, 23 (WSI, joint foundation models, multi-omics fusion, TCGA modality reps, SALI, inverse folding, the `finetune_antibody_lm` wrapper). Each is big new work — and most are **already partly/fully built** (e.g. TITAN WSI + BindingDB_Kd already have results), so "not in this pass" mostly means "already there, nothing to do."
- **55 run-dependent** (e.g. "c02 VEGF −0.027", "c08 underfit", "c24 errored") = results-validity conclusions whose underlying code bugs are already fixed; **only the ARC rerun produces the corrected numbers.**
- **2 REFUTED** (MISS-21 nucleotide-in-cache-exclusion, DEAD-53 fed herg branch live) = code-verified non-bugs.

---

# §6 — FED + OPIG (62 IDs)

**2 results-changing items to confirm (both rule-mandated, recommend KEEP):**
1. **FED-3-tail — fed03 Ridge→RandomForest** in silo feature-selection (`fed03:813-837`). The old code silently fell back to Ridge (the forbidden evaluator); the fix uses RF. Mandated by the no-ridge rule (which says replace silently, don't ask). Results-changing (different features selected), clean revert. **Recommend KEEP.**
2. **OPIG-19 — chembl_qsar InChIKey dedup** (`chembl_qsar.py:99,227`). Adds molecule de-duplication before splitting, closing a train/test duplicate-compound leak. Mandated by the standing dedup rule. Results-changing (n shrinks), clean revert. **Recommend KEEP.**

**OOS federated/OPIG items — nothing now-or-never:** FED-13/14 already documented; FED-30 is a non-bug (live dispatch, REFUTED); FED-20 / FED-22-dead-field / OPIG-20 / OPIG-26 are optional cosmetic smells; FED-23/24/31 + the FED-22 model-catalogue are design/scope changes that need a rerun → mark "not in this pass." **Federated files are active work — no deletions.**

---

# §7 — PCB + BRK

**Confirmed clean.** PCB-19 = the c09 3→5 fold change (= CACHE-5, already decided KEEP). **Zero other unauthorized, zero botched** in PCB/BRK. The PCB cache-key cluster (`_make_search_meta`) and the BRK mutual-information-collapse fixes are byte-identical to the parent commit — i.e. genuinely fixed *before* this session, not new. The REFUTED items (PCB-26 `budget=` is live, BRK-11 s04 molecular-gate is intentional) are real non-bugs. No decision needed.

---

# ════ CONSOLIDATED — what's actually waiting for you across all 8 sections ════

The trust-critical "~16 UNAUTH + 1 botched" review resolves much smaller than feared:

**A. Rule-mandated keeps (content right, only the per-item sign-off was skipped) — recommend KEEP each:**
- ISS-29 antibody near-dup grouping (no-leakage) · ISS-84 balanced folds for clf (standard) · SIMA-20 ridge-removed (no-ridge, already KEEP) · CXA-12 + EVAL-3 random→raise (no-random) · SIMA-5 s11 blueprint recipe · FED-3 ridge→rf (no-ridge) · OPIG-19 dedup (no-leakage). Several are results-changing → the affected datasets need the rerun, but the changes are *your own rules*.

**B. Inert / non-functional (no results impact) — revert-for-clean-tree OR leave:**
- 8 forward-compat scaffolds (only ProSST works; 6 are unreachable dead code; Boltz-protein orphaned) · ISS-71 Boltz-mol stub · ISS-85 Butina option · the 7 loop-ordering sorts (default-OFF) · TIME-c17 once-slice (bit-identical) · SIMA-10 (failure-path only).

**C. The ONE active-harm item — recommend a small fix:**
- **IgBlend** is registered in the default antibody run-list but only errors → remove it from the default list (or delete the stub).

**D. False alarm cleared:**
- **n10/4a is NOT botched** (verified) — keep code, fix docs. (So: **zero** genuine botches across all 413 IDs.)

**E. Feature-gaps → "not in this pass" (most already built):** TabPFN, per-residue CNN, PaDEL, Yield-BERT, MOJO, AlphaFold reps, Kermut, S2ALM, WSI Virchow2/CONCH, multi-omics fusion, Metz.

**F. The master lever:** the **Phase-5 ARC rerun** (ISS-81/82/83) — server-only, you drive it — is what turns every rule-mandated change + every coded-but-unmeasured rep (MolE, MiniMol, TITAN…) into final numbers. The rerun commands are in `NEXT_SESSION_PROMPT.md`.

**Your decision list, minimized:** (1) confirm KEEP on the bucket-A rule-mandated changes; (2) bucket-B inert items — revert-all for a clean tree, or leave?; (3) approve removing IgBlend from the default antibody list; (4) acknowledge n10 was a false alarm (keep code); (5) confirm the bucket-E feature-gaps are "not in this pass."
