# Rep & Dataset Pruning Log

Started 2026-07-18. Records the per-dataset pruning of the KIRBy roster (60 datasets in
`DATASET_REPS`). Decisions are the USER's; this log records the evidence Claude produced and the
call the user made. Worked **one dataset at a time**, dataset cut and rep cut together.

## The per-dataset procedure (3 steps)

For each dataset, under the dataset's real CV strategy (scaffold-group K-fold for molecular,
group-aware for the rest), scoring model = **lightgbm** (decided internal evaluator; ridge removed):

1. **Validity gate.** Score every base rep individually. Does the best base rep give a *meaningful*
   result — clearly better than a trivial predictor (regression: Spearman ρ well above 0 and RMSE
   well below the target's std; classification: AUROC well above 0.5)? **If no rep is meaningful,
   the dataset is DROPPED** — nothing downstream can work.

2. **Prediction correlation among reps.** For reps that pass, compare their **prediction vectors**
   (out-of-fold), NOT their raw features (roadmap rule). Reps that make the same predictions carry
   the same signal → keep only one.
   - Primary flag = **Spearman (rank) correlation** of prediction vectors (robust to scale; "same
     ordering of samples" = same signal). Pearson (absolute) computed as a cross-check.
   - Confirmation = **concat test** (user's idea): concatenate two candidate reps and re-score. If
     `concat(A,B) ≈ best(A,B)`, B adds nothing beyond A → redundant. If it improves, keep both.

3. **Define the general pool (≤10 reps).** Using steps 1–2, choose **≤10** reps that span distinct
   signal. **The `top_tier` concept is removed — one pool per dataset.** Selection method (proposed,
   pending user sign-off): **greedy forward by marginal CV gain** — start from the best rep, add the
   rep that most improves the concatenated CV score, stop when a rep adds negligible gain or at 10.
   Each addition IS a concat test, so it directly measures marginal signal.

**Open method question (user asked for help):** rank vs absolute correlation, and how heavily to
lean on the (expensive) concat test vs the (cheap) correlation flag. Current proposal: correlation
to rank/flag redundancy cheaply, concat-greedy to make the actual ≤10 cut. To be settled on the
first dataset before scaling to ~40.

**Cache note:** changing a dataset's rep pool invalidates its search/baseline caches (reps are NOT
in the cache key) — each finalized pool ships with scoped delete instructions before re-scoring.

---

## Dataset-cut decisions (2026-07-18, in progress)

Criterion (user): cut ONLY when the numbers are genuinely bad (no signal even under an
honest/near-random split). A broken split is NOT grounds to cut — fix the split (use the
literature standard) and re-judge on the honest number.

| dataset | finding | decision |
|---|---|---|
| `cdr_flexibility` | ITsFlexible (OPIG, Nat. Mach. Intell. 2025). Literature splits by **CDR sequence-identity clusters (MMseqs2, 80–90%)**, not CDR3-dedup (which made groups size-1). Fixable with the repo's existing `sequence_identity_clusters`. | FIX split → identity clusters, then re-judge. NOT an auto-cut. |
| `flab_binding_phillips_h3` / `flab_binding_adams` / `flab_expression_adams` | Survey best: phillips_h3 **0.984**, adams **0.806**, expression_adams **0.560** — NOT bad numbers. But likely leakage-inflated by the combinatorial split. | Re-split with sequence-identity clusters, re-score. Cut only if the honest number is bad. |
| `flab_immunogenicity_marks` | Survey best **0.775** (decent). Not a Jain duplicate (marks2021). | **CUT per user call** (noted: signal is decent + rationale for coverage-inflation was wrong — flagged for reconsideration). |
| `tz_her2_binary` / `tz_her2_multiclass` | Field commonly uses random splits (optimistic). | Implement **sequence-similarity hold-out**: cluster on the varying **CDRH3** (not the 227-aa concat, which collapses to 1 cluster). |
| DTA: `kiba`, `bindingdb_kd`, `davis` | **NOT cut** (Claude's earlier "axed" was wrong). `kiba` (CI 0.692) + `bindingdb_kd` (CI 0.726) DONE with REAL signal — docs state per-drug-mean degeneracy stops winning at their protein diversity (1,219 groups). `davis` PARKED pending degeneracy investigation (job 7568428). | KEEP kiba + bindingdb_kd; davis pending. Better reps may resolve davis's degeneracy — on-thesis. |
| Nucleotide | DNA foundation reps failed (gena_lm ~0.02 MCC); classicals won at modest MCC ~0.35–0.55 **with ridge**. | KEEP the ridge exception FOR NUCLEOTIDE ONLY (user, 2026-07-18) — see [[feedback_no_ridge]] exception. Curate 21→6 (coverage + weak signal). |

**Sources:** [ITsFlexible / CDR flexibility (Nat. Mach. Intell. 2025)](https://www.nature.com/articles/s42256-025-01131-6) · [FlAbDab (OPIG)](https://opig.stats.ox.ac.uk/webapps/flabdab)

## Path to the FINAL dataset list (dataset side only — do this BEFORE any feature work)

The final list is NOT a big pile of judgement calls. It is: **fix the broken splits → run step-1
validity across the roster → cut the no-signal datasets → read off the survivors.** Breakdown of
the 60-roster:

- **Decided CUT (1):** `flab_immunogenicity_marks` (user).
- **Needs a SPLIT FIX first, then step-1 validity (6):** `cdr_flexibility`,
  `flab_binding_phillips_h3`, `flab_binding_adams`, `flab_expression_adams` → CDR/sequence-identity
  cluster split (`sequence_identity_clusters`, exists); `tz_her2_binary`, `tz_her2_multiclass` →
  cluster on the 10-residue CDRH3 (`df['seq']` in the loader) at ~0.8 identity. Test each with one
  cheap rep (onehot/blosum) to confirm the split is non-degenerate and the honest number.
- **KEEP, no action (3 DTA):** `kiba`, `bindingdb_kd` (real signal), `davis` (pending degeneracy
  job 7568428 — keep, better reps may fix it).
- **Nucleotide (21):** step-1 validity **with the ridge exception**, then curate (was ~6/21 on the
  old ridge survey; re-confirm honestly now).
- **Everything else (~29):** step-1 validity (server batch) decides keep/cut on signal.

**So the remaining dataset-side work is exactly: (1) 6 split fixes + cheap-rep test, (2) one step-1
validity batch on `medium`, (3) nucleotide curation from the honest numbers. Output = final list.**
Only then do we touch features.

### Split-fix results + decisions (2026-07-18, tested with aac)

Key finding: the flab combinatorial sets + tz_her2 are **single-WT mutational libraries** (e.g.
flab_binding_phillips_h3 = 5000 seqs, all 230 aa, 94–99% identical, ~7.5 mutations from one WT).
Identity-clustering is the WRONG tool for these — everything is >90% identical, so it collapses or
leaks (the 0.9-threshold AUROCs were leakage). So the doc was right: **single-WT combinatorial
libraries have no clean grouped split.** Identity-clustering only works for genuinely diverse panels.

| dataset | nature | decision |
|---|---|---|
| `flab_binding_phillips_h3` | single-WT + saturated (aac 0.98 while leaking) | **CUT** (user) |
| `flab_binding_adams` / `flab_expression_adams` / `tz_her2_binary` / `tz_her2_multiclass` | single-WT | **KEEP on `edit_distance`** (mutation-shell hold-out) with stated "mild adjacent-shell leakage" caveat; step-1 validity judges the honest number (user) |
| `cdr_flexibility` | diverse panel (clusters merge as threshold drops) | **KEEP, fix to identity-cluster split** (aac ~0.67 honest). Needs a load-path added — it currently routes only through `opig_cdr_flexibility.py`, not `load_data`. |
| `flab_immunogenicity_marks` | — | CUT (user) — removed from DATASET_REPS |

**Roster now = 43** (60 − 15 nucleotide − immunogenicity_marks − phillips_h3).
**Remaining dataset-side code: wire cdr_flexibility's load path (identity split).** Then step-1
validity batch → final list.

## Molecular domain (8 datasets: herg, herg_ki, caco2, logd, qm9, adme_hlm, adme_rlm, adme_mdr1)

Starting subset (user): qm9, logd, herg, herg_ki.

| dataset | n | task | metric | #general reps | data staged |
|---|---|---|---|---|---|
| herg_ki | 1,412 | regression | RMSE | 15 | local ✓ |
| logd | 5,028 | regression | RMSE | 15 | local ✓ |
| qm9 | 10,000 | regression (HOMO-LUMO gap) | RMSE | 16 | server |
| herg | 7,772 | classification (FLuID) | AUROC/MCC | 18 | server |

### herg_ki — STEP 1 done, STEP 2 done, STEP 3 pending method sign-off

- **Step 1 (validity): PASS.** lightgbm, 3-fold scaffold-group CV. Best rep `graph_kernel`
  ρ +0.745 (RMSE 0.608); tight band down to `graphmvp` ρ +0.655. Real, usable signal.
  Full ranking: graph_kernel .745 · mhg_gnn .744 · ecfp4 .728 · pdv .727 · atom_pair .724 ·
  mordred .721 · avalon .714 · mol2vec .713 · top_torsion .700 · chembert .696 · grover .692 ·
  smited .690 · chemberta .689 · molformer .687 · graphmvp .655.
- **Step 2 (prediction correlation): NO duplicates.** Zero rep pairs exceed |Spearman| 0.9 — the
  molecular reps make genuinely different predictions despite similar scores. (Contrast: antibody
  tresanco had 10/28 reps redundant.) So correlation ALONE does not cut this pool at all.
- **Step 3 (≤10 pool): DONE 2026-08-20 (user call).** Nothing exceeds 0.90 pred-corr, so the cut is a
  solo-vs-redundancy judgement. **CUT 5: chemberta, graphmvp, molformer, mol2vec, top_torsion.**
  **FINAL 10:** mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, ecfp4, atom_pair, chembert, smited.

### logd — STEP 1 done, STEP 2 done, STEP 3 pending method sign-off

- **Step 1 (validity): PASS (strong).** Best rep `mordred` ρ +0.894 (RMSE 0.521); tight band down to
  `graphmvp` +0.782. High, clean signal.
  Ranking: mordred .894 · mhg_gnn .889 · pdv .882 · grover .868 · avalon .865 · ecfp4 .860 ·
  graph_kernel .857 · mol2vec .850 · atom_pair .848 · smited .844 · molformer .835 · top_torsion
  .830 · chembert .809 · chemberta .786 · graphmvp .782.
- **Step 2 (prediction correlation): THRESHOLD-SENSITIVE.** Highest pair mordred≡pdv 0.967.
  - at |Spearman| > 0.95 → 1 redundant pair → pool 15 → 14
  - at |Spearman| > 0.90 → 21 pairs, 10 drop-candidates → pool 15 → 5
  A bare threshold swings the pool from 14 to 5 between 0.90 and 0.95 — arbitrary. This is the
  clearest case that the **concat test** (does pdv actually improve CV on top of mordred?) is needed,
  not a threshold guess.
- **Step 3 (≤10 pool): DONE 2026-08-20 (user call).** Heavily redundant (top-9 mutually >0.90); kept
  one representative per family + the distinct pretrained/graph reps.
  **CUT 5: pdv, mol2vec, chemberta, graph_kernel, atom_pair.**
  **FINAL 10:** mhg_gnn, mordred, grover, ecfp4, smited, molformer, avalon, top_torsion, chembert, graphmvp.

### caco2 — STEP 3 DONE 2026-08-20 (user call)
- Only one pred-corr pair >0.90 (pdv≡mordred 0.907); rest genuinely distinct → cutting 12→10 drops
  2 near-non-redundant reps on solo/redundancy tradeoff.
  **CUT 2: mordred, mol2vec.**
  **FINAL 10:** graph_kernel, mhg_gnn, ecfp4, pdv, grover, avalon, top_torsion, smited, atom_pair, molformer.

### qm9 — STEP 1/2 data staged (server), STEP 3 pending user call
### herg — awaiting rerun (silent multiclass/loader failure — resubmitted 2026-08-20)

---

## How we are doing STEP 3 (finalized method, 2026-08-20)

Supersedes the earlier "concat-greedy pending sign-off" proposal. For each dataset, from the
`rep_prune/<ds>/` server evidence (group-CV OOF, lightgbm), Claude presents **three verified numbers
per rep IN THE CHAT (plain tables, no artifacts, re-read from the files every time)**:
  1. **solo** — the rep's own score (Spearman for regression, AUROC for classification).
  2. **avg corr** — mean prediction-correlation to all other reps (higher = more redundant).
  3. **nearest twin (+ corr)** — the single rep it most duplicates.
Datasets are worked **by domain, 3–4 at a time.** The **user picks the 10** (no auto-selection).
Each finalized pool → edit `complexity_helpers.py` `DATASET_REPS[ds]['general']` (and `top_tier`,
which is retired → set equal to `general`, one pool per dataset). Record CUT + FINAL 10 here.
**Cache:** reps are not in the cache key, so a pool change invalidates that dataset's search/baseline
caches — wipe + re-score those datasets before trusting new hybrid numbers (user runs the deletion).

### FINAL POOLS so far
| dataset | pool (10) | cut |
|---|---|---|
| caco2 | graph_kernel, mhg_gnn, ecfp4, pdv, grover, avalon, top_torsion, smited, atom_pair, molformer | mordred, mol2vec |
| logd | mhg_gnn, mordred, grover, ecfp4, smited, molformer, avalon, top_torsion, chembert, graphmvp | pdv, mol2vec, chemberta, graph_kernel, atom_pair |
| herg_ki | mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, ecfp4, atom_pair, chembert, smited | chemberta, graphmvp, molformer, mol2vec, top_torsion |
