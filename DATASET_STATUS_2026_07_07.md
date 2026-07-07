# KIRBy Dataset Status — OPIG Antibody Presentation Prep (2026-07-07)

**Purpose:** Honest, per-dataset, code-verified status for the OPIG presentation. What is
viable, what must be fixed, what to drop. Every methodological claim below is checked against
the actual code (file:line given) or against a fetched OPIG/field paper — not memory.

**Sources:** 21-agent research+audit fleet (`w0qe4prsr`, 1.45M tokens, 12 OPIG-paper research
angles + 8 code-audit angles + adversarial synthesis) — plus my own direct code reads to
verify the load-bearing claims. OPIG anchor numbers are from papers the fleet actually fetched.

## HARD RULES
1. **NO POOLING.** Never "worked on x/20 antibody datasets." Each dataset stands alone.
2. **Own every caveat before they say it.** Saturated / tiny / synthetic / leaky — say it first.
3. **This is not small-molecule territory.** QSAR intuitions about "good numbers," activity
   cliffs, and Tanimoto similarity actively mislead on antibodies. See §6.

---

# 0. THE ONE-PARAGRAPH TRUTH (read this first)

Your antibody **skeleton is strong and genuinely OPIG-aligned**: group-aware CV that *raises*
instead of silently falling back to random KFold, real 95%-identity single-linkage clustering
on concatenated heavy+light with mandatory Levenshtein (this is literally the AbLang/IgT5
recipe), leave-one-PDB-complex-out for ΔΔG, ProteinGym `position_modulo` for single-site DMS,
Ginkgo's own hierarchical folds, and domain-correct metrics (Spearman / AUROC-F1w) with
one-hot & BLOSUM baselines included. **That splitting rigor is your single best card.**

The **value proposition mostly cannot survive this audience**: the core KIRBy deliverable is a
**0.01–0.05 hybrid-vs-baseline Spearman delta**, and on antibody data (n = 94–1780, highly
redundant) that delta is **within noise**. OPIG's own 2024–2026 message is deflationary
(FLAb2: PLMs beat nothing on ~80% of developability sets; Ginkgo: CV ρ~0.65 collapses to blind
ρ~0.35; Graphinity: random-split Pearson 0.87 collapses ~63% under CDR-identity cutoffs). Two
things you do **not** have that they **will** ask for: **bootstrap CIs / multi-seed** on every
delta, and a **germline-edit-distance control** (~40% of apparent PLM "skill" is just germline
distance).

**Highest-leverage move:** reframe from *"our hybrids win"* to *"here is a leakage-controlled
evaluation harness, and here is how little signal survives it."* That negative result is
exactly what OPIG respects.

---

# 1. THE SHORTLIST — present at most these 3–4, all with CIs, framed as *ranking* not absolute

| Dataset | Task | n | Split (code-verified) | OPIG-credible range | Verdict |
|---|---|---|---|---|---|
| `abcov_kd` | anti-SARS-CoV-2 Ab −log Kd (reg) | ~1780 | 95%-identity heavy+light clustering | modest ρ w/ CIs (NbBench affinity ρ<0.20 for all) | **PRESENT** (strongest) |
| `flab_thermo_tresanco` | **nanobody (VHH)** Tm (reg) | 556 uniq | identity-clustered | ρ 0.3–0.6 (NbBench Tm ρ~0.59) | **PRESENT** (say "VHH") |
| `ginkgo_hic` | GDPa1 developability HIC (reg) | 242 | benchmark hierarchical folds | ρ~0.35–0.7 (blind HIC 0.708) | **PRESENT** (the one predictable property) |
| `ddg_combined` | Ab–Ag ΔΔG (reg) | ~429 / ~20–32 complexes | leave-one-complex-out | modest ρ that survives cutoffs | **PRESENT only if** you add k-fold + state leakage |

Everything else is a supporting sanity-check, needs a fix, or should be cut. **Lead with the
splitting methodology as a contribution**, then show these with CIs.

---

# 2. ANTIBODY DATASETS — full code-verified status (no pooling)

## 2.1 PRESENT (with caveats)

### `abcov_kd` — **PRESENT (top card)**
- **Code:** `abcov.py:41` ~1780 antibodies; inter-family (diverse, not one lineage);
  `_identity_groups` 95%-identity single-linkage on heavy+light. Metric Spearman.
- **Why safe:** largest antibody set, real near-duplicate control (OPIG's exact recipe), diverse.
- **Caveats to state:** report Spearman **with bootstrap CIs across folds**; do **not** lead
  with the hybrid-vs-baseline delta (within noise); you have **no germline-distance control**.
- **OPIG anchor:** NbBench nanobody affinity ρ<0.20 for ALL models — frozen-embedding affinity
  is near-unsolved, so **modest numbers with CIs are the honest result**, not a headline ρ.

### `flab_thermo_tresanco` — **PRESENT** (say it is a nanobody)
- **Code:** `flab.py` (Tresanco 2023), ~672 rows / 556 unique; identity-clustered; Spearman.
- **Why safe:** best FLAb task; thermostability is a learnable *intrinsic* property. **But it is
  a VHH single-domain nanobody** — say so explicitly, don't conflate with paired mAbs. Spearman
  only (never R²/RMSE on raw Tm). Add CIs + germline control.
- **OPIG anchor:** NbBench thermostability best ρ~0.59; Ginkgo Tm2 blind 0.392. **Defensible =
  ρ 0.3–0.6**, not higher.

### `ginkgo_hic` — **PRESENT** (headline developability)
- **Code:** `ginkgo_dev.py` pre-computed hierarchical-cluster, IgG-isotype-stratified folds, n=242.
- **Why safe:** THE benchmark (GDPa1), authors' authoritative folds, HIC is the one genuinely
  predictable developability property (tracks computable CDR hydrophobicity).
- **Caveats:** n=242 → ~48/fold; add CIs; note the documented CV→blind-test collapse, so your
  5-fold CV ρ is optimistic vs a true blind number.
- **OPIG anchor:** Ginkgo GDPa3 blind HIC ρ = **0.708** (the "easy" one). Near that = credible;
  much higher on CV reads as CV-optimism.

### `ginkgo_tm2 / titer / pr_cho / acsins / colloidal / purity / aggregation` — **PRESENT deflationarily only**
- **Code:** `ginkgo_dev.py` GDPa1 properties, pre_split folds, n≈197–242.
- **Framing:** these are the HARD properties. Present as *"we reproduce the field finding that
  these are weakly predictable and hybrids don't meaningfully move them"* — an honest negative.
  Do NOT claim a per-property rep winner from a small delta at n~200.
- **OPIG anchor (blind ρ):** Tm2 0.392, polyreactivity 0.356, self-association 0.337, titer
  0.310. **ρ 0.3–0.4 IS state-of-the-art here**; higher on your CV = optimism to flag, not celebrate.

### `ddg_combined` — **PRESENT only if you add k-fold + own the ceilings**
- **Code:** `ddg_combined.py:5-10,25-26` SKEMPI 2.0 Ab-Ag subset + AB-Bind (~754 entries / 32
  Ab-Ag complexes); leave-one-PDB-complex-out GroupKFold; **code itself documents "residual
  structural leakage across similar interfaces."**
- **Caveats:** only ~20–32 complexes → coarse, high-variance CV; labels **mix** direct AB-Bind
  ΔΔG with Kd-derived SKEMPI ΔΔG (different noise semantics); reverse-mutation augmentation
  inflates n with perfectly anti-correlated pairs. Lead with a modest ρ that *survives* the split.
- **OPIG anchor:** Graphinity — random-split Pearson 0.87 collapses ~63% under CDR-identity
  cutoffs. A modest honest Spearman is the win.

### `humanness_therapeutic` — **PRESENT as sanity check, NOT a headline**
- **Code:** `humanness.py` therapeutic preset (BioPhi IMGT/mAb-DB, n=564); 95%-identity clusters,
  StratifiedGroupKFold. This is the *meaningful* humanness axis (human vs humanized, not vs mouse).
- **Reality:** near-saturated (AUROC >0.92, many reps ≥0.90). Reps within <0.05 AUROC on
  ~110-point folds are noise — do not claim a rep winner. VH-only (state it). The genuinely hard
  task (clinical immunogenicity / ADA, R²~0.3) is not addressed.
- **OPIG anchor:** off-the-shelf humanness sits 0.96–0.98, within ~0.01 of each other.
- **Attribution guard:** BioPhi/OASis = **Merck**; AbNatiV = **Vendruscolo (Cambridge)**. Only
  **Hu-mAb / Humatch** are OPIG. Do not misattribute in the talk.

### `abagym_EGFR / lysozyme / VEGF / Ang2` — **PRESENT in the prediction family ONLY**
- **Code:** `abagym.py` single-antibody DMS; `position_modulo` = ProteinGym `fold_modulo_5`,
  correct for single-site DMS (`antibody_baseline_survey.py:351` uses first mutation position).
- **Framing:** these are **single-antibody affinity DMS** (every variant is a point mutant of ONE
  parent), so whole-Fv PLM reps barely move and "which representation helps" has little signal.
  Present as hard single-antibody DMS in **complexity / forward-compat only** — NOT in similarity
  (s05/s12) or noise-cliff (n*), where the premise is circular (§6). Loader docstring itself says
  FoldX beats ML and near-zero ρ is expected.
- **`abagym_all`** (leave-one-interface-out, 5 groups) is the most interesting framing (cross-Ab
  transfer) but only 5 folds → high variance + z-score pooling can be gamed. Exploratory only.

## 2.2 CUT (from the presentation; several also from run scripts)

### `tz_her2_binary` / `tz_her2_multiclass` — **CUT** (was my "reframe" — I was wrong)
- **Code-verified leak:** `antibody_baseline_survey.py:255-259` — `groups = edit_distances`
  (integer edit-distance from WT). Two CDRH3s differing by one residue can fall in adjacent
  edit-distance shells → opposite folds → **maximal near-duplicate leakage**, bypassing the
  95%-identity clustering every other dataset gets.
- **Second problem:** it is a **single-antibody CDRH3 deep-mutational scan** (fixed framework +
  identical VL; ~110/120 residues constant) mislabeled as "antibody classification."
- **The Chinery angle still matters but doesn't save it:** your winners are blosum/onehot
  (registry line 468/480), which *does* reproduce "Baselining the Buzz." But OPIG's first
  question is "does it beat edit-distance-to-WT, on a clean split?" — and the split is leaky.
- **Fix to salvage (optional):** re-split by CDRH3 sequence-identity clusters (hold out clusters),
  add an edit-distance-to-WT baseline, then it becomes presentable *as a single-antibody DMS*.

### `cdr_flexibility` — **CUT** (code-verified random-split-in-disguise)
- **Code:** `opig_cdr_flexibility.py:140-153` — comment: "the caller dedups CDR3 sequences
  upstream so every group is size-1 → StratifiedGroupKFold degenerates to stratified per-sample";
  and an explicit `assert group_sizes.max() == 1`. This is a **random split wearing a group-CV
  label** — a direct violation (in spirit) of the lab's no-random-split rule. Do not present.

### `humanness_species` — **CUT** (presentation AND run scripts)
- Human vs mouse/rhesus/camelid VH is separable from **germline alone** (~0.99 AUROC), so it
  can't discriminate representations. Mixing camelid VHH into "non-human" makes it *more* trivial.
- **OPIG anchor:** BioPhi — Hu-mAb 0.977 vs a plain germline-content baseline 0.968.
- **Action:** remove `humanness_species` from `DATASET_REPS` (`complexity_helpers.py:957`) and
  `DATASET_REGISTRY` (`antibody_baseline_survey.py`) + any DOMAIN_GROUPS.

### `ginkgo_heparin` — **CUT** (n=94)
- `ginkgo_dev.py` n=94 → ~19/test fold. Spearman on 19 points is uninformative; no rep-ranking
  or hybrid claim survives. Split legitimacy doesn't rescue this n. (Even Ginkgo's blind test
  used 80 antibodies for a *single* number, not 5-fold rep-ranking on 19-point folds.)

### Jain clinical-panel FLAb regressions — **CUT all but ONE illustrative**
`flab_thermo_jain`, `flab_aggregation_jain`, `flab_aggregation_jain_hicrt`,
`flab_aggregation_jain_sas`, `flab_polyreact_jain_bvp`, `flab_polyreact_jain_cic`,
`flab_pk_jain_fcrnrt`, `flab_pk_jain_heprt`, `flab_immunogenicity_marks`, `flab_thermo_hutchinson`
- **Two problems:** (1) each is only ~130–137 antibodies → ~27/test fold → Spearman CI ≈ ±0.3, so
  deltas are pure sampling noise; (2) **~9–10 of them are carved from the SAME Jain 2017
  137-antibody panel** — showing them as separate results is coverage inflation an antibody
  reviewer spots instantly. **Present at most ONE** Jain property as an illustration.
- **Action:** definitely drop `flab_pk_jain_fcrnrt` (n=132) and `flab_pk_jain_heprt` from run
  scripts (as you asked); treat the rest as one panel, not independent tasks.
- **OPIG anchor:** FLAb2 flags datasets this small (n=2–200) as underpowered and excludes those
  with <5 significant correlations.

### `abagym_HER2` — **CUT** (n=146)
- `abagym.py` HER2_2021_trastuzumab, n=146 → ~29/fold; whole-Fv embeddings near-constant across
  146 point mutants of trastuzumab. Any hybrid delta is noise.

### `flab_binding_phillips_h3` — **CUT** (code-verified leakage bug)
- **Code:** Phillips 2021 CR9114 is a **combinatorially-complete multi-mutation** library, but
  `position_modulo` groups on the **first mutation position only** (`:351`). A variant mutated at
  (5,6) is grouped by pos 5 yet shares its pos-6 mutation with fold-1 variants → **individual
  mutation effects leak across the split**. Do not present until re-split by full-genotype cluster.

### `flab_binding_adams` / `flab_expression_adams` — **CUT** (combinatorial → same leak, VERIFIED)
- Adams 2017 Tite-Seq (eLife 23156) varies **CDR1H and CDR3H together** → multi-mutation
  variants, so `position_modulo` (first-mutation-only, `:351`) **leaks** exactly like phillips_h3.
  Re-split by full-genotype cluster before any use.

### `flab_binding_koenig` / `flab_expression_koenig` — **KEEP** (single-site, split VALID — corrected)
- **Verified against Koenig 2017 PNAS (E486):** these are **single-site saturated mutagenesis**
  libraries (all amino-acid substitutions, one position at a time) of the G6.31 anti-VEGF Fab. So
  `position_modulo` is the correct ProteinGym split and does **not** leak. My earlier "cut pending
  verification" was wrong. Caveat is the usual single-antibody-DMS one (whole-Fv reps barely move,
  limited rep-ranking signal) — present in the prediction family with CIs, same class as `abagym`.

### `ddg_combined_full` — **CUT** (off-domain)
- `ddg_combined.py` `ab_ag_only=False` feeds non-antibody PPIs (barnase-barstar, T4-lysozyme) to
  antibody-specific reps. Not an antibody result; at most a control ablation, never shown as
  antibody work.

### `graphinity_foldx / graphinity_rosetta / graphinity_augmented` — **CUT** (physics emulation)
- **Code:** `graphinity.py` labels are FoldX/Rosetta OUTPUT (foldx ~50K of 943K; rosetta ~21K).
  Predicting them well = learning FoldX (which correlates only r~0.34 with experiment).
  `graphinity_augmented` concatenates experimental + synthetic ΔΔG on different scales with no
  harmonization. Already excluded from the run plan.
- **OPIG anchor:** Graphinity is OPIG's *own* synthetic data, built to argue you need ~90k real
  ΔΔG points — a good FoldX-emulation number invites exactly that rebuttal.

## 2.3 IGNORE (dormant / not real tasks — just don't resurrect)
- **`sabdab`** (`sabdab.py`): only a valid *unlabelled pretraining corpus*. Its supervised
  "tasks" (predict crystal resolution, or predict rule-computed humanness/dev labels that are
  deterministic functions of the input) are circular; `remove_redundancy` only warns → would leak.
- **`abbind`** (`abbind.py`): dead code, no caller; AB-Bind reaches experiments only via
  `ddg_combined`. Prune so no one mistakes it for an independent benchmark.

---

# 3. WHOLE EXPERIMENT FAMILIES — antibody applicability

The complexity/similarity/noise/federated code is domain-generic: it runs the same battery on
any dataset. On antibodies:

| Family | On antibody data | Verdict |
|---|---|---|
| **complexity (c01–c24)** | prediction/rep-ranking with the group-CV above | **KEEP** — the defensible core; present with CIs + germline control |
| **forward-compat (c13)** | per-new-rep symbiotic value inside hybrid | KEEP (same caveats); it's a rep-addition ablation |
| **similarity (s01,s02,s03,s07,s13)** | in-sample, no held-out split | **CUT** — circular. s02 fits importance on the FULL label vector then measures similarity-activity corr vs the SAME labels; "hybrid improves SAC 100%+" is definitional |
| **noise cliffs (n01,n03,n04,n10)** | activity-cliff / SALI / SAS premise | **CUT** — transplanted QSAR; "structural similarity" = Euclidean distance in each rep's OWN embedding → cliffs defined by rep A, then rep A scored on resolving them (circular) |
| **noise injection (n05,n07–n13)** | synthetic Gaussian / label-flip noise | **CUT** — biophysically meaningless; real antibody assay noise is heteroscedastic & platform-dependent (Ginkgo's whole point). n06 (real replicate noise) runs on BindingDB, not antibody |
| **federated (fed05)** | 3 "silos" = unrelated assays | **CUT** — contrived; the 3 silos (Ab-Ag ΔΔG kcal/mol, single-Ab DMS enrichment, VHH Tm °C) share no task, so the actual federated FLuID baselines (C7/C8) are **skipped**; single 80/20/one-seed on n=146 |

**Net:** for OPIG, present the **complexity/forward-compat prediction results** on the shortlist
datasets. The similarity, noise, and federated families are where the small-molecule DNA of the
codebase shows — they do not transfer to antibodies and will draw fire.

---

# 4. CONCRETE FIXES (code changes)

1. **CUT `humanness_species`** — remove key from `complexity_helpers.py:957` `DATASET_REPS`,
   from `antibody_baseline_survey.py` `DATASET_REGISTRY`, and any DOMAIN_GROUPS references.
2. **CUT `flab_pk_jain_fcrnrt` + `flab_pk_jain_heprt`** — remove from `DATASET_REPS`
   (lines 236, 224) and `DATASET_REGISTRY` + DOMAIN_GROUPS.
3. **`tz_her2`** — no cheap fix; the `edit_distance` grouping is leaky. Either CUT for the talk,
   or (to salvage later) re-split by CDRH3 identity clusters + add edit-distance-to-WT baseline.
4. **`abagym_EGFR` split** — NO CHANGE (ProteinGym-standard; I was wrong earlier). Action is
   framing only. Consider dropping `abagym_lysozyme` (registry: only 3 reps beat ρ>0.15) and
   `abagym_HER2` (n=146).
5. **Add before any "beats baseline" claim:** bootstrap CIs / multi-seed on every delta, and a
   **germline-edit-distance partial-correlation control**. Without it OPIG asks and you can't answer.
6. **Verify-before-trust** (registry rep lists copied from a sibling, so never independently
   tuned): `cdr_flexibility` (also a bad split — cut), `ginkgo_tm2/aggregation/purity`,
   `chembl_qsar` (copied from `caco2`, line 974). Confirm a real survey exists or label provisional.

---

# 5. NON-ANTIBODY DATASETS (your home turf — CODE-VERIFIED 2026-07-07)

**Correction to an earlier lazy pass:** the molecular datasets are **methodologically sound and
verified**, not "flag and move on." `_shared.py:2553-2571` splits them by **Murcko scaffold**
(`assign_scaffold_groups`), with Butina clustering as the ISS-85 alternative, and the CV code
**raises rather than falling back to random** (`:594`, `:651`) — the group-required standard.
Result-file counts confirm they are real, heavily-run surveys.
- **Molecular (Murcko-scaffold split, RMSE — SOUND):** `caco2` (36 result files), `logd` (59),
  `herg` (clf, 74), `herg_ki`, `qm9` (15), `adme_hlm/rlm/mdr1`. Correct split, correct metrics,
  real data. These ARE up to standard.
  - **Only real caveats:** `chembl_qsar` has **0 result files** — registered but never run
    (build it or drop it, don't cite it as a result); `herg`'s `graph_kernel` value is from a
    stale-merge bug (`complexity_helpers.py:99`) — don't quote that one number.
- **DTA (cold-protein split, CI):** `kiba` (n30k, CI 0.6924), `bindingdb_kd` (n30k, CI 0.7260),
  `davis` (CI ceiling 0.7567, flagged possible per-drug-mean degeneracy). `kiba` drug-side spread
  is 0.0006 (near-tied/degenerate under 229 protein groups) — don't over-claim drug-side ranking.
- **TCGA (clf):** `tcga_luad_vs_lusc`, `tcga_brca_pam50`. Sound; off-topic — omit.
- **Reaction (R²):** `buchwald_hartwig`, `bh_doyle_plates`. Off-topic — omit.
- **Nucleotide (MCC):** 18 `nt_*` + 4 `gb_*`. Note classical k-mers beat DNA foundation models on
  several (e.g. `gb_human_or_worm`: tetranuc_freq 0.9135). Off-topic — omit.

---

# 6. WHY SMALL-MOLECULE INTUITION MISLEADS HERE (so you're not blindsided in Q&A)
- **"Similarity" ≠ embedding distance.** Antibody similarity = germline + CDR-H3 + epitope. An
  "antibody activity cliff" computed from embedding geometry marks the work as off-domain.
- **Redundancy is worse than in QSAR.** ~98–99% conserved framework; diversity is in the CDRs.
  Naive splits leak massively — which is why every credible number is on a clustered split.
- **Germline confound.** ~40% of apparent PLM "skill" on antibody tasks is just germline distance.
  A partial-correlation control is expected before any "rep beats baseline" claim.
- **n is tiny.** Most real antibody property data is hundreds of points → CIs mandatory; a 0.03 ρ
  delta is noise, not a finding.

---

# 7. VERIFIED OPIG / FIELD REFERENCES (spot-check these before the talk)
- **AbLang** — Olsen, Moal, Deane, *Bioinformatics Advances* 2022. Training-data recipe: CDR3-then-70%
  Linclust clustering. (OPIG)
- **IgBert/IgT5** paired LMs — Kenlay, Deane et al., *PLOS Comp Biol* 2024 (arXiv 2403.17889). 95%-identity Linclust. (OPIG)
- **OAS** — Olsen, Boyles, Deane, *Protein Science* 2022. Filter by Redundancy + ANARCI_status. (OPIG)
- **SAbDab / SAbDab-nano / Thera-SAbDab** — Dunbar/Schneider/Raybould, Deane, *NAR* 2014/2022/2019. (OPIG)
- **TAP (5 developability guidelines)** — Raybould, Deane, *PNAS* 2019. Structure+statistics, thresholds from Thera-SAbDab. (OPIG)
- **ImmuneBuilder / ABodyBuilder2** — Abanades, Deane, *Comm Bio* 2023. CDR-H3 RMSD 2.81 Å. (OPIG)
- **Paragraph** (paratope) — Chinery, Deane, *Bioinformatics* 2023. Reports PR-AUC (class-imbalanced). (OPIG)
- **Graphinity** (synthetic ΔΔG) — Hummer, Deane, *Nat Comput Sci* 2025. Random-split Pearson 0.87 → −63% under cutoffs. (OPIG)
- **Baselining the Buzz** (Tz-HER2) — Chinery, Hummer, Deane 2024 (bioRxiv 2024.03.26.586756). Simple baselines hard to beat. (OPIG)
- **FLAb2** — Fitness Landscapes for Antibodies benchmark (2024–). PLMs beat ~nothing on ~80% of dev sets; underpowered-dataset exclusion.
- **BioPhi/OASis (Merck), Hu-mAb/Humatch (OPIG), AbNatiV (Vendruscolo)** — humanness saturated at 0.96–0.98.
- **NbBench** — nanobody benchmark: affinity ρ<0.20 all models; thermostability best ρ~0.59.
- **Ginkgo GDPa1/GDPa3** — blind developability: HIC 0.708; Tm2 0.392; polyreactivity 0.356; self-assoc 0.337; titer 0.310.

*(Full per-agent research incl. more citations + reported numbers:
`/private/tmp/claude-501/.../tasks/w0qe4prsr.output` and the workflow `journal.jsonl`.)*

---

# 8. BOTTOM LINE
- **Viable to present (with CIs, as ranking):** `abcov_kd`, `flab_thermo_tresanco` (VHH),
  `ginkgo_hic`, `ddg_combined` (if k-folded). Plus the hard Ginkgo panel as an honest negative.
- **Lead with:** your leakage-controlled splitting harness as a methods contribution.
- **CUT:** `tz_her2` (leaky split), `humanness_species` (trivial), `cdr_flexibility` (random-in-
  disguise), `ginkgo_heparin` (n=94), `abagym_HER2` (n=146), `flab_binding_phillips_h3` (leak),
  the Jain panel down to one, `graphinity_*`, `ddg_combined_full`, and the entire
  similarity/noise-cliff/noise-injection/federated antibody families.
- **Verify before trusting:** `flab_*_koenig/adams` split integrity, the copied registry entries.
- **Add before claiming a winner:** CIs/multi-seed + a germline-distance control.

---

# 9. THREE-BUCKET TRIAGE — ALL DOMAINS COMBINED (science × live compute, 2026-07-07)

The talk is not antibody-only, so all 74 datasets are triaged together. **Bucket = min(science,
compute):** immediate only if BOTH scientifically defensible AND precompute built. Scientific
CUTs go to Bucket 3 regardless of compute; fixable-but-leaky/partial go to Bucket 2.
`Dom`: Ab=antibody, Mol=molecular, Rxn=reaction, Nuc=nucleotide. **cx** = complexity experiments
run out of 20 (the presentable family); for antibody, ignore the `sim`/`noise` columns (§3 —
those families are cut/under review), for other domains sim/noise still pending the noise audit.

> **⚡ Precompute reality — the `--distance` artifact (2026-07-07 correction).** The pasted
> status dump was run with `--distance`, which the status script says is OFF by default
> (`check_precompute_status.py:15`). It pads each dataset's denominator to 8 by adding the four
> `*_distance` tiers. But **only the `similarity` family and the single `c03` complexity
> experiment consume the distance pool** (`complexity_helpers.py:1324`: "c03 is the only
> complexity experiment that uses the distance pool"); **all other complexity experiments AND
> the noise family use the non-distance `general`/`top` tiers** (`noise_helpers.py:224,246`).
> So a dataset "partial N/8" whose only missing tiers are `*_distance` is **fully precomputed for
> complexity** (all of c01–c24 except c03). This promotes several datasets into Bucket 1 for the
> complexity story — they were only ever "partial" for similarity.

## 🟢 BUCKET 1 — precompute done + verified sound → FOCUS for immediate results

| Dataset | Dom | Science | Precomp | cx | Action |
|---|---|---|---|---|---|
| `abcov_kd` | Ab | **top antibody card** (n~1780, identity-clustered) | DONE 4/4 | 20/20 | Ready — CIs + germline control |
| `abagym_Ang2` | Ab | single-Ab DMS (valid split) | DONE 4/4 | 20/20 | Ready — frame as single-Ab DMS |
| `abagym_VEGF` | Ab | single-Ab DMS | DONE 4/4 | 20/20 | Ready |
| `abagym_EGFR` | Ab | single-Ab DMS | DONE 4/4 | 16/20 | Finish 4 cx |
| `flab_expression_koenig` | Ab | **single-site DMS, split VALID** (Koenig 2017) | DONE 8/8 | 3/20 | Run cx |
| `humanness_therapeutic` | Ab | saturated → sanity check only | DONE 4/4 | 19/20 | Not a headline |
| `flab_thermo_tresanco` | Ab | **headline (nanobody/VHH) Tm** | DONE 8/8 | 3/20 | **Run cx now** |
| `ginkgo_hic` | Ab | **headline developability** | DONE 9/9 | 5/20 | **Run cx now**; CIs |
| `ginkgo_tm2/titer/pr_cho/acsins/colloidal/purity/aggregation` | Ab | honest-negative panel | DONE 8–9/9 | 3–6/20 | Run cx; "field says hard" |
| `caco2` | Mol | Murcko-scaffold, sound (36 result files) | DONE 8/8 | 3/20 | Run cx |
| `herg` | Mol | scaffold clf, sound (74 files) | DONE 8/8 | 1/20 | Run cx (skip stale graph_kernel) |
| `herg_ki` | Mol | scaffold, sound | DONE 8/8 | 3/20 | Run cx |
| `adme_mdr1` | Mol | scaffold, sound (Polaris ADME-Fang) | DONE 8/8 | 3/20 | Run cx |
| `qm9` | Mol | scaffold, sound — **cx precompute COMPLETE** (only `*_distance` missing) | cx-DONE† | 0/20 | Run cx now; similarity needs distance build |
| `logd` | Mol | scaffold, sound — **cx precompute COMPLETE** | cx-DONE† | 0/20 | Run cx now; similarity ~ready (only `top_distance` missing) |
| `adme_hlm` | Mol | scaffold, sound — **cx precompute COMPLETE** | cx-DONE† | 3/20 | Run cx; similarity ~ready |
| `adme_rlm` | Mol | scaffold, sound — **cx precompute COMPLETE** | cx-DONE† | 1/20 | Run cx; similarity ~ready |
| `buchwald_hartwig` | Rxn | leave-additive-out, sound | DONE 4/4 | 13/20 | Strongest non-Ab; finish cx |
| `bh_doyle_plates` | Rxn | leave-additive-out, sound — **cx precompute COMPLETE** | cx-DONE† | 0/20 | Run cx; similarity needs distance |
| `tcga_brca_pam50` | TCGA | TSS-grouped clf, sound | DONE 8/8 | 3/20 | Run cx |
| `tcga_luad_vs_lusc` | TCGA | TSS-grouped clf, sound | DONE 8/8 | 3/20 | Run cx |
| `nt_H3K27me3` `nt_H3K4me3` `nt_H3K9ac` `nt_H3K9me3` `nt_promoter_tata` `gb_human_enhancers_cohn` | Nuc | chrom/natural-grouped, sound | DONE 8/8 | 0–3/20 | Run cx |

**Immediate-results plan:** ready-now (cx substantially done) = `abcov_kd`, `abagym_Ang2/VEGF/EGFR`,
`humanness_therapeutic`, `buchwald_hartwig`. Everything else in Bucket 1 has precompute built but
few cx runs — those runs are cheap now, so **kick off cx for your headliners first**
(`flab_thermo_tresanco`, `ginkgo_hic`, then `caco2`/`herg` for the molecular half of the talk).

## 🟡 BUCKET 2 — precompute partial/missing OR needs a modification → complete later

| Dataset | Dom | Issue | Compute | Fix |
|---|---|---|---|---|
| `ddg_combined` | Ab | coarse leave-one-complex-out (~20–32 groups); documented residual leakage | DONE, cx 3/20 | Add k-fold + state leakage; then run cx |
| `tz_her2_binary` / `tz_her2_multiclass` | Ab | **leaky edit_distance grouping** | DONE, cx 7/20 | Re-split by CDRH3 identity clusters + add edit-dist-to-WT baseline; rerun |
| `flab_binding_koenig` | Ab | single-site VALID, precompute unfinished | partial 7/8 | Finish precompute + cx |
| `flab_binding_phillips_h3` | Ab | **combinatorial leak, UNFIXABLE → see Bucket 3** | partial 6/8 | No valid grouped split exists (genotype clustering → 1 group) |
| `flab_binding_adams` / `flab_expression_adams` | Ab | **combinatorial leak, UNFIXABLE → see Bucket 3** | 6/8 & DONE | No valid grouped split exists |
| `davis` / `kiba` / `bindingdb_kd` | DTA | sound (cold-protein CI) but precompute barely built | partial 1/8 | Finish precompute |
| `adme_hlm` / `adme_rlm` / `logd` | Mol | **SIMILARITY only** (complexity → Bucket 1); `top_distance` tier missing | 6/8 | Build `top_distance` for similarity |
| `qm9` | Mol | **SIMILARITY only** (complexity → Bucket 1); all `*_distance` missing | 4/8 | Build distance tiers for similarity |
| `chembl_qsar` | Mol | **0 result files** (never run) | MISSING | Build precompute + run, or drop |
| `bh_doyle_plates` | Rxn | **SIMILARITY only** (complexity → Bucket 1); `*_distance` missing | 4/8 | Build distance tiers for similarity |
| 15× partial `nt_*` / `gb_demo_*` | Nuc | sound; precompute unfinished | partial | Finish precompute (top_distance/uncapped mostly) |

*(Global modification for ALL antibody results: add bootstrap CIs / multi-seed and a
germline-distance partial-correlation control before any "rep beats baseline" claim.)*

## 🔴 BUCKET 3 — CUT (do not present; several also stop building precompute)

| Dataset | Dom | Why cut | Compute |
|---|---|---|---|
| `humanness_species` | Ab | trivial (~0.99, germline-separable) | MISSING — don't build; remove from registry |
| `cdr_flexibility` | Ab | **random-split-in-disguise** (group size==1 assert) | MISSING — don't build |
| `graphinity_foldx` / `graphinity_rosetta` | Ab | predicting FoldX/Rosetta, not biology | MISSING — don't build |
| `ginkgo_heparin` | Ab | n=94 → ~19/fold, uninformative (unfixable) | DONE but drop |
| `abagym_HER2` | Ab | n=146 single-Ab DMS (unfixable) | DONE but drop |
| `abagym_lysozyme` | Ab | near-dead signal (3/20 reps clear ρ>0.15) | DONE but drop |
| Jain panel: `flab_thermo_jain`, `flab_aggregation_jain(_hicrt/_sas)`, `flab_polyreact_jain_bvp/_cic`, `flab_pk_jain_fcrnrt/_heprt`, `flab_immunogenicity_marks`, `flab_thermo_hutchinson` | Ab | ~130 antibodies each, **all one Jain 2017 panel** → coverage inflation + noise | mostly DONE — present **≤1** illustrative (candidate `flab_immunogenicity_marks`, cx 19/20) |
| `flab_binding_phillips_h3` / `flab_binding_adams` / `flab_expression_adams` | Ab | **combinatorial libraries, position_modulo leaks, NO valid grouped split possible** (all genotypes >86% identical → clustering yields 1 group → CV fails) | some DONE — do not present position_modulo numbers |
| `ddg_combined_full` | Ab | non-antibody PPIs (off-domain) | — |

**Experiment-family note (separate axis from datasets):** the antibody `similarity`
(s01/s02/s03/s07/s13) and `noise`-cliff/injection families are cut for antibody (§3), and
`fed05` federated is cut (contrived). The **noise family viability across ALL domains** is under
a dedicated read (workflow `wc3sxb23t`) — findings will be appended. Forward-compat (`c13`) and
the nine `opig_*` experiments are under verification (`witdz8dms`).

† **cx-DONE** = all non-distance (`general`/`top` × `none`/`uncapped`) tiers built, so the whole
complexity battery except `c03` is precompute-ready; the "partial" label came only from the
opt-in `--distance` tiers that solely feed similarity + c03. `cx N/20` = how many complexity
experiments have been *run* so far (cheap now that precompute exists).

**Precompute headline (corrected):** the `--distance` run understated readiness. For the
**complexity** story, the ready set is not 8 datasets but ~**12**: the 8 marked DONE plus `qm9`,
`logd`, `adme_hlm`, `adme_rlm`, `bh_doyle_plates` (cx-complete; only distance/similarity tiers
outstanding). Genuinely missing precompute: `graphinity_foldx/rosetta`, `humanness_species`,
`cdr_flexibility`, `chembl_qsar` — 4 of 5 are Bucket-3 CUTs, so **only `chembl_qsar` is a
"build if you want it."** The remaining "partial" datasets need only their `*_distance` tiers,
and **only if you want the similarity family** on them.

---

# 10. EXPERIMENT VERIFICATION — noise / forward-compat / OPIG (code+paper grounded, 2026-07-07)

Two verification fleets read the actual experiment code (and the published protocols the OPIG
experiments claim to reproduce). Results below are decisions, not hedges.

## 10.1 Noise experiments (n01–n13) — viability

**Structural fact for ALL of them:** the noise family runs at **1-fold CV, fast search**
(`noise_helpers.py:17`) — a single split, no multi-fold variance. That alone means no noise
result should carry a confidence claim.

| Exp | Verdict | Why | Valid where |
|---|---|---|---|
| **n06** BindingDB replicate noise | ✅ **VIABLE** | The ONLY experiment with real ground truth: correlates estimator score vs actual per-sample replicate std | DTA/molecular only — **no antibody analogue** |
| n02 importance-weighted weighting | 🟡 caveat | Soundest of the rest: real labels, held-out CV, no leakage — but measures *predictive improvement*, not noise detection | all domains (as "does weighting help", not "noise") |
| n07 remediation | 🟡 caveat | Sound *within* synthetic-noise regime; no fiction-scored detector | synthetic-noise only |
| n09 LNL survey | 🟡 caveat | Honest relative question (does hybrid change LNL ranking); synthetic noise on train only | synthetic-noise only |
| n11 estimation×remediation grid | 🟡 caveat | Real held-out recovery metric, multi-seed — but synthetic noise; NOT the hybrid-vs-base claim (oracle-biased) | molecular, synthetic only |
| n12 importance-weighted distance | 🟡 caveat | Legit narrow ablation; effects likely within run-to-run noise | molecular only |
| n13 alt noise distributions | 🟡 caveat | Honest *synthetic-noise sensitivity study* — the soundest framing; drop multiplicative noise | any, as sensitivity study only |
| n01 activity-cliff framework | 🟡 caveat | Sub-analyses 1–3 **circular** (cliffs from each rep's own embedding; supervised hybrid wins by construction); only sub-4 (rep-independent cliff ref + held-out CV) is defensible, molecular only | molecular sub-4 only |
| **n03** SAS landscape | 🔴 **NOT VIABLE** | Circular zone-shift + the one predictive test (GP bridge) is leakage-biased AND an n=4 correlation | — |
| **n04** GP-LOO "killer experiment" | 🔴 **NOT VIABLE** | **Rigged**: ground truth = the Gaussian injection; hybrid fit on the clean labels it's asked to separate → its win is by construction | — |
| **n05** noise-injection benchmark | 🔴 **NOT VIABLE** | F1 scored against an artifact mask (\|noise\|>MAD of the injection itself); homoscedastic Gaussian ≠ real assay noise | — |
| **n08** noise-estimation cliff-discrimination | 🔴 **NOT VIABLE** | Circular: cliffs and noise scores both derived from the same embedding distances + labels | — |
| **n10** hybrid-neighborhood LNL | 🔴 **NOT VIABLE** | Synthetic + self-fulfilling: injected Gaussian IS a neighbor-deviation, so a distance detector "wins" trivially | — |

**Decision:** CUT n03, n04, n05, n08, n10 entirely (circular or scored against a fiction).
Present n06 as the one real-noise result — **but only for DTA/molecular, never antibody**. The
rest (n01/n02/n07/n09/n11/n12/n13) are presentable ONLY as *synthetic-noise sensitivity studies*
on molecular data, explicitly labelled as such, never as real-noise detection, never on antibody
(small-n + no cliff analogue). **The noise story is a molecular/DTA story, not an antibody one.**

## 10.2 Forward-compatibility (c13) — presentable premise, but the core metric needs a fix

The rep-novelty framing is sound (flagged reps genuinely post-date the KIRBy build) and it's
defensible on **molecular + DTA**. But the *symbiosis* claim has real problems:
- **Core "new rep helps" metric (D−C) is selection-on-same-data** → biased positive by
  construction (D searches a superset of C on the same data). Present only as an *in-sample upper
  bound* unless you add a held-out/nested split or a **hybrid-level random-Gaussian null**
  (`D_random − C`); if `D_real−C` isn't clearly above `D_random−C`, the effect is noise.
- **The other half of the headline ("hybrid improves the new rep") has no working metric** —
  condition E is null by design, yet the code prints a spurious "E>B: 0/N (0%)". Implement `D−B`
  or delete that row; as-is a slide would claim a direction the code can't substantiate.
- **Stats are vaporware:** Friedman/Nemenyi dead, Wilcoxon on 5 non-independent folds is
  underpowered — do not put a significance p-value on a slide; drop the Demšar citation.
- **Driver divergence:** `submit_forward_compat.sh` runs a *weaker* experiment (no "hybrid
  without r" control) than the c13 module. Decide which is "the" forward-compat result and
  present the c13 module version. (VERIFY_ON_SERVER which produced the results you have.)
- **Antibody caveat:** three marquee "new" models (MINT, xTrimoPGLM, AIDO.Protein) are dead on
  antibody — lean on SaProt/IgT5/ESM3 and say so.

## 10.3 OPIG experiments — several are faithful reproductions = your SAFEST slides

These are `opig_*.py` self-contained experiments that reproduce published protocols. The fleet
verified faithfulness against the actual papers. **This is the most defensible material you have
— lead with these, not the raw hybrid deltas.**

| Experiment | Reproduces | Decision | Note |
|---|---|---|---|
| `opig_activity_cliff_qsar` | Dablander 2023 (J Cheminf 15:47) | ✅ **KEEP / present** | Most faithful: thresholds, half-cliff exclusion, MCC, compound/AC counts all match exactly. Molecular |
| `opig_ginkgo_developability` | Ginkgo GDPa1 competition, van Niekerk 2026 | ✅ **KEEP / present** | Faithful CV reproduction, baselines match to 3 decimals. **Caveat: reproduces CV leaderboard (`cv_top`), NOT the blinded GDPa3 test; anchor to `cv_top` and state the large CV→test gaps** |
| `opig_graphinity_ddg` | Hummer 2025 (Nat Comput Sci 5:635) | ✅ **KEEP / present** | Faithful; correct leave-one-complex-out. Frame `within_foldx/rosetta` as "reproduce FoldX," not affinity |
| `opig_trastuzumab_enrichment` | Chinery 2024 "Baselining the Buzz" | 🟠 **FIX framing first** | Sound + honest docstring, BUT argparse help (`:535`) overclaims "Replicates Chinery." **Chinery IS OPIG** — change to "uses the Chinery dataset for a supervised rep-quality question"; drop the non-comparable predicted-binder-% block |
| `opig_blueprint_transfer` | (original KIRBy method) | ✅ KEEP | Clean, leakage-aware; present as a novel contribution, not a reproduction |
| `opig_cgr_comparison` | (original KIRBy method) | ✅ KEEP | Present as KIRBy's own CGR-vs-SMILES analysis; verify neural-CGR reps actually embed on server |
| `opig_humanness` | BioPhi (Prihoda 2022) | 🟡 KEEP (later) | Saturated + tiny → sanity only, don't lead |
| `opig_cdr_flexibility` | ITsFlexible (Spoendlin 2025) | 🟠 FIX | Same random-split-in-disguise as the dataset (assert `:149`); needs a real V-gene/germline group split before OPIG |
| `opig_kingsmith_component_splits` | HTE component-LOO | 🟠 FIX attribution | Methodology sound (true LOO GroupKFold), but **misattributed** — King-Smith 2024 is an HTE *analyzer*, not a component-LOO benchmark. Re-attribute to Saebi 2023 / Bradshaw before OPIG |
| `imrie_transfer_learning` | Imrie 2018 (inspired-by) | ⚠ VERIFY_ON_SERVER | All 3 questions need pretrained `.pt` weights; if absent, TL conditions silently skip. Check `ls /Volumes/seagate/kirby/chembl_pretrain/*_256_128_64.pt` — run `imrie_pretrain.sh` first if missing |

**Server checks that would close the remaining opens:** (1) Ginkgo — confirm the GDPa1_v1.2 CSV
has column `hierarchical_cluster_IgG_isotype_stratified_fold` (5 folds, n~242); (2) Imrie
pretrained weights exist; (3) `opig_cgr` neural reps (`chytorch_graphormer`, `cgr_smiles_lm`)
actually produce embeddings; (4) which forward-compat driver produced your results.

## 10.4 Revised bottom line for the talk
1. **Lead with the OPIG faithful reproductions** (`opig_activity_cliff_qsar`,
   `opig_ginkgo_developability` anchored to CV, `opig_graphinity_ddg` framed as FoldX-emulation)
   + your leakage-controlled splitting harness. This is your most defensible ground.
2. **Then the Bucket-1 prediction results** with CIs (`abcov_kd`, `flab_thermo_tresanco`,
   `ginkgo_hic`, the abagym/koenig single-Ab DMS, plus the molecular `caco2`/`herg`/
   `buchwald_hartwig`).
3. **Forward-compat:** present as rep-novelty on molecular/DTA; do NOT put the D−C symbiosis
   number up as an unbiased effect without the held-out null.
4. **Noise:** a molecular/DTA story via n06 only; the antibody noise experiments are out.
5. **Fix before showing:** the `opig_trastuzumab_enrichment` "replicates Chinery" overclaim and
   the `opig_kingsmith` misattribution — both are the exact hand-wavy-attribution traps an OPIG
   audience catches.

# 11. DOMAIN-MECHANIC VIABILITY — does each experiment's *setup* make sense for the *data*? (code-verified, 2026-07-07)

A 6-agent fleet re-read every experiment file (noise, complexity, similarity, forward-compat,
OPIG, federated) asking a narrower question than §10: **these harnesses were written with small
molecules in mind — does the mechanic itself still make sense when pointed at antibody / protein /
DTA / omics data?** Watching specifically for Murcko-scaffold CV, Tanimoto/ECFP similarity,
RDKit/Morgan/SMILES calls, descriptor-space perturbation, and molecular-only "complexity/noise"
axes applied blindly to sequences.

> **⚠️ This is a DIFFERENT axis from §10.1.** "Domain-mechanic OK" means the code does something
> *sensible* for antibody data — it does **not** override the scientific cuts in §10.1. Example:
> n04/n05/n08/n10 are *domain-agnostic* (they inject noise on the label, not on SMILES, so they
> run fine on antibodies) yet still **CUT** because they're scientifically circular. A slide needs
> to clear **both** gates.

**The reassuring headline (say this if challenged in Q&A):** the harness is far more
domain-adaptive than its molecular origin implies. Three design facts do the heavy lifting —
- **Splits** come from per-dataset `data['groups']` (`_shared.py:545`, hard-fails with no random
  fallback): antibody → sequence-identity clusters, molecular → Murcko scaffolds, DTA → cold-target,
  TCGA → TSS site. The word "scaffold" in the code is legacy naming; the *mechanic* adapts.
- **"Similarity"** defaults to standardized **Euclidean/cosine in embedding space**
  (`noise_mitigation_methods.py:67`), *not* Tanimoto. Fingerprint distance is a guarded, opt-in
  branch that only fires when `domain=='molecular'`.
- **"Noise"** is injected on the **scalar label `y`**, never on SMILES/descriptors; **rep pools**
  are per-domain (`DATASET_REPS`), so antibody tasks search antibody reps.

Where a genuinely molecular mechanic exists, it is almost always **guarded to skip** on antibody
rather than silently produce garbage. The exceptions — the real traps — are §11.2.

## 11.1 Molecular / omics / reaction / DTA BY DESIGN — meaningless on antibody, but they SKIP cleanly (safe: absent, not wrong)

These are correctly domain-locked. On an antibody dataset they return a `{'skipped'}` marker (or
raise before producing numbers), so they **cannot** put a misleading antibody figure on a slide —
just don't claim antibody coverage for them.

| Exp | Locked to | Guard (verified) |
|---|---|---|
| **n06** BindingDB replicate noise | DTA only (needs drug×target replicates) | not in per-dataset suite; run only for `bindingdb_kd` (`noise_helpers.py:58`) |
| **c12** TCGA modality ablation | TCGA multi-omics only | `MODALITY_GROUPS` = fixed gene/methyl/CNV/histology map (`c12:71`); nothing to ablate off-TCGA |
| **s04** ECFP4-Tanimoto activity cliffs | molecular only | `domain!='molecular' → skipped` (`s04:66`); s12 is the Euclidean companion for antibody |
| **s11** Bayesian-opt / library screening | molecular/reaction only | `domain not in (molecular,reaction) → skipped` (`s11:785`); Tanimoto-GP never touches embeddings |
| `opig_activity_cliff_qsar` | molecular (Dablander) — **correct** | hardcoded d2/fxa/mpro; Murcko + MMP by design |
| `opig_cgr_comparison` | reaction (CGR/DRFP) — **correct** | hardcoded Buchwald-Hartwig/Suzuki |
| `opig_kingsmith_component_splits` | reaction component-LOO — **correct** | hardcoded BH; components = aryl-halide/ligand/base/additive |
| **fed01 / fed03A / fed04** | molecular (Morgan-FP silos + `pKi≥6.0`) | KeyError on `data['smiles']` — can't run on antibody |
| **fed02 / fed03B** | molecular (hardcoded ECFP4/mordred/molformer silo reps) | `K<3 → ValueError` on antibody; nothing silently produced |

## 11.2 ⚠️ THE REAL TRAPS — run on antibody but produce MISLEADING output (fix or caveat before presenting)

Only three. These don't crash and don't skip — they emit a plausible-looking number that means
less than it appears:

1. **`opig_cdr_flexibility`** — antibody-appropriate reps, but the CDR3-dedup makes every CV group
   size-1, so `StratifiedGroupKFold` silently degrades to **stratified-random KFold**
   (`asserts group_sizes.max()==1`, `:141-152`). It prints an AUROC, but the split is effectively
   random. **Already CUT/FIX in §2.2 + §10.3** — reconfirmed here from the mechanic side.
2. **`c07` (data-complexity partitions) on antibody** — of its three "complexity" axes, the
   `molecular_complexity` one is RDKit `HeavyAtomCount(SMILES)` and **silently drops** when no
   SMILES (`:131-133`); the other two run but the output mislabels sequence-cluster frequency as
   **"scaffold_frequency"** (`:50`, `:121-127`). Result on antibody: a 2-of-3 experiment with a
   chemistry label on a non-chemistry axis. **Fix:** rename the axis or gate c07 to molecular; at
   minimum never show the "scaffold" wording on an antibody slide.
3. **`fed04`** — force-binarizes the target at **`pKi≥6.0`** (`:150,191`) regardless of endpoint,
   which would be nonsensical on a kcal/mol or °C antibody label. In practice it crashes on
   `data['smiles']` first, so it *fails* rather than misleads — but do not "fix the smiles crash"
   and let the binarization through. **It's a molecular experiment; leave it molecular.**

## 11.3 Antibody-native / genuinely domain-agnostic — SAFE on antibody (the bulk of the suite)

Everything not in §11.1/§11.2 clears the domain gate. Verified specifically:
- **Complexity:** c01–c06, c08–c11, c13–c19, c21–c24 — all operate on the numeric rep matrix +
  label + group-aware CV. "Augmentation" (c21) = concatenating complementary reps via per-domain
  `AUG_GROUPS`, **not** SMILES enumeration. c15's docstring says "Tanimoto" but the code builds an
  RBF kernel only (`:387`) — no fingerprint path runs.
- **Similarity:** s01–s03, s05–s10, s12, s13 — Euclidean/cosine in embedding space; s01/s02/s07's
  cliff sweeps have an **explicit antibody branch** (edit distance on heavy chains). s12 is the
  purpose-built Euclidean replacement for s04.
- **Forward-compat c13:** domain-agnostic mechanic with **per-domain legacy-rep sets**
  (`_LEGACY_ANTIBODY/_DTA/_TCGA/_REACTION`); the D−C *selection bias* flagged in §10.2 is a
  methodology issue that applies equally to all domains, **not** a molecular-mechanic mismatch.
- **Federated:** **fed03C** (antibody datasets as silos) and **fed05** (the explicit antibody port
  — silos clustered on **ESM2 embeddings**, not Morgan FPs; Gaussian noise on the scalar target;
  self-skips the C7/C8 cross-silo-predict conditions that genuinely don't transfer). These are the
  federated slides if you want one.
- **OPIG antibody scripts:** `opig_ginkgo_developability` (competition cluster folds),
  `opig_graphinity_ddg` (leave-one-complex-out), `opig_humanness`, `opig_trastuzumab_enrichment`,
  `opig_blueprint_transfer` — all use antibody reps + proper group-aware splits, no scaffold/RDKit.

## 11.4 One-line reconciliation
Passing §11 (domain-mechanic sensible) is **necessary but not sufficient**. The final "present on
antibody?" verdict = §11 (mechanic fits the domain) **AND** §10.1/§2 (scientifically sound + the
dataset itself is real). The noise family is the clearest case: domain-fine, science-cut.
