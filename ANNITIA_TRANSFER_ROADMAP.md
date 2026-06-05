# ANNITIA Transfer-Learning + Foundation-Model Roadmap

**Purpose:** Lift hepatic and death C-index on the Trustii ANNITIA submission by
(a) plugging in tabular foundation models with survival heads, (b) augmenting
the rare-event class synthetically, and (c) pretraining on
external MASLD-adjacent cohorts before fine-tuning.

**Date:** 2026-05-07
**Trustii deadline:** 2026-05-08 (T1/T2 only feasible pre-deadline; T3 starts now,
lands post-deadline for v2 / paper).

**Endpoints:** hepatic (47 events / 1253) and death (~750 events / 990 after
21% blank exclusion). Score weight = 0.7 hep + 0.3 death.

---

## Background

Current pipeline (from `tests/experiments/annitia/`):
- 6 hand-crafted feature views × 3 sksurv models × 5-fold StratifiedKFold per endpoint.
- Existing branches in flight: R2 main, xgbcox, extra-views, stacking (24h),
  weighted (kNN-reliability + inverse-freq), pycox/tabpfn/tabicl/finegray/D9,
  catboost_cox+aft.
- Best CV so far: hep CV 0.9079 → LB 0.8898 (R1 baseline). Hep CV is the
  bottleneck — 47 events makes the noise floor high.
- Rank-ensembling across branches via `combine_branches.py`.

What's missing:
1. **Foundation tabular models with survival semantics.** TabPFN/TabICL classifiers
   exist in the folder but aren't used as survival predictors.
2. **Class-conditional synthetic augmentation** of the 47 hepatic positives.
3. **External pretraining** on a larger MASLD cohort.
4. **Competing-risk handling.** Currently hep and death are modelled
   independently; SurvTRACE-style joint modelling is unexplored.

---

## Tiers

| Tier | What | Feasibility before 2026-05-08 |
|------|------|-------------------------------|
| **T1** | Foundation-model survival heads + TabSurv + synthetic augmentation | Feasible if started immediately |
| **T2** | NHANES pretrain → ANNITIA fine-tune | Feasible (open data); ~6h preprocessing |
| **T3** | NIDDK CR / SteatoSITE / LITMUS apply-and-wait | Post-deadline; v2 + paper work |

---

## T1 — Foundation models, survival heads, augmentation (no external data)

### T1.1 — MITRA + TabPFN-2.5 survival head via discretised classification

**Source:** "Tabular Foundation Models Can Do Survival Analysis" (arxiv 2601.22259).
Reformulate (X, time, event) as (n_patients × K) binary-classification problems:
- Discretise time into K bins of equal *event count* (not equal width — keeps the
  rare-event bins from being empty).
- For patient i and bin k, label = 1 if event observed by bin-k upper edge,
  0 if censored after bin-k upper edge, NaN (skip) if censored before.
- Run MITRA (HuggingFace `autogluon/mitra-classifier`) and TabPFN-2.5 (
  `priorlabs/TabPFN`) per-bin in-context. Stack the K predictions into S(t).

**Files:**
- `tests/experiments/annitia/_survival_discretiser.py` — shared bin builder
  (`SurvivalDiscretiser(strategy='equal_event'|'quantile', K=K)`), label matrix.
- `tests/experiments/annitia/mitra_survival.py` — `MitraSurvival(K, ckpt)` class
  with `fit(X, y_struct).predict_risk(X)` interface matching `survival.py`.
- `tests/experiments/annitia/tabpfn25_survival.py` — same interface, TabPFN-2.5.
- `tests/experiments/annitia/run.py` — register `mitra_surv` and `tabpfn25_surv`
  alongside `cox`, `coxnet`, `rsf`.

**Endpoints:** Both. K=10 for hep (~4-5 events/bin), K=20 for death.
Run as ablation; if K=10 wins on death too, drop K=20.

**Conditions to evaluate:**
| ID | Endpoint | Model | K | Expected lift |
|----|----------|-------|---|---------------|
| T1.1a | hep   | MITRA      | 10 | High — rare-event regime is MITRA's strength |
| T1.1b | hep   | TabPFN-2.5 | 10 | High |
| T1.1c | death | MITRA      | 20 | Medium — already plenty of events |
| T1.1d | death | TabPFN-2.5 | 20 | Medium |
| T1.1e | both  | MITRA      | 5  | Sensitivity check, K small |
| T1.1f | both  | MITRA      | 15 | Sensitivity check, K large |

Hyperparams: zero-shot first, then `n_estimators=8` ensembling for both models if it fits in compute. No fine-tuning yet (that's T1.2).

**Deliverable:** Two new branches (`feat/mitra-surv`, `feat/tabpfn25-surv`)
plugged into `combine_branches.py`.

**Compute:** ~1–2h on CPU per fold per endpoint with ensemble-of-8;
~10min on V100. Submit as separate SLURM jobs on `medium` (4h).

**Status:** Pending.

### T1.2 — TabTune-based fine-tuning of TabPFN-v2.6 / MITRA

**Source:** TabTune (`github.com/Lexsi-Labs/TabTune`). Wraps TabPFN-v2.6, MITRA,
TabICL with PEFT and full fine-tuning APIs.

Run T1.1 zero-shot first; if it doesn't outperform `cox+nit_view`, fine-tune via
TabTune in PEFT mode (cheap). Per-bin classification loss, bin K from T1.1
winner.

**Files:**
- `tests/experiments/annitia/tabtune_survival.py` — wraps TabTune `TabularPipeline`
  with the per-bin classification → S(t) reformulation.

**Conditions:**
| ID | Endpoint | Model | Strategy | Notes |
|----|----------|-------|----------|-------|
| T1.2a | hep   | TabPFN-v2.6 | PEFT     | LoRA-style adapters |
| T1.2b | hep   | MITRA       | PEFT     | |
| T1.2c | death | TabPFN-v2.6 | PEFT     | |
| T1.2d | hep   | TabPFN-v2.6 | base SFT | If PEFT wins |

**Deliverable:** Branch `feat/tabtune-surv`.

**Compute:** PEFT ~30min/fold/endpoint on V100. SLURM `medium`, 4h.

**Status:** Pending. Run only if T1.1 zero-shot is competitive (≥ best current
branch on CV).

### T1.3 — TabSurv (modern tabular NN with survival head)

**Source:** TabSurv (arxiv 2605.03944). FT-Transformer / SAINT family with
survival output. Claims to beat DeepSurv, DeepHit, SurvTRACE.

**Files:**
- `tests/experiments/annitia/tabsurv.py` — `TabSurv(backbone='ft_transformer')`
  wrapper.

**Endpoints:** Both. Single config first (FT-Transformer, default hparams).
Tune later only if it lands in the top 3 branches.

**Deliverable:** Branch `feat/tabsurv`.

**Compute:** ~1–2h/fold on V100, ~5–8h on CPU. SLURM `medium`, 8h to be safe.

**Status:** Pending. Implementation availability **needs verification** — the
arxiv paper code link wasn't surfaced in research; if not on GitHub, deprioritise
or reimplement minimally.

### T1.4 — Synthetic augmentation of the 47 hepatic positives

**Source:** TabDDPM (TKDD 2025), TabSyn (ICLR 2024), TabEBM (OpenReview 2024).

**Approach:**
- Train generator on the n=47 event=1 patients × all features (snapshot + slope
  + summary + delta + missingness, concatenated).
- Generate n_synth ∈ {50, 100, 200, 500} synthetic positives.
- Append synthetic positives **only to the training fold**, never the test
  fold. Re-fit the existing winning `(views, model)` candidates from R2.
- Keep the original 47 in training too — augmentation, not replacement.

**Conditions:**
| ID | Generator | n_synth | Endpoint | Notes |
|----|-----------|---------|----------|-------|
| T1.4a | TabDDPM  | 100     | hep   | First pass, conservative |
| T1.4b | TabDDPM  | 500     | hep   | Aggressive |
| T1.4c | TabSyn   | 100     | hep   | Compare diffusion architectures |
| T1.4d | TabEBM   | 100     | hep   | Class-EBM, designed for rare-class |
| T1.4e | best of (a–d) | best | death | Only run if hep wins |

**Calibration check** (Goorbergh JAMIA 2022): augmentation can wreck calibration.
Report Brier score and integrated Brier score alongside C-index. **Do not submit
augmented model if calibration degrades > 10%.** C-index alone is rank-based
and may look fine while underlying probabilities are wrong.

**Files:**
- `tests/experiments/annitia/synthetic_aug.py` — generator interface
  (`SyntheticAugmenter(method='tabddpm', n_synth=100).fit(X_pos).sample(n)`).
- `tests/experiments/annitia/run.py` — `--synthetic-aug` flag.

**Deliverable:** Branch `feat/synthetic-aug-hep`.

**Compute:** Generator training ~10–30min on V100. SLURM `medium`, 4h.

**Status:** Pending. Risk: TabDDPM with n=47 may collapse modes — fall back to
TabEBM (designed for class-conditional, low-N regime).

### T1.5 — SurvTRACE for competing-risk joint modelling

**Source:** SurvTRACE (`github.com/RyanWangZf/SurvTRACE`).

Hep and death are **competing risks** — once a patient dies (without a hepatic
event) they cannot have one. Modelling them independently double-counts the
censoring. SurvTRACE handles this jointly with shared transformer trunk.

**Conditions:**
| ID | Setup | Notes |
|----|-------|-------|
| T1.5a | (hep, death) joint | Default SurvTRACE |
| T1.5b | (hep, death, blank-death-imputed) | Use estimated death times for the 21% blank-death patients |

**Files:**
- `tests/experiments/annitia/survtrace_branch.py` — wraps SurvTRACE on
  concatenated views; produces per-endpoint risk for `combine_branches.py`.

**Deliverable:** Branch `feat/survtrace-competing`.

**Compute:** ~2h/endpoint pair on V100, SLURM `medium`, 4h.

**Status:** Pending. Lower priority than T1.1 — adds a model class but may
not beat existing ensemble's hep score.

---

## T2 — NHANES pretrain → ANNITIA fine-tune

The only open-access MASLD-adjacent cohort with downloadable longitudinal-ish data
+ linked mortality. Roadmap below assumes you can pull NHANES today.

### T2.0 — NHANES preprocessing

**Source:** [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/) +
NHANES Linked Mortality File (LMF, public-use through 2019-12-31 follow-up).

**Cycles to pull:** 1999–2018 (10 cycles). FibroScan only available 2017–2018.

**Variables to align with ANNITIA's 13 longitudinal labs + static covariates:**
| ANNITIA | NHANES variable | NHANES file | Notes |
|---------|-----------------|-------------|-------|
| age      | RIDAGEYR | DEMO     | Direct |
| BMI      | BMXBMI   | BMX      | Direct |
| ALT      | LBXSATSI | BIOPRO   | Direct |
| AST      | LBXSASSI | BIOPRO   | Direct |
| platelets| LBXPLTSI | CBC      | Direct |
| albumin  | LBXSAL   | BIOPRO   | Direct |
| GGT      | LBXSGTSI | BIOPRO   | Direct |
| glucose  | LBXGLU   | GLU      | Fasting |
| HbA1c    | LBXGH    | GHB      | Direct |
| triglycerides | LBXTR | TRIGLY  | Fasting |
| HDL      | LBDHDD   | HDL      | Direct |
| total chol | LBXTC  | TCHOL    | Direct |
| bilirubin | LBXSTB  | BIOPRO   | Direct |
| (no NIT) | compute FIB-4 = (age × AST) / (platelets × √ALT) | derived | proxy |

**NAFLD filter (subset ablation):**
- US Fatty Liver Index ≥ 30 (validated proxy on NHANES; Ruhl & Everhart 2015)
  — primary subset.
- Alternative: HSI > 36 ∪ FLI ≥ 60 ∪ TyG ≥ 8.5 (broader, less specific).

**Endpoint mapping:**
| ANNITIA endpoint | NHANES proxy | Pseudo events available |
|------------------|--------------|-------------------------|
| death | LMF all-cause death | ~5000+ events on NAFLD subset |
| hepatic | LMF underlying cause ICD-10 K70-K77 + C22 (HCC) | ~150-300 events on NAFLD subset (rough) |

The hep proxy is a **liver-specific death** flag, not a composite event with
non-fatal decompensation. It's directional but not identical; this is the main
risk in T2.

**Files:**
- `src/kirby/datasets/nhanes_nafld.py` — loader. Outputs same wide-format CSV
  schema as ANNITIA so feature views drop in unchanged.
- `tests/experiments/annitia/nhanes_pretrain.py` — fits CoxPH / RSF / GBSA /
  TabPFN-survival on NHANES, saves checkpoints.

**Deliverable:** `data/nhanes_nafld/{train,val,test}.csv` matching ANNITIA
column names.

**Compute (your laptop):** ~2-4h to download all NHANES files (xpt format,
moderate size) + preprocess. Suggest running locally during T1 SLURM jobs.

**Status:** Blocked on user kicking off the download.

### T2.1 — NHANES alone (sanity check)

Can NHANES-trained models predict ANNITIA without any ANNITIA training data?
This is the upper bound on cross-cohort generalisation; if it scores < 0.55
C-index then NHANES is too distributionally different to help.

| ID | Pretrain | Fine-tune | Endpoint |
|----|----------|-----------|----------|
| T2.1a | NHANES NAFLD subset | none (zero-shot) | death |
| T2.1b | NHANES NAFLD subset | none (zero-shot) | hep (liver-cause-death proxy) |

**Decision rule:** If T2.1a > 0.65 → NHANES has signal for death and T2.2 is
worth doing. If T2.1b > 0.55 → same for hep (lower bar because of proxy mismatch).

### T2.2 — NHANES → ANNITIA transfer

**Recipes (apply to whichever models passed T2.1):**
| ID | Recipe | Endpoint |
|----|--------|----------|
| T2.2a | Pretrain CoxPH on NHANES → use coefficients as Bayesian prior on ANNITIA Cox | both |
| T2.2b | Pretrain RSF on NHANES → tree-warm-start RSF on ANNITIA (refit only top half of trees) | both |
| T2.2c | Pretrain GBSA on NHANES → continue training on ANNITIA (lower LR, smaller n_estimators) | both |
| T2.2d | Pretrain TabPFN-survival on NHANES via PEFT → PEFT on ANNITIA | both |
| T2.2e | Train CoxPH on NHANES ∪ ANNITIA jointly with cohort-indicator covariate | both |
| T2.2f | Stacking: NHANES-trained risk-score as one feature view; ANNITIA model uses NHANES risk + own views | both |

**Note (T2.2f):** This is the safest one if cohort-shift is an issue — adds NHANES
as a *feature* rather than a *prior*, so even a poorly-transferring NHANES model
can still help if its risk score is monotone with hep risk. Closest to KIRBy's
existing "feature view" design.

### T2.3 — NHANES subset ablation

Don't assume "all NHANES" is best. Test:
| ID | Subset | Rationale |
|----|--------|-----------|
| T2.3a | All cycles 1999–2018, USFLI ≥ 30 | Maximum N |
| T2.3b | Cycles 2007–2018 only, USFLI ≥ 30 | Better label fidelity (HbA1c routine, ELT methods modern) |
| T2.3c | Cycles 2017–2018 only, FibroScan-confirmed CAP ≥ 285 dB/m | Best phenotype match (closest to ANNITIA's biopsy-/imaging-based MASLD), small N |
| T2.3d | T2DM subset only (HbA1c ≥ 6.5 OR self-report) | Match ANNITIA's diabetes-enriched cohort if it is one — verify against `dictionary.csv` |
| T2.3e | Excluding patients with viral hepatitis (HBsAg+, anti-HCV+) | Cleaner MASLD signal |

**Decision rule:** Pick the subset that maximises pretrain-then-fine-tune CV
on ANNITIA hep. If two subsets tie, prefer the larger one for stability.

**Compute:** Each ablation = full T2.2 grid. Submit as a single SLURM array job
(5 subsets × 6 recipes × 2 endpoints = 60 cells) on `medium`, 8h, V100.

**Status:** Pending T2.0 download.

### T2.4 — Best NHANES recipe → submission branch

The winner of T2.3 becomes branch `feat/nhanes-pretrain`. Plug into
`combine_branches.py`.

---

## T3 — Apply-and-wait cohorts (for v2 / paper)

Action items for the user **today**, regardless of submission outcome.

### T3.1 — NIDDK Central Repository

**Apply via:** [repository.niddk.nih.gov](https://repository.niddk.nih.gov/) → Data Request portal.

**What to request:**
| Study | N | Follow-up | Why |
|-------|---|-----------|-----|
| NAFLD Adult Database 2 | 2,501 | up to 10y annual | **Most ANNITIA-shaped**: longitudinal labs, hepatic + ALL-cause death endpoints |
| PIVENS | 247 | 96 weeks + post-trial | Histology-anchored, treatment arm |
| FLINT | ~283 | 72 weeks | OCA, similar |
| NAFLD Adult Database 1 | ~1500 | varies | Baseline cohort, predecessor |
| NAFLD Adult Database 3 | TBD | active | Latest cycle |
| TONIC | pediatric | varies | Skip unless pediatric extension planned |

**Realistic timeline:** Weeks-to-months. NIDDK CR expects an analysis plan
+ DUA. Expect ~6-12 weeks from submission to data delivery.

**Subset strategy when data lands:** mirror T2.3 — try Database 2 alone vs
Database 2 ∪ PIVENS+FLINT. PIVENS and FLINT are RCT cohorts and may have
treatment-driven trajectories that distort transfer if pooled naively.

### T3.2 — SteatoSITE (Edinburgh / PMS-IC)

**Apply via:** [regeneration-repair.ed.ac.uk/research/research-resources/steatosite](https://regeneration-repair.ed.ac.uk/research/research-resources/steatosite)
(application portal at Precision Medicine Scotland-Innovation Centre).

**What you get:** 940 patients × 5.67M days of EHR + bulk RNA-seq + histopath.
Outcome-rich, longitudinal, time-stamped — *the* closest external cohort to
ANNITIA's design.

**Subset strategy:**
- Histology-stratified: F0-F2 vs F3-F4 → does fibrosis-stage stratification of
  the NHANES/SteatoSITE pretrain set improve hep transfer? (Hypothesis: yes,
  since hep events are concentrated in F3-F4.)
- Tissue-RNA cohort vs full cohort — does adding transcriptomic features
  (whether as direct features or as a VAE-compressed embedding) help where
  ANNITIA only has clinical features? Probably no for direct use (ANNITIA
  doesn't have RNA), but yes for *cross-modal pretraining* — train a model
  on SteatoSITE clinical features → hep with RNA as auxiliary loss → fine-tune
  on ANNITIA clinical-only.

**Realistic timeline:** Weeks; less paperwork than NIDDK in our experience but
still expect 2-4 weeks.

### T3.3 — LITMUS / European NAFLD Registry

**Apply via:** European Collaborative Research Network on MASLD (post-LITMUS
governance). Contact via LITMUS project page.

**What you get:** 1,430 biopsy-confirmed NAFLD across 13 EU countries.
Histology-anchored, multi-site (so heterogeneity is real — useful for transfer
robustness testing).

**Realistic timeline:** Months. Lowest priority unless v2 needs it.

### T3.4 — UK Biobank

**Apply via:** ukbiobank.ac.uk; ~3-6 month review for new project.

**What you get:** 127K-179K MASLD-relevant patients with HES-linked hepatic
events. Largest cohort but blunt phenotype (ICD codes, not biopsy).

**Realistic timeline:** Quarters. Only worth doing if KIRBy direction continues
to need a public-cohort gold standard for MASLD survival.

### T3.5 — TARGET-NASH

**Apply via:** Site application via AASLD CQC registry (
`aasld.org/cqcregistryapp`) or contact `jpeter@targetrwe.com`.

**What you get:** 7,000+ NASH (sub-cohort of 400K liver-disease registry).
Real-world EHR + biospecimens.

**Realistic timeline:** Industry collab; depends on whether your institution
already has access. Skip unless someone at Oxford already has a TARGET RWE
collab.

---

## Sequencing

**Today (2026-05-07):**
- User: kick off NHANES download (T2.0)
- User: submit NIDDK CR data request (T3.1)
- User: submit SteatoSITE application (T3.2)
- Claude: scaffold T1.1 (MITRA + TabPFN-2.5 survival head), T1.4 (TabDDPM/TabEBM),
  T1.3 (TabSurv if available)

**Tonight → tomorrow morning:**
- Submit SLURM jobs for T1.1, T1.3, T1.4, T1.5
- T2.0 preprocess complete locally → push to server
- Submit T2.1 sanity check (cheap, ~30min on V100)

**Tomorrow morning (deadline day):**
- Triage: which T1 branches beat current best CV?
- If T2.1 hep > 0.55 OR T2.1 death > 0.65: submit T2.2 grid (T2.3 ablation)
- Re-rank-ensemble winners via `combine_branches.py`
- Submit final v2 to Trustii before deadline

**Post-deadline:**
- Wait on T3 cohorts
- Write up the v1 submission for paper
- v2 work: T2.4 with NHANES + NIDDK NAFLD-DB2 transfer, multi-cohort joint
  pretraining

---

## What the user needs to do (manual)

1. **Today, ~30 min:** Submit NIDDK CR data request for NAFLD Adult Database 2
   (+ optionally PIVENS, FLINT, NAFLD Adult Database 1, 3). Account at
   `repository.niddk.nih.gov`. Need an analysis plan + a DUA — I can draft both
   if you want.
2. **Today, ~30 min:** Submit SteatoSITE application via the PMS-IC portal at
   `regeneration-repair.ed.ac.uk/research/research-resources/steatosite`.
   Lighter paperwork.
3. **Today, ~5 min:** Confirm whether NHANES download runs locally (your laptop,
   ~4h) or on the server. If server, I'll write a SLURM script; if local, I'll
   write a python script you point at a target dir.
4. **Today, ~5 min:** Confirm whether the existing in-flight branches
   (R2 / xgbcox / extra-views / stacking / weighted / pycox / tabpfn / tabicl /
   finegray / D9 / catboost) should keep running or be cancelled to free queue
   priority for T1/T2 jobs.
5. **Tomorrow, ~15 min:** Review T1 CV results before final submit; you decide
   the rank-ensemble basket.

Optional but useful for paper-quality v2:
- Apply for UK Biobank access (T3.4) — start within the next week.
- Decide whether TARGET-NASH (T3.5) is worth chasing through institutional
  channels.

---

## Journal / writeup readiness

Trustii's score is 70% C-index + **30% qualitative methodology**, and the
challenge writeup will likely seed a paper. Cleanliness checklist per phase:

### What's clean by construction
- **T1.1, T1.2, T1.3 (foundation-model survival heads):** Clean. Pretrained
  weights are public (HuggingFace), the discretised-classification reduction
  has a citable source (arxiv 2601.22259), and no external data crosses the
  ANNITIA train/val boundary. **Fully reproducible from a `requirements.txt`
  pin + checkpoint hash.** Only caveat: license check on each model
  (TabPFN-2.5 is non-commercial OSS; MITRA is Apache via AutoGluon; TabSurv
  TBD). If submission is non-commercial research, all are fine.
- **T1.5 (SurvTRACE):** Clean. MIT-licensed, peer-reviewed (Wang et al. 2022).
- **T3 cohorts (NIDDK / SteatoSITE / LITMUS / UKBB / TARGET-NASH):**
  Clean *by design* — these are the gold-standard pretraining cohorts that
  reviewers expect. Each comes with a DUA, citation, and acknowledgement
  requirements that you'll satisfy as a side effect of the access process.
  This is the path that makes the paper bulletproof.

### What needs care
- **T1.4 (synthetic augmentation):** This is the most reviewer-sensitive piece.
  Generative augmentation of survival data is a live methodological debate
  (calibration concerns, double-dipping risk, mode collapse on n=47). For a
  *messy first submission* it's fine, but the cleanup pass should:
  - Report results with *and* without augmentation side-by-side.
  - Report calibration metrics (Brier, IBS, slope, intercept) not just C-index.
  - Cite the synthetic generator paper and version-pin the implementation.
  - Pre-specify the augmentation ratio in the methods section, not pick post-hoc.
  - **Best-practice fallback:** if a reviewer balks, drop T1.4 from the
    final ensemble — the rest stands without it.
- **T2 (NHANES pretraining):** Methodologically defensible but needs
  documentation discipline:
  - The hep endpoint mapping (NHANES LMF liver-cause-death → ANNITIA hep
    composite) is a *proxy*. State this explicitly. A reviewer will ask.
  - Subset-selection ablation (T2.3) must be reported in full — picking the
    best subset post-hoc and reporting only that is selective reporting.
    Report the full ablation grid; the chosen subset is a *prespecified
    decision rule output*, not a free choice.
  - Cohort indicator (T2.2e) and stacking (T2.2f) are textbook transfer-
    learning methods — easy to cite and defend.
  - Variable mapping (NHANES → ANNITIA) belongs in supplementary as a table.

### Provenance hygiene (do during the messy pass, save the cleanup)
- **Hash every input.** Add SHA256 of `train.csv`, `val_test.csv`,
  `dictionary.csv`, NHANES download bundle, any T3 data. Saves arguments later
  about which rev was used.
- **Pin every package.** `requirements.lock` for the final submission run, not
  just `requirements.txt`. Foundation-model behaviour shifts across point
  releases.
- **Save model checkpoints**, not just predictions. Reviewers may want to
  re-run on hold-out.
- **Log random seeds per fold per branch.** You're already running seed-bagged
  ensembles via `bagging.py` — make sure the seed list lands in the manifest.
- **Manifest file** per submission: `submission_<id>.json` listing branches
  used, their CV scores, the rank-ensemble weights, the data hashes, the
  package lock, the git SHA. `combine_branches.py` should write this.
- **Train/val/test boundary audit.** For T2 specifically: the NHANES → ANNITIA
  transfer must never see ANNITIA's `val_test.csv` during pretraining. Easy to
  get right; easy to silently break if fine-tuning loops touch the wrong CSV.
  Add an explicit assertion.

### Concrete cleanup workflow for v2 / paper

1. After deadline, freeze the messy v1 codebase as a git tag (`v1-trustii`).
2. Branch `v2-clean` and re-run *only* the components you want in the paper,
   from a fresh worktree with pinned deps.
3. Drop anything that didn't materially help (post-hoc, but transparent — log
   the decision).
4. Add the supplementary tables: variable mapping (T2), full ablation grid
   (T2.3), generator config (T1.4), calibration metrics for every reported
   model.
5. Once T3 cohorts arrive, re-run T2.3-style ablation on each, then a
   joint-cohort condition. The paper's headline claim should be the joint
   cohort, with NHANES-only and NIDDK-only as ablations.

**Bottom line for the deadline:** Submit the messy v1 with everything in T1+T2.
For the *Trustii qualitative writeup* (the 30%), describe the approach honestly
— foundation-model survival head + synthetic augmentation + NHANES transfer —
and flag the proxy-endpoint caveat and calibration check. That earns
methodology points even if a single component (e.g. T1.4) is borderline. The
v2 cleanup happens once T3 data arrives.

---

## Open questions for the user

- **NIDDK DUA wording:** Use stat-cadd Oxford as the data steward, or your own
  named institution? Affects who signs.
- **Hep event composition:** ANNITIA's `dictionary.csv` defines the hep composite
  endpoint as `{decompensation, HCC, liver-related death, transplant}`. NHANES
  LMF only gives liver-related death. Is the rank-correlation between
  liver-related-death-only and the full composite high enough that NHANES
  pretrain helps hep, or should we restrict NHANES use to the death endpoint
  only?
- **Calibration for Trustii:** Does the leaderboard score use rank C-index
  only, or any calibration component? If rank-only, T1.4 (synthetic aug)
  becomes safer.
- **Compute budget:** How many SLURM hours can we burn tonight on `medium` /
  `long`? Affects whether we run the full T2.3 ablation (60 cells) or just the
  most promising 2-3 subsets.
