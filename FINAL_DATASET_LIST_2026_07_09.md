# KIRBy — Final Dataset List & Restart Plan (2026-07-09)

Supersedes the triage in `DATASET_STATUS_2026_07_07.md` §9 where they disagree. Every
claim below is code-verified with file:line, or explicitly marked as needing server output.

---

## 0. What changed since yesterday

1. **`check_precompute_status.py` is fixed** (commit `c61fc62`). It now reports readiness
   **per experiment family** instead of one union verdict. This is the qm9 problem.
2. **The tz_her2 "rerun with identity" plan is broken.** Arithmetic proof in §3.1. It would
   silently produce a *random split*. This reverses `DATASET_STATUS` §12.4.
3. **The 20-dataset keep-set contradicts the science doc** in two places (§2.3).

---

## 1. The precompute-status bug, explained

`check_precompute_status.py` had ONE `DONE/partial` verdict computed over a union of two
different condition pools:

| Pool | Conditions |
|---|---|
| **MAIN** | `general/{none,uncapped}`, `top/{none,uncapped}`, `top/capped` (only when n ≤ 250) |
| **DIST** | `general_distance/{none,uncapped}`, `top_distance/{none,uncapped}` |

But the families do not consume the same pool. Verified by static read of every
`c*`/`s*`/`n*` experiment file:

| Family | Needs | Evidence |
|---|---|---|
| **complexity** | MAIN | only `c03_cross_dataset_transfer.py:108,118` reads DIST |
| **noise** | MAIN | no noise experiment passes `distance=True`; `noise_helpers.run_all` never precomputes a distance tier |
| **similarity** | MAIN + DIST | `s04,s05,s06,s07,s08,s10,s13` are distance consumers |

So a dataset whose **only** gap is the distance tiers (qm9, logd, adme_hlm, adme_rlm,
bh_doyle_plates) was ready for complexity *and* noise, but the tool called it "not done" —
and `--distance` made it worse by padding the denominator from 4 to 8.

**Now:**

```
  dataset   cx-pre  sim-pre  nz-pre   rows  fresh   missing-for-complexity
  qm9       DONE    4/8      DONE     4     24h
  logd      DONE    7/8      DONE     7     24h
  caco2     1/4     1/8      1/4      1     24h     general/uncapped, top/none, top/uncapped
```

`--family {complexity,similarity,noise,all}` selects what `--done/--incomplete/--names-only`
filter on (default `complexity`). `--distance` is kept as an alias for `--family all`, so
`submit_rerun_wave.sh:58-59` is unchanged.

Two side-fixes: the docstring claimed `general/capped` was expected — it is never
precomputed nor pooled (`_shared.py:2828-2846`, `:3263`). And `--priority` help said 33
while the list has been 20 since `aa8ff46`.

> **Caveat.** The 4/8 and DONE values above come from a synthetic DB I built to test the
> logic. The *real* qm9 state must come from the server — see §6.

### 1.1 Which job builds what (this drives the submission order)

| Job | Builds |
|---|---|
| `run_dataset.py complexity <ds>` | MAIN **and** DIST (`complexity_helpers.run_all:1293,1315,1339,1362`) |
| `run_dataset.py similarity <ds>` | MAIN and DIST (`similarity_helpers.run_all:868,890,916,939`) |
| `run_dataset.py noise <ds>` | MAIN only (`noise_helpers.run_all:212,234`) |

**Complexity is the universal builder.** Chaining `complexity → afterany → similarity` (as
`submit_notdone.sh:71-73` already does) is not a correctness requirement — similarity can
build its own pools — it is a *compute*-efficiency guard so two jobs don't recompute the
same expensive searches. The same argument applies to noise, which is currently unchained.

---

## 2. THE FINAL LIST

### 2.1 RUN SET — 33 datasets

This is exactly `DATASET_STATUS` §9 Bucket 1, and exactly the `presentation` preset in
`tests/analysis/blueprint_trace.py:60-77`. Those two were derived independently and agree,
which is the reason to trust it.

| Domain | n | Datasets |
|---|---|---|
| Antibody | 15 | `abcov_kd`, `abagym_Ang2`, `abagym_VEGF`, `abagym_EGFR`, `flab_expression_koenig`, `humanness_therapeutic`, `flab_thermo_tresanco`, `ginkgo_hic`, `ginkgo_tm2`, `ginkgo_titer`, `ginkgo_pr_cho`, `ginkgo_acsins`, `ginkgo_colloidal`, `ginkgo_purity`, `ginkgo_aggregation` |
| Molecular | 8 | `caco2`, `herg`, `herg_ki`, `adme_mdr1`, `qm9`, `logd`, `adme_hlm`, `adme_rlm` |
| Reaction | 2 | `buchwald_hartwig`, `bh_doyle_plates` |
| TCGA | 2 | `tcga_brca_pam50`, `tcga_luad_vs_lusc` |
| Nucleotide | 6 | `nt_H3K27me3`, `nt_H3K4me3`, `nt_H3K9ac`, `nt_H3K9me3`, `nt_promoter_tata`, `gb_human_enhancers_cohn` |

Plus **`ddg_combined` — conditional**, only after the k-fold fix (§3.2). It is in the
presentation shortlist but not in Bucket 1's ready set.

Note `herg`: sound, but its `graph_kernel` value is from a stale-merge bug
(`complexity_helpers.py:99`) — do not quote that single number.

### 2.2 CUT — 25 datasets, do not run, do not build precompute

| Dataset | Why |
|---|---|
| `humanness_species` | trivial (~0.99 AUROC, germline-separable) |
| `cdr_flexibility` | random-split-in-disguise (`opig_cdr_flexibility.py:148-152` *asserts* group size == 1) |
| `graphinity_foldx`, `graphinity_rosetta`, `graphinity_augmented` | labels are FoldX/Rosetta output, not biology |
| `ginkgo_heparin` | n=94 → ~19/fold |
| `abagym_HER2` | n=146 single-Ab DMS |
| `abagym_lysozyme` | near-dead signal (3/20 reps clear ρ>0.15) |
| `flab_binding_phillips_h3`, `flab_binding_adams`, `flab_expression_adams` | combinatorial libraries; `position_modulo` groups on first mutation only → leaks. No valid grouped split exists (same collapse maths as §3.1) |
| `ddg_combined_full` | non-antibody PPIs, off-domain |
| Jain panel (10): `flab_thermo_jain`, `flab_aggregation_jain`, `flab_aggregation_jain_hicrt`, `flab_aggregation_jain_sas`, `flab_polyreact_jain_bvp`, `flab_polyreact_jain_cic`, `flab_pk_jain_fcrnrt`, `flab_pk_jain_heprt`, `flab_immunogenicity_marks`, `flab_thermo_hutchinson` | ~130 Abs each, 9–10 carved from the SAME Jain 2017 panel → coverage inflation. Present ≤1 illustrative |
| `chembl_qsar` | 0 result files, never run — build it or drop it, don't cite it |
| `tz_her2_binary`, `tz_her2_multiclass` | see §3.1 — **decision required** |

Four of these are **already absent** from `DATASET_REPS` (so already out of the
complexity/precompute path, but still live in the antibody survey):
`graphinity_augmented`, `ddg_combined_full`, `flab_aggregation_jain_sas`,
`flab_thermo_hutchinson`.

**To actually stop a dataset running, it must be removed from all three:**
1. `tests/antibody_baseline_survey.py` `DATASET_REGISTRY` (:53) **and** every
   `DOMAIN_GROUPS` sub-list (:107-135) — `DOMAIN_GROUPS['all']` derives from the registry
   keys, but `all_dms`/`all_inter`/`all_clf`/`all_ginkgo`/`all_ddg` are hardcoded.
2. `tests/experiments/complexity/complexity_helpers.py` `DATASET_REPS` (:94).
3. The hardcoded slurm lists — `submit_precompute.sh:122-138`,
   `submit_missing_htc.sh:104-174`, `resubmit_clean.sh:20`. **These bypass the registries**
   and will resubmit a "removed" dataset.

### 2.3 The 20-dataset keep-set is wrong in two places

`cancel_to_keepset.sh:21-31` (mirrored into `check_precompute_status.py PRIORITY_DATASETS`)
was cut to 20 for compute yesterday. Against the science doc it:

- **keeps `abagym_HER2`** — an explicit Bucket-3 cut (n=146, §2.2 above);
- **drops `flab_thermo_tresanco`** — the doc's **#2 headline** (the VHH/nanobody Tm card,
  `DATASET_STATUS` §1);
- also drops `buchwald_hartwig`, called "strongest non-Ab" in §9's immediate-results plan.

**Minimum correction if you stay at 20:** swap `abagym_HER2` → `flab_thermo_tresanco`.
If a 21st slot exists, add `buchwald_hartwig`.

---

## 3. DATASETS NEEDING A FIX (blocked on your call)

### 3.1 `tz_her2` — the identity rerun does NOT work

`DATASET_STATUS` §12.4 and my memory both say "rerun with `'identity'` to remove all
doubt." **That is wrong, and it fails silently.**

- `_identity_groups` (`antibody_baseline_survey.py:314-336`) concatenates heavy+light and
  calls `sequence_identity_clusters(seqs, identity_threshold=0.95)` — real single-linkage
  Levenshtein clustering (so the doc was right about *what it does*).
- tz_her2 geometry (`src/kirby/datasets/trastuzumab.py:25-32`): VH = 120 aa, VL = 107 aa
  and **identical for every variant**; only a **10-residue CDRH3** varies. Concatenated
  length = 227.
- Worst case, two variants differ at all 10 CDRH3 positions →
  identity = 1 − 10/227 = **0.9559 ≥ 0.95**.
- So **every pair merges**. Single-linkage yields **1 cluster**.
- `sequence_identity_clusters` then hits its degenerate guard
  (`_shared.py:916-924`): `n_clusters < min_groups` →
  `groups = sha256(seq) % n_buckets`.

That is a **per-sequence hash bucketing — a random split** — wearing a group-CV label. It
is exactly the `cdr_flexibility` failure mode, and it violates the lab's no-random-splits
rule.

Corollary: the `edit_distance` grouping that §2.2 of the status doc called "leaky" is in
fact the **only non-degenerate grouping this dataset currently has**.

**Options (your call):**
- **(a) Present as-is** with edit-distance grouping, stated honestly as stricter than every
  published tz_her2 evaluation (Mason/MAMMAL/Chinery are all random 70/15/15), and weaker
  than your own 95%-identity standard. Add an edit-distance-to-WT baseline.
- **(b) Build a real CDRH3-cluster grouping** — cluster on the 10-residue CDRH3 alone (not
  the 227-aa concat), with a threshold that actually partitions. New code, needs design.
- **(c) Cut.**

I recommend **(a)** for the talk and **(b)** later. Do not do the identity rerun.

> **Separate, wider issue:** the `min_groups` hash fallback at `_shared.py:916-924` silently
> converts *any* near-clonal panel into a random split, and it is reachable from the
> `pre_split` / `edit_distance` / `experiment` / `position_modulo` fallback paths in
> `_assign_groups` (`:243-294`) whenever their metadata is missing. Every current
> `identity`-strategy dataset is an inter-family panel (`abcov_kd`, the Jain sets,
> `flab_thermo_tresanco`, `humanness_*`) so none collapse today — but this should probably
> raise instead of hash. **Not changed without your signoff.**

### 3.2 `ddg_combined` — needs k-fold + a stated ceiling
Leave-one-PDB-complex-out over only ~20–32 complexes → coarse, high-variance CV. The code
itself documents residual structural leakage across similar interfaces. Labels mix AB-Bind
direct ΔΔG with Kd-derived SKEMPI ΔΔG. Add k-fold, state the leakage, then run.

### 3.3 `cdr_flexibility` — needs a germline/V-gene group key
The loader dedups on CDR3 (`itsflexible.py:135-137`), so every group is size-1 and
`StratifiedGroupKFold` degenerates. The experiment *asserts* this
(`opig_cdr_flexibility.py:148-152`). A fix needs a coarser cluster key (V-gene or
framework-similarity), relaxing the assert, and ≥ n_folds clusters. Currently **CUT**.

### 3.4 `chembl_qsar` — build or drop
Registered in `DATASET_REPS:974` (reps copied from `caco2`), **0 result files**. Either
build precompute and run it, or drop it. Do not cite it as a result.

---

## 4. SUBMISSION PLAN

### 4.1 The script you were thinking of
`tests/slurm_scripts/where_to_submit.sh` — read-only, touches no jobs. It prints the
allowed accounts, **FairShare per account** (§2), each partition's **wall limit + idle
CPUs** (§3), your pending jobs by reason/wall/partition (§4), and global queue contention
(§5).

**Account and partition are live state. I will not guess them.** Run this and paste the
output, then we pick:

```bash
cd /data/stat-ecr/scat9264/KIRBy && git pull --ff-only origin main \
  && bash tests/slurm_scripts/where_to_submit.sh
```

Its own decision rule: bill to whichever account has the **higher FairShare** (§2); skip a
partition with a huge pending backlog (§5); and if your pending reason is `(Priority)` while
idle CPUs exist, the blocker is the **7-day wall** — a 7-day job cannot backfill into a
short idle gap.

### 4.2 Then, the two buckets (drives directly off the fixed script)

```bash
# Bucket A — needs the builder. complexity builds MAIN + DIST.
python tests/check_precompute_status.py --family complexity --incomplete --names-only

# Bucket B — ready now. Run complexity + noise immediately.
python tests/check_precompute_status.py --family complexity --done --names-only

# Of bucket B, these still need DIST before similarity is cheap:
python tests/check_precompute_status.py --family similarity --incomplete --names-only
```

- **Bucket A** → submit `complexity`; chain `similarity` and `noise` on `afterany`.
  `submit_notdone.sh:71-73` already does this for complexity→similarity. **Noise is
  currently HELD in that script and would need adding** — it duplicates MAIN work if run
  unchained on a cold dataset.
- **Bucket B** → submit complexity + noise in parallel now. Similarity will build its own
  DIST tiers where they're missing (qm9, logd, adme_hlm, adme_rlm, bh_doyle_plates), so
  expect it to be slower on those five.

Existing launchers are all **hardcoded lists with count guards** (`submit_notdone.sh` `-ne
27`, `submit_done.sh` `-ne 42`, `submit_priority.sh` `-ne 33`). Those lists predate this
cut. They must be regenerated from the final 33 before use, or the guards will fire.

`submit_rerun_wave.sh` is the one launcher that reads status dynamically
(`:58-59`) — it is the natural place to drive the new `--family` flag.

### 4.3 Federated + OPIG — partition

Requested walls today:

| Family | Script | Wall | Partition |
|---|---|---|---|
| OPIG (most) | `submit_opig_experiments.sh:43-74` | `47:59:00` | long |
| OPIG `graphinity_ddg` | same | `7-00:00:00` | long |
| OPIG priority (6) | `submit_opig_priority.sh:20-21` | `1-00:00:00` | long |
| OPIG surveys | `submit_opig_surveys.sh:32-35` | `47:59:00` + `gpu:1` | long |
| Federated | via `submit_priority.sh:30-31` | `1-00:00:00` | **medium** |
| Federated | via `submit_dataset.sh:6-7` | `7-00:00:00` | long |

Everything except `opig_graphinity_ddg` asks for ≤ 48 h, so it does **not** need the long
partition, and `where_to_submit.sh` notes long/medium **share the same nodes** — a shorter
wall backfills sooner. **Recommendation: move OPIG and federated to medium**, keep
`opig_graphinity_ddg` on long.

**I cannot confirm medium's wall limit from here** — `where_to_submit.sh` §3 prints it
(`sinfo -o "%.12l"`). If medium's limit is under 48 h, the `47:59:00` OPIG jobs must stay
on long or be split.

Note `opig_graphinity_ddg` (a faithful reproduction, **KEEP**) is a different thing from
the `graphinity_*` **datasets** (CUT). Don't cut the experiment.

### 4.4 Blueprint tracing

`tests/slurm_scripts/run_blueprint_trace.sh` runs `tests/analysis/blueprint_trace.py`
(**not** `c24_blueprint_transfer`, **not** `opig_blueprint_transfer.py`). Its
`presentation` preset is exactly the 33-dataset final list. Submit per domain so each
finishes inside the wall:

```bash
for P in presentation_ab presentation_mol presentation_rxn presentation_tcga presentation_nuc; do
  sbatch --account=<ACCT> --export=PRESET=$P tests/slurm_scripts/run_blueprint_trace.sh
done
```

Header is `--partition=short --time=11:59:00` (`run_blueprint_trace.sh:2-10`). Two gotchas:
it sets **no `--account`** (pass it at submit), and the all-33 single job needs the
documented `--partition=long --time=23:59:00` override — which the per-domain split avoids.

---

## 5. Already-broken things found in passing (not fixed)

- `resubmit_missing.sh:64` and `resubmit_missing_bundled.sh:45` import `_searches_dir` and
  `_check_file` from `check_precompute_status`. **Those names no longer exist** after the
  SQLite redesign. Both scripts are dead. I preserved `expected_conditions`'s signature so
  I did not widen the breakage, but they need a rewrite before use.

---

## 6. What I need from you

1. **Server output** — `bash tests/slurm_scripts/where_to_submit.sh` (account + partition
   are live state).
2. **Server output** — the real qm9 state, to confirm the fix reads correctly on the
   actual cache:
   ```bash
   python tests/check_precompute_status.py --dataset qm9 --verbose
   python tests/check_precompute_status.py --priority --no-experiments
   ```
3. **Decision: tz_her2** — (a) present as-is with edit_distance, (b) build a real CDRH3
   cluster split, or (c) cut. §3.1.
4. **Decision: run-set size** — the full 33, or a corrected 20 (swap `abagym_HER2` →
   `flab_thermo_tresanco`). §2.3.
5. **Decision: `chembl_qsar`** — build or drop. §3.4.
6. **Signoff to edit the registries** — removing the 25 cuts touches
   `antibody_baseline_survey.py`, `complexity_helpers.py`, and 3 slurm lists. I have not
   touched them.
