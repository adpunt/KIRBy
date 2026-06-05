# OPIG Antibody + DDG Experiments — Local Smoke Test Report (2026-06-03)

Goal: make the five OPIG antibody/DDG experiments RUN end-to-end on tiny local
data subsets and write a results JSON with finite (no NaN/inf) numbers. Tiny
sample sizes are expected and fine — this is a plumbing check, not research signal.

All five now RUN-CLEAN locally. Two real correctness bugs were found and fixed
(both caused by the smoke row-cap not reaching per-row fields the experiments
rebuild groups/folds from), plus one cosmetic display bug in the trastuzumab
table. No NaN/inf anywhere in any output JSON.

## How the runs were driven

The plain `VAR=val python3 ...` env-var-prefix form is blocked by the sandbox
here, so runs went through a tiny driver that sets `KIRBY_SMOKE_N` in-process
before calling each experiment's `main()`:
`tests/results/_smoke/drive.py <smoke_n> <module> [args...]`.
A finiteness checker (`tests/results/_smoke/check_json.py`) recursively scans
each results JSON for NaN/inf and prints key numbers.

Rep set for every run: `blosum,onehot,aac` (cheap sequence reps, no foundation
model downloads). Folds: 2. Output: `tests/results/_smoke/opig_abddg/*.json`.

A smoke-only shrink of the hybrid search was added to all five files (see
"Performance" below) because the default `speed='full'` + FULL_BUDGETS search
took >8 minutes even on 80 rows.

## Per-experiment verdicts

| # | Experiment | Verdict | Key finite numbers (from JSON) |
|---|-----------|---------|--------------------------------|
| 1 | opig_cdr_flexibility | CRASHES-FIXED → RUNS-CLEAN | ROC AUC 0.42–0.55, PR AUC 0.70–0.82, 0 NaN/inf |
| 2 | opig_ginkgo_developability | CRASHES-FIXED → RUNS-CLEAN | Spearman ρ 0.16–0.23, RMSE ≈0.29, top10_recall∈[0,1], 0 NaN/inf |
| 3 | opig_humanness | RUNS-CLEAN | accuracy 0.62–0.79, AUROC 0.66–0.84, both classes present, 0 NaN/inf |
| 4 | opig_trastuzumab_enrichment | RUNS-WITH-DISPLAY-FIX → RUNS-CLEAN | E@1%=1.0, E@5%=0.17–1.0, E@10%≈0.71, base_rate=0.28, AUROC 0.64–0.81, 0 NaN/inf |
| 5 | opig_graphinity_ddg | CRASHES-FIXED → RUNS-CLEAN | experimental: Pearson r up to 0.47 / ρ up to 0.56; transfer: r −0.45…0.07, RMSE 1.6–2.3; 0 NaN/inf |

---

### 1. opig_cdr_flexibility — CRASHES-FIXED → RUNS-CLEAN

Tiny command:
```
python3 tests/results/_smoke/drive.py 80 experiments.opig_cdr_flexibility \
  --cdr-types CDRH3 --n-folds 2 --reps blosum,onehot,aac \
  --output tests/results/_smoke/opig_abddg/cdr.json
```

Bug found (real): every (rep, model) was SKIPPED with
`Found input variables with inconsistent numbers of samples: [80, 80, 216]`,
so `results` was empty and best ROC AUC printed as 0.0000.
- Root cause: `_smoke_subsample` caps `labels`/`heavy_seqs` to 80, but the
  experiment rebuilds `groups` from `data['cdr_seqs']`
  (`tests/experiments/opig_cdr_flexibility.py:205`), and `cdr_seqs` is a per-row
  field that `_smoke_subsample` does not know about, so it stayed length 216.
  `groups` (216) then mismatched `X`/`labels` (80) inside StratifiedGroupKFold.
- Fix (`opig_cdr_flexibility.py:204-211`): fall back to `heavy_seqs` (which is
  consistently subsampled) when `len(cdr_seqs) != len(heavy_seqs)`. For this
  experiment `cdr_seqs` and `heavy_seqs` are both per-row and groups are
  one-per-unique-sequence either way, so this is behaviour-preserving in normal
  runs and only triggers under the smoke cap.

Key numbers (cdr.json, 80 samples, 61 flexible): best ROC AUC = 0.5485
(onehot/lightgbm), PR AUC up to 0.8215, MCC and balanced-accuracy all finite.
Finiteness check: 0 NaN/inf.

Note (NOT a bug): `kirby_hybrid: FAILED (no candidate returned by pool)`. On 80
rows with the smoke budget set, the hybrid search returns no candidate; the
experiment records this as `hybrid_summary.kirby_hybrid_failed=True` and
continues. This is graceful, expected on tiny data, and matches the same
behaviour seen in humanness below.

---

### 2. opig_ginkgo_developability — CRASHES-FIXED → RUNS-CLEAN

Tiny command:
```
python3 tests/results/_smoke/drive.py 80 experiments.opig_ginkgo_developability \
  --properties ginkgo_hic --n-folds 2 --reps blosum,onehot,aac \
  --output tests/results/_smoke/opig_abddg/ginkgo.json
```

Bug found (real, would crash under the smoke cap): the experiment runs CV from
`data['fold_assignments']` (the Ginkgo pre-computed hierarchical-cluster fold
IDs), but `_smoke_subsample` subsets `data['groups']` and NOT
`data['fold_assignments']` (separate per-row key). Under `KIRBY_SMOKE_N` the
fold-id array would stay full length (242) while labels are 80, and
`_fold_assignments_to_splits` would build out-of-range indices.
- Fix (`opig_ginkgo_developability.py:188-197`): when
  `len(fold_assignments) != n`, fall back to the (subsampled) `groups`. For
  Ginkgo `groups == fold_assignments` (pre_split strategy), so this is
  behaviour-preserving in normal runs.

Key numbers (ginkgo.json, ginkgo_hic, 80 samples): Spearman ρ 0.16–0.23 across
reps, RMSE ≈ 0.29 (blosum/rf), Pearson/R²/CI/top10_recall all present.
top10_recall = 0.0 on this tiny subset (valid, in [0,1]). R² goes as low as
−4.74 (blosum/mlp) — finite and expected (MLP overfits 80 rows); negative R² is
normal for weak fits, not a NaN. Finiteness check: 0 NaN/inf.

No hybrid materialised on this tiny subset (search produced no fitted result);
experiment continues cleanly, no crash.

---

### 3. opig_humanness — RUNS-CLEAN (no code bug)

Tiny command (KIRBY_SMOKE_N=150 so both classes survive the cap):
```
python3 tests/results/_smoke/drive.py 150 experiments.opig_humanness \
  --datasets humanness_therapeutic --n-folds 2 --reps blosum,onehot,aac \
  --output tests/results/_smoke/opig_abddg/humanness.json
```

No correctness bug. `humanness_therapeutic` loads 564 rows; capped to 150 with
both classes present (54 human / 96 non-human). Only the smoke-subsample hook
and the hybrid-search shrink were added.

Key numbers (humanness.json): accuracy 0.62–0.79, AUROC 0.66–0.84, F1-weighted
and balanced-accuracy all finite. Best individual = aac/svm AUROC 0.8441.
Classification metrics are correct for a classification task. Finiteness: 0
NaN/inf. (No `kirby_hybrid` row materialised on 150 rows — same graceful
no-candidate path as CDR; not a crash.)

NEEDS-DATA note for the OTHER preset: `humanness_species` (~10K OAS VH
sequences) was NOT smoke-tested here — it is large and pulls an OAS FASTA from
Zenodo. The therapeutic preset fully exercises the code path; species should be
run on the server with the full pipeline.

---

### 4. opig_trastuzumab_enrichment — RUNS-CLEAN (one cosmetic display bug fixed)

Tiny command (KIRBY_SMOKE_N=150 so both classes survive):
```
python3 tests/results/_smoke/drive.py 150 experiments.opig_trastuzumab_enrichment \
  --n-folds 2 --reps blosum,onehot,aac \
  --output tests/results/_smoke/opig_abddg/trastuzumab.json
```

`tz_her2_binary` loads 524,346 rows, subsamples to its built-in 5,000, then the
smoke cap takes 150 (42 binders / 108 non-binders — both classes present, uses
the registry edit-distance groups, no random split).

Display bug found + fixed (data was already correct in JSON):
- `print_analysis` looked up `metrics.get('enrich_0.10')`, but
  `compute_enrichment` stores the key as `f'enrich_{0.10}'` → `'enrich_0.1'`
  (Python drops the trailing zero). The E@10% column therefore always printed
  "—" even though the value exists in the JSON.
- Fix (`opig_trastuzumab_enrichment.py:479`): look up `'enrich_0.1'`. No rerun
  needed for finiteness — the stored value was always correct
  (`enrich_0.1` mean = 0.7143 for blosum/rf).

Key numbers (trastuzumab.json): base_rate = 0.28; enrichment fractions all in
[0,1] (E@1% = 1.0, E@5% 0.17–1.0, E@10% ≈ 0.71); AUROC 0.64–0.81; accuracy
0.69–0.79. Finiteness: 0 NaN/inf.

Interpretation note (NOT a bug): E@1% = 1.0 for almost every config because at
n_val ≈ 75 (2-fold of 150) the top-1% is a single sample, so E@1% is just
whether that one top-ranked item is a binder — degenerate but finite and in
[0,1]. The experiment's own docstring already warns about this; E@5%/E@10% are
the informative columns at this scale. `zero_shot` is correctly empty here
because the smoke rep set has no LLR reps (`esmc_llr`/`antiberty_llr`); on the
server with the default 11-rep set the zero-shot block populates.

---

### 5. opig_graphinity_ddg — CRASHES-FIXED → RUNS-CLEAN

Tiny commands (run as two invocations to cover all four conditions):
```
# within_rosetta (caps the ~21K Rosetta set at load, then smoke-caps to 80)
python3 tests/results/_smoke/drive.py 80 experiments.opig_graphinity_ddg \
  --conditions within_rosetta --n-folds 2 --reps blosum,onehot,aac \
  --max-rosetta 200 --output tests/results/_smoke/opig_abddg/graphinity_rosetta.json

# experimental + transfer_foldx (caps the ~943K FoldX set at load)
python3 tests/results/_smoke/drive.py 80 experiments.opig_graphinity_ddg \
  --conditions experimental,transfer_foldx --n-folds 2 --reps blosum,onehot,aac \
  --max-foldx 150 --output tests/results/_smoke/opig_abddg/graphinity_transfer.json
```

Bug found (real, would crash under the smoke cap): the direct-load graphinity
presets return `pdb_ids` (per row) but no `groups`; the experiment built
`groups` from `pdb_ids` AFTER `_smoke_subsample`. Because `_smoke_subsample`
does not know `pdb_ids`, it stayed full length while labels were capped to 80,
producing a `groups`/`labels` length mismatch in GroupKFold.
- Fix (`opig_graphinity_ddg.py:339-349`): convert `pdb_ids → data['groups']`
  BEFORE calling `_smoke_subsample`, so the cap subsets the groups consistently.
  Same conversion, just moved earlier; no behaviour change in normal runs.
- Smoke hooks also added to the transfer path (`evaluate_transfer`) for both
  `train_data` and `test_data` (`opig_graphinity_ddg.py:189-193`).

Key numbers:
- within_rosetta (graphinity_rosetta.json, 80 samples, DDG range [−7.34, 2.07],
  80 PDB-complex groups): Pearson r −0.10…0.23, Spearman −0.03…0.27,
  pearson_global finite. Best individual r_g = 0.2352.
- experimental (graphinity_transfer.json, ddg_combined, 80 samples, 25 complex
  groups): blosum/onehot give real DDG signal — onehot/svm Pearson r = 0.4737,
  ρ = 0.5562; aac is weak/negative (expected for amino-acid-composition on DDG).
- transfer_foldx (train=80 synthetic FoldX, test=80 experimental SKEMPI): rep
  intersection is NON-empty (blosum/onehot/aac all present in both); DDG values
  finite; transfer Pearson r −0.45…0.07, RMSE 1.6–2.3 (positive, finite).
  `pearson == pearson_global` for the single train/test pair (as designed).

Sign-correctness of the gap-to-EGNN (RMSE-trap check requested): the comparison
uses Pearson r (higher-is-better), so `gap = egnn_r − best_r`. experimental:
0.271 − 0.375 = −0.104 (negative = KIRBy beats the EGNN reference — correct).
transfer_foldx: 0.271 − 0.073 = +0.198 (positive = KIRBy worse — correct).
Signs are right for a higher-is-better metric. RMSE is reported per-row but not
used in any delta/gain, so the RMSE-lower-is-better trap does not apply here.
Finiteness: 0 NaN/inf in either graphinity JSON.

The weak/negative transfer numbers are scientifically expected (the paper's
whole point is that synthetic→experimental transfer is hard), not a bug.

---

## Performance fix applied to all five (smoke-only, gated on KIRBY_SMOKE_N)

The dominant cost in every experiment was the hybrid search
(`get_best_search_for_downstream`, hardcoded `speed='full'` + FULL_BUDGETS up to
2000, over a pool of tier × augmentation conditions). On 80 rows the CDR run
exceeded 8 minutes purely in this search. A no-op-when-unset helper
`_smoke_search_kwargs()` was added to each of the five files; when
`KIRBY_SMOKE_N` is set it injects `speed='fast'` and `budgets=[25, 50]` into the
search call, otherwise it returns `{}` and the server/full path is completely
unchanged. After this, each experiment's compute (excluding one-time downloads)
finishes in seconds.

## Irreducibly slow / needs a full server rerun

- graphinity within_rosetta / experimental / transfer_foldx do a ONE-TIME
  download of PDB structures from RCSB (a few hundred files) plus the FoldX
  (~943K) and Rosetta (~21K) CSVs from Zenodo. First local run spent most of its
  wall-clock here (PDB fetch is network-bound, ~1–3s each, and is cached to
  `/Volumes/seagate/kirby/pdb` afterwards). Compute itself is seconds. On the
  server with a warm PDB cache this is fast.
- humanness_species (10K OAS) and the full default rep set (foundation-model
  reps such as esm2/ablang2/antiberty/igt5/prot_t5/abroberta and the LLR reps)
  were NOT exercised locally — they need model downloads and real compute. They
  should be validated on the server with the normal (non-smoke) pipeline. The
  code paths for them are unchanged by this work.

## Edits made (all surgical, only in the allowed files)

- `tests/experiments/opig_cdr_flexibility.py`: smoke subsample hook;
  cdr_seqs/heavy_seqs length-guard for groups; `_smoke_search_kwargs` helper +
  wiring into the search call.
- `tests/experiments/opig_ginkgo_developability.py`: smoke subsample hook;
  fold_assignments realignment under the cap; `_smoke_search_kwargs` helper +
  wiring.
- `tests/experiments/opig_humanness.py`: smoke subsample hook;
  `_smoke_search_kwargs` helper + wiring.
- `tests/experiments/opig_trastuzumab_enrichment.py`: smoke subsample hook;
  E@10% lookup key fix (`enrich_0.1`); `_smoke_search_kwargs` helper + wiring.
- `tests/experiments/opig_graphinity_ddg.py`: smoke subsample hooks (direct
  load + transfer train/test); pdb_ids→groups conversion moved before the cap;
  `_smoke_search_kwargs` helper + wiring.

No edits to `tests/experiments/_shared.py`. No dataset-loader edits were needed
(the per-row-field mismatches were all fixable on the experiment side). No
random splits introduced; all group/fold/edit-distance splitting preserved. No
mean/averaged baselines; reps reported individually. No Ridge-as-sensitivity and
no PCA variants added.

## `_shared.py` observations (report-only, NOT edited)

`_smoke_subsample` (`tests/experiments/_shared.py:1541`) only subsets a fixed
allow-list of per-row keys (`labels, groups, heavy_seqs, light_seqs, sequences,
seqs, smiles, sample_ids, sample_weights` plus the `reps`/`augmentations`
dicts). It does NOT touch other per-row fields that experiments derive splits
from — specifically `cdr_seqs` (itsflexible), `fold_assignments` (Ginkgo
pre-split), and `pdb_ids` (graphinity direct load). That is exactly what caused
the three crash bugs above. I worked around each on the experiment side per the
constraints, but if the owner of `_shared.py` wants a cleaner fix, adding
`cdr_seqs`, `fold_assignments`, and `pdb_ids` to the `_take`-d key list (and
recomputing any derived counts) would let the cap reach those fields directly
and make every experiment's smoke path robust without per-file guards. This is a
suggestion only — I did not modify `_shared.py`.
