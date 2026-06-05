# Post-Precompute Command Queue

Single source of truth for everything that is **blocked on the precompute jobs**
finishing. Run top to bottom once precompute is verified complete.

All commands run from `/data/stat-ecr/scat9264/KIRBy` on ARC after `git pull`.

---

## 0. Gate: precompute must be done first

These all depend on the standard/distance precompute cache being populated.
Verify precompute jobs are in terminal state before submitting anything below:

```
sacct -X --format=JobID,JobName,State,Elapsed,ExitCode | grep -i precompute
```

---

## 1. Complexity + Similarity  (run with `all` reps — pair with precompute)

These were always meant to run with the full rep set *after* the cache is warm.
Datasets: **logd, caco2, herg, qm9** (the `full_cross` part loads all four itself).
Submit one job per dataset × part.

```
# Complexity — all reps
for DS in logd caco2 herg qm9; do
    sbatch tests/slurm_scripts/run_complexity_experiments.sh full_a $DS all
    sbatch tests/slurm_scripts/run_complexity_experiments.sh full_b $DS all
done
sbatch tests/slurm_scripts/run_complexity_experiments.sh full_cross "" all

# Similarity — all reps
for DS in logd caco2 herg qm9; do
    sbatch tests/slurm_scripts/run_similarity_experiments.sh full_a $DS all
    sbatch tests/slurm_scripts/run_similarity_experiments.sh full_b $DS all
done
sbatch tests/slurm_scripts/run_similarity_experiments.sh full_cross "" all
```

---

## 2. OPIG experiments  (blocked on precompute)

```
# 6 find_best_hybrid comparison experiments
bash tests/slurm_scripts/submit_opig_experiments.sh

# 4 survey jobs + imrie pretrain
bash tests/slurm_scripts/submit_opig_surveys.sh
# imrie eval is manual AFTER imrie pretrain finishes:
sbatch tests/slurm_scripts/imrie_eval.sh
```

Asset note: `qsar` + `imrie` use **mol2vec** — symlink `~/model_300dim.pkl`
is in place, verified.

### NEW OPIG experiments (incoming — not yet wired in)

When the new OPIG `.py` files land, they need a `submit ...` block added to
`submit_opig_experiments.sh` (guarded by a `FILTER` name). TODO — paste files here.

---

## 3. PT-Rep pretraining  (blocked on precompute)

```
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_tm2
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_titer
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_pr_cho
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_acsins
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_purity
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_aggregation
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_colloidal
sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_heparin
```

---

## 4. Noise experiments

Run via the per-dataset suite runner, noise category only. Same dataset set
as complexity/similarity:

```
CATEGORIES="noise" bash tests/slurm_scripts/submit_full_suite.sh logd caco2 herg qm9
```

---

## 5. Federated experiments

Same runner, federated category:

```
CATEGORIES="federated" bash tests/slurm_scripts/submit_full_suite.sh logd caco2 herg qm9
```

To run noise + federated together in one pass:

```
CATEGORIES="noise federated" bash tests/slurm_scripts/submit_full_suite.sh logd caco2 herg qm9
```

---

## 6. Fine-tuning — ALREADY DONE, no action

FT precompute is complete and cached — it is **not** blocked on this precompute
and needs nothing re-run. Per `PRECOMPUTE_JOB_TRACKER.md` (Phase B, 2026-04-20→23)
and `PIPELINE_READINESS_TODO.md`:

- `submit_ft_precompute.sh` — molecular, antibody, davis/kiba/bindingdb_kd v2: all COMPLETED
- `submit_ft_precompute_adme.sh` — adme_hlm, adme_rlm: COMPLETED
- `submit_molecular_ft_fill.sh` — all 4 (qm9, adme_hlm/rlm/mdr1): COMPLETED
- `submit_antibody_ft_fill.sh` — all 29 datasets: COMPLETED

FT embeddings live in `$KIRBY_CACHE_DIR/finetuned_reps/` and are picked up
automatically by downstream hybrid search via `load_or_compute()`.

---

## 7. Verify the `metz` drug-target dataset (NEW 2026-06-05 — run BEFORE any metz job)

`metz` was added to the DTA loader + registry, but the dev box's PyTDC ships only
6 DTI sets and **did not include Metz** — so the exact TDC name (`Metz`) is
UNCONFIRMED. Verify the cluster's PyTDC has it before relying on it:

```
# (a) list this PyTDC build's DTI datasets + whether metz is present
python -c "from tdc.utils import retrieve_dataset_names; n=retrieve_dataset_names('DTI'); print('DTI sets:', sorted(n)); print('metz present:', 'metz' in [x.lower() for x in n])"

# (b) if present, confirm it loads and inspect columns
python -c "from tdc.multi_pred import DTI; d=DTI(name='Metz'); df=d.get_data(); print('metz rows:', len(df)); print(df.columns.tolist())"

# (c) if (a) says present: False -> upgrade and re-check (a)
pip install -U PyTDC
```

If Metz is still unavailable after upgrade, just skip it — the loader raises a
clear TDC error and nothing else is affected. If it IS available, run the DTA
baseline survey on it:

```
python tests/dta_baseline_survey.py metz --max-pairs 30000
```
