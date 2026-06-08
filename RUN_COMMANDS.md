# KIRBy re-run — canonical commands (2026-06-08)

Turnkey. Run on the ARC login node from `/data/stat-ecr/scat9264/KIRBy`. Everything below is on the
`long` partition with sufficient walltime (`run_full_suite.sh` = 7 days; opig = 48h / graphinity 7d;
forward-compat = long 24–48h). The per-dataset jobs run their own `git pull` + `. ./setup.sh` internally.

This run uses the fixed code: precompute cache drift is real (the cache rebuilds with the corrected
methodology — ridge kept in the distance evaluators, leakage-grouping, rep changes), the status tool
is now drift-aware (DONE means the current code actually accepts the cache), the 12h→7d limit is fixed,
and the experiment bugs (s12 no-cliff, c24 few-group, c06/c10/c15/c17 clf-skip, c15 empty-blueprint
crash, genomic download race) are fixed.

---

## 0. Once, before anything
```bash
cd /data/stat-ecr/scat9264/KIRBy && git pull && . ./setup.sh
# Purge the corrupt Graphinity rep cache (values changed, shape didn't → silently reused otherwise):
python tests/purge_rep_cache.py graphinity_foldx graphinity_rosetta graphinity_augmented --all-caches            # dry-run
python tests/purge_rep_cache.py graphinity_foldx graphinity_rosetta graphinity_augmented --all-caches --delete
```

## 1. Classify by precompute status (now drift-aware — DONE = current code will reuse it)
```bash
DONE=$(python tests/check_precompute_status.py --done --distance --names-only)
# NOT-DONE for the cx/sim/noise wave EXCLUDES the 5 OPIG-only datasets (they have no
# complexity/similarity/noise loader — they run via OPIG in step 4):
NOTDONE=$(python tests/check_precompute_status.py --incomplete --distance --names-only \
  | grep -vE '^(cdr_flexibility|chembl_qsar|graphinity_foldx|graphinity_rosetta|humanness_species)$')
echo "DONE: $(echo $DONE | wc -w)   NOTDONE(wave): $(echo $NOTDONE | wc -w)"
```

## 2. Precompute DONE → one combined job per dataset (complexity + similarity + noise)
```bash
CATEGORIES="complexity similarity noise" bash tests/slurm_scripts/submit_full_suite.sh $DONE
```

## 3. Precompute NOT done → forward complexity, then reverse similarity+noise (2 jobs/dataset)
```bash
# forward: complexity (builds general/top precompute)
CATEGORIES="complexity" bash tests/slurm_scripts/submit_full_suite.sh $NOTDONE
# reverse: similarity + noise (similarity builds the distance tier; reverse condition order so it
# doesn't cold-build the same dataset's precompute at the same instant as the forward job)
KIRBY_PRECOMPUTE_REVERSE_ORDER=1 CATEGORIES="similarity noise" bash tests/slurm_scripts/submit_full_suite.sh $NOTDONE
```

## 4. OPIG (own datasets/reps; long partition, sufficient time)
```bash
bash tests/slurm_scripts/submit_opig_experiments.sh all
bash tests/slurm_scripts/submit_opig_surveys.sh
# imrie eval only AFTER imrie pretrain finishes:
# sbatch tests/slurm_scripts/imrie_eval.sh
```

## 5. Federated (reuses the precompute the wave built — run once DONE datasets exist)
```bash
CATEGORIES="federated" bash tests/slurm_scripts/submit_full_suite.sh $DONE
```

## 6. Forward-compatibility (foundation-model reps; own submitter, long partition)
```bash
bash tests/slurm_scripts/submit_forward_compat.sh
```

## 7. Monitoring — pre-scheduled, emails you (no babysitting)
```bash
bash tests/slurm_scripts/schedule_snapshots.sh            # default: every 30 min × 48 = 24 h
# every 30 min for 48 h:
bash tests/slurm_scripts/schedule_snapshots.sh 30 96
# send to a different address:
EMAIL=adelaidepunt@gmail.com bash tests/slurm_scripts/schedule_snapshots.sh
```
Each snapshot emails a concise summary (queue counts + precompute totals); full detail (timing +
audit) is appended to `monitor_<date>.log`. Stop with `scancel --name=kirby_snapshot`.
First confirm mail works: `echo test | mail -s test adelaide.punt@stcatz.ox.ac.uk`.

---

## Notes / watch
- **All on `long`, sufficient time.** run_full_suite=7d; opig=48h (graphinity 7d); forward-compat long.
- **The 5 OPIG-only datasets** (cdr_flexibility, chembl_qsar, graphinity_foldx, graphinity_rosetta,
  humanness_species) are excluded from the cx/sim/noise wave — they only run via OPIG (step 4).
- **Drift is expected and correct** now: jobs rebuild precompute that was built with the old
  methodology, and write the corrected cache. The drift-aware status tool will show DONE climb as
  real (reusable) caches accumulate — no more false "DONE: 33".
- **nt_* caveat:** all nucleotide tasks share one HuggingFace dataset; if concurrent nt_ jobs throw
  `stale file handle`, they just need a resubmit (the data caches after the first success).
- **test-less-on-large-n is NOT applied** (proposed only — needs your sign-off on which reps/models
  for DTA n=30k / nucleotide n=100k). Until then those giants rely on the SVM/MI gates to fit 7 days;
  if they still time out, that proposal is the next lever.
```
