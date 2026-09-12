#!/bin/bash
# Local ranking-rule test. One dataset, three donor sources in sequence so no
# single run passes three hours. The first run also generates whatever
# representations are missing from the local cache, once.
set -u
ds="$1"
base="$2"   # PIN IT. Without rep_prune scores in the local cache f24 falls back
            # to the widest representation in the pool, which is not what the
            # server ran: caco2 took grover at 4800 features here against
            # graph_kernel at 317 there, and flab_expression_koenig took
            # esm3_cdr at 3072 against prot_t5 at 2048. It says so in its own
            # output -- "chosen by: NOT MEASURED" -- and 24 hours went into the
            # wrong experiment on 2026-09-08 because nobody read that line.
export KIRBY_LGBM_THREADS=1
for src in curated borrow both; do
  echo "=== $(date +%H:%M) $ds $src"
  python -u tests/experiments/framework/f24_augmentation_admission.py \
      --dataset "$ds" --base-rep "$base" --placement single --methods uncapped \
      --sources "$src" --caps 64,off \
      >> "scratchpad/rank_${ds}.log" 2>&1 \
    || echo "!!! $ds $src FAILED with $?"
done
echo "=== $(date +%H:%M) $ds done"
