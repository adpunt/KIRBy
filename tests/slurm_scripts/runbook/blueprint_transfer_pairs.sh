#!/bin/bash
# Run every blueprint transfer pair in tests/experiments/blueprint_transfer_pairs.txt
# (Adelaide, 2026-10-30). Plain commands: run it inside an srun shell or locally.
#
#   bash tests/slurm_scripts/runbook/blueprint_transfer_pairs.sh <output dir> [pairs file]
#
# One pair is one run of opig_blueprint_transfer.py. Its result lands at
# <output dir>/opig_blueprint_transfer/<source>_to_<target>.json, and a pair
# whose .json already exists is skipped, so the same command can be restarted.
# A pair that stops partway resumes from its .partial.jsonl.
set -uo pipefail
OUT="${1:?usage: blueprint_transfer_pairs.sh <output dir> [pairs file]}"
PAIRS="${2:-tests/experiments/blueprint_transfer_pairs.txt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"
cd "$REPO" || exit 2
mkdir -p "$OUT/opig_blueprint_transfer" "$OUT/logs"
export PYTHONWARNINGS=ignore
n_done=0; n_run=0; n_fail=0
while read -r SRC TGT _; do
    case "$SRC" in ''|\#*) continue ;; esac
    RES="$OUT/opig_blueprint_transfer/${SRC}_to_${TGT}.json"
    if [ -f "$RES" ]; then
        echo "=== $SRC -> $TGT: result exists, skipped"; n_done=$((n_done+1)); continue
    fi
    LOG="$OUT/logs/blueprint_${SRC}_to_${TGT}.log"
    echo "=== $SRC -> $TGT: started $(date -Iseconds), log $LOG"
    python tests/experiments/opig_blueprint_transfer.py \
        --source "$SRC" --target "$TGT" --mode full --output "$OUT" > "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== $SRC -> $TGT: done $(date -Iseconds)"; n_run=$((n_run+1))
    else
        echo "=== $SRC -> $TGT: FAILED rc=$rc, see $LOG"; n_fail=$((n_fail+1))
    fi
done < "$PAIRS"
echo "finished: $n_run run, $n_done already done, $n_fail failed"
