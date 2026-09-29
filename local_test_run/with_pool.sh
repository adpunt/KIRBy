#!/bin/bash
# with_pool.sh <dataset> <command...>: restrict the representation pool to the dataset's test pool
# (top-5 by rf, which the default hybrid uses, plus top-2 by svm and top-2 by lightgbm).
ds="$1"; shift
f=$LT/pools/$ds.json
[ -f "$f" ] || { echo "### STOP: no pool file $f, so every representation would be searched"; exit 2; }
if [ -f "$f" ]; then export KIRBY_LOCAL_REPS=$(python -c "import json;print(','.join(json.load(open('$f'))['pool_dirty']))"); echo "### KIRBY_LOCAL_REPS=$KIRBY_LOCAL_REPS"; fi
exec bash -c "$*"
