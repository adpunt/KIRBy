#!/bin/zsh
cd /Users/apunt/repos/KIRBy
mkdir -p tests/results/_smoke/opig_abddg
export KIRBY_SMOKE_N=80
export PYTHONUNBUFFERED=1
python3 tests/experiments/opig_cdr_flexibility.py \
  --cdr-types CDRH3 --n-folds 2 --reps blosum,onehot,aac \
  --output tests/results/_smoke/opig_abddg/cdr.json \
  > tests/results/_smoke/opig_abddg/cdr.log 2>&1
echo "EXIT=$?"
