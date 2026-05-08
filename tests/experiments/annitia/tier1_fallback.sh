#!/bin/bash
# Tier-1 fallback: rebuild the proven 0.9302 recipe from EXISTING pool only.
# Use this if auto_chain failed overnight and we need a guaranteed submission.
#
#   bash tests/experiments/annitia/tier1_fallback.sh

set -eu
REPO=/Users/apunt/repos/KIRBy
cd "$REPO"

LOG=/tmp/annitia_runs/tier1_fallback.log
echo "=== tier1_fallback at $(date) ===" > "$LOG"

# Rebuild meta_avg_aware_weighted from the existing 15 variants (no Round 4
# dependency, no new variants). This is the EXACT recipe that scored 0.9302.
python -u << 'PY' >> "$LOG" 2>&1
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# 15 variants that produced the 0.9302 submission. Order matters for the
# weight pattern (10 base x1, 5 patient-aware x2).
variants = [
    'rank_mean', 'rank_median', 'rank_trimmed', 'residual_greedy',
    'uncertainty_route', 'anomaly_blend', 'disagreement_route',
    'cluster_route', 'specialist_override', 'soft_blend_disagreement',
    'anomaly_router_meta', 'disagreement_router_meta', 'anomaly_blend_meta',
    'weighted_anomaly_per_patient', 'cluster_router_meta',
]
weights = np.array([1.0]*10 + [2.0]*5)
weights /= weights.sum()

dfs = {}
for v in variants:
    p = f'tests/results/annitia/full/sub_master_{v}.csv'
    try:
        dfs[v] = pd.read_csv(p)
    except FileNotFoundError:
        print(f'  MISSING: {v}')

assert len(dfs) >= 10, f"only {len(dfs)} variants present — pool too thin"

trustii_id = next(iter(dfs.values()))['trustii_id'].values
n = len(trustii_id)

def rank01(x):
    return rankdata(x) / len(x)

h = np.zeros(n)
d = np.zeros(n)
present = list(dfs.keys())
present_w = np.array([weights[variants.index(v)] for v in present])
present_w /= present_w.sum()

for w, v in zip(present_w, present):
    h += w * rank01(dfs[v]['risk_hepatic_event'].values)
    d += w * rank01(dfs[v]['risk_death'].values)

out = pd.DataFrame({
    'trustii_id': trustii_id,
    'risk_hepatic_event': h,
    'risk_death': d,
})
dest = 'tests/results/annitia/full/sub_master_tier1_fallback.csv'
out.to_csv(dest, index=False)
print(f'wrote {dest} from {len(present)} variants')
PY

# Pre-submit check on the rebuilt CSV
python -u tests/experiments/annitia/pre_submit_check.py \
    tests/results/annitia/full/sub_master_tier1_fallback.csv >> "$LOG" 2>&1

# Build journal + zip
python -u tests/experiments/annitia/build_journal.py \
    --csv tests/results/annitia/full/sub_master_tier1_fallback.csv \
    --output-dir tests/results/annitia/submissions/ >> "$LOG" 2>&1

echo "=== tier1_fallback done at $(date) ===" >> "$LOG"
ls -la tests/results/annitia/submissions/sub_master_tier1_fallback*
