# Similarity-metric study — server package

Tests **which molecular similarity metric best reflects real (activity)
similarity**, and **how to evaluate that** — the two questions from the research
brief (`compass_artifact_wf-e835d62c…`, Deliverables 1 & 2), extended with the
**noise-mitigation** use case (Deliverable 3).

**The design principle** (why this isn't circular): a similarity metric is good
if the molecules it calls similar really behave alike — "alike" is defined by
**measured activity**, never by ECFP4 Tanimoto. ECFP4 Tanimoto is one competitor
in the grid, judged by the same yardstick as everything else, never the judge.

## What it runs

| metric family | what | notes |
|---|---|---|
| classical grid | {ECFP4/6, FCFP4, atom-pair, torsion, RDKit, MACCS, ECFP4-count, atom-pair-count, descriptors} × {Tanimoto, Dice, Cosine, Soergel, Tversky, Euclidean} | RDKit, CPU |
| transformer | ChemBERTa, MolFormer embeddings + cosine/euclidean | GPU (V100) |
| learned metric | triplet metric-learner (the "molecules as triplets" idea), structural + activity modes | pure torch (no PyG) |

Every metric is scored by the **activity-grounded battery** (`simeval.py`):
kNN read-across / continuous-MODI, MODI, Patterson neighbourhood-behaviour,
SALI cliff-AUROC, BEDROC enrichment, violation rate, **noise-detection AUROC**,
**noise-recovery corr**. All on held-out scaffold folds (honest scoring, both
arms), aggregated across the 30 MoleculeACE targets with **paired Wilcoxon + SRD**.

The noise evals inject calibrated Gaussian label noise (σ≈0.6, 15% of labels;
Kramer 2012 / Kalliokoski 2013 / Cortés-Ciriano 2015) and measure whether each
metric's neighbourhoods **expose** (detection) and **correct** (recovery) it —
the direct test for the KIRBy noise use case. Recovery uses the same validity
metric as `kirby.utils.noisy_labels.noise_recovery_corr`: corr(applied
correction, −injected ε).

## Prerequisites (login node, once, inside the venv)

```bash
. setup.sh
pip install MoleculeACE                      # ships the 30 benchmark CSVs
python - <<'PY'                              # pre-download HF models (compute nodes may lack net)
from transformers import AutoModel, AutoTokenizer
for m in ['seyonec/ChemBERTa-zinc-base-v1','ibm/MoLFormer-XL-both-10pct']:
    AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    AutoModel.from_pretrained(m, trust_remote_code=True)
PY
```

## Run

```bash
sbatch tests/slurm_scripts/submit_similarity_metrics.sh
# skip GPU/transformers (classical + triplet only, CPU):
WHICH="" sbatch tests/slurm_scripts/submit_similarity_metrics.sh
```

Pipeline: `build_moleculeace.py` → `learned_embeddings.py` →
`run_battery_moleculeace.py`. Outputs land in
`$KIRBY_CACHE_DIR/similarity_metrics/sim_data/`; rsync back
`moleculeace_summary.json` + `moleculeace_rows.json` (JSON-only, per server rules).

## Caveats / knobs

- **Triplet encoder is an MLP over ECFP4**, not a message-passing GNN, because
  `torch_geometric` is ABI-broken in the current env. If a working PyG/DGL is
  available, swapping the encoder in `learned_embeddings.train_triplet` for a GIN
  is a drop-in change; the scientific question (does a *learned* distance beat the
  classical winners) is answered either way.
- **No random splits**: scaffold groups are computed here (Bemis-Murcko); the
  MoleculeACE `split` column is ignored.
- **No pipeline changes**: this is a standalone study. It does NOT touch
  `hybrid.py` / `find_best_hybrid.py`. The local pilot (2 datasets) found the
  search-objective swap gives <0.03 — so a hybrid.py change is not on the table
  until this 30-target run says otherwise.
