#!/bin/bash
# Everything still open on the server, in one place. 2026-09-18.
#
# Run from /data/stat-ecr/scat9264/KIRBy with the environment sourced:
#     cd /data/stat-ecr/scat9264/KIRBy && git pull && . setup.sh
#
# Each block is independent. Run them in any order, or one at a time.
# Nothing here deletes anything.
#
#     bash tests/tools/close_open_items.sh torch
#     bash tests/tools/close_open_items.sh freshness
#     bash tests/tools/close_open_items.sh nested
#     bash tests/tools/close_open_items.sh qsar
#     bash tests/tools/close_open_items.sh f29
#     bash tests/tools/close_open_items.sh all      # the three cheap ones
set -uo pipefail
WHAT="${1:-all}"

run_torch() {
    echo "=============== is torch working on ARC ==============="
    # If this fails, chemberta, molformer, selformer, smited, grover, mhg_gnn
    # and the dnn model are all unavailable here — which would explain dnn's
    # absence from the per-model timings in precompute_coverage better than the
    # fast-tier model list does.
    python - <<'PY'
try:
    import torch
    print(f'  torch {torch.__version__}   cuda available: {torch.cuda.is_available()}')
    import torch.fft            # libcufft lives behind this one
    print('  torch.fft imports — libcufft is fine')
except Exception as exc:
    print(f'  ⛔ {type(exc).__name__}: {exc}')
PY
    echo
    echo "  and whether the torch-backed builders load:"
    python - <<'PY'
import sys, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import kirby.representations.molecular as M
print(f'  molecular.py imported; torch handle present: {M._torch_module is not None}')
if M._torch_module is None:
    print(f'  reason: {type(M._TORCH_IMPORT_ERROR).__name__}: {M._TORCH_IMPORT_ERROR}')
    print('  RDKit builders still work; chemberta and the rest do not.')
PY
}

run_freshness() {
    echo "=============== does the cache match the code ==============="
    echo "  NO grep this time — chembert is the one being looked for, and a"
    echo "  filter on 'pdv' would hide it."
    echo "  ⛔ chembert (256 features) is NOT chemberta (384). chembert is what"
    echo "     adme_hlm, herg_ki, logd and qm9 carry; chemberta is in qm9 alone."
    python tests/experiments/opig_shared_vectors_census.py \
        --datasets caco2 herg_ki logd adme_mdr1 --check-freshness 200
    echo
    echo "  Read every 'CACHED MATRIX NO LONGER MATCHES THE CODE' line above."
    echo "  pdv on caco2/herg_ki was fixed 2026-09-18 and should NOT appear."
    echo
    echo "  ⛔ AN ABSENT WARNING IS NOT A PASS. A representation the check never"
    echo "     rebuilt produces no line either way. What was actually COVERED:"
    python - <<'PY'
import json, os
p = 'tests/results/comparisons/opig_shared_vectors_census.json'
if not os.path.exists(p):
    print(f'     no {p}'); raise SystemExit
blob = json.load(open(p))
# The census writes its per-dataset block under 'per_dataset'. Asking for
# 'meta' and then falling back to "the first value that is a list" picked
# 'per_representation' instead -- one entry per (dataset, representation),
# none of which carries cache_matches_the_code. Every one printed
# 'freshness NOT RUN', 47 lines on 4 datasets, while the check had in fact
# run. Read the key by name, and say so if it is missing.
metas = blob.get('per_dataset') if isinstance(blob, dict) else None
if metas is None:
    print('     no per_dataset block in the census JSON -- '
          'it was written by an older version of the census')
    raise SystemExit
for m in (metas or []):
    if not isinstance(m, dict):
        continue
    ds = m.get('dataset', '?')
    checks = m.get('cache_matches_the_code')
    if not checks:
        print(f'     {ds:12s} freshness NOT RUN')
        continue
    ok = [r for r, v in checks.items() if isinstance(v, dict) and v.get('matches') is True]
    bad = [r for r, v in checks.items() if isinstance(v, dict) and v.get('matches') is False]
    skip = [r for r, v in checks.items() if not isinstance(v, dict) or v.get('matches') is None]
    print(f'     {ds:12s} matched {len(ok)}: {", ".join(sorted(ok))}')
    if bad:
        print(f'     {ds:12s} ⛔ STALE {len(bad)}: {", ".join(sorted(bad))}')
    if skip:
        print(f'     {ds:12s} not checked {len(skip)}: {", ".join(sorted(skip))}')
    # ⛔ AND THE ONES THE CHECKER CANNOT REBUILD AT ALL never reach
    # cache_matches_the_code, so "matched 5" reads as "all 5 are fine" when the
    # pool holds 10. Name them.
    pool = m.get('representations_counted') or []
    never = [r for r in pool if r not in checks]
    if never:
        print(f'     {ds:12s} NO BUILDER in the checker, never verifiable '
              f'here ({len(never)}): {", ".join(sorted(never))}')
PY
}

run_nested() {
    echo "=============== nested CV against per-fold reselection ==============="
    # Nineteen experiments report a hybrid-minus-baseline number built by
    # re-picking the winner per fold over ONE whole-dataset search. Five run a
    # real nested CV. Nobody has measured how far apart the two are.
    #
    # ⛔ Wants a compute node and a sourced environment. A node that comes up in
    # `base` has no lightgbm, and under speed='fast' lightgbm_gain is the only
    # importance method, so every configuration fails. Since 2026-09-18 that
    # stops the run instead of failing 2,160 times in silence.
    python tests/compare_nested_vs_reselection.py herg_ki
    echo
    echo "  Result: tests/results/nested_vs_reselection/herg_ki.json"
    echo "  The row that matters is whether the two AGREE ON THE SIGN for each"
    echo "  downstream model. If they do everywhere, the nineteen stand."
}

run_qsar() {
    echo "=============== chembl_qsar's pdv, per target ==============="
    # chembl_qsar loads per TARGET, not by name, so pdv_cache_audit.py cannot
    # reach it. No result database exists for it yet, so nothing is
    # contaminated — but one will be built.
    python - <<'PY'
import sys, os, hashlib, warnings
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests'); sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
from kirby.datasets.chembl_qsar import load_chembl_qsar

root = os.environ.get('KIRBY_CACHE_DIR', os.path.expanduser('~/.cache/kirby'))


def zscored(X):
    X = np.asarray(X, dtype=np.float64)
    keep = X.std(axis=0) > 1e-9
    if not keep.any():
        return None
    return bool(np.abs(X[:, keep].mean(axis=0)).max() < 1e-3
                and np.abs(X[:, keep].std(axis=0) - 1.0).max() < 1e-3)


for target in ('d2', 'fxa', 'mpro'):
    try:
        d = load_chembl_qsar(target)
    except Exception as exc:
        print(f'  {target:6s} could not load — {type(exc).__name__}: {exc}')
        continue
    sm = d.get('smiles') or d.get('SMILES')
    if sm is None:
        print(f'  {target:6s} no smiles key; keys are {sorted(d)[:8]}')
        continue
    h = hashlib.sha256('\n'.join(list(sm)).encode()).hexdigest()[:16]
    p = f'{root}/reps/{h}/pdv.npy'
    if not os.path.exists(p):
        print(f'  {target:6s} {len(sm)} molecules — no cached pdv, nothing to fix')
        continue
    print(f'  {target:6s} {len(sm)} molecules — reps/{h}/pdv.npy '
          f'z-scored={zscored(np.load(p))}')
    print(f'         mv {p} $KIRBY_CACHE_DIR/pdv_stale_20260918/reps_{target}_{h}_pdv.npy')
PY
    echo
    echo "  A 'z-scored=True' line means that target's pdv is stale. Run the mv"
    echo "  it prints, then load that target once to rebuild."
}

run_f29() {
    cat <<'TXT'
=============== f29 needs its dnn and cnn rows re-run ===============

  WHY. Until 2026-09-18 dnn and cnn carved their early-stopping holdout at
  random, not by scaffold group, so near-duplicate scaffolds sat on both sides
  and early stopping stopped too late. Every f29 dnn and cnn score is at the
  wrong LEVEL — not contaminated, the carve is inside the training fold, but
  not comparable with the tree models beside it. f29's "DNN wins 4/5" rests on
  those rows.

  f29 reads and writes NOTHING in result_db, searches/ or the hybrid caches —
  its own header says so — so there is nothing to delete first.

  ⛔ --arms default ONLY, and that is not a cut. Neither dnn nor cnn has an
  entry in _HP_SEARCH_SPACES, so tuned_per_rep and tuned_per_model fit exactly
  the same estimator as default and would store the same number three times.
  Running one arm is a third of the work and loses nothing.

  srun ONLY — a queued job waits about a week. One dataset per session, each
  well inside a 4-hour wall. Run them in any order; each writes its own file.

    for DS in tcga_brca_pam50 caco2 buchwald_hartwig flab_expression_koenig logd; do
      srun --time=04:00:00 --cpus-per-task=8 --mem=64G --pty bash -c "
        cd /data/stat-ecr/scat9264/KIRBy && . setup.sh &&
        python tests/experiments/framework/f29_model_and_tuning_survey.py \
          --datasets $DS --models dnn cnn --arms default --tag groupaware"
    done

  Ordered cheapest first from f29's own stored seconds-per-fit: tcga 1.22s,
  caco2 3.78s, buchwald 5.88s, koenig 8.11s, logd 37.86s. logd is the one that
  might not fit; if it runs out of wall, do it a fold at a time:

    for F in 0 1 2 3 4; do
      srun --time=04:00:00 --cpus-per-task=8 --mem=64G --pty bash -c "
        cd /data/stat-ecr/scat9264/KIRBy && . setup.sh &&
        python tests/experiments/framework/f29_model_and_tuning_survey.py \
          --datasets logd --models dnn cnn --arms default \
          --fold $F --tag groupaware_f$F"
    done

  --tag keeps the new rows beside the old ones rather than overwriting them, so
  the size of the shift can be read rather than guessed.
TXT
}

case "$WHAT" in
    torch)     run_torch ;;
    freshness) run_freshness ;;
    nested)    run_nested ;;
    qsar)      run_qsar ;;
    f29)       run_f29 ;;
    all)       run_torch; echo; run_qsar; echo; run_f29 ;;
    *)         echo "usage: $0 {torch|freshness|nested|qsar|f29|all}"; exit 2 ;;
esac
