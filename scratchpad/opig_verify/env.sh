export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
# ⛔ OMP_NUM_THREADS must be 1. A torch scorer running after a LightGBM fit
# segfaults (exit 139) at any higher value -- two OpenMP runtimes in one
# process, libiomp5 via torch and libomp via LightGBM. Measured here 2026-09-18
# at 2 threads, and recorded before at 8.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KIRBY_LGBM_THREADS=1
# Search-space shrink with every row kept: each _smoke_subsample cap is
# `if n > n_keep`, so a cap above every dataset's row count subsamples nothing
# while _smoke_search_kwargs() still cuts the search to budgets=[25], the rf
# internal evaluator alone and lightgbm_gain importance alone.
export KIRBY_SMOKE_N=100000000
export RUNS=/Users/apunt/repos/KIRBy/scratchpad/opig_verify
# joblib/loky honours this for n_jobs=-1. Six concurrent experiments each
# asking for every core is 8x oversubscription on an 8-core laptop.
export LOKY_MAX_CPU_COUNT=2
