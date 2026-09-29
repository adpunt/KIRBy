# Server copy of the 2026-09-28 local test settings. Source AFTER `KIRBY_SKIP_INSTALLS=1 . ./setup.sh`.
# Never run a plain `. setup.sh` in this folder: its `pip install -e .` would point the shared
# venv's kirby at this folder, and your real queued jobs would import this code.
export LT=/data/stat-ecr/scat9264/KIRBy_localtest/local_test_run
export RUN=/data/stat-ecr/scat9264/localtest_run
# Own cache: the folders this run writes are its own; everything else is a link to the real cache.
export KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby_localtest
export PYTHONPATH=/data/stat-ecr/scat9264/KIRBy_localtest/src:/data/stat-ecr/scat9264/KIRBy_localtest/tests
# The default hybrid: one configuration, searched by the real engine.
export KIRBY_LOCAL_GRID=1
export KIRBY_LOCAL_ALLOCATIONS=performance_weighted
export KIRBY_LOCAL_IMPORTANCE=lightgbm_gain
export KIRBY_LOCAL_EVALUATORS=rf
export KIRBY_LOCAL_REP_COUNTS=5
export KIRBY_LOCAL_BUDGETS=250
export KIRBY_LOCAL_MODELS=rf,svm,lightgbm
export KIRBY_LOCAL_FOLDS=3
export KIRBY_LOCAL_AUG_CONDITIONS=none
export KIRBY_RF_JOBS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE KIRBY_LGBM_THREADS=1
export KIRBY_NOISE_CONTROLS=1
