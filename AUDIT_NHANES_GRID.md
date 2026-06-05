# AUDIT — NHANES → ANNITIA Transfer Grid (T2.0–T2.4)

Date: 2026-05-08
Auditor: agent run, 90-min cap, local Mac (no SLURM, no server SSH).

## Executive summary

The prior `external` stage (round3.log + external.log on auto-chain 2026-05-07) was a **silent failure**. All five subsets reported `hep_c=n/a death_c=n/a` because of two bugs (below). The bugs are now fixed and one cell of the grid completed locally before the time cap; the rest of the grid (4 subsets × 6 recipes × 2 endpoints minus what finishes in the trailing run) still needs a server-side run.

## Bugs fixed in this audit

1. **Zero-shot eval crashed with `'NoneType' object is not callable`.** In `tests/experiments/annitia/experiment_external.py::_zeroshot_eval`, the code called `m._load_encoder(...)` (which *returns* `(trunk, head)` but does **not** assign `m._trunk`) and then immediately tried `m._trunk(Xt)`. Fixed by capturing the return value and assigning `m._trunk` and `m._head` before use; also calling `m._trunk.eval()`.
2. **T2.3 subset variation never reached the NHANES encoder.** `_ensure_nhanes_encoder` shelled out to `pretrain_nhanes.py --restrict <restrict>` only — no `--cycle-range`, no `--t2dm-only`, no `--no-viral` — so all five subset encoders trained on identical NHANES (n=7291, all cycles, no T2DM filter, no viral exclusion). Fixed by:
   - Adding `--cycle-range`, `--t2dm-only`, `--no-viral` flags to `pretrain_nhanes.py` (wired through `load_nhanes_liver`).
   - Forwarding the corresponding env vars (`ANNITIA_NHANES_CYCLE_RANGE`, `ANNITIA_NHANES_T2DM`, `ANNITIA_NHANES_NOVIRAL`) to the subprocess `cmd` in `_ensure_nhanes_encoder`.

The ANNITIA-side recipe code in `nhanes_transfer.py` was already env-aware via `_env_subset_kwargs`, so the recipes themselves had been seeing the right NHANES cohorts. The encoder-MLP path is the one that was broken.

## Coverage matrix

Re-run launched at `tests/results/annitia/external_v2/` (the prior `external/` is left intact for diff). Per-cell state at audit close:

| subset / recipe          | t2-a-bayesprior | t2-b-rsf-warmstart | t2-c-gbsa-continued | t2-d-peft-mlp | t2-e-joint-cohort | t2-f-nhanes-risk-feat |
|--------------------------|-----------------|--------------------|---------------------|---------------|-------------------|-----------------------|
| fib4_high                | DONE (hep+dth)  | running            | pending             | pending       | pending           | pending               |
| fib4_high_2007           | pending         | pending            | pending             | pending       | pending           | pending               |
| fib4_high_2017           | pending         | pending            | pending             | pending       | pending           | pending               |
| t2dm_only                | pending         | pending            | pending             | pending       | pending           | pending               |
| no_viral                 | pending         | pending            | pending             | pending       | pending           | pending               |

T2.1 zero-shot for fib4_high: `hep_c = 0.5033`, `death_c = 0.4162`. Both below the gate (0.55 / 0.65), but `--ignore-t21-gate` was set so T2.2 ran anyway. (For the remaining subsets, zero-shot will compute on the same path now that the bug is fixed.)

The local Mac is throughput-bound on RSF (recipe b) at NHANES n≈7291, n_estimators=200 — wall-clock ~3 min per fold, ~12 min per endpoint, ~24 min per recipe-b cell. The full grid (60 cells) is therefore not finishable in 90 min on this machine; it must be re-launched on the server.

## Top cells by weighted LB (0.7·hep + 0.3·death)

Only one (subset × recipe) pair has both endpoints completed:

| subset    | recipe              | hep_C  | death_C | weighted_LB |
|-----------|---------------------|--------|---------|-------------|
| fib4_high | t2-a-bayesprior-cox | 0.7788 | 0.7826  | 0.7799      |

Reference ANNITIA leaderboard high-water mark on the same training set is wLB ~0.93 (cf. `project_annitia_2026_05_08`); a single-recipe NHANES-prior CoxPH at wLB 0.78 on the train-set 5-fold is in the right ballpark for an unaugmented Cox baseline but **is not yet competitive with the existing best ANNITIA branch**.

## T2.4 winning recipe — provisional

With one cell complete the winner is trivially `t2-a-bayesprior-cox @ subset=fib4_high`. This is **provisional** — running the full grid on ARC is required before a real T2.4 decision.

## Wiring check — is T2.4 plugged into refuse_meta_aware / experiment_full?

- `tests/experiments/annitia/experiment_full.py` already includes `external` in its default `--extra-pools` (`be,cd,hep_boost,branches,external`). It will pool any `oof_*.npy`/`val_*.npy` it finds under `tests/results/annitia/external/`.
- `tests/experiments/annitia/refuse_meta_aware.py` similarly hardcodes the pool list `('be','cd','hep_boost','branches','external','hparam')` at line 185.

So T2.4 is *structurally wired* — once the full grid is re-run **into `tests/results/annitia/external/`** (not `external_v2/`), it will show up. Two options:
1. After the server run, copy/rsync the `external_v2/` outputs back into `external/`.
2. Or change the local run target dir back to `external/` (currently it's set to `external_v2` only because we wanted to avoid clobbering the existing-but-broken outputs while fixes were unverified).

Right now the only thing in `external/` is the broken zeroshot.json + 5 dummy encoders that all match `fib4_high` (per-subset variation never took effect). These should be deleted before the new `external_v2` results are copied in.

## Files updated by this audit

- `tests/experiments/annitia/experiment_external.py` — fixed zero-shot, forwarded subset env vars to pretrain CLI.
- `tests/experiments/annitia/pretrain_nhanes.py` — added `--cycle-range`, `--t2dm-only`, `--no-viral`.
- `tests/results/annitia/full/nhanes_grid.csv` — partial grid (1 row pair).
- `tests/results/annitia/full/_build_grid.py` — aggregator script (re-runnable).

## Recommended server-side command

```
sbatch tests/slurm_scripts/<external_resubmit>.sh   # if it exists, otherwise:
python -u tests/experiments/annitia/experiment_external.py \
    --validation-csv /data/.../annitia/val_test.csv \
    --output-dir tests/results/annitia/external \
    --t23-subsets fib4_high,fib4_high_2007,fib4_high_2017,t2dm_only,no_viral \
    --ignore-t21-gate \
    --n-folds 5 --seed 42
```

NHANES raw files already exist at `/Volumes/seagate/kirby/nhanes/{continuous,mortality}` locally; on the server, set `NHANES_DATA_DIR=/data/stat-ecr/scat9264/KIRBy/data/nhanes` and pre-stage the same XPT/.dat files (or let the script download — the URLs in `nhanes_liver.py` are public CDC).

## Open questions for user

- Is wLB 0.78 from a single CoxPH+NHANES-prior recipe useful, or is the bar effectively the existing ~0.93 fusion result? If the former, we should keep going on T2.4. If the latter, T2.4 is unlikely to move the needle and you may want to deprioritise NHANES transfer in favour of NIDDK/SteatoSITE (T3).
- Confirm whether the existing broken `external/` directory contents can be deleted before the new run lands.
