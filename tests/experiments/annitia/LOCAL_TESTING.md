# ANNITIA local testing — what works, what needs server

The pipeline has 5 stages, each writes per-condition cached OOFs and
resumes safely on re-run. **Everything below works on a Mac with the
default `setup.sh` deps.**

```
BE → CD → HB → BRANCHES → EXTERNAL → FUSION
```

Run the whole thing locally with:

```
bash tests/experiments/annitia/run_all_local.sh
```

Or one stage at a time (each is RESUME-friendly):

```
python tests/experiments/annitia/experiment_be.py
python tests/experiments/annitia/experiment_cd.py
python tests/experiments/annitia/experiment_hep_boost.py
python tests/experiments/annitia/experiment_branches.py
python tests/experiments/annitia/experiment_external.py
python tests/experiments/annitia/experiment_full.py --skip-be --skip-cd
```

## What works locally (Mac, CPU)

| Stage | Condition | Local? | Notes |
|---|---|---|---|
| BE | All 8 algos × 6 view-sets × 2 endpoints × 5-fold | ✅ | ~25 min |
| CD | C framings + D tricks + Fine-Gray + D9 cross-pollinate | ✅ | ~30 min |
| HB | RSF-1000, GBSA-500, CatBoost, weighted/bagged/snapshot tricks (hep) | ✅ | ~30 min |
| BRANCHES → DSM | DSM + DSM-CR | ✅ | needs torch (already installed) |
| BRANCHES → NHANES MLP (uses pretrained encoder) | ✅ | first time downloads NHANES (~4h) |
| BRANCHES → TSFM view × 6 algos | ✅ | FFT backend default; chronos/timesfm if installed |
| BRANCHES → TabPFN landmark | ✅ | needs `pip install tabpfn` |
| BRANCHES → MITRA discretised (T1.1) | ⚠️ | needs `pip install "autogluon.tabular[mitra]"` (~2 GB weights). If absent, condition is skipped, doesn't break pipeline. |
| BRANCHES → TabPFN-2.5 discretised (T1.1) | ✅ | reuses installed `tabpfn` package |
| BRANCHES → TabSurv (T1.3) | ✅ | torch-only, no extra deps |
| BRANCHES → Synthetic-aug GMM (T1.4) | ✅ | sklearn fallback always works |
| BRANCHES → Synthetic-aug TabDDPM/TabSyn | ⚠️ | needs `pip install synthcity`. Falls back to GMM otherwise. |
| BRANCHES → Synthetic-aug TabEBM (T1.4) | ⚠️ | needs `pip install tab-ebm`. Falls back to GMM. |
| BRANCHES → SurvTRACE (T1.5) | ⚠️ | needs `pip install survtrace`. Skipped if absent. |
| BRANCHES → TabTune PEFT (T1.2) | ⚠️ | needs `pip install git+https://github.com/Lexsi-Labs/TabTune.git`. Skipped if absent. |
| EXTERNAL → NHANES pretrain (T2.0) | ✅ | first run downloads + assembles cohort, saves cache. ~30 min after download. |
| EXTERNAL → T2.1 zero-shot gate | ✅ | trivial post-hoc on encoder |
| EXTERNAL → T2.2 (a-f) recipes | ✅ | sksurv-only; 5-fold each, ~5-15 min per recipe per endpoint per subset |
| EXTERNAL → T2.3 subset ablation | ✅ | 5 subsets × 6 recipes × 2 endpoints = 60 cells (~3-5 h on Mac) |
| FUSION | All 5 modes + stack_meta over union pool | ✅ | ~5 min |

**Bottom line: every stage runs locally.** The only conditions that get
skipped without warning are the optional foundation-model packages above
(MITRA, TabTune, SurvTRACE, TabEBM, synthcity). The pipeline degrades
gracefully — those skip individually and the rest of the run continues.

## When to bother with SLURM

Only when you specifically want:
1. **Wall-clock parallelism with the queue** — short / medium / long are
   identical pipelines, so submit all three concurrently and whichever
   gets furthest before its walltime hits writes the most into the cache.
   Re-running picks up from the latest cache state.
2. **More CPU than your Mac** for the slow stages: BRANCHES with all
   foundation models + EXTERNAL T2.3 ablation can take 5-10h on Mac CPU,
   under 2h on a 16-core SLURM node.
3. **The NHANES download** — the first run pulls ~3-4 GB of XPT files and
   the linked mortality flat-files; doing this on the server with the
   download caching to `/data/stat-ecr/scat9264/KIRBy/data/nhanes/` is
   nicer than on local Wi-Fi.

## Quick local sanity (5 min, no installs needed)

```
# 1. Touch every loader / view / model
python -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'tests')
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import VIEW_BUILDERS, build_views
from experiments.annitia.discretised_survival import DiscretisedSurvival
from experiments.annitia.synthetic_aug import SyntheticAugmentedSurvival
from experiments.annitia.survival import SurvivalCoxPH, concordance
print('views:', sorted(VIEW_BUILDERS))
"

# 2. One fold of one condition end-to-end
python tests/experiments/annitia/experiment_be.py --models cox --n-folds 3
```

If both succeed, the layered pipeline is wired correctly. From there,
`bash tests/experiments/annitia/run_all_local.sh` is the same code path
the SLURM scripts execute.

## Optional install targets (in order of value-for-money)

```
# Highest lift on the rare hep endpoint:
pip install tabpfn                                                       # T1.1 (TabPFN-2.5)
pip install "autogluon.tabular[mitra]"                                   # T1.1 (MITRA)
pip install synthcity                                                     # T1.4 (TabDDPM, TVAE/ARF)

# Medium lift:
pip install tab-ebm                                                       # T1.4 (TabEBM, low-N regime)
pip install git+https://github.com/Lexsi-Labs/TabTune.git                # T1.2 (PEFT fine-tune)
pip install survtrace                                                     # T1.5 (joint competing risks)

# Already installed via setup.sh:
torch, sksurv, lightgbm, ngboost, catboost, xgboost, pycox, dsm
```

## Env vars to control the pipeline

```
ANNITIA_VAL_TEST                — path to val_test.csv (must exist)
ANNITIA_TSFM_BACKEND            — fft (default, no deps) | chronos | timesfm
ANNITIA_K_DISC                  — bins for discretised classifier reductions (10)
ANNITIA_N_SYNTH                 — comma-list for T1.4 sweep (50,100,200,500)
ANNITIA_T23_SUBSETS             — comma-list of T2.3 subsets to run
                                  (fib4_high,fib4_high_2007,fib4_high_2017,t2dm_only,no_viral)

# Skip flags (set to "1" to disable a section)
ANNITIA_SKIP_TABPFN, ANNITIA_SKIP_MITRA, ANNITIA_SKIP_TABPFN25_DISC,
ANNITIA_SKIP_TABSURV, ANNITIA_SKIP_SYNTHETIC, ANNITIA_SKIP_SURVTRACE,
ANNITIA_SKIP_TABTUNE, ANNITIA_SKIP_T21, ANNITIA_SKIP_T22, ANNITIA_SKIP_T23,
ANNITIA_IGNORE_T21_GATE         — if set to 1, run T2.2 even if T2.1 zero-shot fails
```
