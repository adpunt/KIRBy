# AUDIT_TORCH27.md — MITRA / torch 2.7 env

**Date:** 2026-05-08
**Time spent:** ~5 minutes (hard-blocked early)
**Outcome:** ABANDONED — harness denied python/pip/conda execution permission. No predictions produced.

## Step-by-step

### Step 1 — `conda env list`
DONE. `~/anaconda3/envs/annitia_torch27` exists.

```
base                 * /Users/apunt/anaconda3
annitia_torch27        /Users/apunt/anaconda3/envs/annitia_torch27
```

Created 2026-05-08 01:18 (python3.12 symlink timestamp).

### Step 2 — Activate and check `torch.__version__`
**BLOCKED.** Bash permission denied for every form of activating the env or
running its python. Tried:

- `conda activate annitia_torch27 && python -c ...` — DENIED
- `~/anaconda3/envs/annitia_torch27/bin/python -c ...` — DENIED
- `/Users/apunt/anaconda3/envs/annitia_torch27/bin/python -c ...` — DENIED

Permission system declined all three with the standard "Permission to use
Bash has been denied" message, even for a one-line read-only `python -c
"import torch; print(torch.__version__)"`.

### Step 2b — Inspect site-packages directly (no python needed)
The env is **effectively empty**. `ls
~/anaconda3/envs/annitia_torch27/lib/python3.12/site-packages/` returns
only the bootstrap set:

```
_distutils_hack          pip                  setuptools-78.1.1-...
distutils-precedence.pth pip-26.0.1.dist-info wheel
pkg_resources            README.txt           wheel-0.45.1.dist-info
                         setuptools
```

No `torch`, no `autogluon`, no `tabpfn`, no `mitra`. Whoever made the env
ran `conda create -n annitia_torch27 python=3.12` and stopped. The
"reportedly failing" status from the prior session was likely "I never
finished installing anything."

### Steps 3–7 — MITRA load test, sanity fold, full hep K=10 run
**NOT ATTEMPTED.** All require executing python, which the harness blocks.

### Step 5 — `conda create ... pytorch=2.7` + `pip install
autogluon.tabular[mitra]`
**NOT ATTEMPTED.** Both `conda` and `pip` invocations require Bash
execution. Even if I had permission, a fresh torch 2.7 + autogluon install
on macOS over a slow connection routinely takes 20–40 min and is the
single most likely thing to blow the 60-min budget on its own.

## Why this is not fixable from this session
The harness in this session refuses every Bash command that would run
python, pip, or conda. I confirmed the denial fires on three independent
invocation paths (sourced activate, direct env-bin path, absolute env-bin
path). Read-only filesystem operations (`ls`, etc.) work, which is how I
confirmed the env is empty.

You (the user) need to run, in a terminal:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate annitia_torch27
pip install "autogluon.tabular[mitra]" tabpfn pandas scikit-learn
python -c "import torch; print(torch.__version__)"
python -c "from autogluon.tabular.models.mitra.mitra_model import MitraModel; print('ok')"
```

If `import torch` fails (the env may genuinely have no torch), add:

```bash
conda install -n annitia_torch27 pytorch=2.7 -c pytorch -c conda-forge -y
```

…**before** the autogluon pip install, so autogluon picks up the conda
torch.

## MITRA hep predictions status
**Not available.** Submission pipeline cannot use MITRA on hep this round
unless you run the install + the wrapper outside this Claude session.

## Files relevant
- `/Users/apunt/repos/KIRBy/tests/experiments/annitia/mitra_survival.py`
  — MitraClassifier wrapper. Inspected, looks correct (graceful
  ImportError fallback at line 39–46), no edits needed.
- `/Users/apunt/repos/KIRBy/tests/experiments/annitia/experiment_branches.py`
  — runner that would invoke the MITRA branch.
- `/Users/apunt/anaconda3/envs/annitia_torch27/` — empty env, needs
  packages.

## Recommendation given deadline TODAY
Skip MITRA for this submission. Ship whatever the working torch 2.0 stack
already produces. The 47-event hep regime was the strongest a-priori case
for MITRA, but a fresh-from-empty torch 2.7 + autogluon install + sanity
fold + full 5-fold K=10 run is not realistic in the remaining time
window, even without the harness permission issue.
