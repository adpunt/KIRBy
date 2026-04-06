# SLURM Job Output Debugging Report

**Date:** 2026-04-06
**Problem:** SLURM jobs show only "Activated venv: ..." in output, with no subsequent output, even after hours of runtime. Affects framework experiments (F15) and possibly precompute jobs.

## Symptom

```
$ cat slurm-11733782.out
Activated venv: /data/stat-cadd/scat9264/py311-kirby (Python 3.11.3)
```

Job ran for 6+ hours in R (running) state. No Python output visible.

## Investigation Summary

Five parallel investigations were conducted:

### 1. SLURM Script Comparison (Working vs Broken)

| Feature | F15 (broken) | Precompute (works) | F08-F14 (work) |
|---------|-------------|-------------------|----------------|
| KIRBY_SKIP_INSTALLS | YES | YES | NO |
| git pull in WRAP | NO | NO | YES |
| PYTHONUNBUFFERED | YES (added in fix) | NO | NO |
| Echo after setup.sh | NO | NO | NO |
| flush=True on prints | YES (added in fix) | NO | NO |

**Key finding:** ZERO submit scripts have diagnostic output between setup.sh and Python. If setup.sh hangs or Python crashes silently, you see only "Activated venv" with no indication of what went wrong.

**Only 2 scripts use KIRBY_SKIP_INSTALLS=1:** submit_precompute.sh and submit_framework_f15.sh.

### 2. setup.sh Hang Analysis (KIRBY_SKIP_INSTALLS=1 path)

After printing "Activated venv" (line 20), the SKIP_INSTALLS path runs:
1. Lines 27-35: Set env vars (HF_HOME, TORCH_HOME, etc.) - cannot hang
2. Lines 43-57: unset PYTHONPATH, set PYTHONNOUSERSITE - cannot hang
3. **Line 62: `module load CUDA 2>/dev/null || true`** - CAN HANG (see below)
4. Lines 64-67: Set LD_LIBRARY_PATH - cannot hang
5. Line 68: `return 0 2>/dev/null || exit 0` - see analysis below

**`module load CUDA` (line 62):**
- On HPC clusters, `module load` interacts with shared filesystem (NFS-backed module database)
- Under cluster load, module operations can be slow
- `2>/dev/null` hides ALL error output
- `|| true` catches failures but NOT hangs
- **Verdict:** Could hang, but unlikely for 6 hours. More likely a few seconds delay.

**`return 0 2>/dev/null || exit 0` (line 68):**
- sbatch --wrap creates a /bin/sh script
- If /bin/sh is bash (typical on RHEL/CentOS): `return 0` works fine, returns from sourced script
- If /bin/sh is dash or strict POSIX sh: `return 0` might fail, `exit 0` would terminate the ENTIRE wrap job
- **BUT:** If exit 0 killed the wrap, the job would complete in seconds, NOT run for 6 hours
- **Verdict:** Not the cause of the 6-hour issue.

### 3. Python Import Chain Analysis

**No hang risk from imports.** The F15 import chain is:
- numpy, scipy, sklearn (stdlib, ~6s cold start, normal)
- kirby.hybrid, kirby.find_best_hybrid (pure Python + sklearn, no network calls)
- experiments._shared, complexity_helpers (config dicts, no I/O)

**NOT imported at module level:** torch, transformers, huggingface_hub. No network calls, no model downloads.

### 4. Python stdout Buffering

When Python's stdout is redirected to a file (SLURM output), it uses full buffering (4-8KB buffer). The F15 script produces ~700 bytes of print output before entering the main computation loop, which was silent (joblib Parallel with zero progress output). The buffer never fills.

**This IS a real problem** but only explains the lack of output, not whether the script is actually running or stuck.

### 5. GPU Resources

28 of 57 SLURM scripts don't request GPU (--gres=gpu). F15 is one of them. Jobs can land on CPU-only nodes where `module load CUDA` fails silently. F15 doesn't need GPU (pure sklearn), so this is fine for computation but the CUDA module load could behave differently.

## Three Remaining Hypotheses

### H1: Python IS running, output is just buffered
- **Evidence for:** 6-hour runtime means process is alive; buffering math checks out (700 bytes < 4KB buffer)
- **Evidence against:** User reports this happens with ALL experiments, including precompute which supposedly works
- **Fix applied:** PYTHONUNBUFFERED=1, sys.stdout.reconfigure(line_buffering=True), flush=True on all prints

### H2: `module load CUDA` hangs intermittently
- **Evidence for:** It's the only system call in the SKIP_INSTALLS path that can block; NFS/Lmod issues are common under load
- **Evidence against:** Unlikely to hang for 6 full hours; precompute uses same path and works
- **Fix applied:** Added diagnostic echo before/after module load in setup.sh

### H3: Python crashes silently before producing flushed output
- **Evidence for:** No echo between setup.sh and Python in any script; bare except clauses could swallow errors
- **Evidence against:** 6-hour runtime means the process didn't crash immediately
- **Fix applied:** Diagnostic script tests each stage independently

## Fixes Applied

### setup.sh
Added diagnostic echo statements:
- `[setup.sh] skip mode: loading CUDA module...` (before module load)
- `[setup.sh] CUDA module done` (after module load)
- `[setup.sh] setup complete (skip mode)` (end of SKIP_INSTALLS path)
- `[setup.sh] setup complete (full mode)` (end of full setup path)

### submit_framework_f15.sh
- Added `export PYTHONUNBUFFERED=1` to WRAP

### f15_adaptive_pruning_validation.py
- Added `sys.stdout.reconfigure(line_buffering=True)` at top
- Added `flush=True` to every print statement
- Added timestamps at every phase (imports, data load, importance, allocation, evaluation)
- Replaced silent joblib parallel with sequential + progress every 100 configs
- **WARNING:** Sequential execution is slower than parallel. Could hit 24h time limit for large datasets.

### Diagnostic script
`tests/slurm_scripts/diagnostic.sh` — submit to test each stage independently:
- Stage 1: cd to KIRBy
- Stage 2: module load CUDA (with stderr visible, not suppressed)
- Stage 3: setup.sh with SKIP_INSTALLS=1
- Stage 4: Python startup
- Stage 5: Full import chain + data load test

## How to Use the Diagnostic

```bash
# On the server:
git pull
sbatch tests/slurm_scripts/diagnostic.sh
# Wait for job, then:
cat slurm-<jobid>.out
```

Expected output tells you exactly which stage hangs:
- Stops at "STAGE 2: module load CUDA" → module system issue
- Stops at "STAGE 3: setup.sh" → setup.sh issue
- Stops at "STAGE 5: Python imports" → specific import hangs
- All stages pass → the issue is in experiment-specific code (buffering was masking it)

## Open Questions

1. Does `module load CUDA` complete in interactive mode? (User testing now)
2. What shell does sbatch --wrap use on ARC? (`readlink -f /bin/sh` on a compute node)
3. Do precompute jobs actually show output promptly, or does output only appear at the end?
4. Is the issue specific to KIRBY_SKIP_INSTALLS=1, or does it also happen with full setup?

## For Future Chats

**DO NOT:**
- Assume it's just Python stdout buffering without evidence
- Assume it's node contention without evidence
- Tell the user to "wait it out"
- Pick one hypothesis and commit without diagnostic proof

**DO:**
- Run `sbatch tests/slurm_scripts/diagnostic.sh` first to identify the failing stage
- Check the setup.sh echo output to see where it stops
- Check if SLURM output file exists and has content vs empty
- Ask user to test interactively on a compute node if batch diagnostics are inconclusive
