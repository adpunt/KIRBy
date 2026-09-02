#!/usr/bin/env python
"""Say why antibody structure prediction is failing, in under a minute.

Jobs 8668608 and 8668609 ran 3706 predictions in 8 minutes 39 seconds and
produced no geometry features at all. Both of the causes that do that are
invisible to `import ImmuneBuilder`, which is what setup.sh used to check:

  1. hmmscan is not on PATH. ANARCI numbers every sequence by running that
     binary, so prediction raises FileNotFoundError in about 0.02 seconds.
  2. A weight file downloaded short. torch.load then raises PytorchStreamReader
     on every call, and the short file is reused forever.

Run this on the same node the job runs on:

    python tests/check_structure_prediction.py
"""

import os
import sys
import time
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

HEAVY = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
         "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS")
LIGHT = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSR"
         "SGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK")
# Real nanobody, flab_thermo_tresanco row 0.
NANOBODY = ("EVQLQASGGGLVRPGGSLRLSCAASGFTFSSYAMMWVRQAPGKGLEWVSAINGGGGSTSYADSVKG"
            "RFTISRDNAKNTLYLQMNSLKPEDTAVYYCAKYQAAVHQEKEDYWGQGTQVTVSS")

failures = []


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


section("1. imports")
for mod in ('torch', 'openmm', 'pdbfixer', 'anarci', 'ImmuneBuilder'):
    try:
        __import__(mod)
        print(f"  {mod:15} OK")
    except Exception as e:
        print(f"  {mod:15} FAIL  {type(e).__name__}: {e}")
        if 'libcub' in str(e) or 'libcud' in str(e):
            print(f"  {'':15}       ^ that is a CUDA library failing to map. "
                  f"Login nodes cannot do this. Run this check inside "
                  f"srun on the node the job will use.")
        failures.append(f"{mod} does not import")


section("2. hmmscan, which ANARCI shells out to")
where = shutil.which('hmmscan')
print(f"  hmmscan on PATH and executable: {where or 'NO'}")
if not where:
    # shutil.which returns None for a file that exists without an execute bit,
    # so absence here does not mean absence. Look again ignoring the bit, and
    # include the ANARCI package's own bin directory, which pip does not
    # always mark executable.
    search = os.environ.get('PATH', '').split(os.pathsep)
    try:
        import anarci as _anarci
        pkg = Path(_anarci.__file__).parent
        search += [str(pkg), str(pkg / 'bin'), str(pkg.parent / 'bin')]
    except Exception:
        pass
    found_here = []
    for d in search:
        if not d:
            continue
        cand = Path(d) / 'hmmscan'
        if cand.exists():
            st = cand.stat()
            found_here.append(cand)
            print(f"  present but NOT executable: {cand}  "
                  f"mode={oct(st.st_mode)[-3:]} uid={st.st_uid} "
                  f"(you are uid {os.getuid()})")
    if found_here:
        owned = [c for c in found_here if c.stat().st_uid == os.getuid()]
        if owned:
            failures.append(
                f"hmmscan exists but has no execute bit — "
                f"chmod +x {' '.join(str(c) for c in owned)}")
        else:
            failures.append(
                f"hmmscan exists at {found_here[0]} but has no execute bit and "
                f"you do not own it — reinstall HMMER into your own environment")
    else:
        failures.append("hmmscan is nowhere on PATH — try 'module load HMMER'")
try:
    from anarci import number
    t = time.time()
    numbering = number(HEAVY)
    if numbering and numbering[0]:
        print(f"  numbering a known heavy chain: OK ({time.time() - t:.2f}s)")
    else:
        print("  numbering a known heavy chain: RETURNED NOTHING")
        failures.append("ANARCI returns no numbering for a known heavy chain")
except Exception as e:
    print(f"  numbering a known heavy chain: {type(e).__name__}: {e}")
    failures.append(f"ANARCI numbering raises {type(e).__name__}")


section("3. downloaded weights")
try:
    import torch
    import ImmuneBuilder
    weights = Path(ImmuneBuilder.__file__).parent / 'trained_model'
    print(f"  {weights}")
    found = False
    for f in sorted(weights.glob('*body_model_*')):
        found = True
        mb = f.stat().st_size / 1e6
        try:
            torch.load(f, map_location='cpu')
            print(f"  {f.name:22} {mb:8.1f} MB  loads")
        except Exception as e:
            print(f"  {f.name:22} {mb:8.1f} MB  {type(e).__name__}")
            failures.append(f"weight file {f.name} is corrupt or short — "
                            f"delete it and let it re-download")
    if not found:
        print("  no weights downloaded yet (the first prediction fetches them)")
except Exception as e:
    print(f"  could not check: {type(e).__name__}: {e}")


section("4. one prediction of each kind")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
try:
    from kirby.representations.structural import (
        predict_structure, prediction_failures)
    for label, light in (("paired H+L", LIGHT), ("nanobody, H only", None)):
        seq = HEAVY if light else NANOBODY
        t = time.time()
        pdb = predict_structure(seq, light)
        dt = time.time() - t
        if pdb is None:
            recorded = prediction_failures()
            why = recorded[-1] if recorded else "returned None, nothing recorded"
            print(f"  {label:18} FAILED in {dt:.2f}s  {why}")
            failures.append(f"{label} prediction failed: {why}")
        else:
            print(f"  {label:18} OK in {dt:.1f}s  {pdb}")
except Exception as e:
    print(f"  could not run: {type(e).__name__}: {e}")
    failures.append(f"predict_structure raised {type(e).__name__}: {e}")


section("verdict")
if failures:
    for f in failures:
        print(f"  BROKEN: {f}")
    sys.exit(1)
print("  Antibody structure prediction works on this node.")
