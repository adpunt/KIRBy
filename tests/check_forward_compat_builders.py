#!/usr/bin/env python3
"""Which forward-compatibility candidates actually build on this machine?

WHY THIS EXISTS. c13 asks whether the hybrid search can absorb a newly published
embedding model. 56 candidates are registered. Each one needs a pip install, or a
git clone, or a weights download, and most need more than one. Finding that out
inside a 30-hour grid job wastes the job. This calls every builder on a handful of
real inputs and prints one line per candidate saying what happened.

It never touches a dataset and never fits a model. A pass here means the builder
returned a correctly shaped matrix; it says nothing about whether the numbers are
good.

WHAT A ROW MEANS
  OK            built, shape and dtype and row count all correct.  Runnable now.
  BAD_SHAPE     built but returned the wrong rows.  A BUG — c13 matches rows to
                labels by position, so this would score one molecule against
                another molecule's label.  Fix before running.
  STUB          deliberately not implemented; the API could not be confirmed from
                a published page.  Needs someone to write it.
  NEEDS_DEP     a package is missing.  The line names it.
  NEEDS_WEIGHTS the code is there, the checkpoint is not.  The line names where
                to get it.
  ERROR         built and raised something else.  A BUG — read the message.
  NEEDS_DATA    cannot be tested without a real dataset (omics and pathology take
                a TCGA expression matrix or extracted slide tiles).  Test those by
                running c13 on tcga_luad_vs_lusc.

USAGE
  python3 tests/check_forward_compat_builders.py                # every domain
  python3 tests/check_forward_compat_builders.py --domain molecular
  python3 tests/check_forward_compat_builders.py --only chemeleon,clamp
  python3 tests/check_forward_compat_builders.py --json out.json

ALWAYS run it with the OpenMP caps set, or a torch builder followed by a LightGBM
fit segfaults with no traceback (exit 139):
  OMP_NUM_THREADS=1 KIRBY_LGBM_THREADS=1 python3 tests/check_forward_compat_builders.py
"""
import os
import sys
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

import numpy as np

# --- Tiny real inputs. Real molecules and real sequences, not random strings:
#     several builders canonicalise with RDKit or tokenise on a fixed alphabet
#     and would fail on nonsense for the wrong reason. -----------------------
SMILES = [
    'CCO',                                  # ethanol
    'CC(=O)Oc1ccccc1C(=O)O',                # aspirin
    'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',         # caffeine
    'c1ccc2c(c1)cccc2O',                    # 2-naphthol
]
HEAVY = [
    'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQRLEWMGWINAGNGNTKYSQKFQG'
    'RVTITRDTSASTAYMELSSLRSEDTAVYYCARDRGYTSSWYPDYWGQGTLVTVSS',
    'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG'
    'RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKGGYYYGMDVWGQGTTVTVSS',
]
LIGHT = [
    'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSG'
    'SGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK',
    'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSG'
    'SGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK',
]
PROTEINS = [
    'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR',
    'MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFVNQHLCGSHLVEALYLVCGERGFFYTPKA',
]
DNA = [
    'ACGTACGTACGTTTGACCATGGCCTAGCTAGCTAGCATCGATCGATCGGGCCTTAAGGCCATATATCGCGCGATATCG' * 2,
    'TTGACCATGGCCTAGCTAGCTAGCATCGATCGATCGGGCCTTAAGGCCATATATCGCGCGATATCGACGTACGTACGT' * 2,
]
RXN = [
    'CC(C)c1ccc(N)cc1.Brc1ccccc1>>CC(C)c1ccc(Nc2ccccc2)cc1',
    'Cc1ccc(N)cc1.Clc1ccccn1>>Cc1ccc(Nc2ccccn2)cc1',
]

# name -> (callable_getter, args, expected_rows)
DOMAINS = {
    'molecular':  ('smiles',  len(SMILES)),
    'antibody':   ('ab',      len(HEAVY)),
    'protein':    ('protein', len(PROTEINS)),
    'nucleotide': ('dna',     len(DNA)),
    'reaction':   ('rxn',     len(RXN)),
    'omics':      ('data',    0),
    'pathology':  ('data',    0),
}


def _resolve(domain):
    """Yield (candidate_name, callable) for a domain, straight off the registry
    c13 will use — so a name that is not wired here fails loudly rather than
    passing a test it would fail in the real run."""
    import experiments.complexity.forward_compat_reps as F
    if domain == 'molecular':
        import inspect, re
        from find_best_hybrid_molecular import get_representations as g
        src = inspect.getsource(g)
        import kirby.representations.forward_compat_molecular as m
        import kirby.representations.molecular as m0
        for name in F.MOLECULAR:
            if f"'{name}':" not in src:
                yield name, None
                continue
            fn = getattr(m, f'create_{name}', None) or getattr(m0, f'create_{name}', None)
            yield name, fn
    elif domain == 'antibody':
        from kirby.representations.antibody import ANTIBODY_REPS
        for name in F.ANTIBODY:
            yield name, ANTIBODY_REPS.get(name)
    elif domain == 'protein':
        import kirby.representations.forward_compat_protein as p
        for name in F.PROTEIN:
            yield name, getattr(p, f'create_{name}_protein_embeddings', None)
    elif domain == 'nucleotide':
        from kirby.representations.nucleotide import NUCLEOTIDE_REPS
        for name in F.NUCLEOTIDE:
            yield name, NUCLEOTIDE_REPS.get(name)
    elif domain == 'reaction':
        import kirby.representations.forward_compat_reaction as r
        for name in list(F.REACTION) + ['concat_chemeleon']:
            yield name, getattr(r, f'create_{name}', None)
    elif domain in ('omics', 'pathology'):
        block = F.OMICS if domain == 'omics' else F.PATHOLOGY_WSI
        for name in block:
            yield name, 'NEEDS_DATA'


def _call(domain, fn):
    if domain == 'molecular':
        return fn(SMILES)
    if domain == 'antibody':
        return fn(HEAVY, LIGHT)
    if domain == 'protein':
        return fn(PROTEINS)
    if domain == 'nucleotide':
        return fn(DNA)
    if domain == 'reaction':
        return fn(RXN)
    raise AssertionError(domain)


def _classify(exc):
    """Turn an exception into an actionable status + the line to act on."""
    msg = str(exc)
    first = msg.strip().splitlines()[0] if msg.strip() else type(exc).__name__
    if isinstance(exc, NotImplementedError):
        return 'STUB', first
    low = msg.lower()
    if isinstance(exc, ImportError) or 'no module named' in low or 'pip install' in low:
        return 'NEEDS_DEP', first
    for token in ('weight', 'checkpoint', 'download', 'not found at', 'gated',
                  'access request', '.pt', '.ckpt', '.pth', 'snapshot'):
        if token in low:
            return 'NEEDS_WEIGHTS', first
    return 'ERROR', f'{type(exc).__name__}: {first}'


def check(domain, only=None):
    rows = []
    for name, fn in _resolve(domain):
        if only and name not in only:
            continue
        if fn == 'NEEDS_DATA':
            rows.append(dict(domain=domain, name=name, status='NEEDS_DATA', detail=(
                'needs a real TCGA expression matrix or extracted slide tiles — '
                'test by running c13 on tcga_luad_vs_lusc'), seconds=0.0, dim=None))
            continue
        if fn is None:
            rows.append(dict(domain=domain, name=name, status='NOT_REGISTERED', detail=(
                'the name does not resolve through the registry c13 uses — '
                'the registration was not applied'), seconds=0.0, dim=None))
            continue
        t0 = time.time()
        try:
            X = _call(domain, fn)
        except Exception as e:  # noqa: BLE001 — classifying every failure is the point
            status, detail = _classify(e)
            if status == 'ERROR' and os.environ.get('FC_TRACEBACK'):
                detail += '\n' + traceback.format_exc()
            rows.append(dict(domain=domain, name=name, status=status,
                             detail=detail, seconds=round(time.time() - t0, 1), dim=None))
            continue
        secs = round(time.time() - t0, 1)
        want = DOMAINS[domain][1]
        problems = []
        X = np.asarray(X)
        if X.ndim != 2:
            problems.append(f'ndim {X.ndim}, expected 2')
        elif X.shape[0] != want:
            problems.append(f'{X.shape[0]} rows for {want} inputs')
        if X.dtype != np.float32:
            problems.append(f'dtype {X.dtype}, expected float32')
        if X.ndim == 2 and X.shape[0] == want and want > 1:
            if np.allclose(X[0], X[1]):
                problems.append('row 0 and row 1 are identical — the encoder is '
                                'not distinguishing inputs (this is how the old '
                                'igbert failed)')
        if not np.all(np.isfinite(X)):
            problems.append('contains non-finite values')
        rows.append(dict(
            domain=domain, name=name,
            status='OK' if not problems else 'BAD_SHAPE',
            detail='; '.join(problems) if problems else f'{X.shape[0]}x{X.shape[1]} float32',
            seconds=secs, dim=int(X.shape[1]) if X.ndim == 2 else None))
    return rows


ORDER = ['OK', 'BAD_SHAPE', 'ERROR', 'NEEDS_WEIGHTS', 'NEEDS_DEP', 'STUB',
         'NOT_REGISTERED', 'NEEDS_DATA']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', default=None,
                    help='one of: ' + ', '.join(DOMAINS) + ' (default: all)')
    ap.add_argument('--only', default=None,
                    help='comma-separated candidate names to test')
    ap.add_argument('--json', default=None, help='also write the rows here')
    a = ap.parse_args()

    if os.environ.get('OMP_NUM_THREADS') not in ('1', None):
        print(f"WARNING: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}. "
              f"A torch builder followed by a LightGBM fit segfaults at exit 139 "
              f"with no traceback. Re-run with OMP_NUM_THREADS=1.\n")

    domains = [a.domain] if a.domain else list(DOMAINS)
    only = set(x.strip() for x in a.only.split(',')) if a.only else None

    rows = []
    for d in domains:
        print(f"\n{'=' * 74}\n  {d.upper()}\n{'=' * 74}")
        for r in check(d, only):
            rows.append(r)
            secs = f"{r['seconds']:6.1f}s" if r['seconds'] else '       '
            print(f"  {r['status']:14s} {r['name']:22s} {secs}  {r['detail'][:150]}")

    print(f"\n{'=' * 74}\n  TOTALS — one candidate per count\n{'=' * 74}")
    for s in ORDER:
        n = sum(1 for r in rows if r['status'] == s)
        if n:
            print(f"  {s:14s} {n:3d}")
    runnable = [r['name'] for r in rows if r['status'] == 'OK']
    print(f"\n  RUNNABLE RIGHT NOW ({len(runnable)}): {', '.join(runnable) or 'none'}")

    broken = [r for r in rows if r['status'] in ('BAD_SHAPE', 'ERROR', 'NOT_REGISTERED')]
    if broken:
        print(f"\n  NEEDS A CODE FIX ({len(broken)}) — these are bugs, not missing installs:")
        for r in broken:
            print(f"    {r['domain']}/{r['name']}: {r['detail'][:160]}")

    if a.json:
        with open(a.json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"\n  wrote {a.json}")

    # Exit non-zero only on a real bug. Missing installs are expected and are not
    # a failure of this script.
    return 1 if broken else 0


if __name__ == '__main__':
    sys.exit(main())
