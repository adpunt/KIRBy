"""SERVER STEP 1 -- build per-target npz for the MoleculeACE 30-target battery.

MoleculeACE (van Tilborg, Alenicheva & Grisoni, JCIM 2022, 62:5938) ships 30
cliff-rich ChEMBL targets. We DISCOVER them from the installed package rather
than hardcode names (anti-hallucination), read the pActivity column, compute our
OWN Bemis-Murcko scaffold groups (the lab forbids random splits; we do not use
MoleculeACE's own train/test split), and featurize with the same representation
set as the local study.

CSV columns (verified): smiles, exp_mean [nM], y, cliff_mol, split, y [pEC50/pKi]
We use `smiles` and `y [pEC50/pKi]` (the interpretable pActivity target).

Output: sim_<TARGET>.npz per target, identical schema to the local study, into
$KIRBY_CACHE_DIR/similarity_metrics/ (or ./sim_data/).
"""
import os, sys, glob, csv, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
NBITS = 1024
OUT = os.path.join(os.environ.get('KIRBY_CACHE_DIR', '.'), 'similarity_metrics', 'sim_data')
os.makedirs(OUT, exist_ok=True)

SMILES_COL = 'smiles'
Y_COL = 'y [pEC50/pKi]'


def find_csvs():
    """Locate the 30 benchmark CSVs shipped inside the installed MoleculeACE."""
    try:
        import MoleculeACE
        base = os.path.dirname(MoleculeACE.__file__)
    except Exception as e:
        print(f'MoleculeACE not importable ({e}); falling back to $MOLECULEACE_DATA')
        base = os.environ.get('MOLECULEACE_DATA', '')
    hits = glob.glob(os.path.join(base, '**', 'benchmark_data', '*.csv'), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(base, '**', 'CHEMBL*.csv'), recursive=True)
    return sorted(hits)


def _sparse(fp, n=NBITS):
    v = np.zeros(n, dtype=np.float32)
    for idx, cnt in fp.GetNonzeroElements().items():
        v[idx % n] += cnt
    return v


def featurize(mols):
    reps = {}
    reps['ecfp4'] = np.array([AllChem.GetMorganFingerprintAsBitVect(m, 2, NBITS) for m in mols], np.uint8)
    reps['ecfp6'] = np.array([AllChem.GetMorganFingerprintAsBitVect(m, 3, NBITS) for m in mols], np.uint8)
    reps['fcfp4'] = np.array([AllChem.GetMorganFingerprintAsBitVect(m, 2, NBITS, useFeatures=True) for m in mols], np.uint8)
    reps['ecfp4_count'] = np.array([_sparse(AllChem.GetHashedMorganFingerprint(m, 2, NBITS)) for m in mols], np.float32)
    reps['atompair'] = np.array([rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=NBITS) for m in mols], np.uint8)
    reps['atompair_count'] = np.array([_sparse(rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=NBITS)) for m in mols], np.float32)
    reps['torsion'] = np.array([rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=NBITS) for m in mols], np.uint8)
    reps['rdkit'] = np.array([np.frombuffer(Chem.RDKFingerprint(m, fpSize=NBITS).ToBitString().encode(), 'u1') - ord('0') for m in mols], np.uint8)
    reps['maccs'] = np.array([MACCSkeys.GenMACCSKeys(m) for m in mols], np.uint8)
    dl = Descriptors._descList
    desc = np.zeros((len(mols), len(dl)), np.float32)
    for i, m in enumerate(mols):
        for j, (_, fn) in enumerate(dl):
            try:
                v = fn(m); desc[i, j] = v if np.isfinite(v) else 0.0
            except Exception:
                desc[i, j] = 0.0
    reps['desc'] = np.nan_to_num(desc)
    return reps


def scaffolds(mols):
    d, out = {}, []
    for m in mols:
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            s = ''
        out.append(d.setdefault(s, len(d)))
    return np.asarray(out, np.int64)


def tanimoto(mols):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, NBITS) for m in mols]
    T = np.zeros((len(fps), len(fps)), np.float32)
    for i in range(len(fps)):
        T[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    return T


def pick_ycol(fields):
    """Activity column, by an ordered fallback: the documented name, then the
    MoleculeACE pActivity name, then any field naming pEC50 or pKi, then the
    bare 'y'. First match wins — file order never decides. None when no
    candidate is present."""
    fields = list(fields or [])
    for cand in (Y_COL, 'y [pEC50/pKi]'):
        if cand in fields:
            return cand
    for c in fields:
        if 'pEC50' in c or 'pKi' in c:
            return c
    return 'y' if 'y' in fields else None


def build(path):
    name = os.path.splitext(os.path.basename(path))[0]
    smis, ys = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        ycol = pick_ycol(r.fieldnames)
        if ycol is None:
            print(f'  SKIP {name}: no activity column (wanted {Y_COL!r}); '
                  f'fields are {list(r.fieldnames or [])}')
            return
        for row in r:
            try:
                ys.append(float(row[ycol])); smis.append(row[SMILES_COL])
            except (ValueError, KeyError):
                continue
    mols, keep = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m); keep.append(i)
    y = np.asarray(ys, float)[keep]
    smis_kept = [smis[i] for i in keep]
    if len(mols) < 100:
        print(f'  SKIP {name}: only {len(mols)} mols'); return
    reps = featurize(mols)
    np.savez_compressed(os.path.join(OUT, f'sim_{name}.npz'),
                        y=y, groups=scaffolds(mols), tanimoto=tanimoto(mols),
                        smiles=np.array(smis_kept), **reps)  # '<U..' unicode, no pickle
    print(f'  {name}: n={len(mols)} scaffolds={len(set(scaffolds(mols)))} -> sim_{name}.npz', flush=True)


def main():
    csvs = find_csvs()
    print(f'found {len(csvs)} MoleculeACE CSVs -> {OUT}')
    for p in csvs:
        try:
            build(p)
        except Exception as e:
            print(f'  FAIL {p}: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
