"""Rebuild the LOCAL pilot datasets (herg_ki, logd) as npz, same schema as the
MoleculeACE builder. Reconstructed after the scratchpad was wiped — this lives in
the repo now so the pilot is never lost again.

Sources (local cache):
  herg_ki : ~/.cache/kirby/chembl_herg_ki.csv   (SMILES, pKi)
  logd    : ~/.cache/kirby/openadmet_train.csv  (SMILES, LogD)

Output: sim_<name>.npz next to this file (or $SIM_DATA_DIR), holding every
representation + activity y + Bemis-Murcko scaffold groups + ECFP4-Tanimoto
reference + SMILES. Featurisation is imported from build_moleculeace so the two
builders can never drift.
"""
import os, sys, csv, numpy as np
from rdkit import Chem, RDLogger
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_moleculeace import featurize, scaffolds, tanimoto  # single source of truth
RDLogger.DisableLog('rdApp.*')

OUT = os.environ.get('SIM_DATA_DIR', HERE)
SPECS = {
    'herg_ki': ('~/.cache/kirby/chembl_herg_ki.csv', 'SMILES', 'pKi'),
    'logd':    ('~/.cache/kirby/openadmet_train.csv', 'SMILES', 'LogD'),
}


def build(name):
    path, scol, ycol = SPECS[name]
    smis, ys = [], []
    with open(os.path.expanduser(path)) as f:
        for row in csv.DictReader(f):
            s, v = row.get(scol), row.get(ycol)
            if not s or v in (None, '', 'NA'):
                continue
            try:
                ys.append(float(v)); smis.append(s)
            except ValueError:
                continue
    mols, keep = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m); keep.append(i)
    y = np.asarray(ys, float)[keep]
    smis_kept = [smis[i] for i in keep]
    reps = featurize(mols)
    out = os.path.join(OUT, f'sim_{name}.npz')
    np.savez_compressed(out, y=y, groups=scaffolds(mols), tanimoto=tanimoto(mols),
                        smiles=np.array(smis_kept), **reps)
    print(f'[{name}] n={len(mols)} scaffolds={len(set(scaffolds(mols)))} -> {out}', flush=True)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or SPECS):
        build(nm)
