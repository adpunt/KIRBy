"""Build the LOCAL 'adme_solubility' similarity dataset as npz, SAME schema as
the MoleculeACE / build_local_datasets pilot (sim_herg_ki.npz).

Source (local cache):
  adme_solubility : ~/.cache/kirby/polaris_adme_fang.csv  (SMILES, LOG_SOLUBILITY)

Rows whose LOG_SOLUBILITY is missing/blank are dropped; rows RDKit cannot parse
are dropped. Output: sim_adme_solubility.npz next to this file (or $SIM_DATA_DIR),
holding every representation + activity y + Bemis-Murcko scaffold groups +
ECFP4-Tanimoto reference + SMILES. Featurisation is imported from
build_moleculeace so this builder can never drift from the pilot.
"""
import os, sys, csv, numpy as np
from rdkit import Chem, RDLogger
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_moleculeace import featurize, scaffolds, tanimoto  # single source of truth
RDLogger.DisableLog('rdApp.*')

OUT = os.environ.get('SIM_DATA_DIR', HERE)
NAME = 'adme_solubility'
CSV_PATH = '~/.cache/kirby/polaris_adme_fang.csv'
SMILES_COL = 'SMILES'
Y_COL = 'LOG_SOLUBILITY'


def build():
    smis, ys = [], []
    with open(os.path.expanduser(CSV_PATH)) as f:
        for row in csv.DictReader(f):
            s, v = row.get(SMILES_COL), row.get(Y_COL)
            if not s or v in (None, '', 'NA', 'NaN', 'nan'):
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(fv):
                continue
            ys.append(fv); smis.append(s)
    mols, keep = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m); keep.append(i)
    y = np.asarray(ys, float)[keep]
    smis_kept = [smis[i] for i in keep]
    reps = featurize(mols)
    grp = scaffolds(mols)
    out = os.path.join(OUT, f'sim_{NAME}.npz')
    np.savez_compressed(out, y=y, groups=grp, tanimoto=tanimoto(mols),
                        smiles=np.array(smis_kept), **reps)
    print(f'[{NAME}] n={len(mols)} scaffolds={len(set(grp.tolist()))} -> {out}', flush=True)


if __name__ == '__main__':
    build()
