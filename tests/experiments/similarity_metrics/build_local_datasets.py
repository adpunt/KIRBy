"""Rebuild the LOCAL pilot datasets (herg_ki, logd, adme_solubility) as npz, same
schema as the MoleculeACE builder. Reconstructed after the scratchpad was wiped —
this lives in the repo now so the pilot is never lost again.

Sources (local cache):
  herg_ki         : ~/.cache/kirby/chembl_herg_ki.csv     (SMILES, pKi)
  logd            : ~/.cache/kirby/openadmet_train.csv    (SMILES, LogD)
  adme_solubility : ~/.cache/kirby/polaris_adme_fang.csv  (SMILES, LOG_SOLUBILITY)

Output: sim_<name>.npz next to this file (or $SIM_DATA_DIR), holding every
representation + activity y + Bemis-Murcko scaffold groups + ECFP4-Tanimoto
reference + SMILES, plus the sha256 of the source CSV and of the kept SMILES in
order. Featurisation is imported from build_moleculeace so the two builders can
never drift.
"""
import os, sys, csv, hashlib, numpy as np
from rdkit import Chem, RDLogger
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_moleculeace import featurize, scaffolds, tanimoto  # single source of truth
RDLogger.DisableLog('rdApp.*')

OUT = os.environ.get('SIM_DATA_DIR', HERE)
SPECS = {
    'herg_ki': ('~/.cache/kirby/chembl_herg_ki.csv', 'SMILES', 'pKi'),
    'logd':    ('~/.cache/kirby/openadmet_train.csv', 'SMILES', 'LogD'),
    'adme_solubility': ('~/.cache/kirby/polaris_adme_fang.csv', 'SMILES', 'LOG_SOLUBILITY'),
}


def _inchikey(m):
    """InChIKey, or the canonical SMILES when InChI generation fails, so a
    molecule the InChI code cannot handle is not merged with every other one."""
    try:
        k = Chem.MolToInchiKey(m)
    except Exception:
        k = ''
    return k or Chem.MolToSmiles(m)


def build(name):
    path, scol, ycol = SPECS[name]
    src = os.path.expanduser(path)
    smis, ys = [], []
    with open(src) as f:
        for row in csv.DictReader(f):
            s, v = row.get(scol), row.get(ycol)
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
    # One row per molecule, first occurrence kept, matching the framework
    # loader. A repeated molecule is otherwise its own zero-distance nearest
    # neighbour in the Tanimoto matrix below.
    seen, uniq = set(), []
    for j, m in enumerate(mols):
        k = _inchikey(m)
        if k in seen:
            continue
        seen.add(k); uniq.append(j)
    n_dup = len(mols) - len(uniq)
    mols = [mols[j] for j in uniq]
    keep = [keep[j] for j in uniq]
    y = np.asarray(ys, float)[keep]
    smis_kept = [smis[i] for i in keep]
    reps = featurize(mols)
    grp = scaffolds(mols)
    # Provenance, so a later rebuild that moves row numbers is detectable: the
    # bytes of the source CSV, and the kept SMILES in the order they are stored.
    # 0-d '<U64' arrays, so np.load needs no pickle.
    with open(src, 'rb') as f:
        csv_sha256 = hashlib.sha256(f.read()).hexdigest()
    smiles_sha256 = hashlib.sha256('\n'.join(smis_kept).encode()).hexdigest()
    out = os.path.join(OUT, f'sim_{name}.npz')
    np.savez_compressed(out, y=y, groups=grp, tanimoto=tanimoto(mols),
                        smiles=np.array(smis_kept),
                        csv_sha256=np.array(csv_sha256),
                        smiles_sha256=np.array(smiles_sha256), **reps)
    print(f'[{name}] n={len(mols)} scaffolds={len(set(grp.tolist()))} '
          f'duplicate molecules dropped={n_dup} -> {out}', flush=True)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or SPECS):
        build(nm)
