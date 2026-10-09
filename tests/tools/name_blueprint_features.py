"""Name the features a stored hybrid selected, for the Chapter 5 blueprint figure (F3).

Reads the features each outer fold's hybrid picked from a c24 result file
(by_model.<model>.stability.feature_sets_per_fold: (representation, position) pairs),
counts on how many folds each was picked, and names it where the representation allows:
  - ecfp4: the atom environment that sets the bit, as a SMILES fragment, taken from the
    first molecule of the dataset that sets it (one hashed bit can stand for several
    environments; only the one drawn is named);
  - mordred: the descriptor name, in the order Calculator(descriptors, ignore_3D=True) gives;
  - pdv: the RDKit descriptor name (DEFAULT_DESCRIPTOR_LIST);
  - every other representation: its position only.
No search and no model fit. Usage:
  python tests/tools/name_blueprint_features.py --dataset logd --out scratchpad/thesis_ch5_8/f3_logd_feature_names.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tests'))
sys.path.insert(0, str(REPO / 'tests' / 'experiments'))
RESULTS = Path('/Volumes/seagate/kirby_results_2026-10-09/complexity/c24_blueprint_transfer')


def ecfp4_names(smiles, cached, positions, radius=2, n_bits=2048):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    names, todo = {}, set(positions)
    for i, smi in enumerate(smiles):
        if not todo:
            break
        hit = [b for b in todo if cached[i, b] > 0]
        if not hit:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        fp = gen.GetFingerprint(mol, additionalOutput=ao)
        info = ao.GetBitInfoMap()
        for b in hit:
            if b not in info:
                # cached bit set but recomputed fingerprint disagrees: report, do not guess
                names[b] = f'MISMATCH: cached bit set, recomputed not (molecule {i})'
            else:
                atom, rad = info[b][0]
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom)
                if rad == 0 or not env:
                    frag = mol.GetAtomWithIdx(atom).GetSymbol() + ' (single atom)'
                else:
                    amap = {}
                    sub = Chem.PathToSubmol(mol, env, atomMap=amap)
                    frag = Chem.MolToSmiles(sub)
                names[b] = f'{frag} | radius {rad} | from molecule {i}: {smi}'
            todo.discard(b)
    for b in todo:
        names[b] = 'no molecule in the dataset sets this bit'
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='logd')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    res = json.load(open(RESULTS / f'{a.dataset}.json'))
    picked = {}  # model -> Counter((rep, pos) -> folds)
    n_folds = {}
    for model, bm in res['by_model'].items():
        fs = bm['stability']['feature_sets_per_fold']
        n_folds[model] = len(fs)
        picked[model] = Counter(tuple(x) for sets in fs.values() for x in sets)

    reps_needed = {r for c in picked.values() for (r, _) in c}
    from experiments._shared import load_data
    data = load_data(a.dataset, rep_names=[r for r in ('ecfp4',) if r in reps_needed])
    smiles = data['smiles']

    names = {}
    if 'ecfp4' in reps_needed:
        pos = {p for c in picked.values() for (r, p) in c if r == 'ecfp4'}
        cached = np.asarray(data['reps']['ecfp4'])
        names['ecfp4'] = ecfp4_names(smiles, cached, pos)
    if 'mordred' in reps_needed:
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        mnames = [str(d) for d in calc.descriptors]
        width = res['_rep_dims']['per_rep'].get('mordred')
        if width != len(mnames):
            sys.exit(f'mordred width {width} in the result, {len(mnames)} from the installed calculator')
        names['mordred'] = dict(enumerate(mnames))
    if 'pdv' in reps_needed:
        from kirby.representations.molecular import DEFAULT_DESCRIPTOR_LIST
        names['pdv'] = dict(enumerate(DEFAULT_DESCRIPTOR_LIST))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'representation', 'position', 'folds_picked', 'folds_total', 'name'])
        for model, c in picked.items():
            for (rep, p), k in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                w.writerow([model, rep, p, k, n_folds[model], names.get(rep, {}).get(p, '')])
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
