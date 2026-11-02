"""ProteinGym deep mutational scanning loader (substitutions, single mutants).

Source: https://github.com/OATML-Markslab/ProteinGym, release v1.3 (MIT).
Downloaded 2026-10-23 from https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/
to PROTEINGYM_ROOT below. Four archives: the assay scores, the cross-validation
fold assignments, the zero-shot model scores, the predicted structures.

What this loader reads is the FOLD files, not the plain assay files. The fold
files carry the same mutant, sequence and score columns plus the three fold
assignments, so one file is enough:

    mutant, mutated_sequence, DMS_score, DMS_score_bin,
    fold_random_5, fold_modulo_5, fold_contiguous_5

Fold scheme (Adelaide, 2026-10-23): modulo. ProteinGym names no single default;
it ships three, defined in Notin et al., NeurIPS 2023, and published supervised
benchmarks report all three side by side. Random is not used.

PER-POSITION ENCODINGS DO NOT WORK UNDER EITHER MODULO OR CONTIGUOUS.
Measured on ESTA_BACSU_Nutschel_2020, 2026-10-23, all three schemes:

    scheme       held-out positions also mutated in training (per fold)
    modulo       0, 0, 0, 0, 0
    contiguous   0, 0, 0, 0, 0
    random       153, 154, 145, 148, 149

Modulo and contiguous both assign POSITIONS to folds, so a held-out position is
never mutated in training; they differ only in which positions land together.
Any encoding with one feature per position per amino acid (onehot, blosum) then
has all its held-out mutants collapse to a single identical pattern on the
features that varied in training -- 1 distinct pattern across 417 to 470 test
rows -- the downstream model predicts a constant, and Spearman comes back nan.

Only 'random' shares positions between training and held-out rows, and random
splitting is not used in this repo.

So a representation is usable here only if it does not index by position:
amino acid composition, protein descriptors, substitution-type mutation
descriptors (create_mutation_features: wildtype and mutant amino acid, BLOSUM
score, property deltas), whole-sequence language model embeddings, and the
ProteinGym per-variant zero-shot scores.

Every mutant is a full-length copy of the protein with one residue substituted,
so a 2,000-mutant assay is 2,000 distinct sequences for a representation to
embed.

KNOWN TRUNCATION (unverified against every assay, read from the code 2026-10-23):
create_esm2_protein_embeddings cuts each sequence at 1,022 residues. Any assay
whose protein is longer than that loses mutations past residue 1,022 — those
mutants become identical to the wildtype under ESM-2. load_proteingym_dataset
raises on such an assay unless allow_truncation=True.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROTEINGYM_ROOT = Path(
    os.environ.get('KIRBY_PROTEINGYM_DIR', '/Volumes/seagate/proteingym_raw')
)
FOLD_DIR = PROTEINGYM_ROOT / 'ex_folds' / 'cv_folds_singles_substitutions'
REFERENCE_FILE = PROTEINGYM_ROOT / 'DMS_substitutions_reference.csv'

# ESM-2's cut in create_esm2_protein_embeddings.
MAX_EMBEDDABLE_LEN = 1022

FOLD_COLUMNS = {
    'contiguous': 'fold_contiguous_5',
    'modulo': 'fold_modulo_5',
    'random': 'fold_random_5',
}

NAME_PREFIX = 'pgym_'


def _reference() -> Dict[str, dict]:
    """Assay metadata, keyed by DMS_id."""
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f'{REFERENCE_FILE} is missing. It is ProteinGym\'s '
            f'reference_files/DMS_substitutions.csv; without it the loader has '
            f'no wildtype sequence and no assay type.'
        )
    with open(REFERENCE_FILE, newline='') as fh:
        return {row['DMS_id']: row for row in csv.DictReader(fh)}


def assay_id(name: str) -> str:
    """'pgym_BLAT_ECOLX_Jacquier_2013' -> 'BLAT_ECOLX_Jacquier_2013'."""
    return name[len(NAME_PREFIX):] if name.startswith(NAME_PREFIX) else name


def list_assays(min_variants: int = 0) -> List[dict]:
    """Every assay with a fold file, newest-first by variant count.

    Each entry: dms_id, n_variants (rows in the fold file), seq_len,
    coarse_selection_type, selection_type.
    """
    ref = _reference()
    out = []
    for path in sorted(FOLD_DIR.glob('*.csv')):
        dms_id = path.stem
        with open(path, newline='') as fh:
            n = sum(1 for _ in fh) - 1
        if n < min_variants:
            continue
        meta = ref.get(dms_id, {})
        out.append({
            'dms_id': dms_id,
            'n_variants': n,
            'seq_len': int(meta['seq_len']) if meta.get('seq_len') else None,
            'coarse_selection_type': meta.get('coarse_selection_type', ''),
            'selection_type': meta.get('selection_type', ''),
        })
    out.sort(key=lambda r: -r['n_variants'])
    return out


def load_proteingym_dataset(
    name: str,
    fold_scheme: str = 'modulo',
    allow_truncation: bool = False,
) -> Dict:
    """Load one ProteinGym substitution assay.

    Args:
        name: 'pgym_<DMS_id>' or a bare DMS_id, e.g. 'pgym_ESTA_BACSU_Nutschel_2020'.
        fold_scheme: 'modulo' (default), 'contiguous' or 'random'.
        allow_truncation: let an assay through whose protein is longer than
            MAX_EMBEDDABLE_LEN residues.

    Returns:
        dict with sequences, labels, groups, n_groups, task_type, mutations,
        wildtype, dms_id, seq_len, selection_type, coarse_selection_type.
        groups holds ProteinGym's own fold index, 0-4, so GroupKFold over them
        reproduces ProteinGym's folds exactly.
    """
    if fold_scheme not in FOLD_COLUMNS:
        raise ValueError(
            f'fold_scheme={fold_scheme!r}; ProteinGym ships '
            f'{sorted(FOLD_COLUMNS)}'
        )
    dms_id = assay_id(name)
    path = FOLD_DIR / f'{dms_id}.csv'
    if not path.exists():
        raise FileNotFoundError(
            f'No fold file for {dms_id!r} at {path}. '
            f'{len(list(FOLD_DIR.glob("*.csv")))} assays are present; '
            f'list_assays() names them.'
        )

    meta = _reference().get(dms_id)
    if meta is None:
        raise KeyError(f'{dms_id!r} has no row in {REFERENCE_FILE}')
    wildtype = meta['target_seq']
    seq_len = len(wildtype)
    if seq_len > MAX_EMBEDDABLE_LEN and not allow_truncation:
        raise ValueError(
            f'{dms_id} is {seq_len} residues; ESM-2 here cuts at '
            f'{MAX_EMBEDDABLE_LEN}, so every mutation past residue '
            f'{MAX_EMBEDDABLE_LEN} would be dropped and those mutants would '
            f'equal the wildtype. Pass allow_truncation=True to override.'
        )

    fold_col = FOLD_COLUMNS[fold_scheme]
    sequences, labels, folds, mutations = [], [], [], []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        if fold_col not in reader.fieldnames:
            raise KeyError(
                f'{path} has no column {fold_col!r}; it has '
                f'{reader.fieldnames}'
            )
        for row in reader:
            mutant = row['mutant']
            # Single substitutions only. ProteinGym writes multiples as
            # 'A1B:C2D'; cv_folds_singles_* should hold none, so a colon here
            # means the wrong archive was unzipped into FOLD_DIR.
            if ':' in mutant:
                raise ValueError(
                    f'{path} row {mutant!r} is a multiple mutant. This loader '
                    f'reads cv_folds_singles_substitutions only.'
                )
            sequences.append(row['mutated_sequence'])
            labels.append(float(row['DMS_score']))
            folds.append(int(row[fold_col]))
            wt_aa, pos, mut_aa = mutant[0], int(mutant[1:-1]), mutant[-1]
            # Key names are the ones create_mutation_features reads
            # (mutation.py: m['wt_aa'], m['mut_aa'], m.get('pos'),
            # m.get('chain')). 'chain' is 'heavy' because these reps were
            # written for antibodies and treat the first chain as heavy; a
            # single-chain protein has no light chain.
            mutations.append([{
                'wt_aa': wt_aa, 'mut_aa': mut_aa, 'pos': pos,
                'chain': 'heavy',
            }])

    labels = np.asarray(labels, dtype=float)
    groups = np.asarray(folds, dtype=int)
    n_groups = int(len(np.unique(groups)))
    if n_groups != 5:
        raise ValueError(
            f'{dms_id} {fold_col} has {n_groups} distinct folds, not 5'
        )

    return {
        'sequences': sequences,
        'labels': labels,
        'groups': groups,
        'n_groups': n_groups,
        'task_type': 'regression',
        'mutations': mutations,
        'wildtype': wildtype,
        'dms_id': dms_id,
        'seq_len': seq_len,
        'fold_scheme': fold_scheme,
        'selection_type': meta.get('selection_type', ''),
        'coarse_selection_type': meta.get('coarse_selection_type', ''),
        'property_name': meta.get('selection_assay', 'DMS_score'),
    }
