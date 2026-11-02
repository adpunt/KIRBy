"""
Matbench v0.1 Dataset Loader

Six regression tasks from the Matbench v0.1 suite (Dunn et al. 2020,
npj Comput. Mater. 6, 138). Five carry a pymatgen Structure per compound;
matbench_expt_gap carries a composition string only.

    task                   compounds  input
    matbench_jdft2d              636  structure
    matbench_phonons            1265  structure
    matbench_expt_gap           4604  composition
    matbench_dielectric         4764  structure
    matbench_log_gvrh          10987  structure
    matbench_perovskites       18928  structure

FOLDS: Matbench defines its own 5 test folds (KFold, shuffled, fixed seed).
They are random, not grouped. They are used here as they are, so scores
compare with the Matbench leaderboard (author's decision, 2026-10-23).
'matbench_fold' gives each row's Matbench test fold. 'groups' gives every
compound its own group, so any group-aware split inside a fold is a split by
compound.

Data files are read from MATMINER_DATA (default /Volumes/seagate/matbench/data).
"""

import os

import numpy as np

MATBENCH_DATA_DIR = '/Volumes/seagate/matbench/data'

MATBENCH_TASKS = [
    'matbench_jdft2d',
    'matbench_phonons',
    'matbench_expt_gap',
    'matbench_dielectric',
    'matbench_log_gvrh',
    'matbench_perovskites',
]

# Tasks whose input is a composition string, not a Structure.
COMPOSITION_ONLY_TASKS = {'matbench_expt_gap'}


def load_matbench(task_name):
    """Load one Matbench v0.1 task with Matbench's own 5 test folds.

    Returns:
        dict: {
            'structures':    list of pymatgen Structure (None for
                             composition-only tasks),
            'compositions':  list of pymatgen Composition (every task),
            'labels':        float array,
            'task_type':     'regression',
            'groups':        np.arange(n), one group per compound,
            'n_groups':      n,
            'matbench_fold': int array, Matbench test fold (0-4) of each row,
            'ids':           Matbench row ids (mb-<task>-<n>),
            'property_name': target column name,
            'input_type':    'structure' or 'composition',
        }
    """
    if task_name not in MATBENCH_TASKS:
        raise ValueError(f"Unknown Matbench task '{task_name}'. "
                         f"Known: {MATBENCH_TASKS}")
    os.environ.setdefault('MATMINER_DATA', MATBENCH_DATA_DIR)

    from matbench.bench import MatbenchBenchmark
    from pymatgen.core import Composition

    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = getattr(mb, task_name)
    task.load()
    df = task.df
    target = task.metadata['target']
    if task.metadata['task_type'] != 'regression':
        raise ValueError(f"{task_name}: task type "
                         f"{task.metadata['task_type']}, only regression here")

    ids = list(df.index)
    row_of = {mbid: i for i, mbid in enumerate(ids)}
    fold = np.full(len(df), -1, dtype=int)
    for k in task.folds:
        test_ids = task.get_test_data(k, include_target=False).index
        for mbid in test_ids:
            if fold[row_of[mbid]] != -1:
                raise RuntimeError(f"{task_name}: {mbid} in two test folds")
            fold[row_of[mbid]] = int(k)
    if (fold < 0).any():
        raise RuntimeError(f"{task_name}: {int((fold < 0).sum())} rows in no "
                           f"test fold")

    if task_name in COMPOSITION_ONLY_TASKS:
        structures = None
        compositions = [Composition(c) for c in df['composition']]
        input_type = 'composition'
    else:
        structures = list(df['structure'])
        compositions = [s.composition for s in structures]
        input_type = 'structure'

    n = len(df)
    return {
        'structures': structures,
        'compositions': compositions,
        'labels': df[target].to_numpy(dtype=float),
        'task_type': 'regression',
        'groups': np.arange(n),
        'n_groups': n,
        'matbench_fold': fold,
        'ids': ids,
        'property_name': target,
        'input_type': input_type,
    }
