"""Executable checks for the experimental pipeline's splits.

    python tests/smoke/smoke_kirby_splits.py

Two properties of the early-stopping validation carve (RERUN_PLAN.md §2.12 in
qsar_qm_models):

  1. the validation rows share no scaffold group with the training rows, so
     early stopping is not scored on frameworks the model just fitted;
  2. across the five folds, every molecule reaches a training set at least once.

The old carve was `np.arange(len(train_idx) // 5)` -- the first fifth of the
fold's training block in dataset order, which is alphabetical by SMILES. Both
properties failed, and the CONTROL below runs that old rule on the same data to
show by how much.

Everything here RUNS the pipeline's own function.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alternative_data_noise_robustness import (  # noqa: E402
    N_FOLDS, RunIntegrityError, assign_scaffold_groups,
    scaffold_validation_carve)

T0 = time.time()
FAILS = []
NCHECK = [0]


def check(name, ok, detail=''):
    NCHECK[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILS.append(name)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# A dataset shaped like the real ones: 600 molecules over 120 scaffold groups,
# ordered so that molecules sharing a scaffold sit next to each other -- which is
# what an alphabetical SMILES ordering does.
rng = np.random.default_rng(0)
N, N_GROUPS = 600, 120
groups = np.repeat(np.arange(N_GROUPS), N // N_GROUPS)
labels = rng.normal(size=N)

section("the early-stopping validation split is scaffold-disjoint")

gkf = GroupKFold(n_splits=N_FOLDS)
seen_in_training = np.zeros(N, dtype=bool)
old_seen_in_training = np.zeros(N, dtype=bool)
shared_new, shared_old = 0, 0
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(N), labels, groups)):
    groups_full = groups[train_idx]
    tr, va = scaffold_validation_carve(labels[train_idx], groups_full, fold_idx)
    shared_new += len(set(groups_full[tr]) & set(groups_full[va]))
    seen_in_training[train_idx[tr]] = True

    # CONTROL: the rule this replaced.
    n_val = len(train_idx) // 5
    old_va = np.arange(n_val)
    old_tr = np.arange(n_val, len(train_idx))
    shared_old += len(set(groups_full[old_tr]) & set(groups_full[old_va]))
    old_seen_in_training[train_idx[old_tr]] = True

check("no scaffold group is in both training and early-stopping validation",
      shared_new == 0, f"{shared_new} shared groups (old rule: {shared_old})")
check("the old rule DID share scaffolds, so this check can tell them apart",
      shared_old > 0, f"{shared_old} shared groups under the old rule")

section("every molecule reaches a training set")

never_new = int((~seen_in_training).sum())
never_old = int((~old_seen_in_training).sum())
check("no molecule is kept out of training in every fold",
      never_new == 0,
      f"{never_new} of {N} never trained on (old rule: {never_old}, "
      f"{100 * never_old / N:.0f}%)")
check("the old rule DID starve a block of molecules",
      never_old > 0, f"{never_old} of {N}")

section("the CV scaffold key is chirality-blind")

# One Bemis-Murcko framework, two enantiomers. Their ECFP4 vectors are identical
# (create_ecfp4 leaves includeChirality False), so a key that separates them can
# put bit-identical molecules on both sides of the split.
# The stereocentre must be IN the framework, or the Murcko core drops it and the
# two keys agree for the wrong reason.
ENANTIOMERS = ['c1ccc([C@H]2CCCN2c2ccccc2)cc1',
               'c1ccc([C@@H]2CCCN2c2ccccc2)cc1']
g, n = assign_scaffold_groups(ENANTIOMERS)
check("two enantiomers of one framework share a CV group",
      n == 1 and g[0] == g[1], f"{n} group(s)")

# CONTROL: the rule this replaced.
from rdkit import Chem  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
aware = {Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)),
                          canonical=True) for s in ENANTIOMERS}
check("the old chirality-aware key DID separate them",
      len(aware) == 2, f"{len(aware)} distinct keys under the old rule")

# Acyclic molecules are singletons, as they are for the noise grouping and on QM9.
ACYCLIC = ['CCCC', 'CCCCC', 'CCO']
ga, na = assign_scaffold_groups(ACYCLIC)
check("each acyclic molecule gets its own CV group",
      na == 3 and len(set(ga)) == 3, f"{na} group(s) for 3 acyclic molecules")

section("the carve refuses a split it cannot make")

try:
    scaffold_validation_carve(labels[:10], np.zeros(10), 0)
except RunIntegrityError as exc:
    check("a single scaffold group cannot be carved, and says so", True, str(exc)[:60])
except ValueError as exc:
    check("a single scaffold group cannot be carved, and says so", True, str(exc)[:60])
else:
    check("a single scaffold group cannot be carved, and says so", False,
          "it returned a split with no groups to separate")

print("\n" + "=" * 70)
print(f"{NCHECK[0]} checks in {time.time() - T0:.1f}s")
if FAILS:
    print("FAILED:", "; ".join(FAILS))
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
