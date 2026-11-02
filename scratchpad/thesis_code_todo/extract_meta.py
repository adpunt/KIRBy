"""Save SMILES, labels and groups for one dataset, reading the cached ecfp4 only.
Building any representation raises, so this can never start a featuriser."""
import sys, numpy as np
sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests'); sys.path.insert(0, '/Users/apunt/repos/KIRBy/src'); sys.path.insert(0, '/Users/apunt/repos/KIRBy')
import find_best_hybrid_molecular as fm
def _no_build(*a, **k): raise RuntimeError('refusing to build a representation')
fm.get_representations = _no_build
from experiments._shared import load_data
ds = sys.argv[1]
d = load_data(ds, rep_names=['ecfp4'])
X = np.load(f'/Volumes/seagate/kirby_cache/rep_cache/{ds}/ecfp4.npy', mmap_mode='r')
assert np.array_equal(np.asarray(X), np.asarray(d['reps']['ecfp4'])), 'ecfp4 differs from cache'
np.savez(f'meta_{ds}.npz', smiles=np.array(d['smiles'], dtype=object), labels=np.asarray(d['labels'], float),
         groups=np.asarray(d.get('groups')), allow_pickle=True)
print(ds, len(d['smiles']), 'labels sd', np.std(d['labels']))
