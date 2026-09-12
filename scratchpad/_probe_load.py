import os, sys, time
sys.path.insert(0,'src'); sys.path.insert(0,'tests'); sys.path.insert(0,'tests/experiments')
from experiments._shared import load_data
ds = sys.argv[1]; reps = sys.argv[2].split(',')
t=time.time()
d = load_data(ds, rep_names=reps)
import numpy as np
print('OK', ds, 'n=', len(d['labels']), 'task=', d.get('task_type'), 'groups=', len(np.unique(d['groups'])), 'reps=', {k:v.shape for k,v in d['reps'].items()}, f'{time.time()-t:.0f}s')
