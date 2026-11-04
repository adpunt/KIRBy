import sys,re
src=open(sys.argv[1]).read()
src=src.split("# check against the smoke")[0]
exec(src.replace("for lab,kw in [('base input","for lab,kw in [] and [('base input"))
import os
os.environ['KIRBY_NOISE_PROMOTE']='1'
from experiments.noise import noise_helpers as nh
print('MODEL_INPUTS',dc.MODEL_INPUTS)
print('FLAG_SHARES',dc.FLAG_SHARES,'PRECISIONS',dc.PRECISIONS,'WINSOR_C',dc.WINSOR_C)
print('CONFORMAL',len(dc.CONFORMAL_FLAG_SETTINGS),'ITLM',dc.n32.ITLM_ALPHAS)
print('dets',len(dc.REGISTRY_DETECTORS),len(dc.RESIDUAL_DETECTORS),len(dc.N41_DETECTORS),len(dc.CONTROL_DETECTORS),'nb',len(dc.NEIGHBOUR_DETECTORS))
print(dc.REGISTRY_DETECTORS+dc.RESIDUAL_DETECTORS+dc.N41_DETECTORS+dc.CONTROL_DETECTORS)
print('neigh',dc.NEIGHBOUR_DETECTORS)
for c in dc.CLEANUPS: print(' ',c,dc.CLEANUPS[c]['reads'],dc.CLEANUPS[c]['neighbours'],len(dc.cleanup_settings(c)))
data={'task_type':'regression','labels':list(range(5028)),'reps':{'r':None},'smiles':[str(i) for i in range(5028)]}
tot=0
for k in dc.default_conditions(data):
    lv=[float(x) for x in dc._condition_levels(data,'logd',k)];dr=len(nh.draws_for(k,(0,1,2)))
    print(k,'levels',lv,'draws',dr,'defined',k in nh.DETECTION_IS_DEFINED)
for lab,kw in [('base',dict(hybrid_input=False)),('hybrid',dict(hybrid_input=True))]:
    t,c,d=summ(unit(**kw));print(lab,t);print(' cl',c.most_common());print(' det',d.most_common())
