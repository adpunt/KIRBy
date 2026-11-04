import sys,os
os.environ['KIRBY_NOISE_PROMOTE']='1'
sys.path[:0]=['src','tests','tests/experiments']
from experiments.noise import n44_detect_and_clean as dc, n45_detector_grading as dg, n46_structure_specific_fixes as st, n43_noise_aware_training as na
from experiments.noise import noise_helpers as nh
from experiments.orchestrate.plan import dataset_roster, dataset_info
def passes(data,ds,c,lv=None):
    levels=[float(x) for x in (lv or dc._condition_levels(data,ds,c))]
    return 1+len([l for l in levels if l>0])*len(nh.draws_for(c,(0,1,2)))
for ds,n in [('caco2',2158),('logd',5028)]:
    data={'task_type':'regression','labels':list(range(n)),'reps':{'r':None},'smiles':[str(i) for i in range(n)]}
    for name,conds,levf in [('n44',dc.default_conditions(data),None),('n45',dg.default_conditions(data),None),('n46',list(nh.decided_conditions(data)),None),('n47',list(nh.decided_conditions(data)),None)]:
        ps={c:passes(data,ds,c) for c in conds}
        print(ds,name,sum(ps.values()),ps)
