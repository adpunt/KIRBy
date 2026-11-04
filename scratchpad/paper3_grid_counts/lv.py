import sys,os
os.environ['KIRBY_NOISE_PROMOTE']='1'
sys.path[:0]=['src','tests','tests/experiments']
from experiments.noise import n44_detect_and_clean as dc
for ds,n in [('caco2',2158),('logd',5028)]:
    data={'task_type':'regression','labels':list(range(n)),'reps':{'r':None},'smiles':[str(i) for i in range(n)]}
    for k in ['gaussian','outlier_p05','grouped_shifted','censoring','shift_only_p10']:
        print(ds,k,[float(x) for x in dc._condition_levels(data,ds,k)])
