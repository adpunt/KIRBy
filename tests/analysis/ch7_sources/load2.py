import json,glob,os,re,sys
import pandas as pd
BAD=re.compile('tcga|annitia|nhanes|pam50',re.I)
DS={'qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins'}
def load(exp, root='/Volumes/seagate/kirby_results_2026-10-09/noise', pat='*.rows.jsonl'):
    fs=[f for f in glob.glob(f'{root}/{exp}/{pat}') if not BAD.search(f)]
    out=[]; global bad
    for f in fs:
        ds=os.path.basename(f).split('__')[0].split('.')[0]
        if ds not in DS: continue
        for line in open(f):
            line=line.strip()
            if not line: continue
            
            try: r=json.loads(line)
            except Exception: bad.append(f); continue
            r['_file']=os.path.basename(f); out.append(r)
    return pd.DataFrame(out)
bad=[]
