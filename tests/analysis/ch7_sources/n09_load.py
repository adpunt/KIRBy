import json,glob,os,re,math
D='/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n09_lnl_survey'
SCOPE=['qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins']
BAD=re.compile('tcga|annitia|nhanes|pam50',re.I)
def files():
    out=[]
    for f in sorted(os.listdir(D)):
        if BAD.search(f) or not f.endswith('.json'): continue
        ds,kind=f[:-5].split('__',1)
        if ds in SCOPE: out.append((ds,kind,os.path.join(D,f)))
    return out
