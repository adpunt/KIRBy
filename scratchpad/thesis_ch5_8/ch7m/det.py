import re
A='/Volumes/seagate/claude_analysis_2026-10-08/answers/'
OK={'qm9','caco2','herg_ki','logd','herg','adme_mdr1','adme_rlm','buchwald_hartwig','flab_expression_koenig','flab_binding_koenig','flab_thermo_tresanco','abagym_EGFR','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins','ginkgo_hic','abagym_Ang2','humanness_therapeutic','herg_ic50'}
Q={'n07':['4.','5.'],'n21':['2.'],'n12':['5.'],'n23':['3.'],'n31':['1.','3.'],'n32':['4.'],'n38':['3.'],'n41':['4.'],'n22':['1.','2.']}
for e,qs in Q.items():
    L=open(A+e+'.md').read().split('\n')
    hdr=None
    for l in L:
        if l.startswith('| dataset'): hdr=[c.strip() for c in l.strip('|').split('|')]; res={i:{} for i,c in enumerate(hdr) if any(c.startswith(q) for q in qs)}; continue
        if hdr and l.startswith('| ') and not l.startswith('|---'):
            c=[x.strip() for x in l.strip('|').split('|')]
            if c[0] in OK:
                for i in res: res[i][c[0]]=c[i]
        elif hdr and not l.startswith('|'):
            for i,d in res.items():
                y=[k for k,v in d.items() if v=='yes']; n=[k for k,v in d.items() if v=='no']
                print(f"{e} | {hdr[i][:90]} | yes: {', '.join(y) or '-'} | no: {', '.join(n) or '-'}")
            hdr=None
