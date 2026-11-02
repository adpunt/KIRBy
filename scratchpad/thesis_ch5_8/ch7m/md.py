import json,re
o=json.load(open('agg.json'))
S={'qm9':'qm9','caco2':'caco2','herg_ki':'hergKi','logd':'logd','herg':'herg','buchwald_hartwig':'BH','flab_thermo_tresanco':'tresanco','ddg_combined':'ddg','ginkgo_titer':'titer','ginkgo_pr_cho':'prCHO','ginkgo_colloidal':'colloid','ginkgo_acsins':'acsins','humanness_therapeutic':'human','herg_ic50':'hergIC50'}
ACT={'weight':'count less','move':'move the label','move the label':'move the label','delete':'delete','none':'—'}
def ex(s):
    m=re.match(r'(\S+), (.*?), (\w+): base (\S+)→(\S+), hybrid (\S+)→(\S+) (\w+)',s)
    if not m: return ''
    ds,noise,model,b0,b1,h0,h1,met=m.groups()
    noise=noise.replace('none added (real labels)','real labels').replace(' (level 1)','').replace(' (25% of labels)','')
    return f"{S.get(ds,ds)}, {noise}, {model}, {met}: base {float(b0):.3g}→{float(b1):.3g}, hybrid {float(h0):.3g}→{float(h1):.3g}"
def row(r,act=True):
    ds=', '.join(S.get(d,d) for d in r['ds'])
    if r['N']==0: return f"| {r['method']} | {r['exp']} | {r['note']} |"+(f" {ACT[r['act']]} |" if act else '')+f" {ds} | no scores | | | |"
    return (f"| {r['method']} | {r['exp']} | {r['note']} |"+(f" {ACT[r['act']]} |" if act else '')+
            f" {ds} | {r['bh']} of {r['N']} | {r['hh']} of {r['N']} | {r['both']} of {r['N']} | {ex(r['ex'])} |")
H1="| method | exp | what it tests | action | datasets with results | base helped | hybrid helped | both helped | largest hybrid gain |\n|---|---|---|---|---|---|---|---|---|"
H2="| method | exp | what it tests | datasets with results | base helped | hybrid helped | both helped | largest hybrid gain |\n|---|---|---|---|---|---|---|---|"
out=[]
def tab(title,f,act=True):
    rs=[r for r in o if f(r)]
    rs.sort(key=lambda r:(ACT[r['act']],r['method'].lower(),r['exp']))
    out.append(f"\n### {title}\n"); out.append(H1 if act else H2); out.extend(row(r,act) for r in rs)
tab('Noise-robust training: selected by nothing, the training changes',lambda r:r['act']=='none' and r['sel']=='none',False)
tab('Noise-robust training: every molecule or a known flag, data untouched',lambda r:r['act']=='none' and r['sel']!='none',False)
tab('Noise mitigation: selected by a detector',lambda r:r['act']!='none' and r['sel']=='detector')
tab('Noise mitigation: selected by a known flag',lambda r:r['act']!='none' and r['sel']=='flag')
tab('Noise mitigation: every molecule',lambda r:r['act']!='none' and r['sel']=='all')
open('tables.md','w').write('\n'.join(out)); print('\n'.join(out))
