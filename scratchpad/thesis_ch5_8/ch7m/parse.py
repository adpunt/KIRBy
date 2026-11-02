import re, json, sys, collections
A='/Volumes/seagate/claude_analysis_2026-10-08/answers/'
OK={'qm9','caco2','herg_ki','logd','herg','adme_mdr1','adme_rlm','buchwald_hartwig','flab_expression_koenig','flab_binding_koenig','flab_thermo_tresanco','abagym_EGFR','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins','ginkgo_hic','abagym_Ang2','humanness_therapeutic','herg_ic50'}
groups=json.load(open(A+'_scripts/method_groups.json'))
G={(g['section'],g['method']):g for g in groups}
sec=None; meth=None; hdr=None; rows=[]; seen_ds=collections.Counter()
for line in open(A+'MEETING_METHODS.md'):
    m=re.match(r'## \d+\. (.*)',line)
    if m: sec=m.group(1).strip(); meth=None; continue
    m=re.match(r'\*\*(.+?)\*\* — group',line)
    if m: meth=m.group(1); continue
    if line.startswith('| dataset'):
        hdr=[c.strip() for c in line.strip().strip('|').split('|')]; continue
    if line.startswith('|---') or not line.startswith('| ') or hdr is None: continue
    c=[x.strip() for x in line.strip().strip('|').split('|')]
    if len(c)!=len(hdr): continue
    d=dict(zip(hdr,c)); ds=d['dataset'].lstrip('*'); strong=d['dataset'].startswith('*')
    run=''
    mm=re.match(r'(\S+) \((.*)\)',ds)
    if mm: ds,run=mm.group(1),mm.group(2)
    seen_ds[ds]+=1
    if ds not in OK: continue
    def pr(s):
        m=re.match(r'(-?[\d.]+) → (-?[\d.]+) ([✓✗])',s)
        return (float(m.group(1)),float(m.group(2)),m.group(3)=='✓') if m else None
    rows.append(dict(section=sec,method=meth,ds=ds,run=run,noise=d['noise'],model=d['model'],metric=d['metric'].split()[0],
        base=pr(d['single representation']),hyb=pr(d['hybrid']),works=d['does it work?'],hb=d['hybrid better than base?'],strong=strong,
        ctl=d.get('random removal (single / hybrid)')))
json.dump(rows,open('rows.json','w'))
print(len(rows)); print(sorted(seen_ds.items()))
miss={(r['section'],r['method']) for r in rows}-set(G)
print('ungrouped',miss)
