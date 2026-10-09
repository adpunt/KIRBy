import json,sys,os,glob,re,collections
BAN=re.compile('tcga|annitia|nhanes|pam50',re.I)
ARMS=['search_asis__fit_asis','search_asis__fit_cleaned','search_cleaned__fit_cleaned','search_cleaned__fit_asis']
def load(root):
    out={}
    for f in sorted(glob.glob(root+'/*.json'))+sorted(glob.glob(root+'/_partial/*.json')):
        if BAN.search(f): continue
        name=os.path.basename(f)[:-5]
        if name in out: continue
        d=json.load(open(f))
        if 'per_regime' not in d: continue
        out[name]=(f,d)
    return out
def recs(d):
    R=[]
    for reg,lv in d['per_regime'].items():
        for lev,bb in lv.items():
            for base,cc in bb.items():
                for cl,rr in cc.items():
                    for rate,mm in rr.items():
                        for mi,aa in mm.items():
                            keys=set()
                            for a,kv in aa.items(): keys|=set(kv)
                            for k in keys:
                                R.append(dict(reg=reg,lev=lev,base=base,cl=cl,rate=rate,mi=mi,key=k,**{a:aa.get(a,{}).get(k) for a in ARMS}))
    return R
lib={}
for f in glob.glob(sys.argv[1]+'/*.rows.jsonl'):
    if BAN.search(f): continue
    l=open(f).readline(); r=json.loads(l); lib[os.path.basename(f).split('.rows')[0]]=r['lower_is_better']
data=load(sys.argv[1])
mode=sys.argv[2] if len(sys.argv)>2 else 'best'
for name,(f,d) in sorted(data.items()):
    low=d.get('lower_is_better', lib.get(name)); metric=d.get('metric')
    R=recs(d)
    if mode=='folds':
        print('##',name,f.replace(sys.argv[1],''),'metric',metric,'low',low,'keys',sorted({r['key'] for r in R}))
        continue
    sgn=-1 if low else 1
    R=[r for r in R if r['cl']!='no_cleaning' and 'random_flag' not in r['cl'] and r['search_asis__fit_asis'] is not None]
    print('##',name,'|',f.split('/')[-2]+'/'+f.split('/')[-1],'| metric',metric,'lower_better',low,'| n',len(R),'| regimes',sorted({r['reg']+':'+r['lev'] for r in R}))
    for a in ARMS[1:]:
        for scope in ['hyb','single']:
            S=[r for r in R if r[a] is not None and (('hybrid' in r['mi'])==(scope=='hyb'))]
            if not S: continue
            b=max(S,key=lambda r:sgn*(r[a]-r['search_asis__fit_asis']))
            print(f"  best {a:28s} {scope:6s}: {b['reg']} lvl {b['lev']} {b['key']} {b['mi']} {b['cl']}@{b['rate']} cleaned_in={b['base']} | "+' / '.join(f"{x.split('__')[0][7:]}-{x.split('__')[1][4:]}={b[x]:.4f}" if b[x] is not None else 'NA' for x in ARMS))
