import pandas as pd,json,sys,glob
root=sys.argv[1]; ds=sys.argv[2]; lv=float(sys.argv[3]); model=sys.argv[4]
rows=[]
for f in glob.glob(f'{root}/n25_cliff_exemption/{ds}__*.rows.jsonl'):
    for l in open(f):
        if l.strip():
            r=json.loads(l); r['_f']=f.split('/')[-1]; rows.append(r)
df=pd.DataFrame(rows)
df=df[(df.level==lv)&(df.model==model)]
df['arm']=df.method_setting.str.extract(r'arm=(\w+)')[0]
df['setting']=df.method_setting.str.replace(r'\|arm=\w+','',regex=True)
pv=df.pivot_table(index=['_f','fold','draw','model_input','method','setting','similarity_setting'],columns='arm',values='score_Dprime').reset_index()
D=df.groupby(['_f','fold','draw','model_input','method','setting','similarity_setting']).score_D.first().reset_index()
pv=pv.merge(D)
pv.to_pickle(f'n25_{ds}_{lv}_{model}_{root.split("/")[-1]}.pkl')
pd.set_option('display.width',300); pd.set_option('display.max_rows',100); pd.set_option('display.max_colwidth',70)
q=pv[(pv.cliff_exempt.round(3).isin([0.591,0.579]))|(pv.plain.round(3).isin([0.607,0.596]))]
print(q.to_string())
