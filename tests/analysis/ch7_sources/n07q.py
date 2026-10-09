import pandas as pd,sys
df=pd.read_pickle(sys.argv[1])
ds,cond,lv,model=sys.argv[2:6]; meths=sys.argv[6].split(',')
s=df[(df.dataset==ds)&(df.noise_condition==cond)&(df.level==float(lv))&(df.model==model)&(df.method.isin(meths))]
s=s[s.model_input.isin([f'{model}:base_1',f'{model}:hybrid',f'{model}:hybrid__general2'])]
for (mi,m,ms,sim),g in s.groupby(['model_input','method','method_setting','similarity_setting'],dropna=False):
    g=g.sort_values(['fold','draw'])
    print(f"{mi:22s} {m:12s} {str(ms):38s} {str(sim)[:34]:34s} f{sorted(g.fold.unique())} D=" + ' '.join(f'{x:.4f}' for x in g.score_D)+' -> '+' '.join(f'{x:.4f}' for x in g.score_Dprime)+'  mean %.4f->%.4f'%(g.score_D.mean(),g.score_Dprime.mean()))
