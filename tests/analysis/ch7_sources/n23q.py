import pandas as pd,sys
d=pd.read_pickle(sys.argv[1])
ds,cond,lv,model,act=sys.argv[2:7]
s=d[(d.dataset==ds)&(d.noise_condition==cond)&(d.level==float(lv))&(d.model==model)&(d.model_input.isin(['base_1','hybrid','hybrid__general2']))&(d.method_setting.str.endswith(act))]
for (m,mi,ms,sim),g in s.groupby(['method','model_input','method_setting','similarity_setting']):
    g=g.sort_values(['fold','draw'])
    if m!='activity_cliff_aware' and 'k=' in ms: continue
    print(f"{m:20s} {mi:17s} {ms[:30]:30s} {sim[:32]:32s} f{list(g.fold)} d{list(g.draw)} " + ' '.join(f'{x:.4f}' for x in g.score_D)+' -> '+' '.join(f'{x:.4f}' for x in g.score_Dprime)+'  mean %.4f->%.4f'%(g.score_D.mean(),g.score_Dprime.mean()))
