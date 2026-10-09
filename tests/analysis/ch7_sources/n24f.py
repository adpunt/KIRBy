import pandas as pd,sys
a=pd.read_pickle('n24.pkl'); b=pd.read_pickle('n24_full.pkl'); a['src']='noise'; b['src']='full'
df=pd.concat([a,b])
ds,cond,lv,model,space,setting=sys.argv[1:7]
s=df[(df.dataset==ds)&(df.noise_condition==cond)&(df.level==float(lv))&(df.model==model)&(df.model_input==space)&(df.method_setting.str.startswith(setting))&(df.similarity_setting=='none')]
s=s.assign(arm=s.method+s.method_setting.str.extract(r'(\|seed=\d)')[0].fillna(''))
s=s.drop_duplicates(['arm','draw','fold','src'])
for (fold,src),g in s.groupby(['fold','src']):
    D=g.groupby('draw').score_D.first()
    piv=g.pivot_table(index='arm',columns='draw',values='score_Dprime')
    print(f'fold {fold} [{src}] draws {list(D.index)} no_cleaning', ' '.join(f'{x:.4f}' for x in D), '| mean %.4f'%D.mean())
    for arm,r in piv.iterrows():
        if arm in ('knn_smooth','oof_smooth'): continue
        print('   ',arm, ' '.join(f'{x:.4f}' for x in r.dropna()), '| mean %.4f'%r.mean())
