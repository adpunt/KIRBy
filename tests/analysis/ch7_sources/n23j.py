import json,sys,statistics as st
f,lv,model,act=sys.argv[1:5]
d=json.load(open(f)); p=d['per_regime']['injected'][lv]
for sp in ['base_1','hybrid','hybrid__general2']:
  for det,v in p[sp].items():
    for a in [k for k in v if k=='none' or k.endswith(act)]:
      x=v[a].get(model)
      if x: print(sp,det,a,{k:round(t,4) for k,t in x.items()},'mean %.4f'%st.mean(x.values()))
