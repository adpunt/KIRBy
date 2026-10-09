import json,sys,statistics as st
f,lv,model=sys.argv[1],sys.argv[2],sys.argv[3]
spaces=sys.argv[4].split(',') if len(sys.argv)>4 else None
d=json.load(open('/Volumes/seagate/kirby_results_2026-10-09/noise/n24_delete_or_keep/'+f))
p=d['per_level'][lv]
for sp,v in p.items():
  if spaces and sp not in spaces: continue
  for det,w in v.items():
    for fr,x in w.items():
      nc=x['no_cleaning'].get(model)
      if not nc: continue
      line=[]
      for arm,y in x.items():
        z=y.get(model)
        if z: line.append(f"{arm.split('/')[-1]}={st.mean(z.values()):.4f}")
      print(sp,det,fr,list(nc.keys()),' '.join(line))
