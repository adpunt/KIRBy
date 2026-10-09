import json,glob,os,re
BAD=re.compile('tcga|annitia|nhanes|pam50',re.I)
roots=['/Volumes/seagate/kirby_results_2026-10-09/noise','/Volumes/seagate/kirby_results_2026-10-09/noise_full','/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise']
out={}
for r in roots:
  for f in glob.glob(r+'/*/*.jsonl'):
    b=os.path.basename(f)
    if BAD.search(f) or not (b.startswith('qm9') or b.startswith('ginkgo_colloidal')): continue
    try:
      for line in open(f):
        if '"dnn"' not in line and 'dnn:' not in line: continue
        try: x=json.loads(line)
        except: continue
        if x.get('model')!='dnn' or x.get('level') not in (0,0.0): continue
        mi=str(x.get('model_input',''))
        cfg='hybrid__general2' if 'hybrid__general2' in mi else ('base_1' if 'base_1' in mi else None)
        if (b.startswith('qm9') and cfg!='hybrid__general2') or (b.startswith('ginkgo') and cfg!='base_1'): continue
        k=(f.replace('/Volumes/seagate/',''), x.get('dataset'), mi, x.get('fold'), x.get('draw'), x.get('noise_condition'))
        out.setdefault(k,set()).add(round(x['score_D'],4) if x.get('score_D') is not None else None)
    except Exception as e: print('ERR',f,e)
for k,v in sorted(out.items(), key=lambda t:(t[0][1],t[0][0])): print(k,sorted(v,key=str))
