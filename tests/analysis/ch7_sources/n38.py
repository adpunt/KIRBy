import json,glob,collections
S=['qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins']
g=collections.defaultdict(list)
for f in sorted(glob.glob('*.rows.jsonl')):
  ds=f.split('__')[0]
  if ds not in S: continue
  for l in open(f):
    r=json.loads(l)
    if r['method']!='robust_loss': continue
    g[(ds,r['method_setting'],r['model'],r['model_input'],r['level'])].append(r)
for k,v in sorted(g.items()):
  if k[4]==0: continue
  D=sum(r['score_D'] for r in v)/len(v); P=sum(r['score_Dprime'] for r in v)/len(v); C=sum(r['score_clean'] for r in v)/len(v)
  lib=v[0]['lower_is_better']; won=sum((r['gain']>0) for r in v)
  print(k, v[0]['metric'], f"noisy {D:.4f} robust {P:.4f} clean {C:.4f} won {won}/{len(v)} folds {sorted(set(r['fold'] for r in v))} draws {sorted(set(r['draw'] for r in v))}")
