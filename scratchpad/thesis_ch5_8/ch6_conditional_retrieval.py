"""Chapter 6 conditional result for retrieval (s13): does the hybrid's lead in label gap of retrieved
neighbours change with distance, number of neighbours or counting; does weighting the hybrid change it;
fold spread at the fixed setting. Read-only on the 9 Oct results. Positive gain = hybrid's neighbours have
closer labels than the best single representation's (retrieval cost is lower-is-better)."""
import json,collections,statistics as st
R='/Volumes/seagate/kirby_results_2026-10-09/similarity/s13_clustering_retrieval/'
DS=['qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','abagym_EGFR','ddg_combined',
    'ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins']
for ds in DS:
    d=json.load(open(R+ds+'.json')); r=d['retrieval']
    single=r['rep_knn1']['features_unweighted']; H=r['hybrid_knn_distance1']
    by=collections.defaultdict(list); wgain=[]
    for key,sv in single.items():
        dist,k,sch=key.split('|')
        hu=H['features_unweighted'].get(key); hw=H['features_importance_weighted'].get(key)
        if not hu: continue
        g=sv['mean_cost']-hu['mean_cost']
        by['dist='+dist].append(g); by[k].append(g); by[sch].append(g); by['all'].append(g)
        if hw: wgain.append(hu['mean_cost']-hw['mean_cost'])
    fx='manhattan|k5|rank'
    pf=[a-b for a,b in zip(single[fx]['fold_scores'],H['features_unweighted'][fx]['fold_scores'])] if fx in single else None
    print(ds, {k:round(st.mean(v),4) for k,v in sorted(by.items())},
          '| weighting the hybrid, mean change', round(st.mean(wgain),4) if wgain else None,
          '| fixed-setting per fold', [round(x,3) for x in pf] if pf else None,
          '| single mean cost', round(st.mean([v['mean_cost'] for v in single.values()]),4))
