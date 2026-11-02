import json
R=json.load(open('/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo/ties/ties_results.json'))
NAMES={'mhg_gnn':'MHG-GNN','mordred':'Mordred','graph_kernel':'Graph kernel','avalon':'Avalon','grover':'GROVER','pdv':'PDV','ecfp4':'ECFP4','atom_pair':'Atom pair','chembert':'ChemBERT','smited':'SMI-TED','molformer':'MoLFormer','top_torsion':'Topological torsion','graphmvp':'GraphMVP','maccs':'MACCS'}
DS={'herg_ki':('hERG $K_i$','pKi'),'logd':('LogD','log units'),'caco2':('Caco-2','log units')}
rows=[]; summary={}
for ds in ['herg_ki','logd','caco2']:
    d=R[ds]; ORDER=['ecfp4','atom_pair','pdv','mordred','smited','avalon','top_torsion','graph_kernel','maccs']; reps=[r for r in ORDER if r in d['general']+d['augmentation']]; assert all((r in ORDER) or not d['reps'][r].get('tied') for r in d['general']+d['augmentation'])
    part={r:sorted(tuple(g['idx']) for g in d['reps'][r]['groups']) for r in reps if d['reps'][r].get('tied')}
    first=True
    for r in reps:
        i=d['reps'][r]
        if not i.get('tied'): continue
        name=NAMES[r]+(' (aug.)' if r in d['augmentation'] else '')
        dsn=f"{DS[ds][0]} ({DS[ds][1]})" if first else ''
        first=False
        rows.append(f"{dsn} & {name} & {i['molecules']:,} & {i['tied']:,} ({i['tied_pct']:.1f}\\%) & {i['stereo_only']:,} & {i['rmse_min']:.3f} & {i['best_r2']:.3f} & {i['max_spread']:.2f} \\\\")
    rows.append('\\midrule')
    summary[ds]={'no_tie':[r for r in reps if d['reps'][r].get('tied')==0],
                 'same_partition_as_ecfp4':[r for r in part if part[r]==part.get('ecfp4')]}
print('\n'.join(rows[:-1])); print(json.dumps(summary,indent=0))
