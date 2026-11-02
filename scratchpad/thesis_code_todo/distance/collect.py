"""Merge every result into results_all.json and print the LaTeX table bodies."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
DS = ['herg_ki', 'logd', 'caco2', 'adme_rlm', 'adme_mdr1', 'adme_hlm', 'qm9']
NAME = {'herg_ki': 'hERG $K_i$', 'logd': 'LogD', 'caco2': 'Caco-2 efflux', 'adme_rlm': 'RLM clearance',
        'adme_mdr1': 'MDR1--MDCK efflux', 'adme_hlm': 'HLM clearance', 'qm9': 'QM9'}
UNIT = {'herg_ki': 'p$K_i$', 'logd': 'log units', 'caco2': 'log units', 'adme_rlm': 'log units',
        'adme_mdr1': 'log units', 'adme_hlm': 'log units', 'qm9': 'eV'}
REP = {'ecfp4': 'ECFP4', 'atom_pair': 'Atom pairs', 'top_torsion': 'Topological torsions', 'avalon': 'Avalon',
       'maccs': 'MACCS keys', 'graph_kernel': 'WL graph kernel', 'pdv': 'PDV', 'mordred': 'Mordred',
       'grover': 'GROVER', 'mhg_gnn': 'MHG-GNN', 'graphmvp': 'GraphMVP', 'molformer': 'MoLFormer',
       'smited': 'SMI-TED', 'chembert': 'ChemBERT', 'chemberta': 'ChemBERTa-2', 'selformer': 'SELFormer',
       'mol2vec': 'Mol2vec'}
allr = {'datasets': {}}
for d in DS:
    r = json.load(open(f'{HERE}/results_{d}.json'))
    m = json.load(open(f'{HERE}/mcs_{d}.json')) if os.path.exists(f'{HERE}/mcs_{d}.json') else None
    r['mcs'] = m
    allr['datasets'][d] = r
ov = json.load(open(f'{HERE}/overlap.json'))
for d in ov:
    allr['datasets'][d]['overlap'] = ov[d]
allr['figure_herg_ki'] = json.load(open(f'{HERE}/figure_herg_ki.json'))
json.dump(allr, open(f'{HERE}/results_all.json', 'w'), indent=1)

f2 = lambda x: f'{x:.2f}'
f3 = lambda x: f'{x:.3f}'
print('%% TABLE 1')
for d in DS:
    r = allr['datasets'][d]; R = r['reps']; o = r['overlap']; desc = o['descriptor']
    c = r['cliff_rule']
    trio = ['ecfp4', desc, 'smited']
    cells = [NAME[d], f"{r['n_molecules']:,}"]
    cells += [f2(R[x]['allpairs_spearman']) for x in trio]
    cells += [f2(R[x]['knn5_spearman']) for x in trio]
    cells += [f"{c['n_cliff_molecules']:,}", f2(c['activity_gap'])]
    print(' & '.join(cells) + r' \\')
print('%% TABLE 2')
for d in DS:
    r = allr['datasets'][d]; o = r['overlap']; m = r['mcs']
    cells = [NAME[d], {'mordred': 'Mordred', 'pdv': 'PDV'}[o['descriptor']],
             f2(o['ecfp4_vs_desc']), f2(o['ecfp4_vs_emb']), f2(o['desc_vs_emb'])]
    if m:
        cells += [f2(m['spearman_tanimoto_vs_mcs_jaccard_completed']), f"{m['n_used_completed']}", f"{m['n_timed_out']}"]
    else:
        cells += ['--', '--', '--']
    print(' & '.join(cells) + r' \\')
print('%% TABLE 3 (long)')
for d in DS:
    r = allr['datasets'][d]
    first = True
    for rep in r['pool']:
        x = r['reps'][rep]
        cells = [NAME[d] if first else '', REP[rep], f2(x['allpairs_spearman']), f2(x['knn5_spearman']),
                 f2(x['knn5_rmse']), f"{x['matched_n_cliff_molecules']:,}",
                 f"{100 * x['frac_ecfp4_similar_pairs_also_closest_here']:.0f}"]
        first = False
        print(' & '.join(cells) + r' \\')
    print(r'\midrule')

# summary over the pool (dataset x representation in the general pool)
rows = [(d, rep, allr['datasets'][d]['reps'][rep]) for d in DS for rep in allr['datasets'][d]['pool']]
ap = [(x['allpairs_spearman'], d, rep) for d, rep, x in rows]
kn = [(x['knn5_spearman'], d, rep) for d, rep, x in rows]
allr['summary'] = {
    'n_dataset_rep_pairs': len(rows),
    'allpairs_min': min(ap), 'allpairs_max': max(ap),
    'knn5_min': min(kn), 'knn5_max': max(kn),
    'n_knn5_gt_allpairs': sum(x['knn5_spearman'] > x['allpairs_spearman'] for _, _, x in rows),
    'n_allpairs_below_0.10': sum(x['allpairs_spearman'] < 0.10 for _, _, x in rows),
    'per_dataset_allpairs_max': {d: max((allr['datasets'][d]['reps'][r]['allpairs_spearman'], r) for r in allr['datasets'][d]['pool']) for d in DS},
    'per_dataset_knn5_range': {d: [min(allr['datasets'][d]['reps'][r]['knn5_spearman'] for r in allr['datasets'][d]['pool']),
                                   max(allr['datasets'][d]['reps'][r]['knn5_spearman'] for r in allr['datasets'][d]['pool'])] for d in DS},
    'per_dataset_matched_cliff_range': {d: [min(allr['datasets'][d]['reps'][r]['matched_n_cliff_molecules'] for r in allr['datasets'][d]['pool'] if r != 'ecfp4'),
                                            max(allr['datasets'][d]['reps'][r]['matched_n_cliff_molecules'] for r in allr['datasets'][d]['pool'] if r != 'ecfp4')] for d in DS},
    'per_dataset_shared_similar_pairs_pct_range': {d: [100 * min(allr['datasets'][d]['reps'][r]['frac_ecfp4_similar_pairs_also_closest_here'] for r in allr['datasets'][d]['pool'] if r != 'ecfp4'),
                                            100 * max(allr['datasets'][d]['reps'][r]['frac_ecfp4_similar_pairs_also_closest_here'] for r in allr['datasets'][d]['pool'] if r != 'ecfp4')] for d in DS},
}
json.dump(allr, open(f'{HERE}/results_all.json', 'w'), indent=1)
print(json.dumps(allr['summary'], indent=1))
