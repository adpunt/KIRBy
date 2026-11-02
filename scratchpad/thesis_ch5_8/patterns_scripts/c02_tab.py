import json,collections
R=json.load(open('/private/tmp/claude-501/c02_rows.json'))
def tab(key, rows, label):
    t=collections.defaultdict(lambda:[0,0,set()])
    for r in rows:
        k=key(r); t[k][0]+=r['win']; t[k][1]+=1; t[k][2].add(r['ds'])
    print(f'  [{label}]')
    for k,(w,n,s) in sorted(t.items(), key=lambda kv:str(kv[0])):
        print(f'    {k}: hybrid better on {w} of {n} fold-model pairs, {len(s)} datasets')
for dom in ['small_molecule','antibody','reaction','gene_expression']:
    rows=[r for r in R if r['dom']==dom]
    print('==',dom, len(rows))
    tab(lambda r:r['alloc'],rows,'allocation of hybrid1')
    tab(lambda r:r['imp'],rows,'importance of hybrid1')
    tab(lambda r:r['ev'],rows,'internal evaluator of hybrid1')
    tab(lambda r:r['rc'],rows,'rep count of hybrid1')
    tab(lambda r:r['budget'],rows,'budget of hybrid1')
    tab(lambda r:('budget>ntrain' if r['budget']>r['ntrain'] else 'budget<=ntrain'),rows,'budget vs training molecules')
    tab(lambda r:r['model'],rows,'downstream model')
    tab(lambda r:r['ev']==r['model'],rows,'evaluator same as model')
