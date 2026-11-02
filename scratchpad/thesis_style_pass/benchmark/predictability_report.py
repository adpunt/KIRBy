import json, math, re
B = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
res = {tuple(k.split('|')): v for k, v in json.load(open(B + 'predictability_cleaned.json')).items()}
CORP = ['buttenschoen', 'dablander', 'vost', 'hers', 'ai']
ST = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
LAB = [('surp', 'GPT-2 surprisal, cleaned text, bits per token (lower = more predictable)'),
       ('surp_sd', 'Surprisal SD across sentences, cleaned text (burstiness)'),
       ('top1', '% tokens that are GPT-2 top-1 prediction, cleaned text (higher = more predictable)')]
f = lambda x: '–' if x is None else (f'{x:.2f}' if abs(x) < 10 else f'{x:.1f}')
def rowfor(s, k, lab):
    vals = [res[(c, s)][k] if (c, s) in res else None for c in CORP]
    refs = [v for v in vals[:3] if v is not None]; a = vals[4]; flag = ''
    if a is not None and refs: flag = 'inside' if min(refs) <= a <= max(refs) else ('above' if a > max(refs) else 'below')
    return f'| {lab} | ' + ' | '.join(f(v) for v in vals) + f' | {flag} |'
tables = {s: [rowfor(s, k, lab) for k, lab in LAB] for s in ST if any((c, s) in res for c in CORP)}
json.dump(tables, open(B + 'predictability_rows.json', 'w'), indent=1)
for s, t in tables.items():
    print(s); print('\n'.join(t))
    print('  n sentences scored:', [res[(c, s)]['n'] if (c, s) in res else '-' for c in CORP])
