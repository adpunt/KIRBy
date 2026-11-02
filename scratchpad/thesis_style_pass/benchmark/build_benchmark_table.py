"""Build the language part of BENCHMARK_TABLE.md (2026-10-10 evening).
Reads the freshly written BENCHMARK_HERS_VS_AI.md, BENCHMARK_EXTRA.md and predictability_rows.json for "AI now",
and before_tonight_rerun/BENCHMARK_TABLE.md for "AI before tonight's fixes". Writes language_tables.md and
status_counts.json; the structure table and header are written by hand into BENCHMARK_TABLE.md."""
import re, json, collections
B = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
ST = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
MOVES = ['method/background restated', 'pointer to figure/table', 'result as numbers', 'result in words', 'comparison of results',
         'interpretation', 'comparison with literature', 'claim of strength', 'limitation', 'link/transition',
         'recommendation/implication']
PREF = {'Words per sentence', '% sentences over 30 words', '% passive sentences', 'AI-guide phrases per 100 words'}
UNCLEANED = ('GPT-2 surprisal, bits per token', 'Surprisal SD across sentences (burstiness)')

def parse(md):
    """{section: {measure: [b, d, v, yours, ai]}} from markdown tables headed '### Section'"""
    out = {}; sec = None
    for ln in md.split('\n'):
        m = re.match(r'### (\w+)', ln)
        if m: sec = m.group(1).lower(); continue
        if sec and ln.startswith('| ') and not ln.startswith('| Measure') and not ln.startswith('|---'):
            c = [x.strip() for x in ln.strip('|').split('|')]
            out.setdefault(sec, {})[c[0]] = c[1:6]
    return out

old_md = open(B + 'before_tonight_rerun/BENCHMARK_TABLE.md').read()
lang_old, rest = old_md.split('## Added measures')[0], old_md.split('## Added measures')[1]
extra_old, pred_old = rest.split('## Predictability on identically cleaned text')
OLD = {}
for part in (lang_old, extra_old, pred_old):
    for s, d in parse(part).items(): OLD.setdefault(s, {}).update(d)
NEW = {}
for s, d in parse(open(B + 'BENCHMARK_HERS_VS_AI.md').read()).items(): NEW.setdefault(s, {}).update(d)
for s, d in parse(open(B + 'BENCHMARK_EXTRA.md').read()).items(): NEW.setdefault(s, {}).update(d)
pr = json.load(open(B + 'predictability_rows.json'))
for s, rows in pr.items():
    for ln in rows:
        c = [x.strip() for x in ln.strip('|').split('|')]
        NEW.setdefault(s, {})[c[0]] = c[1:6]
size = {}
for ln in open(B + 'BENCHMARK_HERS_VS_AI.md'):
    pass
cur = None
for ln in open(B + 'BENCHMARK_HERS_VS_AI.md').read().split('\n'):
    m = re.match(r'### (\w+)', ln)
    if m: cur = m.group(1).lower()
    if ln.startswith('Size:') and cur: size[cur] = ln[len('Size: '):].replace('hers:', 'yours:').replace('ai:', 'AI-written:')

num = lambda x: None if x in ('–', '', None) else float(x.replace(',', ''))
def fmt_range(lo, hi):
    g = lambda x: f'{x:.3g}' if abs(x) < 10 else f'{x:.1f}'
    return f'{g(lo)}–{g(hi)}'

def status(meas, refs, y, a, b):
    if a is None or not refs: return '', ''
    lo, hi = min(refs), max(refs)
    if lo <= a <= hi: return 'done', ''
    if y is not None and ((y < lo and y <= a < lo) or (y > hi and hi < a <= y)): return 'done', ''
    if meas in PREF: return 'not a target', ''
    side = 'below' if a < lo else 'above'
    if y is not None and ((side == 'below' and y < lo) or (side == 'above' and y > hi)):
        return 'not a target', ''
    dist = lambda v: 0 if v is None else (lo - v if v < lo else v - hi if v > hi else 0)
    yours = '' if y is None else f', yours {y:g}'
    if b is not None and dist(a) < dist(b): return 'closer', ''
    moved = '' if b is None else (' unchanged' if abs(a - b) < 1e-9 else f'; was {b:g}, moved away' if dist(a) > dist(b) else f'; was {b:g}')
    return 'not done', f'{side} the references ({fmt_range(lo, hi)}{yours}){moved}'

GROUPS = [('Sentences and words', lambda m: m not in MOVES and not m.startswith(('Lexical div', 'Sentence length', 'Mean change', 'Connectives',
                                    'Distinct', '% sentences opening', 'Three-item', 'Pronouns', 'GPT-2', 'Surprisal', '% tokens'))),
          ('Rhetorical moves, % of coded sentences', lambda m: m in MOVES),
          ('Diversity, variability and connectives', lambda m: m.startswith(('Lexical div', 'Sentence length', 'Mean change', 'Connectives',
                                    'Distinct', '% sentences opening', 'Three-item', 'Pronouns'))),
          ('Predictability, GPT-2 small on identically cleaned text', lambda m: 'cleaned text' in m)]
out = []; counts = {}; notdone = []
for s in ST:
    if s not in NEW: continue
    out.append(f'\n### {s.capitalize()}\n')
    if s in size: out.append(f'Size now: {size[s]}\n')
    cnt = collections.Counter()
    for gname, pick in GROUPS:
        ms = [m for m in NEW[s] if pick(m) and not m.startswith(UNCLEANED)]
        if not ms: continue
        out.append(f'**{gname}**\n')
        out.append('| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight\'s fixes | AI now | Status |')
        out.append('|---|---|---|---|---|---|---|---|')
        for m in ms:
            v = NEW[s][m]; ov = OLD.get(s, {}).get(m)
            refs = [num(x) for x in v[:3] if num(x) is not None]
            y, a = num(v[3]), num(v[4]); b = num(ov[4]) if ov else None
            st, why = status(m, refs, y, a, b)
            if st: cnt[st] += 1
            cell = st + (f': {why}' if why else '')
            if st == 'not done': notdone.append((s, m, why))
            out.append(f'| {m} | {v[0]} | {v[1]} | {v[2]} | {v[3]} | {ov[4] if ov else "–"} | {v[4]} | {cell} |')
        out.append('')
    counts[s] = dict(cnt)
open(B + 'language_tables.md', 'w').write('\n'.join(out) + '\n')
json.dump(dict(counts=counts, notdone=notdone), open(B + 'status_counts.json', 'w'), indent=1)
print(json.dumps(counts, indent=1))
for x in notdone: print(x)
