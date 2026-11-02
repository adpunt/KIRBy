"""Benchmark split by author (2026-10-10): the author's own paragraphs vs AI-marked paragraphs, against the three
reference theses, by section type. AI-marked = a '% TODO: manual review' (or review marker) line directly above that is
not a '-- fixed:' / '-- moved:' note. Floats, captions and equations are excluded by segment_full_author.py."""
import json, csv, re, sys, collections
sys.path.insert(0, '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/9f687a4a-c71f-4415-b6e9-5c2f5f842ef6/scratchpad/pylib')
sys.path.insert(0, '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark')
from compute_metrics import nlp, normalise, sent_metrics, B
AI = re.compile(r'%\s*TODO:\s*(manual review|review[,:\s-])', re.I)
def who(ml):
    lines = [x for x in ml.split(' | ') if AI.search(x)]
    plain = [x for x in lines if not re.search(r'--\s*(fixed|moved):', x)]
    return 'ai' if plain else 'hers'
seg = {json.loads(l)['sentence_id']: json.loads(l)['text'] for l in open(B + 'segments_author_current.jsonl')}
moves = {seg[r['sentence_id']]: r['move'] for r in csv.DictReader(open(B + 'moves_author_current.csv')) if r['sentence_id'] in seg}
moves.update({r['text']: r['move'] for r in csv.DictReader(open(B + 'moves_after_new_sentences.csv'))})
moves.update({r['text']: r['move'] for r in csv.DictReader(open(B + 'moves_tonight_new_sentences.csv'))})  # 2026-10-10 evening re-coding
rows = []
R = [json.loads(l) for l in open(B + 'full_segments_author_current.jsonl')]
texts = [normalise(r['text'], False) for r in R]
for r, t, doc in zip(R, texts, nlp.pipe(texts, batch_size=256)):
    m = sent_metrics(doc, t)
    rows.append(dict(corpus=who(r.get('marker_lines', '')), paragraph_id=r['paragraph_id'], section_type=r['section_type'],
                     move=moves.get(r['text']), **m))
for l in open(B + 'archive_first_run/metrics_sentences.jsonl'):
    d = json.loads(l)
    if d['corpus'] in ('buttenschoen', 'vost', 'dablander'): rows.append(d)
json.dump(rows, open(B + 'split_hers_ai_sentences.json', 'w'))
CORP = ['buttenschoen', 'dablander', 'vost', 'hers', 'ai']
ST = ['abstract', 'introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
MEAS = [('sent_len', 'Words per sentence'), ('long30', '% sentences over 30 words'), ('dc', 'Dependent clauses per sentence'),
        ('para_len', 'Words per paragraph'), ('lexd', 'Lexical density %'), ('nom', 'Nominalizations per 100 words'),
        ('passive', '% passive sentences'), ('self', 'we/our/I per 100 words'), ('hedge', 'Hedges per 100 words'),
        ('boost', 'Boosters per 100 words'), ('ai', 'AI-guide phrases per 100 words'), ('num', 'Numbers per 100 words'),
        ('num3', '% sentences with 3+ numbers')] + [(f'M{i}', n) for i, n in enumerate(
        ['method/background restated', 'pointer to figure/table', 'result as numbers', 'result in words', 'comparison of results',
         'interpretation', 'comparison with literature', 'claim of strength', 'limitation', 'link/transition',
         'recommendation/implication'], 1)]
def agg(rs):
    if not rs: return None
    n = len(rs); w = sum(r['words'] for r in rs) or 1
    paras = collections.Counter(r['paragraph_id'] for r in rs)
    coded = [r for r in rs if r.get('move')]
    v = dict(sent_len=w / n, long30=100 * sum(r['words'] > 30 for r in rs) / n, dc=sum(r['dep_clauses'] for r in rs) / n,
             para_len=w / len(paras), lexd=100 * sum(r['content'] for r in rs) / (sum(r['alpha'] for r in rs) or 1),
             nom=100 * sum(r['nominal'] for r in rs) / w, passive=100 * sum(r['passive'] for r in rs) / n,
             self=100 * sum(r['selfmention'] for r in rs) / w, hedge=100 * sum(r['hedges'] for r in rs) / w,
             boost=100 * sum(r['boosters'] for r in rs) / w, ai=100 * sum(r['ai_hits'] for r in rs) / w,
             num=100 * sum(r['numbers'] for r in rs) / w, num3=100 * sum(r['num3'] for r in rs) / n,
             n=n, words=w, coded=len(coded))
    for i in range(1, 12):
        v[f'M{i}'] = 100 * sum(r['move'] == f'M{i}' for r in coded) / len(coded) if len(coded) >= 20 else None
    return v
A = {(c, s): agg([r for r in rows if r['corpus'] == c and r['section_type'] == s]) for c in CORP for s in ST}
out = []
for s in ST:
    if not any(A[(c, s)] for c in CORP): continue
    out.append(f'\n### {s.capitalize()}\n')
    size = ' · '.join(f"{c}: {A[(c,s)]['words']:,} words" for c in CORP if A[(c, s)])
    out.append(f'Size: {size}\n')
    out.append('| Measure | Buttenschoen | Dablander | Vost | Yours | AI-written | AI vs references |')
    out.append('|---|---|---|---|---|---|---|')
    for k, lab in MEAS:
        vals = [A[(c, s)][k] if A[(c, s)] else None for c in CORP]
        if all(x is None for x in vals): continue
        refs = [x for x in vals[:3] if x is not None]
        f = lambda x: '–' if x is None else f'{x:.1f}' if abs(x) >= 1 or x == 0 else f'{x:.2f}'
        flag = ''
        if vals[4] is not None and refs:
            lo, hi = min(refs), max(refs)
            flag = 'inside' if lo <= vals[4] <= hi else ('above' if vals[4] > hi else 'below')
        out.append(f'| {lab} | ' + ' | '.join(f(x) for x in vals) + f' | {flag} |')
open(B + 'BENCHMARK_HERS_VS_AI.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
