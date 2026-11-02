"""Aggregate metrics_sentences_after.jsonl: per section type, reference range (min..max over the three theses),
before this pass (author_before_edits) and now (author_current), and whether each gap closed, narrowed, held or widened.
Gap = distance outside the reference range in SDs of reference paragraphs of that section type (as in BENCHMARK_REPORT.md).
Writes after_section_types.csv and after_tables.json."""
import json, math, collections, statistics as st, csv
B = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
REFS = ['buttenschoen', 'vost', 'dablander']
TYPES = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
MOVES = [f'M{i}' for i in range(1, 12)]
METRICS = [('sent_len', 'words', 'n', 1), ('dc', 'dep_clauses', 'n', 1), ('nom', 'nominal', 'words', 100),
           ('num', 'numbers', 'words', 100), ('num3', 'num3', 'n', 100), ('lexd', 'content', 'alpha', 100),
           ('hedge', 'hedges', 'words', 100), ('boost', 'boosters', 'words', 100), ('ai', 'ai_hits', 'words', 100),
           ('passive', 'passive', 'n', 100), ('self', 'selfmention', 'words', 100), ('para_len', 'words', 'paras', 1)]
S = [json.loads(l) for l in open(B + 'metrics_sentences_after.jsonl')]
nan = float('nan')
def isnan(x): return x is None or (isinstance(x, float) and math.isnan(x))
def agg(rows):
    a = collections.Counter(); paras = set()
    for r in rows:
        for k in ('words', 'dep_clauses', 'nominal', 'numbers', 'num3', 'content', 'alpha', 'hedges', 'boosters', 'ai_hits', 'passive', 'selfmention'):
            a[k] += r[k]
        a['n'] += 1; paras.add(r['paragraph_id'])
    a['paras'] = len(paras); out = {}
    for k, num, den, sc in METRICS: out[k] = sc * a[num] / a[den] if a[den] else nan
    coded = [r['move'] for r in rows if r['move']]; mc = collections.Counter(coded)
    out['coded'] = len(coded)
    for m in MOVES: out[m] = 100 * mc[m] / len(coded) if coded else nan
    d = mc['M2'] + mc['M3'] + mc['M4']
    out['m3share'] = 100 * mc['M3'] / d if d else nan
    out['n_sent'] = a['n']; out['n_words'] = a['words']; out['n_paras'] = len(paras)
    return out
P = collections.OrderedDict()
for r in S: P.setdefault((r['corpus'], r['paragraph_id']), []).append(r)
para = {k: dict(agg(v), section_type=v[0]['section_type']) for k, v in P.items()}
KEYS = [m[0] for m in METRICS] + ['m3share'] + MOVES
refdist = {}
for t in TYPES:
    rp = [m for (c, _), m in para.items() if c in REFS and m['section_type'] == t and m['n_words'] >= 20]
    refdist[t] = {}
    for k in KEYS:
        if k in MOVES: v = [m[k] for m in rp if m['coded'] > 0 and not isnan(m[k])]
        else: v = [m[k] for m in rp if not isnan(m[k])]
        refdist[t][k] = st.pstdev(v) if len(v) >= 2 else nan
sec = {c: {t: agg([r for r in S if r['corpus'] == c and r['section_type'] == t]) for t in TYPES} for c in REFS + ['author_current', 'author_before_edits']}
def gap(v, lo, hi, sd):
    if isnan(v) or isnan(lo): return None, 0.0
    if v > hi: return 'high', (v - hi) / sd if sd else nan
    if v < lo: return 'low', (lo - v) / sd if sd else nan
    return None, 0.0
rows = []
for t in TYPES:
    for k in KEYS:
        rv = [sec[c][t][k] for c in REFS if not isnan(sec[c][t][k])]
        if len(rv) < 2: continue
        lo, hi = min(rv), max(rv); sd = refdist[t][k]
        b, n = sec['author_before_edits'][t][k], sec['author_current'][t][k]
        if isnan(b) or isnan(n): continue
        sb, gb = gap(b, lo, hi, sd); sn, gn = gap(n, lo, hi, sd)
        if sb is None and sn is None: verdict = 'inside both times'
        elif sb is not None and sn is None: verdict = 'closed'
        elif sb is None and sn is not None: verdict = 'opened'
        elif sb != sn: verdict = 'crossed the range'
        elif gn < gb - 0.05: verdict = 'narrowed'
        elif gn > gb + 0.05: verdict = 'widened'
        else: verdict = 'held'
        rows.append(dict(section_type=t, measure=k, ref_lo=round(lo, 2), ref_hi=round(hi, 2), before=round(b, 2), now=round(n, 2),
                         side_before=sb or 'inside', gap_before=round(gb, 2), side_now=sn or 'inside', gap_now=round(gn, 2), verdict=verdict,
                         nref=len(rv)))
with open(B + 'after_section_types.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
sizes = {c: {t: (sec[c][t]['n_sent'], sec[c][t]['n_paras'], sec[c][t]['coded'], sec[c][t]['n_words']) for t in TYPES} for c in sec}
# per chapter words in the measured prose
chap = {}
for c in ('author_before_edits', 'author_current'):
    chap[c] = {ch: sum(r['words'] for r in S if r['corpus'] == c and r['chapter'] == ch) for ch in range(1, 9)}
json.dump(dict(rows=rows, sizes=sizes, chap_words=chap), open(B + 'after_tables.json', 'w'), indent=1)
print(collections.Counter(r['verdict'] for r in rows))
