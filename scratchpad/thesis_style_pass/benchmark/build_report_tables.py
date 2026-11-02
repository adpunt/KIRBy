"""Aggregate metrics_sentences.jsonl into section-type, chapter and paragraph tables.
Writes: metrics_section_types.csv, metrics_chapters.csv, metrics_paragraphs.csv, worklist.csv, report_parts.json"""
import json, csv, re, math, collections, statistics as st
B = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
TEX = '/Users/apunt/repos/KIRBy/thesis.tex'
AUX = '/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo/build/main.aux'
REFS = ['buttenschoen', 'vost', 'dablander']
AUTH = ['author_current', 'author_backup']
TYPES = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
MOVES = [f'M{i}' for i in range(1, 12)]
MOVE_NAMES = dict(M1='method or background restated', M2='pointer to a figure or table', M3='result given as numbers',
                  M4='result in words', M5='comparison of results', M6='interpretation', M7='comparison with literature',
                  M8='claim of strength', M9='limitation', M10='link or transition', M11='recommendation or implication')
METRICS = [  # key, label, numerator, denominator, scale
 ('sent_len', 'Words per sentence', 'words', 'n', 1),
 ('dc', 'Finite subordinate clauses per sentence', 'dep_clauses', 'n', 1),
 ('nom', 'Nominalizations per 100 words', 'nominal', 'words', 100),
 ('num', 'Numbers per 100 words', 'numbers', 'words', 100),
 ('num3', 'Sentences with 3 or more numbers (%)', 'num3', 'n', 100),
 ('lexd', 'Lexical density (%)', 'content', 'alpha', 100),
 ('hedge', 'Hedges per 100 words', 'hedges', 'words', 100),
 ('boost', 'Boosters per 100 words', 'boosters', 'words', 100),
 ('ai', 'AI-phrasing hits per 100 words', 'ai_hits', 'words', 100),
 ('passive', 'Passive sentences (%)', 'passive', 'n', 100),
 ('self', 'We, our, I per 100 words', 'selfmention', 'words', 100),
 ('para_len', 'Words per paragraph', 'words', 'paras', 1),
]
WORK = ['sent_len', 'dc', 'nom', 'num', 'num3', 'lexd', 'hedge', 'boost', 'ai', 'm3share']
LABEL = {k: l for k, l, *_ in METRICS}
LABEL['m3share'] = 'Share of M3 among M2+M3+M4 sentences (%)'
for m in MOVES: LABEL[m] = f'{m} {MOVE_NAMES[m]} (% of sentences)'

S = [json.loads(l) for l in open(B + 'metrics_sentences.jsonl')]

def agg(rows):
    a = collections.Counter()
    paras = set()
    for r in rows:
        for k in ('words', 'dep_clauses', 'nominal', 'numbers', 'num3', 'content', 'alpha', 'hedges', 'boosters', 'ai_hits', 'passive', 'selfmention'):
            a[k] += r[k]
        a['n'] += 1; paras.add(r['paragraph_id'])
    a['paras'] = len(paras)
    out = {}
    for k, l, num, den, sc in METRICS:
        out[k] = sc * a[num] / a[den] if a[den] else float('nan')
    coded = [r['move'] for r in rows if r['move']]
    mc = collections.Counter(coded)
    out['coded'] = len(coded)
    for m in MOVES: out[m] = 100 * mc[m] / len(coded) if coded else float('nan')
    d = mc['M2'] + mc['M3'] + mc['M4']
    out['m3share'] = 100 * mc['M3'] / d if d else float('nan')
    out['n_sent'] = a['n']; out['n_words'] = a['words']; out['n_paras'] = len(paras)
    return out

nan = float('nan')
def isnan(x): return x is None or (isinstance(x, float) and math.isnan(x))

# ---- paragraphs
P = collections.OrderedDict()
for r in S: P.setdefault((r['corpus'], r['paragraph_id']), []).append(r)
para = {}
for key, rows in P.items():
    m = agg(rows); m.update(corpus=key[0], paragraph_id=key[1], chapter=rows[0]['chapter'], section_type=rows[0]['section_type'],
                            section=rows[0]['section'], subsection=rows[0]['subsection'], tex_line=rows[0]['tex_line'], marker=rows[0]['marker'])
    para[key] = m

# reference paragraph distribution per section type (paragraphs of 20+ words, each paragraph weighted equally)
refdist = {}
for t in TYPES:
    rp = [m for (c, _), m in para.items() if c in REFS and m['section_type'] == t and m['n_words'] >= 20]
    refdist[t] = {}
    for k in [x[0] for x in METRICS] + ['m3share'] + MOVES:
        v = [m[k] for m in rp if not isnan(m[k]) and (k not in MOVES and k != 'm3share' or m['coded'] > 0)]
        if k == 'm3share': v = [m[k] for m in rp if not isnan(m[k])]
        refdist[t][k] = (st.mean(v), st.pstdev(v), len(v)) if len(v) >= 2 else (nan, nan, len(v))

# ---- section types per corpus
sec = {c: {t: agg([r for r in S if r['corpus'] == c and r['section_type'] == t]) for t in TYPES} for c in REFS + AUTH}

def compare(val, refvals, t, k):
    rv = [x for x in refvals if not isnan(x)]
    if isnan(val) or len(rv) < 2: return dict(lo=nan, hi=nan, flag='', gap_z=0.0, nref=len(rv))
    lo, hi = min(rv), max(rv)
    sd = refdist[t][k][1]
    if val > hi: return dict(lo=lo, hi=hi, flag='OUTSIDE high', gap_z=(val - hi) / sd if sd else nan, nref=len(rv))
    if val < lo: return dict(lo=lo, hi=hi, flag='OUTSIDE low', gap_z=(lo - val) / sd if sd else nan, nref=len(rv))
    return dict(lo=lo, hi=hi, flag='', gap_z=0.0, nref=len(rv))

KEYS = [x[0] for x in METRICS] + ['m3share'] + MOVES
rows_sec = []
for t in TYPES:
    for k in KEYS:
        refv = [sec[c][t][k] for c in REFS]
        cur, bak = sec['author_current'][t][k], sec['author_backup'][t][k]
        cc, cb = compare(cur, refv, t, k), compare(bak, refv, t, k)
        rows_sec.append(dict(section_type=t, metric=k, label=LABEL[k], buttenschoen=refv[0], vost=refv[1], dablander=refv[2],
                             ref_min=cc['lo'], ref_max=cc['hi'], n_refs=cc['nref'], current=cur, backup=bak,
                             current_flag=cc['flag'], current_gap_sd=cc['gap_z'], backup_flag=cb['flag'], backup_gap_sd=cb['gap_z'],
                             ref_para_sd=refdist[t][k][1]))
with open(B + 'metrics_section_types.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_sec[0])); w.writeheader(); [w.writerow(r) for r in rows_sec]
sizes = {c: {t: (sec[c][t]['n_sent'], sec[c][t]['n_paras'], sec[c][t]['coded']) for t in TYPES} for c in REFS + AUTH}

# ---- chapters
rows_ch = []
for c in AUTH:
    for (ch, t) in sorted({(r['chapter'], r['section_type']) for r in S if r['corpus'] == c}):
        a = agg([r for r in S if r['corpus'] == c and r['chapter'] == ch and r['section_type'] == t])
        for k in KEYS:
            cc = compare(a[k], [sec[x][t][k] for x in REFS], t, k)
            rows_ch.append(dict(corpus=c, chapter=ch, section_type=t, metric=k, label=LABEL[k], value=a[k], ref_min=cc['lo'], ref_max=cc['hi'],
                                flag=cc['flag'], gap_sd=cc['gap_z'], n_sent=a['n_sent'], coded=a['coded']))
with open(B + 'metrics_chapters.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_ch[0])); w.writeheader(); [w.writerow(r) for r in rows_ch]

# ---- paragraph worklist (author_current, Chapters 1-8)
tex = open(TEX).read().split('\n')
auxlab = dict(re.findall(r'\\newlabel\{((?:tab|fig):[^}]*)\}\{\{([^}]*)\}', open(AUX).read()))
sec_lines = [i for i, l in enumerate(tex) if re.match(r'\s*\\(chapter|section)\*?\{', l)]
def para_raw(line1):
    i = line1 - 1; out = []
    while i < len(tex) and tex[i].strip():
        out.append(tex[i]); i += 1
    return '\n'.join(out)
def nearest_table(line1):
    a = max([s for s in sec_lines if s < line1] or [0]); b = min([s for s in sec_lines if s > line1] or [len(tex)])
    labs = []
    for i in range(a, b):
        for lab in re.findall(r'\\label\{(tab:[^}]*)\}', tex[i]): labs.append((abs(i - line1), lab))
    return sorted(labs)[0][1] if labs else None
def tabname(lab):
    return f'Table {auxlab[lab]}' if lab in auxlab else f'table {lab}'

SR = collections.defaultdict(list)
for r in S:
    if r['corpus'] == 'author_current': SR[r['paragraph_id']].append(r)
seg_text = {json.loads(l)['sentence_id']: json.loads(l)['text'] for l in open(B + 'full_segments_author_current.jsonl')}

def advice(k, z, rows, m, raw, line1):
    hi = z > 0
    if k in ('num', 'num3') and hi and m['section_type'] in ('conclusion', 'introduction', 'discussion'):
        n = sum(r['numbers'] for r in rows)
        return f"cut its {n} numbers and keep the finding in words (reference {m['section_type']}s carry {refdist[m['section_type']]['num'][0]:.1f} per 100 words)"
    if k in ('num', 'num3') and hi:
        n = sum(r['numbers'] for r in rows)
        labs = re.findall(r'\\(?:ref|cref)\{(tab:[^}]*)\}', raw)
        lab = labs[0] if labs else nearest_table(line1)
        where = tabname(lab) if lab else 'a table'
        return f'move most of its {n} numbers to {where} and state the pattern in words'
    if k in ('num', 'num3'):
        return 'give the one number that carries the claim'
    if k == 'm3share' and hi:
        return 'turn number-reporting sentences into one sentence saying what the numbers show'
    if k == 'm3share':
        return 'back the stated result with its key number'
    if k == 'sent_len' and hi:
        long = [r['words'] for r in rows if r['words'] > 30]
        return f'split the {len(long)} sentences over 30 words (longest {max(r["words"] for r in rows)} words)' if long else 'shorten sentences'
    if k == 'sent_len':
        return 'sentences are shorter than the references; join only where two sentences make one point'
    if k == 'dc' and hi:
        return f'break up the {sum(r["dep_clauses"] for r in rows)} subordinate clauses (which/that/because/when) into separate sentences'
    if k == 'dc':
        return 'fine to leave; few subordinate clauses'
    if k == 'nom' and hi:
        ex = collections.Counter(w for r in rows for w in r['nom_list']).most_common(3)
        return 'turn nouns into verbs (' + ', '.join(w for w, _ in ex) + ')'
    if k == 'nom':
        return 'fine to leave; few -tion/-ment nouns'
    if k == 'lexd' and hi:
        return 'unpack noun stacks into phrases with verbs'
    if k == 'lexd':
        return "shorten 'the X of the Y' chains (articles and prepositions)"
    if k == 'hedge' and hi:
        ex = collections.Counter(w for r in rows for w in r['hedge_list']).most_common(4)
        return 'cut hedges (' + ', '.join(w.replace('_', ' ') for w, _ in ex) + ')'
    if k == 'hedge':
        return 'add a hedge where the claim goes beyond the data'
    if k == 'boost' and hi:
        ex = collections.Counter(w for r in rows for w in r['booster_list']).most_common(4)
        return 'cut boosters (' + ', '.join(w.replace('_', ' ') for w, _ in ex) + ')'
    if k == 'boost':
        return 'fine to leave; no boosters'
    if k == 'ai' and hi:
        ex = collections.Counter(w for r in rows for w in r['ai_list']).most_common(4)
        return 'replace AI-guide phrases (' + ', '.join(w for w, _ in ex) + ')'
    return ''

work = []
for pid, rows in SR.items():
    m = para[('author_current', pid)]
    if m['n_words'] < 40: continue
    t = m['section_type']; zs = {}
    for k in WORK:
        mu, sd, n = refdist[t][k]
        if isnan(m[k]) or isnan(mu) or not sd: continue
        zs[k] = (m[k] - mu) / sd
    if not zs: continue
    zc = {k: max(-5, min(5, v)) for k, v in zs.items()}
    dist = math.sqrt(sum(v * v for v in zc.values()) / len(zc))
    top = sorted(zs, key=lambda k: -abs(zs[k]))[:3]
    raw = para_raw(m['tex_line'])
    adv = [advice(k, zs[k], rows, m, raw, m['tex_line']) for k in top]
    adv = list(dict.fromkeys(a for a in adv if a and not a.startswith('fine to leave')))
    first = ' '.join(seg_text[rows[0]['sentence_id']].split()[:8])
    mk = {'ai': 'AI-marked', 'hers': 'hers', 'hers_fixed': 'hers (fix note above)'}[m['marker']]
    work.append(dict(distance=round(dist, 2), chapter=m['chapter'], section_type=t, section=m['section'], subsection=m['subsection'] or '',
                     tex_line=m['tex_line'], first_words=first, marker=mk, sentences=m['n_sent'], words=m['n_words'],
                     **{f'top{i+1}': f'{LABEL[k]}: {m[k]:.1f} vs ref mean {refdist[t][k][0]:.1f} (z {zs[k]:+.1f})' for i, k in enumerate(top)},
                     change='; '.join(adv[:2]) if adv else 'no single measure dominates; read for flow',
                     **{f'v_{k}': round(m[k], 2) if not isnan(m[k]) else '' for k in WORK},
                     **{f'z_{k}': round(zs[k], 2) if k in zs else '' for k in WORK}))
work.sort(key=lambda r: -r['distance'])
for i, r in enumerate(work): r['rank'] = i + 1
cols = ['rank'] + [k for k in work[0] if k != 'rank']
with open(B + 'worklist.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in work[:150]]
with open(B + 'worklist_all_paragraphs.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in work]
json.dump(dict(sizes=sizes, refdist={t: {k: v for k, v in d.items()} for t, d in refdist.items()}, n_work=len(work)),
          open(B + 'report_parts.json', 'w'), indent=1, default=str)
print('paragraphs ranked', len(work))
print(collections.Counter(r['marker'] for r in work[:150]))
print(collections.Counter(r['chapter'] for r in work[:150]))
