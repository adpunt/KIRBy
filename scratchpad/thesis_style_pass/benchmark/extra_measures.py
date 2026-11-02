"""Five extra measures (2026-10-10), same corpora and section types as split_hers_ai.py:
lexical diversity (MTLD, McCarthy & Jarvis 2010; MATTR window 50, Covington & McFall 2010),
sentence-length variability (SD, coefficient of variation, mean change between consecutive sentences),
predictability (GPT-2 small mean token surprisal in bits, and its SD across sentences),
connective scaffolding (connectives per 100 words, distinct connectives / connectives used, share of sentences opening
with a formulaic connector, three-item lists per 100 sentences, pronouns and demonstratives per 100 words)."""
import json, re, sys, math, statistics, collections
sys.path.insert(0, '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/9f687a4a-c71f-4415-b6e9-5c2f5f842ef6/scratchpad/pylib')
sys.path.insert(0, '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark')
from compute_metrics import normalise, B
AI = re.compile(r"%\s*TODO:\s*(manual review|review[,:\s-])", re.I)
def who(ml):
    lines = [x for x in ml.split(" | ") if AI.search(x)]
    plain = [x for x in lines if not re.search(r"--\s*(fixed|moved):", x)]
    return "ai" if plain else "hers"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('gpt2'); lm = AutoModelForCausalLM.from_pretrained('gpt2').eval()
rows = []
for l in open(B + 'full_segments_author_current.jsonl'):
    r = json.loads(l)
    rows.append(dict(corpus=who(r.get('marker_lines', '')), pid=r['paragraph_id'], st=r['section_type'], text=normalise(r['text'], False)))
for l in open(B + 'archive_first_run/metrics_sentences.jsonl'):
    d = json.loads(l)
    if d['corpus'] in ('buttenschoen', 'vost', 'dablander'):
        rows.append(dict(corpus=d['corpus'], pid=d['paragraph_id'], st=d['section_type'], text=d['norm_text']))
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
CONN = ['however', 'moreover', 'furthermore', 'additionally', 'in addition', 'therefore', 'thus', 'hence', 'consequently',
        'as a result', 'in contrast', 'by contrast', 'on the other hand', 'whereas', 'while', 'although', 'though', 'because',
        'since', 'so', 'then', 'also', 'instead', 'rather', 'similarly', 'likewise', 'nevertheless', 'nonetheless', 'yet',
        'still', 'overall', 'in summary', 'in short', 'taken together', 'together', 'notably', 'importantly', 'indeed',
        'first', 'second', 'third', 'finally', 'next', 'for example', 'for instance', 'in particular', 'that is', 'namely',
        'specifically', 'meanwhile', 'otherwise', 'accordingly', 'in turn', 'as such', 'crucially', 'interestingly']
FORMULAIC = ['however,', 'moreover,', 'furthermore,', 'additionally,', 'in addition,', 'overall,', 'notably,', 'importantly,',
             'taken together,', 'together,', 'in summary,', 'crucially,', 'interestingly,', 'in contrast,', 'consequently,', 'as a result,']
PRON = {'it', 'its', 'they', 'them', 'their', 'this', 'these', 'that', 'those', 'which', 'such', 'he', 'she', 'his', 'her', 'we', 'our', 'us', 'i', 'my'}
TRIPLE = re.compile(r"\b[\w-]+(?: [\w-]+){0,3}, [\w-]+(?: [\w-]+){0,3},? (?:and|or) [\w-]+")
def mtld_dir(toks, th=0.72):
    f, types, n = 0.0, set(), 0
    for t in toks:
        n += 1; types.add(t)
        if len(types) / n <= th: f += 1; types, n = set(), 0
    if n: f += (1 - len(types) / n) / (1 - th)
    return len(toks) / f if f else float('nan')
def mtld(toks): return (mtld_dir(toks) + mtld_dir(toks[::-1])) / 2
def mattr(toks, w=50):
    if len(toks) < w: return float('nan')
    return statistics.mean(len(set(toks[i:i + w])) / w for i in range(0, len(toks) - w + 1, 5))
@torch.no_grad()
def surprisal(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        batch = [tok.bos_token + t for t in texts[i:i + bs]]
        enc = tok(batch, return_tensors='pt', padding=False, truncation=True, max_length=200) if len(batch) == 1 else None
        for t in batch:
            ids = tok(t, return_tensors='pt', truncation=True, max_length=200).input_ids
            if ids.shape[1] < 3: out.append(float('nan')); continue
            logits = lm(ids).logits[0, :-1]
            lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
            out.append(float(-lp.mean() / math.log(2)))
    return out
CORP = ['buttenschoen', 'dablander', 'vost', 'hers', 'ai']
ST = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
res = {}
for c in CORP:
    for s in ST:
        rs = [r for r in rows if r['corpus'] == c and r['st'] == s]
        if len(rs) < 20: continue
        toks = [w.lower() for r in rs for w in WORD.findall(r['text'])]
        lens = [len(WORD.findall(r['text'])) for r in rs]
        diffs = [abs(lens[i] - lens[i - 1]) for i in range(1, len(rs)) if rs[i]['pid'] == rs[i - 1]['pid']]
        low = [r['text'].lower() for r in rs]
        conn = []
        for t in low:
            for cnt in CONN:
                conn += [cnt] * len(re.findall(r'\b' + re.escape(cnt) + r'\b', t))
        sample = rs if len(rs) <= 400 else [rs[i] for i in range(0, len(rs), max(1, len(rs) // 400))]
        sp = [x for x in surprisal([r['text'] for r in sample]) if not math.isnan(x)]
        nw = len(toks) or 1
        res[(c, s)] = dict(
            mtld=mtld(toks), mattr=100 * mattr(toks), sd=statistics.pstdev(lens), cv=statistics.pstdev(lens) / statistics.mean(lens),
            dlen=statistics.mean(diffs) if diffs else float('nan'), surp=statistics.mean(sp), surp_sd=statistics.pstdev(sp),
            conn=100 * len(conn) / nw, conn_div=len(set(conn)) / len(conn) if conn else float('nan'),
            formulaic=100 * sum(any(t.startswith(f) for f in FORMULAIC) for t in low) / len(low),
            triple=100 * sum(len(TRIPLE.findall(t)) for t in low) / len(low),
            pron=100 * sum(w in PRON for w in toks) / nw)
        print(c, s, 'done', flush=True)
LAB = [('mtld', 'Lexical diversity, MTLD (higher = more varied)'), ('mattr', 'Lexical diversity, MATTR-50 %'),
       ('sd', 'Sentence length SD (words)'), ('cv', 'Sentence length coefficient of variation'),
       ('dlen', 'Mean change in length between consecutive sentences'),
       ('surp', 'GPT-2 surprisal, bits per token (lower = more predictable)'), ('surp_sd', 'Surprisal SD across sentences (burstiness)'),
       ('conn', 'Connectives per 100 words'), ('conn_div', 'Distinct connectives / connectives used'),
       ('formulaic', '% sentences opening with a formulaic connector'), ('triple', 'Three-item lists per 100 sentences'),
       ('pron', 'Pronouns and demonstratives per 100 words')]
out = ['\n## Added measures: diversity, variability, predictability, connectives\n',
       'Same corpora as above. Surprisal from GPT-2 small on up to 400 sentences per cell (evenly spaced).\n']
for s in ST:
    if not any((c, s) in res for c in CORP): continue
    out += [f'\n### {s.capitalize()}\n', '| Measure | Buttenschoen | Dablander | Vost | Yours | AI-written | AI vs references |', '|---|---|---|---|---|---|---|']
    for k, lab in LAB:
        vals = [res[(c, s)][k] if (c, s) in res else None for c in CORP]
        refs = [v for v in vals[:3] if v is not None and not math.isnan(v)]
        f = lambda x: '–' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.2f}' if abs(x) < 10 else f'{x:.1f}')
        a = vals[4]; flag = ''
        if a is not None and refs and not math.isnan(a):
            flag = 'inside' if min(refs) <= a <= max(refs) else ('above' if a > max(refs) else 'below')
        out.append(f'| {lab} | ' + ' | '.join(f(v) for v in vals) + f' | {flag} |')
open(B + 'BENCHMARK_EXTRA.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
