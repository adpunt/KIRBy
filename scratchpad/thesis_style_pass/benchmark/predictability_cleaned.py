"""GPT-2 predictability on identically cleaned text (2026-10-10). Same corpora and section types as extra_measures.py.
One cleaning for all five corpora: bracketed citations, 'et al.', URLs, placeholders ([CIT], [REF], MATH), tokens with
digits, and acronyms (2+ capitals) removed; spaces collapsed; sentences under 8 words (after cleaning) dropped.
Measures per corpus and section type, on up to 400 evenly spaced sentences: GPT-2 small mean surprisal (bits/token),
its SD across sentences, and the share of tokens that are GPT-2's top-1 prediction (GLTR, Gehrmann et al. 2019)."""
import json, re, sys, math, statistics
sys.path.insert(0, '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/9f687a4a-c71f-4415-b6e9-5c2f5f842ef6/scratchpad/pylib')
sys.path.insert(0, '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark')
from compute_metrics import normalise, B
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0)
AI = re.compile(r"%\s*TODO:\s*(manual review|review[,:\s-])", re.I)
def who(ml):
    lines = [x for x in ml.split(" | ") if AI.search(x)]
    plain = [x for x in lines if not re.search(r"--\s*(fixed|moved):", x)]
    return "ai" if plain else "hers"
rows = []
for l in open(B + 'full_segments_author_current.jsonl'):
    r = json.loads(l)
    rows.append(dict(corpus=who(r.get('marker_lines', '')), st=r['section_type'], text=r['text']))  # raw: cleaned below
for l in open(B + 'archive_first_run/metrics_sentences.jsonl'):
    d = json.loads(l)
    if d['corpus'] in ('buttenschoen', 'vost', 'dablander'):
        rows.append(dict(corpus=d['corpus'], st=d['section_type'], text=d['norm_text']))
CITE_PAREN = re.compile(r'\((?:see |e\.g\.,? )?[^()]{0,160}?\b(?:19|20)\d\d[a-z]?(?:;[^()]*)?\)')
CITE_NUM = re.compile(r'\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]')
URL = re.compile(r'(https?://|www\.)\S+', re.I)
ETAL = re.compile(r'\bet al\.?', re.I)
def clean(t):
    t = t.replace('\\%', '%').replace('--', '–').replace('\\', '')
    t = re.sub(r'\[(CIT|REF)\]|MATH\w*', '', t); t = URL.sub(' ', t); t = CITE_PAREN.sub(' ', t); t = CITE_NUM.sub(' ', t); t = ETAL.sub(' ', t)
    keep = []
    for w in t.split():
        core = re.sub(r'^[^\w]+|[^\w]+$', '', w)
        if re.search(r'\d', w) or re.search(r'\[CIT\]|\[REF\]|MATH', w): continue
        if len(re.findall(r'[A-Z]', core)) >= 2: continue      # acronyms, all-caps or mixed case (QSAR, GPs, ChEMBL)
        keep.append(w)
    t = ' '.join(keep)
    t = re.sub(r'\(\s*[,;]*\s*\)|\[\s*[,;]*\s*\]', ' ', t)     # brackets emptied by the removals
    t = re.sub(r'\s+([,.;:)])', r'\1', t); t = re.sub(r'\(\s+', '(', t)
    return re.sub(r'\s+', ' ', t).strip()
for r in rows: r['c'] = clean(r['text'])
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
rows = [r for r in rows if len(WORD.findall(r['c'])) >= 8]
tok = AutoTokenizer.from_pretrained('gpt2'); lm = AutoModelForCausalLM.from_pretrained('gpt2').eval()
@torch.no_grad()
def score(t):
    ids = tok(tok.bos_token + t, return_tensors='pt', truncation=True, max_length=200).input_ids
    logits = lm(ids).logits[0, :-1]; tgt = ids[0, 1:]
    lp = torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0]
    top1 = (logits.argmax(-1) == tgt).sum().item()
    return float(-lp.mean() / math.log(2)), top1, len(tgt)
CORP = ['buttenschoen', 'dablander', 'vost', 'hers', 'ai']
ST = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
res = {}
for c in CORP:
    for s in ST:
        rs = [r for r in rows if r['corpus'] == c and r['st'] == s]
        if len(rs) < 20: continue
        sample = rs if len(rs) <= 400 else [rs[int(i * len(rs) / 400)] for i in range(400)]
        sc = [score(r['c']) for r in sample]
        sp = [x[0] for x in sc]
        res[(c, s)] = dict(surp=statistics.mean(sp), surp_sd=statistics.pstdev(sp),
                           top1=100 * sum(x[1] for x in sc) / sum(x[2] for x in sc), n=len(sample), kept=len(rs))
        print(c, s, len(rs), len(sample), {k: round(v, 3) for k, v in res[(c, s)].items()}, flush=True)
json.dump({f'{c}|{s}': v for (c, s), v in res.items()}, open(B + 'predictability_cleaned.json', 'w'), indent=1)
