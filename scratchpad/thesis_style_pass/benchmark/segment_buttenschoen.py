"""Segment Buttenschoen (2025) DPhil thesis text (pdftotext -layout) into results/discussion/conclusion
paragraphs and sentences, chapters 2-6. Line ranges read by hand from the text file's headings."""
import re, json, sys
SRC = '/Users/apunt/Documents/ReferencePapers/Theses/text/Buttenschoen_2025_Deep_learning-based_proteinligand.txt'
OUT = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/segments_buttenschoen.jsonl'
L = open(SRC, encoding='utf-8').read().split('\n')
ref_line = next(i for i, l in enumerate(L, 1) if l.strip() == 'References' and i > 3700)
# (chapter, section_type, section heading, first line, last line) 1-indexed inclusive
RANGES = [
 (2,'results','2.3 Results',1185,1579),(2,'discussion','2.4 Discussion',1580,1692),(2,'conclusion','2.5 Conclusion',1693,1729),
 (3,'results','3.3 Results',1952,2176),(3,'discussion','3.4 Discussion',2177,2273),(3,'conclusion','3.5 Conclusion',2274,2287),
 (4,'results','4.3 Results',2577,2788),(4,'discussion','4.4 Discussion',2789,2905),(4,'conclusion','4.5 Conclusion',2906,2921),
 (5,'results','5.3 Results',3346,3521),(5,'discussion','5.4 Discussion',3522,3583),(5,'conclusion','5.5 Conclusion',3584,3603),
 (6,'conclusion','6 Conclusion',3604,ref_line-1),
]
head_re = re.compile(r'^\s*\d+(\.\d+)+\s+[A-Z]')
cap_re = re.compile(r'^\s*(Figure|Table|Algorithm) [A-Z0-9]+\.\d+:|^\s*\([a-z]\) ')
def paragraphs(a, b):
    paras, cur, sub, in_cap = [], [], None, False
    for raw in L[a-1:b]:
        line = raw.replace('\f', '')
        s = line.strip()
        if not s:
            in_cap = False
            if cur: paras.append((sub, cur)); cur = []
            continue
        if in_cap: continue
        if cap_re.match(line): in_cap = True; continue
        if re.fullmatch(r'\d+', s) or s.startswith('Continued'): continue
        if head_re.match(line) and len(s) < 90 and not s.endswith('.'):
            if cur: paras.append((sub, cur)); cur = []
            sub = s; continue
        if s in ('Chapter 6', 'Conclusion'): continue
        if re.search(r'\S {3,}\S', s): continue           # table rows / layout columns
        if len(s) < 40 and not re.search(r'[.!?:;,]$', s) and not cur: continue  # stray labels
        if line.startswith('    ') and cur:
            paras.append((sub, cur)); cur = []
        cur.append(s)
    if cur: paras.append((sub, cur))
    return paras
def join(lines):
    t = ''
    for s in lines:
        if t.endswith('-') and s[:1].islower() and not re.search(r'(protein|non|DL|re|well|co|so|high|low|state|PB)-$', t):
            t = t[:-1] + s
        else:
            t = (t + ' ' + s) if t else s
    return re.sub(r'\s+', ' ', t).strip()
JUNK = [re.compile(r'^Minimum protein-ligand distance too small.*?Volume overlap with protein '),
        re.compile(r'^r al u c el o m.*?Molecular bonds ')]
allp = []
for ch, st, sec, a, b in RANGES:
    for sub, lines in paragraphs(a, b):
        allp.append(dict(chapter=ch, section_type=st, section=sec, subsection=sub, text=join(lines)))
# merge paragraphs broken by page/figure interruptions: previous does not end a sentence
merged = []
for p in allp:
    if merged and merged[-1]['section'] == p['section'] and not re.search(r'[.!?:]["”’)]?$', merged[-1]['text']) and p['text'][:1].islower():
        merged[-1]['text'] += ' ' + p['text']
    elif merged and merged[-1]['section'] == p['section'] and not re.search(r'[.!?:]["”’)]?$', merged[-1]['text']):
        merged[-1]['text'] += ' ' + p['text']
    else:
        merged.append(p)
for p in merged:
    for j in JUNK: p['text'] = j.sub('', p['text'])
ABBR = r'(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bal\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bcf\.)(?<!\bapprox\.)(?<!\betc\.)(?<!\b[A-Z]\.)(?<!\bp\.)'
out = open(OUT, 'w'); pid = sid = 0
for p in merged:
    if len(p['text']) < 40: continue
    pid += 1
    sents = re.split(ABBR + r'(?<=[.!?])\s+(?=[A-Z(0-9“"])', p['text'])
    for i, s in enumerate(sents):
        s = s.strip()
        if len(s) < 4: continue
        sid += 1
        out.write(json.dumps(dict(sentence_id=f'BU{sid:04d}', paragraph_id=f'BUP{pid:03d}', sent_in_para=i+1,
            chapter=p['chapter'], section_type=p['section_type'], section=p['section'], subsection=p['subsection'], text=s), ensure_ascii=False) + '\n')
print(pid, sid)
