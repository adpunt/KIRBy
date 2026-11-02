"""Segment the three reference DPhil theses (pdftotext -layout) into paragraphs and sentences over
ALL chapters 1 to the last, each section given a type: introduction, background, methods, results,
discussion, conclusion. Line ranges read by hand from each text's headings.
Paragraph rules: Buttenschoen as segment_buttenschoen.py (blank line or 4-space indent starts one);
Vost and Dablander as segment_vost.py (indented first line starts one). Sentence split identical to
those scripts, so sentence texts match moves_buttenschoen.csv / moves_vost.csv.
Extra cleaning (all three): paragraphs that are only a publication note ('This chapter is based on',
doi links) dropped; lines with < 50% letters (equations, table rows) dropped."""
import re, json
T = '/Users/apunt/Documents/ReferencePapers/Theses/text/'
OUTD = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
SRC = {'buttenschoen': T + 'Buttenschoen_2025_Deep_learning-based_proteinligand.txt',
       'vost': T + 'Vost_2025_Machine_learning_for.txt',
       'dablander': T + 'DPhil_Thesis_Markus_Dablander--Submitted.txt'}
RANGES = {
 'buttenschoen': [
  (1,'introduction',272,284),(1,'background',285,623),(1,'introduction',624,709),
  (2,'introduction',710,853),(2,'methods',854,1184),(2,'results',1185,1579),(2,'discussion',1580,1692),(2,'conclusion',1693,1729),
  (3,'introduction',1730,1854),(3,'methods',1855,1951),(3,'results',1952,2176),(3,'discussion',2177,2273),(3,'conclusion',2274,2287),
  (4,'introduction',2288,2364),(4,'methods',2365,2576),(4,'results',2577,2788),(4,'discussion',2789,2905),(4,'conclusion',2906,2921),
  (5,'introduction',2922,3009),(5,'methods',3010,3345),(5,'results',3346,3521),(5,'discussion',3522,3583),(5,'conclusion',3584,3603),
  (6,'conclusion',3604,3772)],
 'vost': [
  (1,'background',636,1776),(1,'introduction',1777,1815),
  (2,'introduction',1864,1980),(2,'methods',1981,2413),(2,'results',2414,2895),(2,'conclusion',2896,2995),
  (3,'introduction',3028,3114),(3,'methods',3115,3356),(3,'results',3357,3671),(3,'conclusion',3672,3740),
  (4,'introduction',3765,3865),(4,'methods',3866,4114),(4,'results',4115,4197),(4,'background',4198,4256),(4,'conclusion',4257,4285),
  (5,'introduction',4303,4405),(5,'methods',4406,4588),(5,'results',4589,4897),(5,'conclusion',4898,4932),
  (6,'conclusion',4935,5059)],
 'dablander': [
  (1,'introduction',283,510),(2,'background',511,2005),
  (3,'introduction',2006,2237),(3,'methods',2238,2692),(3,'results',2693,2879),(3,'conclusion',2880,2928),
  (4,'introduction',2929,3028),(4,'methods',3029,3881),(4,'results',3882,4164),(4,'conclusion',4165,4219),
  (5,'introduction',4220,4408),(5,'methods',4409,5140),(5,'results',5141,5309),(5,'conclusion',5310,5371),
  (6,'discussion',5372,5620),(7,'conclusion',5621,5714)],
}
head_re = re.compile(r'^\s*\d+(\.\d+)+\s+[A-Z]')
def letters_ok(s):
    s = re.sub(r'\[[\d,\s–-]+\]', '', s)
    if not s.strip(): return True
    return sum(c.isalpha() for c in s) >= 0.5 * len(s.replace(' ', ''))

def paras_butt(L, a, b):
    cap_re = re.compile(r'^\s*(Figure|Table|Algorithm) [A-Z0-9]+\.\d+:|^\s*\([a-z]\) ')
    paras, cur, sub, in_cap = [], [], None, False
    for raw in L[a-1:b]:
        line = raw.replace('\f', ''); s = line.strip()
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
        if re.fullmatch(r'Chapter \d+', s): continue
        if re.search(r'\S {3,}\S', s): continue
        if len(s) < 40 and not re.search(r'[.!?:;,]$', s) and not cur: continue
        if not letters_ok(s): continue
        if line.startswith('    ') and cur:
            paras.append((sub, cur)); cur = []
        cur.append(s)
    if cur: paras.append((sub, cur))
    return paras

def paras_indent(L, a, b, ff_is_header=True):
    cap_re = re.compile(r'^\s*(Figure|Table|Algorithm) \d+\.\d+:')
    paras, cur, sub, in_cap = [], [], None, False
    for raw in L[a-1:b]:
        if raw.startswith('\f'):
            if ff_is_header: continue
            raw = raw[1:]
        line = raw; s = line.strip()
        if not s: in_cap = False; continue
        if in_cap: continue
        if cap_re.match(line): in_cap = True; continue
        if re.fullmatch(r'\d+', s): continue
        if head_re.match(line) and len(s) < 90 and not s.endswith('.'):
            if cur: paras.append((sub, cur)); cur = []
            sub = re.sub(r'\s+', ' ', s); continue
        if s in ('Contents', 'Conclusions and outlook', '6', 'Model') or '. . .' in s: continue
        if re.fullmatch(r'Chapter \d+\.?', s): continue
        if 'Fragments successfully' in s: continue
        gaps = re.findall(r'\S {3,}\S', s)
        prose_gap = len(gaps) == 1 and len(s) >= 70 and not re.search(r'\d\s*$', s) and not re.match(r'\d+ ', s)
        if gaps and not prose_gap: continue
        indent = len(line) - len(line.lstrip(' '))
        if indent > 12 and len(s) < 70: continue
        if len(s) < 40 and not re.search(r'[.!?:;,]$', s) and not cur: continue
        if not letters_ok(s): continue
        if indent >= 3 and cur:
            paras.append((sub, cur)); cur = []
        cur.append(s)
    if cur: paras.append((sub, cur))
    return paras

def join(lines):
    t = ''
    for s in lines:
        if t.endswith('-') and s[:1].islower() and not re.search(r'(protein|non|DL|re|well|co|so|high|low|state|PB|ligand|fragment|structure|graph|task|half)-$', t):
            t = t[:-1] + s
        else:
            t = (t + ' ' + s) if t else s
    return re.sub(r'\s+', ' ', t).strip()

ABBR = r'(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bal\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bcf\.)(?<!\bapprox\.)(?<!\betc\.)(?<!\b[A-Z]\.)(?<!\bp\.)'
NOTE = re.compile(r'doi\.org|^(This|The) chapter (is based|contains work)|^\*?Authors contributed|is based on (my|the following|our) (paper|publication)', re.I)
for name, src in SRC.items():
    L = open(src, encoding='utf-8').read().split('\n')
    fn = paras_butt if name == 'buttenschoen' else paras_indent
    allp = []
    for ch, st, a, b in RANGES[name]:
        for sub, lines in (fn(L, a, b) if name != 'dablander' else fn(L, a, b, ff_is_header=False)):
            t = join(lines)
            if allp and allp[-1]['key'] == (ch, st, a) and not re.search(r'[.!?:]["”’)]?$', allp[-1]['text']):
                allp[-1]['text'] += ' ' + t; continue
            allp.append(dict(key=(ch, st, a), chapter=ch, section_type=st, subsection=sub, text=t, line=a))
    pfx = name[:2].upper()
    out = open(OUTD + f'full_segments_{name}.jsonl', 'w'); pid = sid = 0
    for p in allp:
        if len(p['text']) < 40 or NOTE.search(p['text']): continue
        pid += 1; k = 0
        for s in re.split(ABBR + r'(?<=[.!?])\s+(?=[A-Z(0-9“"])', p['text']):
            s = s.strip()
            if len(s) < 4: continue
            sid += 1; k += 1
            out.write(json.dumps(dict(sentence_id=f'F{pfx}{sid:05d}', paragraph_id=f'F{pfx}P{pid:04d}', sent_in_para=k,
                chapter=p['chapter'], section_type=p['section_type'], subsection=p['subsection'], text=s), ensure_ascii=False) + '\n')
    print(name, 'paragraphs', pid, 'sentences', sid)
