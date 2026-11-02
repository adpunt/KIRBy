"""Segment Vost (2025) DPhil thesis text (pdftotext -layout) into results/discussion/conclusion
paragraphs and sentences, chapters 2-6. Line ranges read by hand from the text file's headings.
In this text a new paragraph is marked by an indented first line, not by a blank line;
blank lines and running headers come from page breaks and floats."""
import re, json
SRC = '/Users/apunt/Documents/ReferencePapers/Theses/text/Vost_2025_Machine_learning_for.txt'
OUT = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/segments_vost.jsonl'
L = open(SRC, encoding='utf-8').read().split('\n')
# (chapter, section_type, section heading, first line, last line) 1-indexed inclusive
RANGES = [
 (2,'results','2.5 Results',2414,2895),(2,'conclusion','2.6 Conclusions',2896,2995),
 (3,'results','3.4 Results and discussion',3357,3671),(3,'conclusion','3.5 Conclusions',3672,3740),
 (4,'results','4.4 Results',4115,4197),(4,'conclusion','4.6 Conclusions',4257,4285),
 (5,'results','5.4 Results',4589,4897),(5,'conclusion','5.5 Conclusions',4898,4932),
 (6,'conclusion','6 Conclusions and outlook',4935,5059),
]
head_re = re.compile(r'^\d+(\.\d+)+\s+[A-Z]')
cap_re = re.compile(r'^\s*(Figure|Table|Algorithm) \d+\.\d+:')
def paragraphs(a, b):
    paras, cur, sub, in_cap = [], [], None, False
    for raw in L[a-1:b]:
        if raw.startswith('\f'): continue                 # first line of a page = running header
        line = raw
        s = line.strip()
        if not s:
            in_cap = False; continue
        if in_cap: continue
        if cap_re.match(line):
            in_cap = True; continue
        if re.fullmatch(r'\d+', s): continue
        if head_re.match(line) and len(s) < 90 and not s.endswith('.'):
            if cur: paras.append((sub, cur)); cur = []
            sub = re.sub(r'\s+', ' ', s); continue
        if s in ('Contents', 'Conclusions and outlook', '6', 'Model') or '. . .' in s: continue
        if 'Fragments successfully' in s: continue           # table header row (Table 2.6)
        if sub and sub.endswith('-') and re.fullmatch(r'[a-z]+', s.split()[0]) and not cur:
            sub = sub[:-1] + s.split()[0]; s = ' '.join(s.split()[1:]); line = s
            if not s: continue
        gaps = re.findall(r'\S {3,}\S', s)
        prose_gap = len(gaps) == 1 and len(s) >= 70 and not re.search(r'\d\s*$', s) and not re.match(r'\d+ ', s)
        if gaps and not prose_gap: continue                # running headers, table rows (justified prose kept)
        indent = len(line) - len(line.lstrip(' '))
        if indent > 12 and len(s) < 70: continue          # centred table rows / equations
        if indent >= 3 and cur:
            paras.append((sub, cur)); cur = []
        cur.append(s)
    if cur: paras.append((sub, cur))
    return paras
def join(lines):
    t = ''
    for s in lines:
        if t.endswith('-') and s[:1].islower() and not re.search(r'(non|re|well|co|so|high|low|state|ligand|protein|fragment|structure)-$', t):
            t = t[:-1] + s
        else:
            t = (t + ' ' + s) if t else s
    return re.sub(r'\s+', ' ', t).strip()
ABBR = r'(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bal\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bcf\.)(?<!\bapprox\.)(?<!\betc\.)(?<!\b[A-Z]\.)(?<!\bp\.)'
allp = []
for ch, st, sec, a, b in RANGES:
    for sub, lines in paragraphs(a, b):
        t = join(lines)
        # a paragraph cut by a float or page break: previous one does not end a sentence
        if allp and allp[-1][2] == sec and not re.search(r'[.!?:]["”’)]?$', allp[-1][4]):
            allp[-1][4] += ' ' + t; continue
        allp.append([ch, st, sec, sub, t])
out = open(OUT, 'w'); pid = sid = 0
for ch, st, sec, sub, t in allp:
    if True:
        if len(t) < 40: continue
        pid += 1
        sents = re.split(ABBR + r'(?<=[.!?])\s+(?=[A-Z(0-9“"])', t)
        k = 0
        for s in sents:
            s = s.strip()
            if len(s) < 4: continue
            sid += 1; k += 1
            out.write(json.dumps(dict(sentence_id=f'VO{sid:04d}', paragraph_id=f'VOP{pid:03d}', sent_in_para=k,
                chapter=ch, section_type=st, section=sec, subsection=sub or sec, text=s), ensure_ascii=False) + '\n')
print(pid, sid)
