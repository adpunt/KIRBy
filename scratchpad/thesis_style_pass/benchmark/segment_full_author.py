"""Segment a thesis.tex version, Chapters 1-8, ALL sections, into paragraphs and sentences.
Same LaTeX cleaning and sentence split as segment_author_current.py, so sentence texts match
moves_author_current.csv / moves_author_backup.csv. Differences: every section is kept and given a
section type (introduction, background, methods, results, discussion, conclusion); a whole-line
comment does not end a paragraph; each paragraph records its first tex line and the comment lines
directly above it.
usage: python segment_full_author.py current|backup|before_edits"""
import re, json, sys
VER = sys.argv[1]
SRC = {'current': '/Users/apunt/repos/KIRBy/thesis.tex',
       'backup': '/Users/apunt/repos/KIRBy/thesis.tex.before_style_apply_2026-10-10',
       'before_edits': '/Users/apunt/repos/KIRBy/thesis.tex.before_benchmark_edits_2026-10-10'}[VER]
PFX = {'current': 'C', 'backup': 'B', 'before_edits': 'E'}[VER]
OUT = f'/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/full_segments_author_{VER}.jsonl'
lines = open(SRC).read().split('\n')
CH = {'chap:intro': 1, 'chap:background': 2, 'chap:robustness': 3, 'chap:flaws': 4, 'chap:method': 5,
      'chap:similarity': 6, 'chap:noise': 7, 'chap:discussion': 8}

def stype_of(chap, sec):
    if chap is None: return None
    if sec is None:   # text between \chapter and the first \section
        return {1: 'introduction', 2: 'background', 8: 'discussion'}.get(chap, 'introduction')
    if chap == 1: return None if sec.startswith('Collaborations') else 'introduction'
    if chap == 2:
        if sec.startswith('Conclusion'): return 'conclusion'
        if sec.startswith(('Implementation of the noise', 'Evaluation methodology', 'NoiseInject')): return 'methods'
        return 'background'
    if chap == 8:
        if VER in ('current', 'before_edits'): return 'conclusion' if 'Concluding' in sec else 'discussion'
        return 'conclusion' if sec.startswith('Conclusion') else 'discussion'
    if sec.startswith('Introduction'): return 'introduction'
    if sec.startswith('Methods'): return 'methods'
    if sec.startswith('Conclusion'): return 'conclusion'
    if sec.startswith('Results'): return 'results'
    if sec in ('Discussion', 'Limitations'): return 'discussion'
    if VER == 'backup' and chap in (3, 4): return 'results'
    return None

chap = None; sec = None; sub = None
buf = []; start = None; paras = []
skip_env = re.compile(r'\\begin\{(figure\*?|table\*?|equation\*?|align\*?|tabular\*?|longtable\*?|tikzpicture|algorithm|verbatim|lstlisting)\}')
end_env = re.compile(r'\\end\{(figure\*?|table\*?|equation\*?|align\*?|tabular\*?|longtable\*?|tikzpicture|algorithm|verbatim|lstlisting)\}')
depth = 0

def comments_above(i):
    """contiguous whole-line comments directly above 0-based line i"""
    out = []; j = i - 1
    while j >= 0 and lines[j].lstrip().startswith('%'):
        out.insert(0, lines[j].strip()); j -= 1
    return out

def flush():
    global buf, start
    t = ' '.join(buf).strip()
    st = stype_of(chap, sec)
    if t and chap and st:
        paras.append(dict(chapter=chap, section_type=st, section=sec, subsection=sub, raw=t,
                          tex_line=start + 1, comments=comments_above(start)))
    buf = []; start = None

for i, ln in enumerate(lines):
    if ln.lstrip().startswith('%'):
        continue                              # whole-line comment: does not end a paragraph
    s = re.sub(r'(?<!\\)%.*', '', ln)
    if re.match(r'\s*\\chapter\*?\{', s):
        flush(); chap = None; sec = None; sub = None
        for k, v in CH.items():
            if k in s: chap = v
        continue
    m = re.match(r'\s*\\section\*?\{([^}]*)\}', s)
    if m: flush(); sec = m.group(1); sub = None; continue
    m = re.match(r'\s*\\(sub)+section\*?\{([^}]*)\}', s)
    if m: flush(); sub = m.group(2); continue
    if skip_env.search(s): flush(); depth += 1
    if depth > 0:
        if end_env.search(s): depth -= 1
        continue
    if re.match(r'\s*\\(begin|end)\{(itemize|enumerate|description)\}', s): flush(); continue
    if re.match(r'\s*\\item\b', s): flush(); s = re.sub(r'^\s*\\item(\[[^\]]*\])?', '', s)
    if not s.strip(): flush(); continue
    if re.match(r'\s*\\(label|paragraph|vspace|noindent|clearpage|input|FloatBarrier|centering)\b', s) and len(s.strip()) < 120:
        mm = re.match(r'\s*\\paragraph\{([^}]*)\}(.*)', s)
        if mm: flush(); s = mm.group(2)
        else: continue
    if not buf: start = i
    buf.append(s.strip())
flush()

def clean(t):
    t = re.sub(r'~?\\(cite[pt]?|citep|citet)\{[^}]*\}', ' [CIT]', t)
    t = re.sub(r'\\(ref|cref|Cref|autoref|eqref)\{[^}]*\}', '[REF]', t)
    t = re.sub(r'\\label\{[^}]*\}', '', t)
    t = re.sub(r'\\(textbf|textit|emph|texttt|mathrm|text|textsc|underline)\{([^{}]*)\}', r'\2', t)
    t = re.sub(r'\$([^$]*)\$', r'\1', t)
    t = re.sub(r'\\[a-zA-Z]+\*?', '', t)
    t = t.replace('{', '').replace('}', '').replace('~', ' ')
    return re.sub(r'\s+', ' ', t).strip()

ABBR = r'(?<!\be\.g)(?<!\bi\.e)(?<!\bet al)(?<!\bvs)(?<!\bFig)(?<!\bEq)(?<!\bapprox)(?<!\bcf)'
AI = re.compile(r'%\s*TODO:\s*(manual review|review[,:\s-])', re.I)
out = open(OUT, 'w'); pid = 0; sid = 0
for p in paras:
    raw = p['raw'].replace('\\aucnorm', 'AUCnorm')
    c = clean(raw)
    if len(c) < 15: continue
    pid += 1
    ai_lines = [x for x in p['comments'] if AI.search(x)]
    plain_ai = [x for x in ai_lines if not re.search(r'manual review --\s*fixed:', x)]
    marker = 'ai' if plain_ai else ('hers_fixed' if ai_lines else 'hers')
    sents = re.split(ABBR + r'(?<=[.!?])\s+(?=[A-Z\[(0-9])', c)
    k = 0
    for st in sents:
        st = st.strip()
        if len(st) < 4: continue
        sid += 1; k += 1
        out.write(json.dumps(dict(sentence_id=f'F{PFX}{sid:05d}', paragraph_id=f'F{PFX}P{pid:04d}',
                                  sent_in_para=k, chapter=p['chapter'], section_type=p['section_type'], section=p['section'],
                                  subsection=p['subsection'], tex_line=p['tex_line'], marker=marker,
                                  marker_lines=' | '.join(p['comments'])[:400], text=st)) + '\n')
print(VER, 'paragraphs', pid, 'sentences', sid)
