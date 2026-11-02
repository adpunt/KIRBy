"""Segment thesis.tex (author_backup, thesis.tex.before_style_apply_2026-10-10) chapters 3-8 into results/discussion/conclusion paragraphs and sentences."""
import re, json, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else '/Users/apunt/repos/KIRBy/thesis.tex.before_style_apply_2026-10-10'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/segments_author_backup.jsonl'
lines = open(SRC).read().split('\n')
CH = {'chap:robustness':3,'chap:flaws':4,'chap:method':5,'chap:similarity':6,'chap:noise':7,'chap:discussion':8}
chap=None; stype=None; sec=None; sub=None
buf=[]; paras=[]
skip_env = re.compile(r'\\begin\{(figure\*?|table\*?|equation\*?|align\*?|tabular|algorithm|verbatim|lstlisting)\}')
end_env = re.compile(r'\\end\{(figure\*?|table\*?|equation\*?|align\*?|tabular|algorithm|verbatim|lstlisting)\}')
depth=0
def flush():
    global buf
    t=' '.join(buf).strip()
    if t and chap and stype: paras.append(dict(chapter=chap, section_type=stype, section=sec, subsection=sub, raw=t))
    buf=[]
for ln in lines:
    s = re.sub(r'(?<!\\)%.*', '', ln)
    m = re.match(r'\s*\\chapter\*?\{', s)
    if m:
        flush(); chap=None; stype=None
        for k,v in CH.items():
            if k in s: chap=v
        continue
    m = re.match(r'\s*\\section\*?\{([^}]*)\}', s)
    if m:
        flush(); sec=m.group(1); sub=None
        if chap is None: stype=None
        elif sec.startswith('Conclusion'): stype='conclusion'
        elif chap==8: stype='discussion'
        elif sec in ('Introduction','Methods') or sec.startswith('Methods'): stype=None
        elif sec.startswith('Results'): stype='results'
        elif sec in ('Discussion','Limitations'): stype='discussion'
        elif chap in (3,4): stype='results'   # backup has no single Results heading in Ch3/Ch4; every section between Methods/Introduction and Conclusions reports results
        else: stype=None
        continue
    m = re.match(r'\s*\\(sub)+section\*?\{([^}]*)\}', s)
    if m: flush(); sub=m.group(2); continue
    if skip_env.search(s): flush(); depth+=1
    if depth>0:
        if end_env.search(s): depth-=1
        continue
    if re.match(r'\s*\\(begin|end)\{(itemize|enumerate|description)\}', s): flush(); continue
    if re.match(r'\s*\\item\b', s): flush(); s=re.sub(r'^\s*\\item(\[[^\]]*\])?','',s)
    if not s.strip(): flush(); continue
    if re.match(r'\s*\\(label|paragraph|vspace|noindent|clearpage|input|FloatBarrier|centering)\b', s) and len(s.strip())<120:
        mm=re.match(r'\s*\\paragraph\{([^}]*)\}(.*)', s)
        if mm: flush(); s=mm.group(2)
        else: continue
    buf.append(s.strip())
flush()
def clean(t):
    t=re.sub(r'~?\\(cite[pt]?|citep|citet)\{[^}]*\}',' [CIT]',t)
    t=re.sub(r'\\(ref|cref|Cref|autoref|eqref)\{[^}]*\}','[REF]',t)
    t=re.sub(r'\\label\{[^}]*\}','',t)
    t=re.sub(r'\\(textbf|textit|emph|texttt|mathrm|text|textsc|underline)\{([^{}]*)\}',r'\2',t)
    t=re.sub(r'\$([^$]*)\$',r'\1',t)
    t=re.sub(r'\\[a-zA-Z]+\*?','',t)
    t=t.replace('{','').replace('}','').replace('~',' ')
    return re.sub(r'\s+',' ',t).strip()
ABBR = r'(?<!\be\.g)(?<!\bi\.e)(?<!\bet al)(?<!\bvs)(?<!\bFig)(?<!\bEq)(?<!\bapprox)(?<!\bcf)'
out=open(OUT,'w'); pid=0; sid=0; n=0
for p in paras:
    c=clean(p['raw'])
    if len(c)<15: continue
    pid+=1
    sents=re.split(ABBR+r'(?<=[.!?])\s+(?=[A-Z\[(0-9])', c)
    for i,st in enumerate(sents):
        st=st.strip()
        if len(st)<4: continue
        sid+=1
        out.write(json.dumps(dict(sentence_id=f'AB{sid:04d}', paragraph_id=f'ABP{pid:03d}', sent_in_para=i+1, chapter=p['chapter'], section_type=p['section_type'], section=p['section'], subsection=p['subsection'], text=st))+'\n')
print(pid, sid)
