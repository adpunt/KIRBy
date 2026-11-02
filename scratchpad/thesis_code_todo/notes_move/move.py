"""Move note bullets out of sections that have real text. Dry run unless --write."""
import re, sys
TEX = '/Users/apunt/repos/KIRBy/thesis_rework/stage1_takeaways_bullets/thesis.tex'
OUT = '/Users/apunt/repos/KIRBy/thesis_rework/stage1_takeaways_bullets/thesis_notes_moved.md'
L = open(TEX).read().split('\n')
HEAD = re.compile(r'^\\(chapter|section|subsection|subsubsection)\*?\{(.*?)\}')
KEEP = re.compile(r'CODE INPUT|TODO|OLD TEXT|SPLIT|====|CHECK:')
BUL = re.compile(r'^%\s*(-|Paragraph \d|Takeaway|Figure:|Table:|Result needed|Change:|Source needed)')
# spans: from each heading to the next heading
heads = [i for i, l in enumerate(L) if HEAD.match(l)]
bounds = list(zip(heads, heads[1:] + [len(L)]))
lead = (0, heads[0])
def is_text(l):
    s = l.strip()
    return bool(s) and not s.startswith('%') and not HEAD.match(s) and not s.startswith('\\label') \
        and not s.startswith('\\todo') and not s.startswith('\\input') and not s.startswith('\\appendix') \
        and not s.startswith('\\begin{document') and not s.startswith('\\end{document')
incode = set(); _c = False
for _i, _l in enumerate(L):
    if 'CODE INPUT: new, needs review' in _l: _c = True
    if _c: incode.add(_i)
    if 'END CODE INPUT' in _l: _c = False
move = set(); report = []; path = {}
cur = {}
for a, b in bounds:
    m = HEAD.match(L[a]); lvl = ['chapter','section','subsection','subsubsection'].index(m.group(1))
    cur = {k: v for k, v in cur.items() if k < lvl}; cur[lvl] = m.group(2)
    title = ' > '.join(cur[k] for k in sorted(cur))
    if not any(is_text(l) for l in L[a+1:b]):
        continue
    mv = []
    i = a + 1
    in_old = False
    while i < b:
        l = L[i]
        if l.startswith('% ---- NOTES') or l.startswith('% ---- TAKEAWAYS'):
            end = 'END NOTES' if 'NOTES' in l else 'END TAKEAWAYS'
            j = i
            while end not in L[j]: j += 1
            blk = list(range(i, j + 1))
            keep_blk = [k for k in blk if L[k].startswith('%') and not KEEP.search(L[k])]
            # a blank line moves only when it sits between two comment lines of the block
            keep_blk += [k for k in blk if not L[k].strip() and k-1 in blk and k+1 in blk
                         and L[k-1].startswith('%') and L[k+1].startswith('%')]
            mv += sorted(keep_blk); i = j + 1; continue
        if '[OLD TEXT' in l: in_old = True
        if not l.startswith('%'): in_old = False
        if l.startswith('%') and not in_old and i not in incode and not KEEP.search(l):
            # a run of bullet-shaped comment lines
            j = i; run = []
            while j < b and L[j].startswith('%') and not KEEP.search(L[j]) and (BUL.match(L[j]) or L[j].strip() == '%'):
                run.append(j); j += 1
            if run and any(BUL.match(L[k]) for k in run):
                mv += run; i = j; continue
        i += 1
    if mv:
        move.update(mv); report.append((title, a, len(mv))); path[a] = (title, mv)
for t, a, n in report: print(f'{a+1:5d} {n:4d}  {t[:110]}')
print('total lines moved', len(move))
if '--write' in sys.argv:
    md = ['# Notes moved out of thesis.tex', '', 'Moved 2026-09-29 from every section or subsection that has written text. '
          'Sections with no text yet keep their notes in thesis.tex. Lines are verbatim and in their original order; '
          'a blank line inside a block marks where text sat between two runs of notes.', '']
    for a in sorted(path):
        t, mv = path[a]
        md += [f'## {t}', '', '```latex']
        prev = None
        for k in mv:
            if prev is not None and k != prev + 1: md.append('')
            md.append(L[k]); prev = k
        md += ['```', '']
    open(OUT, 'w').write('\n'.join(md))
    new = []
    for i, l in enumerate(L):
        if i in move:
            if i in path: pass
            continue
        new.append(l)
        if i in path:
            new.append('% [notes moved to thesis_notes_moved.md]')
    open(TEX, 'w').write('\n'.join(new))
