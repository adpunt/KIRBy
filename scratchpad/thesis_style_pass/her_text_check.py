"""List changed paragraphs in her text (no 'manual review' marker in the backup) that lack an 'edited your text' TODO."""
import sys, difflib, re
old_p, new_p = sys.argv[1], sys.argv[2]
def paras(path):
    lines = open(path).read().split('\n')
    out, cur_c, cur_t = [], [], []
    for l in lines + ['']:
        if l.strip() == '':
            if cur_t or cur_c: out.append(('\n'.join(cur_c), '\n'.join(cur_t)))
            cur_c, cur_t = [], []
        elif l.lstrip().startswith('%') and not cur_t: cur_c.append(l)
        else: cur_t.append(l)
    return out
AI = re.compile(r'%\s*TODO:\s*(manual review|review[,:\s-])', re.I)
old = paras(old_p); new = paras(new_p)
old_texts = {t: c for c, t in old}
old_ai = {t for c, t in old if AI.search(c)}
sm = difflib.SequenceMatcher(None, [t for _, t in old], [t for _, t in new], autojunk=False)
n = 0
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal': continue
    olds = old[i1:i2]
    for c, t in new[j1:j2]:
        if not t.strip() or t in old_texts: continue
        was_ai = any(AI.search(oc) for oc, ot in olds) or AI.search(c) and 'edited your text' not in c and any(AI.search(oc) for oc,_ in olds)
        if any(AI.search(oc) for oc, _ in olds): continue          # replaced an AI paragraph
        if re.search(r'edited your text|title was', c, re.I): continue
        if AI.search(c) and not olds: continue                     # pure new AI paragraph
        # her paragraph changed with no note: show a short diff
        best = difflib.get_close_matches(t, [ot for _, ot in olds], n=1, cutoff=0.3)
        n += 1
        print(f'--- #{n}')
        if best:
            d = [x for x in difflib.ndiff(best[0].split(), t.split()) if x[:1] in '+-']
            print('changed words:', ' '.join(d)[:600])
        else:
            print('NEW/UNMATCHED (marker:', c[-120:].replace('\n',' | ') or 'none', ')')
            print(t[:300])
print('total flagged', n)
