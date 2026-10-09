"""Parse the rebuilt MEETING_METHODS.md (scratch copy) into rows.json. Based on scratchpad/thesis_ch5_8/ch7m/parse.py."""
import re, json, sys, collections
P = sys.argv[1]
BAD = re.compile(r'tcga|annitia|nhanes|pam50', re.I)
SECEXP = {'DivideMix': 'n09', 'Fixing labels clipped': 'n30', 'Label smoothing toward': 'n24',
          'kNN label replacement': 'n07', 'Detect and remove': 'n18'}
NOHDR = {'n09': 'DivideMix', 'n24': 'label_smooth', 'n07': 'knn_smooth', 'n18': 'remove'}
sec = exp = meth = hdr = None; rows = []; dropped = collections.Counter()
def pr(s):
    m = re.match(r'(-?[\d.]+) → (-?[\d.]+) ([✓✗])', s)
    return (float(m.group(1)), float(m.group(2)), m.group(3) == '✓') if m else None
for line in open(P):
    m = re.match(r'## (\d+)\. (.*)', line)
    if m:
        sec = m.group(2).strip(); meth = None; hdr = None
        e = re.search(r'\((n\d+)', sec)
        exp = e.group(1) if e else next((v for k, v in SECEXP.items() if sec.startswith(k)), None)
        if sec.startswith('DivideMix'): exp = 'n09'
        continue
    m = re.match(r'\*\*(.+?)\*\* — group', line)
    if m: meth = m.group(1); continue
    if line.startswith('| dataset'):
        hdr = [c.strip() for c in line.strip().strip('|').split('|')]; continue
    if line.startswith('|---') or not line.startswith('| ') or hdr is None or sec is None: continue
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    if len(c) != len(hdr): dropped['bad width'] += 1; continue
    d = dict(zip(hdr, c)); raw = d['dataset']
    if BAD.search(raw): dropped['patient'] += 1; continue
    ds = raw.lstrip('*'); run = ''
    mm = re.match(r'(\S+) \((.*)\)', ds)
    if mm: ds, run = mm.group(1), mm.group(2)
    ctl = [k for k in hdr if '(single / hybrid)' in k]
    rows.append(dict(sec=sec, exp=exp, method=meth or NOHDR.get(exp), ds=ds, run=run, noise=d['noise'],
                     model=d['model'], metric=d['metric'].split()[0], low='lower' in d['metric'],
                     base=pr(d['single representation']), hyb=pr(d['hybrid']), works=d['does it work?'],
                     hb=d['hybrid better than base?'], ctl=d[ctl[0]] if ctl else None))
json.dump(rows, open(sys.argv[2], 'w'), indent=0)
print(len(rows), dict(dropped))
print(collections.Counter((r['exp'], r['method']) for r in rows))
