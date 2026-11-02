"""Cuts per axis for one search. usage: python axes.py TIER AXES dataset...   AXES = comma list of budget,importance,allocation,evaluator,repcount
TIER general uses the decided cuts from DECISIONS.md for the axes it does not recompute; TIER general_distance recomputes every listed axis in cascade."""
import sqlite3, re, sys, json

P = '/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/DECISIONS.md'
SMALL = {'abagym_Ang2', 'abagym_EGFR', 'abagym_VEGF', 'abcov_kd', 'ddg_combined', 'flab_thermo_tresanco',
         'ginkgo_acsins', 'ginkgo_colloidal', 'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_titer',
         'humanness_therapeutic', 'tcga_brca_pam50', 'tcga_luad_vs_lusc'}


def decided():
    L = open(P).read().split('\n')
    sec = lambda p: [j for j, x in enumerate(L) if x.startswith(p)][0]
    s2, s3, s4, s5 = sec('## Step 2 (v3'), sec('## Step 3 (v2'), sec('## Step 4: alloc'), sec('## Step 5 (v3')
    pair, rng, ic, ac, ec = {}, {}, {}, {}, {}
    for i, l in enumerate(L):
        m = re.match(r'\| (\S+) \|.*\*\*(\S+\+\S+)\*\* \(2026', l)
        if m: pair[m.group(1)] = m.group(2).split('+')
        if s2 < i < s3:
            m = re.match(r'\| (\S+) \| \d% \|.*\*\*(\d+)-(\d+)\*\* \(2026', l)
            if m: rng[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        for lo, hi, d in ((s3, s4, ic), (s4, s5, ac), (s5, len(L), ec)):
            if lo < i < hi:
                m = re.match(r'\| (\S+) \|.*\*\*cut ([^*]+)\*\*', l)
                if m: d[m.group(1)] = set(m.group(2).split('/'))
    return pair, rng, ic, ac, ec


def top10(v, lower):
    return sum(sorted(v, reverse=not lower)[:10]) / min(10, len(v))


def categorical(rows, idx, lower, band, minrows=10):
    """rows: tuples with score last. idx: position of the category. returns cells, cut."""
    cats = sorted({r[idx] for r in rows}, key=str)
    top = [r[idx] for r in sorted(rows, key=lambda r: r[-1], reverse=not lower)[:30]]
    avg = {c: top10([r[-1] for r in rows if r[idx] == c], lower) for c in cats
           if sum(1 for r in rows if r[idx] == c) >= minrows}
    if len(avg) < 2:
        return {c: (avg.get(c), top.count(c)) for c in cats}, []
    best = (min if lower else max)(avg.values())
    cut = [c for c in avg if abs(avg[c] - best) > band * abs(best) and top.count(c) <= 1]
    return {c: (avg.get(c), top.count(c)) for c in cats}, cut


def run(ds, tier, axes, D):
    pair, rng, ic, ac, ec = D
    pr = pair[ds]
    c = sqlite3.connect(f'slim/{ds}.slim.db')
    metric = c.execute('select metric from results limit 1').fetchone()[0]
    lower = metric == 'rmse'
    band = 0.03 if ds in SMALL else 0.01
    # row = (budget, importance, allocation, evaluator, model, repcount, score)
    rows = [(int(b), im, al, ev, mo, rc or 'all', s) for b, im, al, ev, mo, rc, s in c.execute(
        "select budget,importance_method,allocation_method,internal_evaluator,model,rep_count,score_mean from results "
        "where tier=? and aug_strategy='none' and score_mean is not null and model in (?,?)", (tier, pr[0], pr[1]))]
    out = {'ds': ds, 'metric': metric, 'pair': pr, 'n': len(rows)}
    if not rows:
        return out
    # budget
    if 'budget' in axes:
        BUD = sorted({r[0] for r in rows})
        avg = {b: top10([r[-1] for r in rows if r[0] == b], lower) for b in BUD}
        best = (min if lower else max)(avg.values())
        band_b = [b for b in BUD if abs(avg[b] - best) <= band * abs(best)]
        i, j = BUD.index(band_b[0]), BUD.index(band_b[-1])
        while j - i + 1 < min(3, len(BUD)):
            cand = []
            if i > 0: cand.append((avg[BUD[i - 1]], i - 1, j))
            if j < len(BUD) - 1: cand.append((avg[BUD[j + 1]], i, j + 1))
            _, i, j = (min if lower else max)(cand, key=lambda x: x[0])
        lo, hi = BUD[i], BUD[j]
        out['budget'] = {'avg': avg, 'range': (lo, hi), 'band_only': (band_b[0], band_b[-1])}
    else:
        lo, hi = rng[ds]
    rows = [r for r in rows if lo <= r[0] <= hi]
    # importance
    if 'importance' in axes:
        cells, cut = categorical(rows, 1, lower, band)
        out['importance'] = {'cells': cells, 'cut': cut}
    else:
        cut = ic.get(ds, set())
    rows = [r for r in rows if r[1] not in cut]
    if 'allocation' in axes:
        cells, cut = categorical(rows, 2, lower, band)
        out['allocation'] = {'cells': cells, 'cut': cut}
    else:
        cut = ac.get(ds, set())
    rows = [r for r in rows if r[2] not in cut]
    if 'evaluator' in axes:
        cells, cut = categorical(rows, 3, lower, band)
        out['evaluator'] = {'cells': cells, 'cut': cut}
    else:
        cut = ec.get(ds, set())
    rows = [r for r in rows if not (r[3] in cut and r[3] != r[4])]
    if 'repcount' in axes:
        cells, cut = categorical(rows, 5, lower, band)
        out['repcount'] = {'cells': cells, 'cut': cut}
    return out


if __name__ == '__main__':
    tier, axes = sys.argv[1], sys.argv[2].split(',')
    D = decided()
    res = [run(ds, tier, axes, D) for ds in sys.argv[3:]]
    print(json.dumps(res, default=str))
