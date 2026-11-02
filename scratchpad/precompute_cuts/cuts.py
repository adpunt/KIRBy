import sqlite3, sys, json, glob, os
from collections import defaultdict

D = '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/32e1346f-f5d3-45a6-9401-35839db6097f/scratchpad/slim'
LOWER = {'rmse', 'mae'}
AXES = ['allocation_method', 'importance_method', 'budget', 'rep_count']


def load(ds):
    c = sqlite3.connect(f'{D}/{ds}.slim.db')
    c.row_factory = sqlite3.Row
    return [dict(r) for r in c.execute(
        'select tier,metric,aug_strategy,allocation_method,importance_method,'
        'budget,rep_count,model,internal_evaluator,score_mean,score_std,time_seconds '
        'from results where score_mean is not null')]


def best(rows, lower):
    return (min if lower else max)(rows, key=lambda r: r['score_mean']) if rows else None


def gap(a, b, lower):  # how much worse b is than a (positive = worse)
    return (b - a) if lower else (a - b)


def analyse(ds, out):
    rows = load(ds)
    metric = rows[0]['metric']
    lower = metric in LOWER
    P = lambda *a: print(*a, file=out)
    P(f'\n=== {ds}  metric={metric} ({"lower" if lower else "higher"} is better)')
    # augmentation check
    for tier in ('general', 'general_distance'):
        t = [r for r in rows if r['tier'] == tier]
        s = []
        for a in ('none', 'capped', 'uncapped'):
            b = best([r for r in t if r['aug_strategy'] == a], lower)
            s.append(f'{a}={b["score_mean"]:.4f}' if b else f'{a}=absent')
        P(f'  augmentation best, tier {tier}: ' + '  '.join(s))
    res = {'ds': ds, 'metric': metric}
    for tier in ('general', 'general_distance'):
        t = [r for r in rows if r['tier'] == tier and r['aug_strategy'] == 'none']
        if not t:
            continue
        bm = {}
        for m in sorted({r['model'] for r in t}):
            bm[m] = best([r for r in t if r['model'] == m], lower)
        rank = sorted(bm, key=lambda m: bm[m]['score_mean'], reverse=not lower)
        P(f'  [{tier}] models by best score: ' + ', '.join(
            f'{m} {bm[m]["score_mean"]:.4f}(+-{bm[m]["score_std"] or 0:.4f})' for m in rank[:5]))
        top2 = rank[:2]
        sub = [r for r in t if r['model'] in top2 and r['internal_evaluator'] in top2]
        ov = best(sub, lower)
        if not ov:
            continue
        spread = ov['score_std'] or 0.0
        P(f'  [{tier}] top-2 {top2}; best config {ov["score_mean"]:.4f}, fold spread {spread:.4f}; '
          f'best = {ov["allocation_method"]}/{ov["importance_method"]}/budget {ov["budget"]}/'
          f'reps {ov["rep_count"] or "all"}/{ov["model"]} scored by {ov["internal_evaluator"]}')
        cuts = defaultdict(list)
        for ax in AXES:
            line = []
            for v in sorted({r[ax] for r in sub}, key=lambda x: (len(str(x)), str(x))):
                b = best([r for r in sub if r[ax] == v], lower)
                g = gap(ov['score_mean'], b['score_mean'], lower)
                cut = g > spread
                if cut:
                    cuts[ax].append(v)
                line.append(f'{v or "all"}:{g:+.4f}{"*" if cut else ""}')
            P(f'    {ax} (gap of its best config to overall best; * = worse than fold spread): ' + '  '.join(line))
        keep = [r for r in sub if all(r[ax] not in cuts[ax] for ax in AXES)]
        kb = best(keep, lower)
        P(f'    all * cuts applied together: {len(keep)} of {len(sub)} configs left, best {kb["score_mean"]:.4f} vs {ov["score_mean"]:.4f}'
          f'; cut {dict(cuts)}')
        # slow models
        P('    seconds per config, median, by model: ' + ', '.join(
            f'{m} {sorted(r["time_seconds"] or 0 for r in t if r["model"]==m)[len([1 for r in t if r["model"]==m])//2]:.1f}'
            for m in rank[:5]))
        res[tier] = {'top2': top2, 'cuts': {k: v for k, v in cuts.items()},
                     'kept_best': kb['score_mean'], 'best': ov['score_mean'], 'spread': spread,
                     'models': [(m, bm[m]['score_mean']) for m in rank[:3]]}
    return res


if __name__ == '__main__':
    out = sys.stdout
    all_res = []
    for ds in sys.argv[1:]:
        all_res.append(analyse(ds, out))
    json.dump(all_res, open(os.environ.get('OUTJSON', '/dev/null'), 'w'), default=str)
