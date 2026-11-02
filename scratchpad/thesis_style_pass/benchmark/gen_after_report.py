"""Write BENCHMARK_AFTER.md from after_tables.json."""
import json
B = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
d = json.load(open(B + 'after_tables.json'))
LAB = dict(sent_len='Words per sentence', dc='Subordinate clauses per sentence', nom='Nominalizations per 100 words',
           num='Numbers per 100 words', num3='% sentences with 3+ numbers', lexd='Lexical density %', hedge='Hedges per 100 words',
           boost='Boosters per 100 words', ai='AI-guide phrases per 100 words', passive='% passive sentences',
           self='we/our/I per 100 words', para_len='Words per paragraph', m3share='% M3 among M2+M3+M4',
           M1='M1 method restated %', M2='M2 pointer to figure/table %', M3='M3 result as numbers %', M4='M4 result in words %',
           M5='M5 comparison of results %', M6='M6 interpretation %', M7='M7 comparison with literature %', M8='M8 claim of strength %',
           M9='M9 limitation %', M10='M10 link %', M11='M11 recommendation %')
TYPES = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
S = d['sizes']
def f(x): return f'{x:.2f}' if abs(x) < 10 else f'{x:.1f}'
def verdict(r):
    v = r['verdict']
    if v == 'inside both times': return 'inside the range both times'
    if v == 'closed': return f"CLOSED (was {r['gap_before']:.2f} SD {'above' if r['side_before']=='high' else 'below'})"
    if v == 'crossed the range': return f"crossed: was {r['gap_before']:.2f} SD above, now {r['gap_now']:.2f} SD below"
    side = 'above' if r['side_now'] == 'high' else 'below'
    if r['gap_now'] != r['gap_now']: return f'{v} ({side}; references all 0, gap in SD undefined)'
    return f"{v} ({side}, {r['gap_before']:.2f} -> {r['gap_now']:.2f} SD)"
o = []
o.append('# Benchmark after the benchmark edits\n')
o.append('Written 2026-10-10. This re-measures the thesis after the edit pass that followed `BENCHMARK_REPORT.md`. No reference thesis was re-read; their numbers are copied from that report\'s run (`archive_first_run/metrics_sentences.jsonl`).\n')
o.append('## What was compared\n')
o.append('- **before** = `thesis.tex.before_benchmark_edits_2026-10-10`, the text the first report measured. Its numbers here equal the first report\'s "yours now" column.')
o.append('- **now** = `thesis.tex`, read 10 Oct after this check. It includes nine sentence splits made during this check (listed under Verification).')
o.append('- **reference range** = lowest and highest of Buttenschoen, Vost and Dablander for that section type. Vost has no discussion section, so the discussion range comes from two theses.')
o.append('- **gap** = distance outside the range, in standard deviations of the reference paragraphs of that section type. Same definition as the first report.')
o.append('- **closed** = outside before, inside now. **narrowed / widened** = outside both times, gap smaller / larger by more than 0.05 SD. **held** = gap moved by 0.05 SD or less. **opened** = inside before, outside now.\n')
o.append('Scripts: `segment_full_author.py` (new version `before_edits`), `compute_metrics_after.py`, `build_after_tables.py`, `gen_after_report.py`. Per-measure CSV: `after_section_types.csv`. The first run\'s outputs are kept in `archive_first_run/`.\n')
o.append('**Moves.** Sentences that did not change keep their old code (matched by exact text). The 276 sentences in results, discussion and conclusion sections that are new or reworded in this pass were coded by reading, with the same eleven moves (`moves_after_new_sentences.csv`). One coder, one pass, no agreement check. 53 sentences now and 62 before were uncoded in the first run and stay uncoded; they are left out of the move shares.\n')
o.append('Sentences / paragraphs / words in the measured prose, before -> now:\n')
o.append('| Section type | before | now |\n|---|---|---|')
for t in TYPES:
    b, n = S['author_before_edits'][t], S['author_current'][t]
    o.append(f'| {t} | {b[0]} / {b[1]} / {b[3]:,} | {n[0]} / {n[1]} / {n[3]:,} |')
o.append('\n## Summary\n')
from collections import Counter
for t in TYPES:
    rr = [r for r in d['rows'] if r['section_type'] == t]
    c = Counter(r['verdict'] for r in rr)
    o.append(f"- **{t}**: {len(rr)} measures. " + ', '.join(f'{k} {v}' for k, v in sorted(c.items(), key=lambda x: -x[1])) + '.')
o.append('')
for t in TYPES:
    o.append(f'## {t.capitalize()}\n')
    o.append('One row is one measure. Moves are shares of coded sentences in that section type.\n')
    o.append('| Measure | reference range | before | now | gap |\n|---|---|---|---|---|')
    for r in [r for r in d['rows'] if r['section_type'] == t]:
        o.append(f"| {LAB[r['measure']]} | {f(r['ref_lo'])} to {f(r['ref_hi'])} | {f(r['before'])} | {f(r['now'])} | {verdict(r)} |")
    o.append('')
o.append(open(B + 'after_tail.md').read())
open(B + 'BENCHMARK_AFTER.md', 'w').write('\n'.join(o) + '\n')
