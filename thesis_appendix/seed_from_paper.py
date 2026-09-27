#!/usr/bin/env python3
"""Start a thesis copy of one of the NoiseInject paper's supplementary items.

    python3 thesis_appendix/seed_from_paper.py <name> [<name> ...]
    python3 thesis_appendix/seed_from_paper.py --list      # what the paper has

Copies qsar_qm_models/supplementary/items/<name>.tex to
thesis_appendix/noise/items/<name>.tex ONCE, and copies the PNGs it includes
into thesis_appendix/noise/figures/. After that the two files are independent:
editing either one never changes the other. That separation is the point; see
thesis_appendix/README.md.

While copying it:
  - adds a header saying where and when the copy came from;
  - renames the label supp:<name> to app:noise:<name>, the thesis label;
  - removes \\SuppFile{}, the paper's "Additional file N," caption prefix,
    because the thesis numbers its own tables and figures;
  - adds a row to thesis_appendix/noise/INDEX.md, carrying the paper's status.

It refuses to overwrite an item the thesis already has. To start again from the
paper's version, delete the thesis copy and its index row by hand first, so that
throwing away thesis edits is always a deliberate act.
"""
import argparse
import os
import re
import shutil
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
KIRBY = os.path.dirname(HERE)
PAPER = os.environ.get('QSAR_QM_MODELS', os.path.join(os.path.dirname(KIRBY), 'qsar_qm_models'))
PAPER_ITEMS = os.path.join(PAPER, 'supplementary', 'items')
PAPER_INDEX = os.path.join(PAPER, 'supplementary', 'INDEX.md')
# Where the paper's figures live, in the order scripts/stage_additional_files.py searches them.
FIG_SEARCH = [os.path.join(PAPER, 'results', d) for d in
              ('decisions_arc/figures', 'decisions_arc_20260916/figures',
               'paper_figures_v2', 'paper_figures')]
NOISE = os.path.join(HERE, 'noise')
ITEMS = os.path.join(NOISE, 'items')
FIGURES = os.path.join(NOISE, 'figures')
INDEX = os.path.join(NOISE, 'INDEX.md')
GRAPHIC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')


def paper_rows():
    rows, header = {}, None
    for line in open(PAPER_INDEX, encoding='utf-8'):
        if not line.strip().startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if header is None:
            header = [c.lower() for c in cells]
        elif not set(''.join(cells)) <= set('-: '):
            row = dict(zip(header, cells))
            rows[row['name']] = row
    return rows


def seed(name, rows):
    src = os.path.join(PAPER_ITEMS, name + '.tex')
    dst = os.path.join(ITEMS, name + '.tex')
    if name not in rows:
        sys.exit(f'{name} is not in {PAPER_INDEX}')
    if not os.path.exists(src):
        sys.exit(f'{name} has no item file in the paper ({src}); its status there is '
                 f'{rows[name]["status"]!r}')
    if os.path.exists(dst):
        sys.exit(f'{dst} already exists. The thesis copy is independent of the paper and is '
                 f'never overwritten. Delete it and its row in {INDEX} first if you really '
                 f'want the paper version back.')
    body = open(src, encoding='utf-8').read()
    body = body.replace(r'\label{supp:' + name + '}', r'\label{app:noise:' + name + '}')
    body = '\n'.join(l if l.lstrip().startswith('%') else
                     l.replace(r'\SuppFile{} ', '').replace(r'\SuppFile{}', '')
                     for l in body.split('\n'))
    header = (f'% Thesis appendix item: {name}\n'
              f'% Seeded on {date.today().isoformat()} from '
              f'qsar_qm_models/supplementary/items/{name}.tex by thesis_appendix/seed_from_paper.py.\n'
              f'% From here on this copy is edited independently of the paper. Rules: '
              f'thesis_appendix/README.md.\n'
              f'% The lines below, down to the first blank line, are the PAPER copy\'s header, kept for\n'
              f'% provenance. They describe the paper file, not this one: in particular a GENERATED\n'
              f'% notice there does not apply here, and this copy may be edited by hand.\n')
    os.makedirs(ITEMS, exist_ok=True)
    open(dst, 'w', encoding='utf-8').write(header + body)

    copied, missing = [], []
    for fig in GRAPHIC.findall(body):
        for d in FIG_SEARCH:
            if os.path.exists(os.path.join(d, fig)):
                os.makedirs(FIGURES, exist_ok=True)
                shutil.copy2(os.path.join(d, fig), os.path.join(FIGURES, fig))
                copied.append(fig)
                break
        else:
            missing.append(fig)

    r = rows[name]
    line = (f'| {name} | {r.get("kind", "")} | {r["title"]} | — | {r["status"]} (at seeding) '
            f'| {date.today().isoformat()} | {"; ".join(copied) or "—"} |\n')
    with open(INDEX, 'a', encoding='utf-8') as fh:
        fh.write(line)
    print(f'seeded {name}' + (f', figures: {", ".join(copied)}' if copied else ''))
    for fig in missing:
        print(f'  MISSING FIGURE {fig}: not found in the paper figure folders')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*')
    ap.add_argument('--list', action='store_true', help="list the paper's items and whether the thesis has each")
    args = ap.parse_args()
    rows = paper_rows()
    if args.list or not args.names:
        for name, r in rows.items():
            have = os.path.exists(os.path.join(ITEMS, name + '.tex'))
            print(f'{name:32s} paper: {r["status"]:8s} thesis: {"has a copy" if have else "—"}')
        return
    for name in args.names:
        seed(name, rows)


if __name__ == '__main__':
    main()
