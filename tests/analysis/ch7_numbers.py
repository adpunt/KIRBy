"""Re-check every number cited in Chapter 7's results against the raw noise result files.

    python tests/analysis/ch7_numbers.py --results-root /Volumes/seagate/kirby_results_<date> \
        [--noise-view /Volumes/seagate/kirby_results_<date>_noise_view] [--only N7-20,N7-21] \
        [--markdown out.md] [--skip-group-table]

Chapter 7 = thesis.tex from "\\section{Results: Does Noise Handling Work at All?}" to
"\\chapter{Discussion and Conclusions}". Every number block there has a comment line
"% SOURCE [N7-xx]: ..." naming the raw file and key path. This script has one function per
N7 ID. Each function re-reads the cited numbers from the raw files and prints

  OK                    the raw value, rounded to the precision the text prints, equals the text
  TEXT DIFFERS FROM RAW the raw value equals the 2026-10-09 baseline but not the text
                        (the text was wrong when written; nothing new arrived)
  CHANGED               the raw value differs from the 2026-10-09 baseline at the text's precision
                        (new results arrived for this exact number)
  MISSING               the file or key is not there
  NEW FOLD              the same quantity on an outer fold the text does not cite
  NEW DATASET           a Chapter 7 dataset with a file for this experiment that the paragraph
                        does not mention ("(new)" if it had no file on 2026-10-09)
  TEXT EDITED?          the value the script expects is no longer in the paragraph of thesis.tex

The text values are typed into each function (EXPECTED, copied from thesis.tex on 2026-10-09;
values the prose prints were retyped at three decimals on 2026-10-10, when Chapter 7 was rounded).
The raw values of 2026-10-09 are in tests/analysis/ch7_numbers_expected.json; rebuild that file
only on purpose, with --write-baseline, against the 2026-10-09 folders.

Then: which experiments have a QM9 result with no noise added, the pending result slots, the
group-table counts rebuilt from the raw files (read-only, into a temp dir), datasets with noise
results that are neither in Chapter 7 nor cut, and the figure command for the clipped-label figure.

Patient data (tcga, annitia, nhanes, pam50) is never listed, opened or printed: every directory
listing in this script goes through files(), which drops those names, and jload() refuses them.

Nothing is written except --markdown and --write-baseline outputs and a temp dir for the group
table. Written by Claude, 2026-10-09.
"""
import argparse, collections, fnmatch, glob, json, math, os, re, shutil, statistics, subprocess, sys, tempfile, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
THESIS = os.path.join(REPO, 'thesis.tex')
BASELINE = os.path.join(HERE, 'ch7_numbers_expected.json')

# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------
# The Chapter 7 dataset list. To add a dataset: append its name here (the name used in result
# file names, the part before the first "__"), and give it a domain in DOMAIN below. The group
# table (section 5) uses the list in group_table/build.py; add it there too, or the group-table
# counts will not include it. Until it is added, the dataset is printed under "datasets with
# noise results that are not on the Chapter 7 list and not cut".
CH7_DATASETS = ['qm9', 'caco2', 'herg_ki', 'logd', 'herg', 'herg_ic50', 'adme_mdr1', 'buchwald_hartwig',
                'flab_thermo_tresanco', 'ddg_combined', 'ginkgo_titer', 'ginkgo_pr_cho',
                'ginkgo_colloidal', 'ginkgo_acsins']
# adme_mdr1 added 2026-10-10 (10-10b results cover n07, n18, n24, n30 and n40 censoring)
DOMAIN = {d: 'small molecules' for d in ['qm9', 'caco2', 'herg_ki', 'logd', 'herg', 'herg_ic50', 'adme_mdr1']}
DOMAIN['buchwald_hartwig'] = 'reaction'
DOMAIN.update({d: 'antibodies' for d in ['flab_thermo_tresanco', 'ddg_combined', 'ginkgo_titer',
                                          'ginkgo_pr_cho', 'ginkgo_colloidal', 'ginkgo_acsins']})
# Cut by the author (the brief of 2026-10-09); the repo's plan.CUT_DATASETS is added at run time.
CUT_DATASETS = {'cdr_flexibility', 'abcov_kd', 'humanness_therapeutic', 'abagym_VEGF', 'adme_hlm',
                'bh_doyle_plates', 'nt_promoter_tata'}
CUT_PREFIXES = ('tz_her2',)
PATIENT = re.compile(r'tcga|annitia|nhanes|pam50', re.I)


def is_cut(ds):
    return ds in CUT_DATASETS or ds.startswith(CUT_PREFIXES)


def _add_plan_cuts():
    try:
        sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))
        from orchestrate.plan import CUT_DATASETS as plan_cut  # noqa
        CUT_DATASETS.update(d for d in plan_cut if not PATIENT.search(d))
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------
# File access (patient data never listed or opened)
# ---------------------------------------------------------------------------------------------
class Ctx:
    """Where the results are. where='noise' -> <root>/noise, 'full' -> <root>/noise_full,
    'view' -> <noise_view>/noise."""

    def __init__(self, root, view):
        self.root = root.rstrip('/')
        self.view = view.rstrip('/') if view else None
        self.dirs = {'noise': os.path.join(self.root, 'noise'), 'full': os.path.join(self.root, 'noise_full'),
                     'view': os.path.join(self.view, 'noise') if self.view else None}
        self._json, self._rows = {}, {}

    def d(self, where):
        return self.dirs.get(where)

    def path(self, where, *parts):
        base = self.d(where)
        return os.path.join(base, *parts) if base else None


CTX = None


def files(directory, pattern='*'):
    """Sorted file names (not paths) in directory matching pattern, patient data dropped."""
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if not PATIENT.search(f) and fnmatch.fnmatch(f, pattern))


def jload(path):
    """json.load or None if missing/unreadable. Refuses patient data."""
    if path is None or PATIENT.search(path):
        return None
    if path in CTX._json:
        return CTX._json[path]
    try:
        with open(path) as fh:
            out = json.load(fh)
    except Exception:
        out = None
    CTX._json[path] = out
    return out


def J(where, sub, name):
    """Load <dir(where)>/<sub>/<name>; None if missing."""
    p = CTX.path(where, sub, name)
    return jload(p)


def rows_file(path):
    """Rows of one .rows.jsonl file (cached); [] if missing. Bad lines skipped."""
    if path is None or PATIENT.search(path):
        return []
    if path in CTX._rows:
        return CTX._rows[path]
    out = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                r['_file'] = os.path.basename(path)
                out.append(r)
    except Exception:
        out = []
    CTX._rows[path] = out
    return out


def rows(sub, pattern='*.rows.jsonl', where=('noise',), **eq):
    """All rows of the matching .rows.jsonl files in noise/<sub> (and noise_full/<sub> if 'full'
    is in where), each tagged with '_src' (the where) and '_file'. Keyword arguments filter
    rows on equality (e.g. dataset='herg', model='xgboost', level=0.0); a list/tuple/set value
    means 'one of'. Rows from noise_full that repeat a (fold, draw, every key) row of noise are
    kept: callers pick folds explicitly."""
    out = []
    for w in where:
        d = CTX.path(w, sub)
        for f in files(d, pattern):
            for r in rows_file(os.path.join(d, f)):
                ok = True
                for k, v in eq.items():
                    rv = r.get(k)
                    if isinstance(v, (list, tuple, set)):
                        if rv not in v:
                            ok = False; break
                    elif rv != v:
                        ok = False; break
                if ok:
                    r = dict(r); r['_src'] = w
                    out.append(r)
    return out


def cfg(model_input):
    """Configuration name from a rows.jsonl model_input: 'dnn:hybrid__general2=rank...' ->
    'hybrid__general2'; 'base_1' -> 'base_1'."""
    return str(model_input).split(':')[-1].split('=')[0]


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def datasets_with_files(sub, where=('noise', 'full', 'view')):
    """Chapter 7 datasets with any result file in <where>/<sub> (fingerprints ignored)."""
    out = set()
    for w in where:
        for f in files(CTX.path(w, sub)):
            if 'fingerprint' in f:
                continue
            ds = f.split('__')[0].split('.')[0]
            out.add(ds)
    return sorted(d for d in out if d in CH7_DATASETS)


# ---------------------------------------------------------------------------------------------
# Thesis paragraphs
# ---------------------------------------------------------------------------------------------
def thesis_blocks():
    """{N7-xx: text from its SOURCE line to the next SOURCE line or section}."""
    try:
        tex = open(THESIS).read()
    except Exception:
        return {}
    # Chapter 7 = from its \section{Results} to the next \chapter (headings renamed 2026-10-10).
    c = tex.find('\\chapter{Detecting and handling label noise}')
    a = tex.find('\\section{Results}', c) if c >= 0 else -1
    b = tex.find('\\chapter{', a + 1) if a >= 0 else -1
    ch = tex[a:b] if a >= 0 and b > a else ''
    out = {}
    parts = re.split(r'(?m)^% SOURCE \[(N7-\d+)\]:', ch)
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        body = re.split(r'(?m)^\\(?:sub)?section\{', body)[0]
        out[parts[i]] = body
    return out


BLOCKS = {}


# ---------------------------------------------------------------------------------------------
# Results of one N7 ID
# ---------------------------------------------------------------------------------------------
def _decimals(text):
    t = text.replace(',', '').replace('$', '').replace('\\%', '').replace('%', '').strip()
    return len(t.split('.')[1]) if '.' in t else 0


def _num(text):
    return float(text.replace(',', '').replace('$', '').replace('\\%', '').replace('%', '').strip())


def _fmt(x, d):
    if x is None:
        return 'None'
    s = f'{x:.{d}f}'
    return '0' if d == 0 and s == '-0' else (s[1:] if s.startswith('-') and float(s) == 0 else s)


class Report:
    """Lines for one N7 ID. Use check() for each number the text cites."""

    def __init__(self, nid, label):
        self.id, self.label = nid, label
        self.lines = []          # (kind, text)
        self.values = {}         # label -> raw value (for the baseline)
        self.new_ds = set()

    def add(self, kind, text):
        self.lines.append((kind, text))

    def check(self, label, text, now, note=''):
        """label: what the number is (dataset, configuration, model, arm, level, fold, draw);
        text: the value exactly as printed in the thesis ('0.6354', '-0.0143', '76', '1,241');
        now: the raw value (float/int) or None if not found."""
        d = _decimals(text)
        self.values[label] = now
        block = BLOCKS.get(self.id)
        # Number-boundary match: '0.838' must not be found inside '0.8381'.
        found = block is None or any(
            re.search(r'(?<![\d.])' + re.escape(t) + r'(?![\d])', block)
            for t in (text, text.replace('-', '')))
        if not found:
            self.add('TEXT EDITED?', f'{label}: expected text "{text}" not found in the paragraph')
        if now is None:
            self.add('MISSING', f'{label}: text {text}, raw value not found' + (f' ({note})' if note else ''))
            return
        nowr = _fmt(now, d)
        textr = _fmt(_num(text), d)
        base = BASE.get(self.id, {}).get(label, 'absent')
        if nowr == textr:
            kind = 'OK'
        elif base not in ('absent', None) and _fmt(base, d) != nowr:
            kind = 'CHANGED'
        else:
            kind = 'TEXT DIFFERS FROM RAW'
        extra = f'  (2026-10-09 raw {_fmt(base, d)})' if kind == 'CHANGED' else ''
        self.add(kind, f'{label}: text {text}, now {nowr}{extra}' + (f'  [{note}]' if note else ''))

    def claim(self, label, holds, detail):
        """A claim in the text that is not one number (e.g. 'LightGBM gives one score in every
        experiment'). holds: bool from the raw files, or None if it cannot be evaluated."""
        self.values[label] = None if holds is None else bool(holds)
        base = BASE.get(self.id, {}).get(label, 'absent')
        if holds is None:
            self.add('MISSING', f'{label}: {detail}')
        elif holds:
            self.add('OK', f'{label}: holds ({detail})')
        elif base is True:
            self.add('CHANGED', f'{label}: no longer holds ({detail})')
        else:
            self.add('TEXT DIFFERS FROM RAW', f'{label}: does not hold ({detail})')

    def missing(self, label, what):
        self.values.setdefault(label, None)
        self.add('MISSING', f'{label}: {what}')

    def new_fold(self, label, fold, text):
        self.add('NEW FOLD', f'{label}, fold {fold}: {text}')

    def new_dataset(self, ds, text):
        known = ds in BASE.get('_datasets', {}).get(self.id, [])
        self.new_ds.add(ds)
        self.add('NEW DATASET', f'{ds}{"" if known else " (new)"}: {text}')

    def info(self, text):
        self.add('note', text)


BASE = {}
CHECKS = []          # (nid, label, function)


def check(nid, label):
    """Decorator: register f(r: Report) as the checker for N7 ID nid."""
    def deco(f):
        CHECKS.append((nid, label, f))
        return f
    return deco


NOT_CHECKABLE = {}   # nid -> reason


# =============================================================================================
# One function per N7 ID
# =============================================================================================

# ---------------------------------------------------------------------------------------------
# N7-03 to N7-06: changing a label against deleting a molecule (n07, n24, n23, n31)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# N7-03 .. N7-06: changing a label against deleting a molecule (n07, n24, n23, n31)
# ---------------------------------------------------------------------------------------------
_A_CLASS_KINDS = {'binary_asymmetric', 'class_dependent', 'class_imbalance', 'confusion_directed',
                  'instance_noise', 'uniform'}


def _a_main_level(kind, levels):
    """The damaged level a paragraph reads for this kind: 0.25 censoring, 0.1 flips, else 1."""
    t = 0.25 if kind == 'censoring' else (0.1 if kind in _A_CLASS_KINDS else 1.0)
    dmg = [L for L in levels if float(L) > 0]
    return min(dmg, key=lambda L: (abs(float(L) - t), float(L))) if dmg else None


def _a_cell(rs, **eq):
    """Mean over draws of score_D and score_Dprime for rows matching eq (cfg() applied to
    model_input). Returns (D, Dprime, n) or (None, None, 0)."""
    sel = []
    for r in rs:
        ok = True
        for k, v in eq.items():
            rv = cfg(r.get('model_input')) if k == 'model_input' else r.get(k)
            if rv != v:
                ok = False; break
        if ok:
            sel.append(r)
    if not sel:
        return None, None, 0
    return mean(r['score_D'] for r in sel), mean(r['score_Dprime'] for r in sel), len(sel)


def _a_dedupe(rs):
    """Rows of noise and noise_full: keep one row per key, preferring noise/."""
    seen, out = set(), []
    for r in sorted(rs, key=lambda r: r.get('_src') != 'noise'):
        k = (r['dataset'], r['noise_condition'], r['level'], r['fold'], r['draw'], r['method'],
             r.get('method_setting'), r.get('similarity_setting'), cfg(r['model_input']), r['model'])
        if k in seen:
            continue
        seen.add(k); out.append(r)
    return out


def _a_n24_fold(rs, fold, space, model, setting, level):
    """n24 rows, one fold: (no cleaning, remove, best random seed value, best seed) for detector
    setting like 'cross_val|flag_rate=0.1'; means over draws."""
    base = [r for r in rs if r['fold'] == fold and cfg(r['model_input']) == space and r['model'] == model
            and r['level'] == level and r.get('similarity_setting') in ('none', None)]
    rem = [r for r in base if r['method'] == 'remove' and r['method_setting'] == setting]
    if not rem:
        return None
    D = mean(r['score_D'] for r in rem); P = mean(r['score_Dprime'] for r in rem)
    rnd = {}
    for r in base:
        if r['method'] == 'delete_at_random' and r['method_setting'].startswith(setting + '|seed='):
            rnd.setdefault(r['method_setting'].split('seed=')[1], []).append(r['score_Dprime'])
    lib = rem[0]['lower_is_better']
    if rnd:
        ms = {s: mean(v) for s, v in rnd.items()}
        s = (min if lib else max)(ms, key=ms.get)
        return D, P, ms[s], s, len(rem)
    return D, P, None, None, len(rem)


def _a_n24_json_line(ds, kind):
    """One-line summary of an n24 file: hybrid, cross_val at flag 0.1, per model:
    no cleaning -> delete, best random deletion (fold 0, mean over draws)."""
    d = J('noise', 'n24_delete_or_keep', f'{ds}__{kind}.json')
    if not d or 'per_level' not in d:
        return None
    L = _a_main_level(kind, list(d['per_level']))
    if L is None:
        return f'{kind}: no damaged level'
    parts = []
    x = d['per_level'][L].get('hybrid', {}).get('cross_val', {}).get('0.1', {})
    for m in d.get('models', []):
        nc = x.get('no_cleaning', {}).get(m); rm = x.get('delete/remove', {}).get(m)
        if not nc or not rm:
            continue
        rnd = [mean(v.get(m, {}).values()) for a, v in x.items() if a.startswith('delete_at_random') and v.get(m)]
        lib = d.get('lower_is_better')
        best = (min if lib else max)(rnd) if rnd else None
        parts.append(f'{m} {mean(nc.values()):.4f} -> delete {mean(rm.values()):.4f}'
                     + (f' vs best random {best:.4f}' if best is not None else ''))
    return f'{kind} level {L}, hybrid, cross_val@0.1, {d.get("metric")}: ' + ('; '.join(parts) or 'no scores')


@check('N7-03', 'hERG flips: delete, move toward neighbors, down-weight; random-deletion control on 3 folds')
def n7_03(r):
    """Section 'Changing a Label Against Deleting a Molecule', first paragraph (hERG under
    confusion-directed flips at 10%, xgboost, fold 0, means over 3 draws).
    n07: noise/n07_remediation/herg__confusion_directed__fold_aligned_2026-10-06.rows.jsonl, level 0.1,
      remove flag_share=0.1; downweight; label_smooth at the setting run on all 3 draws.
    n24: noise/n24_delete_or_keep/herg__confusion_directed.json per_level.0.1.hybrid.
      neighbor_residual@screen1.0.1.{delete/remove, delete_at_random/seed0-4}.xgboost;
      folds 0-2: n24 rows (noise + noise_full), cross_val|flag_rate=0.1, best of 5 random seeds."""
    EXPECTED = {
        'n07 hybrid untreated': '0.8028', 'n07 hybrid remove@0.1': '0.8356',
        'n07 hybrid label_smooth (k=10 inverse_square, forest_proximity)': '0.8338',
        'n07 hybrid downweight': '0.8140',
        'n07 base_1 untreated': '0.8060', 'n07 base_1 remove@0.1': '0.8253',
        'n07 base_1 label_smooth (k=20 inverse_square, hybrid_similarity manhattan)': '0.8250',
        'n07 base_1 downweight': '0.8108',
        'n24 hybrid neighbor_residual@screen1 0.1 delete': '0.838',
        'n24 hybrid neighbor_residual@screen1 0.1 best random deletion': '0.818',
        'n24 hybrid cross_val@0.1 fold 0 untreated': '0.8028', 'n24 hybrid cross_val@0.1 fold 0 delete': '0.8356',
        'n24 hybrid cross_val@0.1 fold 0 best random': '0.8127',
        'n24 hybrid cross_val@0.1 fold 1 untreated': '0.8164', 'n24 hybrid cross_val@0.1 fold 1 delete': '0.8258',
        'n24 hybrid cross_val@0.1 fold 1 best random': '0.8122',
        'n24 hybrid cross_val@0.1 fold 2 untreated': '0.819', 'n24 hybrid cross_val@0.1 fold 2 delete': '0.835',
        'n24 hybrid cross_val@0.1 fold 2 best random': '0.828',
    }
    pre = 'herg confusion_directed level 0.1 xgboost fold 0 (mean of draws) '
    rs = rows('n07_remediation', 'herg__confusion_directed__*.rows.jsonl', where=('noise', 'full'),
              model='xgboost', level=0.1)
    rs = _a_dedupe(rs)
    if not rs:
        r.missing(pre + 'n07', 'n07_remediation herg__confusion_directed rows not found')
    f0 = [x for x in rs if x['fold'] == 0]
    sm = {'hybrid': ('k=10|scheme=inverse_square|flag_share=0.1', 'forest_proximity|none|forest'),
          'base_1': ('k=20|scheme=inverse_square|flag_share=0.1', 'hybrid_similarity|none|manhattan')}
    for sp, lab in (('hybrid', 'n07 hybrid'), ('base_1', 'n07 base_1')):
        D, rem, _ = _a_cell(f0, model_input=sp, method='remove', method_setting='flag_share=0.1')
        r.check(pre + lab + ' untreated', EXPECTED[lab + ' untreated'], D)
        r.check(pre + lab + ' remove@0.1', EXPECTED[lab + ' remove@0.1'], rem)
        k = [k for k in EXPECTED if k.startswith(lab + ' label_smooth')][0]
        _, ls, n = _a_cell(f0, model_input=sp, method='label_smooth', method_setting=sm[sp][0],
                           similarity_setting=sm[sp][1])
        r.check(pre + k, EXPECTED[k], ls, note=f'{n} draws')
        _, dw, _ = _a_cell(f0, model_input=sp, method='downweight')
        r.check(pre + lab + ' downweight', EXPECTED[lab + ' downweight'], dw)
    for fo in sorted({x['fold'] for x in rs} - {0}):
        fr = [x for x in rs if x['fold'] == fo]
        parts = []
        for sp in ('hybrid', 'base_1'):
            D, rem, _ = _a_cell(fr, model_input=sp, method='remove', method_setting='flag_share=0.1')
            _, dw, _ = _a_cell(fr, model_input=sp, method='downweight')
            if D is not None:
                parts.append(f'{sp} {D:.4f} -> remove@0.1 {rem:.4f}, downweight {dw if dw is None else round(dw, 4)}')
        r.new_fold('n07 herg confusion_directed xgboost', fo, '; '.join(parts) or 'rows without the cited arms')

    d = J('noise', 'n24_delete_or_keep', 'herg__confusion_directed.json')
    x = (((d or {}).get('per_level') or {}).get('0.1') or {}).get('hybrid', {}).get('neighbor_residual@screen1', {}).get('0.1')
    lab = 'n24 hybrid neighbor_residual@screen1 0.1'
    if not x:
        r.missing(pre + lab, 'n24 herg__confusion_directed.json per_level.0.1.hybrid.neighbor_residual@screen1.0.1 not found')
    else:
        rm = x.get('delete/remove', {}).get('xgboost')
        rnd = [mean(v['xgboost'].values()) for a, v in x.items() if a.startswith('delete_at_random') and v.get('xgboost')]
        r.check(pre + lab + ' delete', EXPECTED[lab + ' delete'], mean(rm.values()) if rm else None)
        r.check(pre + lab + ' best random deletion', EXPECTED[lab + ' best random deletion'], max(rnd) if rnd else None)

    n24 = _a_dedupe(rows('n24_delete_or_keep', 'herg__confusion_directed__*.rows.jsonl', where=('noise', 'full'),
                         model='xgboost', level=0.1))
    folds = sorted({x['fold'] for x in n24})
    for fo in (0, 1, 2):
        lab = f'n24 hybrid cross_val@0.1 fold {fo}'
        c = _a_n24_fold(n24, fo, 'hybrid', 'xgboost', 'cross_val|flag_rate=0.1', 0.1)
        if c is None:
            for s in ('untreated', 'delete', 'best random'):
                r.check(f'herg confusion_directed level 0.1 xgboost {lab} {s}', EXPECTED[f'{lab} {s}'], None)
            continue
        D, P, rb, seed, n = c
        p = f'herg confusion_directed level 0.1 xgboost ({n} draws) '
        r.check(p + lab + ' untreated', EXPECTED[lab + ' untreated'], D)
        r.check(p + lab + ' delete', EXPECTED[lab + ' delete'], P)
        r.check(p + lab + ' best random', EXPECTED[lab + ' best random'], rb, note=f'seed {seed}')
    for fo in [f for f in folds if f not in (0, 1, 2)]:
        c = _a_n24_fold(n24, fo, 'hybrid', 'xgboost', 'cross_val|flag_rate=0.1', 0.1)
        if c:
            r.new_fold('n24 herg confusion_directed xgboost hybrid cross_val@0.1', fo,
                       f'untreated {c[0]:.4f} -> delete {c[1]:.4f} vs best random {c[2]:.4f} ({c[4]} draws)')

    for ds in datasets_with_files('n24_delete_or_keep', where=('noise',)):
        if ds == 'herg':
            continue
        lines = []
        for f in files(CTX.path('noise', 'n24_delete_or_keep'), f'{ds}__*.json'):
            if 'fingerprint' in f:
                continue
            kind = f[:-5].split('__', 1)[1]
            s = _a_n24_json_line(ds, kind)
            if s:
                lines.append(s)
        r.new_dataset(ds, ' | '.join(lines) if lines else 'n24 files without per_level')


@check('N7-04', 'LogD outliers: actions closer together; deletion against random deletion')
def n7_04(r):
    """Section 'Changing a Label Against Deleting a Molecule', second paragraph (LogD, 5%
    outliers at level 1.0, dnn, fold 0, means over 3 draws).
    n07: noise/n07_remediation/logd__outlier_p05__fold_aligned_2026-10-06.rows.jsonl, base_1 and
      hybrid: remove flag_share=0.05, downweight, label_smooth (k=20 inverse_square,
      hybrid_similarity manhattan), js_shrink (the score-scaled shrink toward an out-of-fold prediction).
    n24: n24 rows (noise + noise_full) logd outlier_p05 dnn level 1.0: hybrid__general2
      cross_val|flag_rate=0.05 and hybrid cross_val|flag_rate=0.1, delete vs best of 5 random seeds;
      fold 1 hybrid__general2 at 0.05."""
    EXPECTED = {
        'n07 base_1 untreated': '0.6273', 'n07 base_1 remove@0.05': '0.6030', 'n07 base_1 downweight': '0.6149',
        'n07 base_1 label_smooth': '0.6124', 'n07 base_1 js_shrink': '0.6005',
        'n07 hybrid untreated': '0.5881', 'n07 hybrid downweight': '0.5634', 'n07 hybrid js_shrink': '0.5900',
        'n24 hybrid__general2 cross_val@0.05 fold 0 untreated': '0.5847',
        'n24 hybrid__general2 cross_val@0.05 fold 0 delete': '0.5460',
        'n24 hybrid__general2 cross_val@0.05 fold 0 best random': '0.5584',
        'n24 hybrid cross_val@0.1 fold 0 delete': '0.5690',
        'n24 hybrid cross_val@0.1 fold 0 best random': '0.5647',
        'n24 hybrid__general2 cross_val@0.05 fold 1 untreated': '0.6242',
        'n24 hybrid__general2 cross_val@0.05 fold 1 delete': '0.6332',
    }
    pre = 'logd outlier_p05 level 1.0 dnn fold 0 (mean of draws) '
    rs = _a_dedupe(rows('n07_remediation', 'logd__outlier_p05__*.rows.jsonl', where=('noise', 'full'),
                        model='dnn', level=1.0))
    if not rs:
        r.missing(pre + 'n07', 'n07_remediation logd__outlier_p05 rows not found')
    f0 = [x for x in rs if x['fold'] == 0]
    LS = dict(method_setting='k=20|scheme=inverse_square', similarity_setting='hybrid_similarity|none|manhattan')
    D, rem, _ = _a_cell(f0, model_input='base_1', method='remove', method_setting='flag_share=0.05')
    r.check(pre + 'n07 base_1 untreated', EXPECTED['n07 base_1 untreated'], D)
    r.check(pre + 'n07 base_1 remove@0.05', EXPECTED['n07 base_1 remove@0.05'], rem)
    r.check(pre + 'n07 base_1 downweight', EXPECTED['n07 base_1 downweight'], _a_cell(f0, model_input='base_1', method='downweight')[1])
    r.check(pre + 'n07 base_1 label_smooth', EXPECTED['n07 base_1 label_smooth'],
            _a_cell(f0, model_input='base_1', method='label_smooth', **LS)[1], note='k=20 inverse_square, hybrid_similarity manhattan')
    r.check(pre + 'n07 base_1 js_shrink', EXPECTED['n07 base_1 js_shrink'], _a_cell(f0, model_input='base_1', method='js_shrink')[1])
    Dh, dwh, _ = _a_cell(f0, model_input='hybrid', method='downweight')
    r.check(pre + 'n07 hybrid untreated', EXPECTED['n07 hybrid untreated'], Dh)
    r.check(pre + 'n07 hybrid downweight', EXPECTED['n07 hybrid downweight'], dwh)
    r.check(pre + 'n07 hybrid js_shrink', EXPECTED['n07 hybrid js_shrink'], _a_cell(f0, model_input='hybrid', method='js_shrink')[1])
    # "down-weighting is best" on the first hybrid, among the four actions the paragraph names
    acts = {'remove@0.05': _a_cell(f0, model_input='hybrid', method='remove', method_setting='flag_share=0.05')[1],
            'downweight': dwh, 'label_smooth': _a_cell(f0, model_input='hybrid', method='label_smooth', **LS)[1],
            'js_shrink': _a_cell(f0, model_input='hybrid', method='js_shrink')[1]}
    ok = {k: v for k, v in acts.items() if v is not None}
    if ok:
        r.claim(pre + 'n07 hybrid: down-weighting best of delete/down-weight/smooth/shrink',
                min(ok, key=ok.get) == 'downweight', ', '.join(f'{k} {v:.4f}' for k, v in ok.items()))
    for fo in sorted({x['fold'] for x in rs} - {0}):
        fr = [x for x in rs if x['fold'] == fo]
        parts = []
        for sp in ('base_1', 'hybrid'):
            D, rm, _ = _a_cell(fr, model_input=sp, method='remove', method_setting='flag_share=0.05')
            dw = _a_cell(fr, model_input=sp, method='downweight')[1]
            js = _a_cell(fr, model_input=sp, method='js_shrink')[1]
            if D is not None:
                parts.append(f'{sp} {D:.4f} -> remove@0.05 {rm:.4f}, downweight {dw}, js_shrink {js}')
        r.new_fold('n07 logd outlier_p05 dnn', fo, '; '.join(parts) or 'rows without the cited arms')

    n24 = _a_dedupe(rows('n24_delete_or_keep', 'logd__outlier_p05__*.rows.jsonl', where=('noise', 'full'),
                         model='dnn', level=1.0))
    if not n24:
        r.missing('logd outlier_p05 n24', 'n24 rows not found')
    for fo, sp, rate, items in ((0, 'hybrid__general2', '0.05', ('untreated', 'delete', 'best random')),
                                (0, 'hybrid', '0.1', ('delete', 'best random')),
                                (1, 'hybrid__general2', '0.05', ('untreated', 'delete'))):
        lab = f'n24 {sp} cross_val@{rate} fold {fo}'
        c = _a_n24_fold(n24, fo, sp, 'dnn', f'cross_val|flag_rate={rate}', 1.0)
        vals = dict(zip(('untreated', 'delete', 'best random'), c[:3])) if c else {}
        for it in items:
            r.check(f'logd outlier_p05 level 1.0 dnn {lab} {it}', EXPECTED[f'{lab} {it}'], vals.get(it),
                    note=(f'seed {c[3]}' if c and it == 'best random' else ''))
    c = _a_n24_fold(n24, 1, 'hybrid__general2', 'dnn', 'cross_val|flag_rate=0.05', 1.0)
    if c and c[2] is not None:
        r.info(f'logd n24 hybrid__general2 cross_val@0.05 fold 1 best random deletion {c[2]:.4f} (seed {c[3]}); '
               'the SOURCE line cites 0.6231, the text gives no number')
    for fo in sorted({x['fold'] for x in n24} - {0, 1}):
        parts = []
        for sp, rate in (('hybrid__general2', '0.05'), ('hybrid', '0.1')):
            c = _a_n24_fold(n24, fo, sp, 'dnn', f'cross_val|flag_rate={rate}', 1.0)
            if c:
                parts.append(f'{sp} cross_val@{rate} {c[0]:.4f} -> delete {c[1]:.4f} vs best random {c[2]:.4f}')
        r.new_fold('n24 logd outlier_p05 dnn', fo, '; '.join(parts) or 'rows without the cited arms')

    # NEW DATASET: n07 rows of other datasets, dnn (or first model) on base_1 and hybrid at the main level
    for ds in datasets_with_files('n07_remediation', where=('noise', 'full')):
        if ds == 'logd':
            continue
        lines = []
        for f in files(CTX.path('noise', 'n07_remediation'), f'{ds}__*.rows.jsonl'):
            kind = f[len(ds) + 2:].split('__')[0].replace('.rows.jsonl', '')
            rr = [x for x in rows_file(os.path.join(CTX.path('noise', 'n07_remediation'), f)) if x.get('fold') == 0]
            if not rr:
                continue
            L = _a_main_level(kind, sorted({x['level'] for x in rr}))
            models = sorted({x['model'] for x in rr})
            m = 'dnn' if 'dnn' in models else models[0]
            rr = [x for x in rr if x['level'] == L and x['model'] == m]
            parts = []
            for sp in ('base_1', 'hybrid'):
                D, rm, _ = _a_cell(rr, model_input=sp, method='remove', method_setting='flag_share=0.05')
                dw = _a_cell(rr, model_input=sp, method='downweight')[1]
                js = _a_cell(rr, model_input=sp, method='js_shrink')[1]
                if D is not None:
                    fm = lambda v: 'n/a' if v is None else f'{v:.4f}'
                    parts.append(f'{sp} {D:.4f} -> remove@0.05 {fm(rm)}, downweight {fm(dw)}, js_shrink {fm(js)}')
            if parts:
                lines.append(f'{kind} L{L} {m}: ' + '; '.join(parts))
        r.new_dataset(ds, ' | '.join(lines) if lines else 'n07 files without fold-0 rows (json/units only)')


@check('N7-05', 'Detector matters more than action: five detectors, delete top 10%')
def n7_05(r):
    """Section 'Changing a Label Against Deleting a Molecule', third paragraph.
    noise/n23_uncertainty_cleaning/herg__confusion_directed.json per_regime.injected.0.1.<config>.
      <detector>.drop@0.1.xgboost, mean over 3 draws (fold 0);
    logd__outlier_p05.json per_regime.injected.1.hybrid.<detector>.drop@0.1.dnn (GP best of five);
    second fold: noise_full/n23 herg confusion_directed rows, fold 1 draw 0, drop@0.1."""
    EXPECTED = {'hybrid none': '0.8028', 'hybrid activity_cliff_aware': '0.8375', 'hybrid gaussian_process': '0.8270',
                'hybrid heteroscedastic_nn': '0.8128', 'hybrid ngboost': '0.8067', 'hybrid hetero_gp': '0.8064',
                'logd hybrid none': '0.5879', 'logd hybrid gaussian_process': '0.5581'}
    DETS = ('activity_cliff_aware', 'gaussian_process', 'heteroscedastic_nn', 'ngboost', 'hetero_gp')
    d = J('noise', 'n23_uncertainty_cleaning', 'herg__confusion_directed.json')
    q = ((((d or {}).get('per_regime') or {}).get('injected') or {}).get('0.1') or {})
    pre = 'herg confusion_directed level 0.1 xgboost fold 0 (mean of 3 draws) '
    if not q:
        r.missing(pre, 'n23 herg__confusion_directed.json per_regime.injected.0.1 not found')
    h = q.get('hybrid', {})
    none = (h.get('activity_cliff_aware', {}).get('none') or {}).get('xgboost')
    r.check(pre + 'hybrid none', EXPECTED['hybrid none'], mean(none.values()) if none else None)
    for det in DETS:
        x = (h.get(det, {}).get('drop@0.1') or {}).get('xgboost')
        r.check(pre + f'hybrid {det} drop@0.1', EXPECTED[f'hybrid {det}'], mean(x.values()) if x else None)
    # "On the single representation the last three make things worse"
    b = q.get('base_1', {})
    bn = (b.get('ngboost', {}).get('none') or {}).get('xgboost')
    if bn:
        vals = {det: mean(((b.get(det, {}).get('drop@0.1') or {}).get('xgboost') or {}).values()) for det in
                ('heteroscedastic_nn', 'ngboost', 'hetero_gp')}
        r.claim(pre + 'base_1: variance-head network, NGBoost, heteroscedastic GP all below untreated',
                all(v is not None and v < mean(bn.values()) for v in vals.values()),
                f'untreated {mean(bn.values()):.4f}; ' + ', '.join(f'{k} {v:.4f}' for k, v in vals.items() if v is not None))
    d2 = J('noise', 'n23_uncertainty_cleaning', 'logd__outlier_p05.json')
    inj = (((d2 or {}).get('per_regime') or {}).get('injected') or {})
    L = _a_main_level('outlier_p05', list(inj)) if inj else None
    hh = inj.get(L, {}).get('hybrid', {}) if L else {}
    pre2 = f'logd outlier_p05 level {L} dnn fold 0 (mean of 3 draws) '
    nn = (hh.get('gaussian_process', {}).get('none') or {}).get('dnn')
    r.check(pre2 + 'hybrid none', EXPECTED['logd hybrid none'], mean(nn.values()) if nn else None)
    gp = (hh.get('gaussian_process', {}).get('drop@0.1') or {}).get('dnn')
    r.check(pre2 + 'hybrid gaussian_process drop@0.1', EXPECTED['logd hybrid gaussian_process'], mean(gp.values()) if gp else None)
    if hh:
        v = {det: mean(((hh.get(det, {}).get('drop@0.1') or {}).get('dnn') or {}).values()) for det in DETS}
        v = {k: x for k, x in v.items() if x is not None}
        r.claim(pre2 + 'hybrid: GP best of the five detectors at drop@0.1', min(v, key=v.get) == 'gaussian_process',
                ', '.join(f'{k} {x:.4f}' for k, x in v.items()))
    # second fold
    fr = rows('n23_uncertainty_cleaning', 'herg__confusion_directed__*.rows.jsonl', where=('noise', 'full'),
              model='xgboost', level=0.1)
    folds = sorted({x['fold'] for x in fr})
    f1 = {}
    for fo in folds:
        if fo == 0:
            continue
        sub = [x for x in fr if x['fold'] == fo]
        draws = sorted({x['draw'] for x in sub})
        for dr in draws:
            parts, beat = [], []
            for sp in ('base_1', 'hybrid'):
                s = [x for x in sub if x['draw'] == dr and cfg(x['model_input']) == sp and x['method_setting'].endswith('drop@0.1')]
                if not s:
                    continue
                D = s[0]['score_D']
                best = {}
                for x in s:
                    best[x['method']] = max(best.get(x['method'], -1e9), x['score_Dprime'])
                beat += [f'{sp} {k} {v:.4f}' for k, v in best.items() if round(v, 3) > round(D, 3)]
                if fo == 1 and dr == 0:
                    f1[sp] = (D, best)
                parts.append(f'{sp} {D:.4f} -> ' + ', '.join(f'{k} {v:.4f}' for k, v in sorted(best.items())))
            txt = '; '.join(parts)
            if fo == 1 and dr == 0:
                # Corrected text (2026-10-09): none of the five improved the single representation
                # (0.8155 untreated, 0.7946 to 0.8147); only the cliff-aware gap improved the hybrid, 0.8314 to 0.8332.
                b1 = f1.get('base_1'); hy = f1.get('hybrid')
                pre = 'herg confusion_directed xgboost fold 1 draw 0 drop@0.1'
                r.check(f'{pre} base_1 untreated', '0.8155', b1[0] if b1 else None)
                r.check(f'{pre} base_1 lowest of the five detectors', '0.7946', min(b1[1].values()) if b1 and b1[1] else None)
                r.check(f'{pre} base_1 highest of the five detectors', '0.8147', max(b1[1].values()) if b1 and b1[1] else None)
                r.claim(f'{pre}: none of the five improved base_1', None if not b1 else
                        all(round(v, 3) <= round(b1[0], 3) for v in b1[1].values()), txt)
                r.check(f'{pre} hybrid untreated', '0.8314', hy[0] if hy else None)
                r.check(f'{pre} hybrid activity_cliff_aware', '0.8332', hy[1].get('activity_cliff_aware') if hy else None)
                r.claim(f'{pre}: only the cliff-aware gap improved the hybrid', None if not hy else
                        sorted(k for k, v in hy[1].items() if round(v, 3) > round(hy[0], 3)) == ['activity_cliff_aware'],
                        '; '.join(beat) or 'none improved')
            else:
                r.new_fold(f'n23 herg confusion_directed xgboost drop@0.1 draw {dr}', fo, txt)
    if 1 not in folds:
        r.missing('herg confusion_directed xgboost fold 1', 'no n23 rows for fold 1')
    for ds in datasets_with_files('n23_uncertainty_cleaning', where=('noise',)):
        if ds in ('herg', 'logd'):
            continue
        lines = []
        for f in files(CTX.path('noise', 'n23_uncertainty_cleaning'), f'{ds}__*.json'):
            if 'fingerprint' in f:
                continue
            kind = f[:-5].split('__', 1)[1]
            dd = J('noise', 'n23_uncertainty_cleaning', f)
            inj = (((dd or {}).get('per_regime') or {}).get('injected') or {})
            L = _a_main_level(kind, list(inj))
            if not L:
                continue
            m = (dd.get('models') or ['?'])[0]
            hy = inj[L].get('hybrid', {})
            nn = (hy.get('gaussian_process', {}).get('none') or {}).get(m)
            if not nn:
                continue
            v = {det: mean(((hy.get(det, {}).get('drop@0.1') or {}).get(m) or {}).values()) for det in DETS}
            lines.append(f'{kind} L{L} {m} hybrid {mean(nn.values()):.4f} -> ' +
                         ', '.join(f'{k} {x:.4f}' for k, x in v.items() if x is not None))
        r.new_dataset(ds, ' | '.join(lines) if lines else 'n23 files without injected-regime scores')


@check('N7-06', 'hERG Ki outliers: down-weighting flags of known quality')
def n7_06(r):
    """Section 'Changing a Label Against Deleting a Molecule', fourth paragraph.
    noise/n31_regime_and_breakeven/herg_ki__fold_aligned_2026-10-06.rows.jsonl, outlier_only_p05 level
    1.0, dnn, downweight: detector/share0.1/downweight (mean of 3 draws) on base_1 and hybrid;
    control flags_share0.1_p0/seed0/downweight on base_1, draws 0, 1, 2 separately. Fold 0."""
    EXPECTED = {'base_1 untreated': '0.689', 'base_1 detector downweight': '0.642',
                'hybrid untreated': '0.6917', 'hybrid detector downweight': '0.6388',
                'base_1 control p0 seed0 draw 0': '0.6921', 'base_1 control p0 seed0 draw 1': '0.7094',
                'base_1 control p0 seed0 draw 2': '0.7171'}
    rs = _a_dedupe(rows('n31_regime_and_breakeven', 'herg_ki__*.rows.jsonl', where=('noise', 'full'),
                        model='dnn', level=1.0, noise_condition='outlier_only_p05', method='downweight'))
    pre = 'herg_ki outlier_only_p05 level 1.0 dnn fold 0 '
    if not rs:
        r.missing(pre, 'n31 herg_ki rows not found')
    f0 = [x for x in rs if x['fold'] == 0]
    for sp in ('base_1', 'hybrid'):
        D, P, n = _a_cell(f0, model_input=sp, method_setting='detector/share0.1/downweight')
        r.check(pre + f'{sp} untreated (mean of {n} draws)', EXPECTED[f'{sp} untreated'], D)
        r.check(pre + f'{sp} detector share 0.1 downweight (mean of {n} draws)', EXPECTED[f'{sp} detector downweight'], P)
    for dr in (0, 1, 2):
        _, P, _ = _a_cell(f0, model_input='base_1', method_setting='flags_share0.1_p0/seed0/downweight', draw=dr)
        r.check(pre + f'base_1 control (0.1 flagged, none damaged, seed 0) draw {dr}',
                EXPECTED[f'base_1 control p0 seed0 draw {dr}'], P)
    # QM9 with no noise added (rows added to the table 2026-10-10 from the 10-10b results): level 0 of the
    # outlier_only_p05 rows, detector/share0.1/downweight, draw 0
    q0 = _a_dedupe(rows('n31_regime_and_breakeven', 'qm9__*.rows.jsonl', model='dnn', level=0.0,
                        noise_condition='outlier_only_p05', method='downweight'))
    for sp, tD, tP in (('base_1', '0.396', '0.402'), ('hybrid', '0.355', '0.359')):
        D, P, n = _a_cell([x for x in q0 if x['fold'] == 0], model_input=sp, method_setting='detector/share0.1/downweight')
        r.check(f'qm9 L0 dnn fold 0 {sp} untreated', tD, D)
        r.check(f'qm9 L0 dnn fold 0 {sp} detector share 0.1 downweight', tP, P)
    for fo in sorted({x['fold'] for x in rs} - {0}):
        fr = [x for x in rs if x['fold'] == fo]
        parts = []
        for sp in ('base_1', 'hybrid'):
            D, P, n = _a_cell(fr, model_input=sp, method_setting='detector/share0.1/downweight')
            _, C, _ = _a_cell(fr, model_input=sp, method_setting='flags_share0.1_p0/seed0/downweight')
            if D is not None:
                parts.append(f'{sp} {D:.4f} -> detector {P:.4f}, no-damaged-flags control {C}')
        r.new_fold('n31 herg_ki outlier_only_p05 dnn downweight share 0.1', fo, '; '.join(parts))
    for ds in datasets_with_files('n31_regime_and_breakeven', where=('noise', 'full')):
        if ds == 'herg_ki':
            continue
        rr = [x for x in rows('n31_regime_and_breakeven', f'{ds}__*.rows.jsonl', where=('noise',), method='downweight')
              if x['fold'] == 0 and x['level'] > 0]
        if not rr:
            r.new_dataset(ds, 'n31 file without rows (summary json only)')
            continue
        models = sorted({x['model'] for x in rr})
        m = 'dnn' if 'dnn' in models else models[0]
        lines = []
        for kind in sorted({x['noise_condition'] for x in rr}):
            k = [x for x in rr if x['model'] == m and x['noise_condition'] == kind]
            L = _a_main_level(kind, sorted({x['level'] for x in k}))
            k = [x for x in k if x['level'] == L]
            parts = []
            for sp in ('base_1', 'hybrid'):
                D, P, n = _a_cell(k, model_input=sp, method_setting='detector/share0.1/downweight')
                _, C, _ = _a_cell(k, model_input=sp, method_setting='flags_share0.1_p0/seed0/downweight')
                if D is not None and P is not None:
                    parts.append(f'{sp} {D:.4f} -> {P:.4f} (no-damaged control {"n/a" if C is None else f"{C:.4f}"})')
            if parts:
                lines.append(f'{kind} L{L} {m}: ' + '; '.join(parts))
        r.new_dataset(ds, ' | '.join(lines) if lines else 'no downweight rows at a damaged level')


# ---------------------------------------------------------------------------------------------
# N7-07 to N7-12: weighting, QM9 deletion, labels as delivered, cliff put-back (n13, n21, n35, n07, n18, n24, n25)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# N7-07 to N7-12 (fragment B): weighting (n13, n21, n35, QM9), deletion on QM9 (n07), labels as
# delivered (n18, n24, n25).
# ---------------------------------------------------------------------------------------------
def _B_cfg(mi):
    """'dnn:base_1' / 'xgboost@base_1' / 'base_1' -> 'base_1' (the '@clean_label_features' tail kept)."""
    return re.split(r'[:@]', str(mi), maxsplit=1)[-1] if re.match(r'^[a-z_]+[:@]', str(mi)) else str(mi)


def _B_better(lib):
    return (lambda a, b: a < b) if lib else (lambda a, b: a > b)


def _B_best_setting(rs, all_draws=True):
    """rs: rows of one model and configuration. Group by (method, method_setting,
    similarity_setting); keep groups present on every draw in rs (if all_draws); return
    (mean score_D, mean score_Dprime, setting) of the group with the best mean score_Dprime."""
    if not rs:
        return None
    lib = rs[0].get('lower_is_better', True)
    draws = {r.get('draw') for r in rs}
    g = collections.defaultdict(list)
    for r in rs:
        g[(r.get('method'), r.get('method_setting'), r.get('similarity_setting'))].append(r)
    cand = []
    for k, v in g.items():
        if all_draws and {r.get('draw') for r in v} != draws:
            continue
        cand.append((mean([r['score_D'] for r in v]), mean([r['score_Dprime'] for r in v]), k))
    if not cand:
        return None
    return (min if lib else max)(cand, key=lambda t: t[1])


def _B_best_single_hybrid(rs, model):
    """'single base_X D->P | hybrid Y D->P' over best settings (all draws), best treated config."""
    out = []
    lib = rs[0].get('lower_is_better', True) if rs else True
    for grp, cfgs in (('single', ('base_1', 'base_2')), ('hybrid', ('hybrid', 'hybrid__general2'))):
        best = None
        for c in cfgs:
            sel = [r for r in rs if r.get('model') == model and _B_cfg(r.get('model_input')) == c]
            b = _B_best_setting(sel) or _B_best_setting(sel, all_draws=False)
            if b and (best is None or _B_better(lib)(b[1], best[1][1])):
                best = (c, b)
        if best:
            out.append(f'{grp} {best[0]} {best[1][0]:.4f} -> {best[1][1]:.4f}')
    return '; '.join(out) or 'no rows'


def _B_rows_desc(sub, ds):
    """One-line description of what a dataset has in an experiment folder."""
    fl = []
    for w in ('noise', 'full', 'view'):
        fl += [f'{w}/{f}' for f in files(CTX.path(w, sub), ds + '*') if 'fingerprint' not in f and f.split('__')[0].split('.')[0] == ds]
    rs = rows(sub, ds + '__*.rows.jsonl', where=('noise', 'full', 'view'), dataset=ds)
    if not rs:
        return f'{len(fl)} files ({", ".join(fl[:4])}{" ..." if len(fl) > 4 else ""}), no rows'
    conds = sorted({str(r.get('noise_condition')) for r in rs})
    return (f'{len(fl)} files; rows: conditions {",".join(conds)}; models {",".join(sorted({str(r.get("model")) for r in rs}))}; '
            f'levels {sorted({r.get("level") for r in rs})}; folds {sorted({r.get("fold") for r in rs})}')


def _B_new_datasets(r, sub, mentioned, describe=None):
    for ds in datasets_with_files(sub):
        if ds in mentioned:
            continue
        try:
            txt = describe(ds) if describe else _B_rows_desc(sub, ds)
        except Exception as e:  # never crash on a new layout
            txt = f'could not summarise ({e!r}); {_B_rows_desc(sub, ds)}'
        r.new_dataset(ds, f'[{sub}] {txt}')


def _B_new_folds(r, label, rs, cited, fn):
    """For rows rs of one cited quantity, print NEW FOLD for each fold not in cited."""
    for f in sorted({x.get('fold') for x in rs} - set(cited), key=lambda x: (x is None, x)):
        r.new_fold(label, f, fn([x for x in rs if x.get('fold') == f]))


@check('N7-07', 'Down-weighting flagged molecules (n13, n21, n35; QM9 limit)')
def check_n7_07(r):
    """Section 'Changing a Label Against Deleting a Molecule', paragraph 'Making flagged molecules
    count less is the gentlest action'.
    n13 rows (herg, confusion_directed 0.1, xgboost, local_variance_ratio): hybrid = k=10 inverse_square
      exponential_0.1 over forest_proximity, mean over draws; single = base_1, the one setting
      present on all three draws (k=20 inverse_square, hybrid_similarity manhattan).
    n21 rows (logd outlier_p05 level 1, dnn) and n35 rows (buchwald gaussian level 1, lightgbm):
      best setting by mean treated score among settings present on every draw.
    QM9 level 0: n21 qm9__censoring rows best setting; n13 qm9__censoring.json
      per_noise_type.censoring.per_strength['0.0'].track_b[<detector fit>][<estimator>][<model>@<config>].denoised,
      minimum over detector fits and estimators."""
    E = {
        'herg n13 xgboost hybrid untreated': '0.8028', 'herg n13 xgboost hybrid weighted': '0.8155',
        'herg n13 xgboost base_1 untreated': '0.8060', 'herg n13 xgboost base_1 weighted': '0.8063',
        'logd n21 outlier dnn base_1 untreated': '0.6273', 'logd n21 outlier dnn base_1 weighted': '0.5915',
        'logd n21 outlier dnn hybrid untreated': '0.5881', 'logd n21 outlier dnn hybrid weighted': '0.5560',
        'buchwald_hartwig n35 gaussian lightgbm hybrid untreated': '22.0472',
        'buchwald_hartwig n35 gaussian lightgbm hybrid weighted': '20.6614',
        'qm9 L0 n21 dnn hybrid__general2 untreated': '0.362', 'qm9 L0 n21 dnn hybrid__general2 neighbour weights': '0.348',
        'qm9 L0 n13 dnn hybrid__general2 detector weights': '0.3488',
        'qm9 L0 n21 lightgbm hybrid untreated': '0.3719', 'qm9 L0 n21 lightgbm hybrid neighbour weights': '0.3707',
        'qm9 L0 n13 lightgbm hybrid detector weights': '0.3724',
    }
    # n13 herg
    rs = rows('n13_noise_distributions', 'herg__*.rows.jsonl', dataset='herg', noise_condition='confusion_directed',
              level=0.1, model='xgboost', method='local_variance_ratio')
    f0 = [x for x in rs if x.get('fold') == 0]
    hy = [x for x in f0 if _B_cfg(x['model_input']) == 'hybrid' and x['method_setting'] == 'k=10|scheme=inverse_square|exponential_0.1'
          and x['similarity_setting'] == 'forest_proximity|none|forest']
    r.check('herg n13 xgboost hybrid untreated', E['herg n13 xgboost hybrid untreated'], mean([x['score_D'] for x in hy]))
    r.check('herg n13 xgboost hybrid weighted', E['herg n13 xgboost hybrid weighted'], mean([x['score_Dprime'] for x in hy]))
    b1 = [x for x in f0 if _B_cfg(x['model_input']) == 'base_1' and x['method_setting'] == 'k=20|scheme=inverse_square|exponential_0.1'
          and x['similarity_setting'] == 'hybrid_similarity|none|manhattan']
    r.check('herg n13 xgboost base_1 untreated', E['herg n13 xgboost base_1 untreated'], mean([x['score_D'] for x in b1]),
            note='mean over 3 draws')
    r.check('herg n13 xgboost base_1 weighted', E['herg n13 xgboost base_1 weighted'], mean([x['score_Dprime'] for x in b1]))
    _B_new_folds(r, 'herg n13 xgboost hybrid k10 forest', [x for x in rs if _B_cfg(x['model_input']) == 'hybrid'], {0},
                 lambda g: _B_best_single_hybrid(g, 'xgboost'))
    # n21 logd
    rs = rows('n21_neighbour_distance', 'logd__outlier_p05__*.rows.jsonl', dataset='logd', level=1.0, model='dnn')
    for cfg in ('base_1', 'hybrid'):
        b = _B_best_setting([x for x in rs if x.get('fold') == 0 and _B_cfg(x['model_input']) == cfg])
        r.check(f'logd n21 outlier dnn {cfg} untreated', E[f'logd n21 outlier dnn {cfg} untreated'], b and b[0])
        r.check(f'logd n21 outlier dnn {cfg} weighted', E[f'logd n21 outlier dnn {cfg} weighted'], b and b[1],
                note=b and f'{b[2][0]} {b[2][1]} {b[2][2]}')
    _B_new_folds(r, 'logd n21 outlier dnn', rs, {0}, lambda g: _B_best_single_hybrid(g, 'dnn'))
    # n35 buchwald
    rs = rows('n35_near_duplicate_disagreement', 'buchwald_hartwig__gaussian.rows.jsonl', dataset='buchwald_hartwig',
              level=1.0, model='lightgbm', method_setting='close_percentile=1.0;weight=0.25')
    b = _B_best_setting([x for x in rs if x.get('fold') == 0 and _B_cfg(x['model_input']) == 'hybrid'])
    r.check('buchwald_hartwig n35 gaussian lightgbm hybrid untreated', E['buchwald_hartwig n35 gaussian lightgbm hybrid untreated'], b and b[0])
    r.check('buchwald_hartwig n35 gaussian lightgbm hybrid weighted', E['buchwald_hartwig n35 gaussian lightgbm hybrid weighted'], b and b[1],
            note=b and b[2][2])
    _B_new_folds(r, 'buchwald_hartwig n35 gaussian lightgbm', rs, {0}, lambda g: _B_best_single_hybrid(g, 'lightgbm'))
    # QM9, level 0, n35 near-identical pairs (rows added 2026-10-10 from the 10-10b results), best setting
    rs = rows('n35_near_duplicate_disagreement', 'qm9__gaussian.rows.jsonl', dataset='qm9', level=0.0)
    for model, cfg, tD, tP in (('dnn', 'hybrid__general2', '0.362', '0.350'), ('lightgbm', 'hybrid', '0.372', '0.390')):
        b = _B_best_setting([x for x in rs if x.get('fold') == 0 and x.get('model') == model and _B_cfg(x['model_input']) == cfg])
        r.check(f'qm9 L0 n35 {model} {cfg} untreated', tD, b and b[0])
        r.check(f'qm9 L0 n35 {model} {cfg} near-identical pairs weighted', tP, b and b[1], note=b and b[2][2])
    # QM9, level 0
    rs = rows('n21_neighbour_distance', 'qm9__censoring__*.rows.jsonl', dataset='qm9', level=0.0)
    for model, cfg in (('dnn', 'hybrid__general2'), ('lightgbm', 'hybrid')):
        b = _B_best_setting([x for x in rs if x.get('fold') == 0 and x.get('model') == model and _B_cfg(x['model_input']) == cfg])
        r.check(f'qm9 L0 n21 {model} {cfg} untreated', E[f'qm9 L0 n21 {model} {cfg} untreated'], b and b[0])
        r.check(f'qm9 L0 n21 {model} {cfg} neighbour weights', E[f'qm9 L0 n21 {model} {cfg} neighbour weights'], b and b[1])
    _B_new_folds(r, 'qm9 L0 n21', rs, {0}, lambda g: 'dnn ' + _B_best_single_hybrid(g, 'dnn') + ' | lightgbm ' + _B_best_single_hybrid(g, 'lightgbm'))
    d = J('noise', 'n13_noise_distributions', 'qm9__censoring.json')
    tb = None
    try:
        tb = d['per_noise_type']['censoring']['per_strength']['0.0']['track_b']
    except Exception:
        r.missing('qm9 n13 track_b', 'n13_noise_distributions/qm9__censoring.json per_strength 0.0 track_b not found')
    if tb is not None:
        for model, cfg in (('dnn', 'hybrid__general2'), ('lightgbm', 'hybrid')):
            vals = [y['denoised'] for k, v in tb.items() if k.startswith(model + ':') for est, x in v.items()
                    for mk, y in x.items() if mk == f'{model}@{cfg}' and isinstance(y, dict) and y.get('denoised') is not None]
            r.check(f'qm9 L0 n13 {model} {cfg} detector weights', E[f'qm9 L0 n13 {model} {cfg} detector weights'],
                    min(vals) if vals else None, note='minimum over detector fits and estimators')
        if (d.get('folds_run') or [0]) != [0]:
            r.new_fold('qm9 L0 n13 track_b', d.get('folds_run'), 'folds_run is no longer [0]; track_b values are means over these folds')
    mentioned = {'herg', 'logd', 'buchwald_hartwig', 'qm9'}
    _B_new_datasets(r, 'n13_noise_distributions', mentioned)
    _B_new_datasets(r, 'n21_neighbour_distance', mentioned, lambda ds: '; '.join(
        f'{c} dnn {_B_best_single_hybrid([x for x in rows("n21_neighbour_distance", ds + "__*.rows.jsonl", dataset=ds, fold=0) if x.get("noise_condition") == c and x.get("level", 0) > 0], "dnn")}'
        for c in sorted({str(x.get('noise_condition')) for x in rows('n21_neighbour_distance', ds + '__*.rows.jsonl', dataset=ds)} - {'None'})) or _B_rows_desc('n21_neighbour_distance', ds))
    def n35desc(ds):
        rs_ = rows('n35_near_duplicate_disagreement', ds + '__*.rows.jsonl', where=('noise', 'full', 'view'), dataset=ds, level=1.0, fold=0)
        if not rs_:
            return _B_rows_desc('n35_near_duplicate_disagreement', ds)
        return 'gaussian level 1: ' + ' | '.join(f'{m} {_B_best_single_hybrid(rs_, m)}' for m in sorted({x["model"] for x in rs_}))
    _B_new_datasets(r, 'n35_near_duplicate_disagreement', mentioned, n35desc)


@check('N7-08', 'Deleting on clean QM9 labels (n07)')
def check_n7_08(r):
    """Section 'Changing a Label Against Deleting a Molecule', paragraph 'Deleting never helped on
    clean labels'. n07 qm9__<kind>__fold_aligned rows, method remove, level 0, fold 0: per model and
    configuration the best treated score over every kind file and flag share; the untreated score
    is the one in the same file. Also counts configurations where the best deletion helped (text: none)."""
    E = {'qm9 L0 n07 dnn hybrid untreated': '0.355', 'qm9 L0 n07 dnn hybrid best deletion': '0.393',
         'qm9 L0 n07 lightgbm hybrid untreated': '0.3719', 'qm9 L0 n07 lightgbm hybrid best deletion': '0.4106'}
    rs = rows('n07_remediation', 'qm9__*.rows.jsonl', where=('noise', 'full'), dataset='qm9', level=0.0, method='remove')
    if not rs:
        for k, v in E.items():
            r.check(k, v, None, note='no n07 qm9 rows')
        return
    best = {}
    for x in rs:
        if x.get('fold') != 0:
            continue
        k = (x['model'], _B_cfg(x['model_input']))
        if k not in best or x['score_Dprime'] < best[k]['score_Dprime']:
            best[k] = x
    for model in ('dnn', 'lightgbm'):
        b = best.get((model, 'hybrid'))
        r.check(f'qm9 L0 n07 {model} hybrid untreated', E[f'qm9 L0 n07 {model} hybrid untreated'], b and b['score_D'])
        r.check(f'qm9 L0 n07 {model} hybrid best deletion', E[f'qm9 L0 n07 {model} hybrid best deletion'], b and b['score_Dprime'],
                note=b and f"{b['method_setting']} in {b['_file'].split('__')[1]}")
    helped = [f'{m}:{c}' for (m, c), b in best.items() if b['score_Dprime'] < b['score_D']]
    r.info(f'configurations where the best deletion improved QM9 (text: none): {helped or "none"}')
    # n23 on QM9 with no noise added (rows added 2026-10-10 from the 10-10b results): qm9__outlier_p05.json
    # per_regime.injected['0'].hybrid.<detector>.{none, drop@0.1}.<model>, mean over the stored passes (fold 0, draw 0)
    d23 = J('noise', 'n23_uncertainty_cleaning', 'qm9__outlier_p05.json')
    T23 = {'dnn': ('0.349', {'activity_cliff_aware': '0.392', 'gaussian_process': '0.409', 'heteroscedastic_nn': '0.423',
                             'ngboost': '0.396', 'hetero_gp': '0.371'}),
           'lightgbm': ('0.372', {'activity_cliff_aware': '0.413', 'gaussian_process': '0.429', 'heteroscedastic_nn': '0.452',
                                  'ngboost': '0.466', 'hetero_gp': '0.392'})}
    w23 = ((((d23 or {}).get('per_regime') or {}).get('injected') or {}).get('0') or {})

    def m23(c, det, arm, model):
        v = (((w23.get(c) or {}).get(det) or {}).get(arm) or {}).get(model) or {}
        return mean(list(v.values())) if v else None
    for model, (tn, dets) in T23.items():
        r.check(f'qm9 L0 n23 {model} hybrid no deletion', tn, m23('hybrid', 'activity_cliff_aware', 'none', model))
        for det, t in dets.items():
            r.check(f'qm9 L0 n23 {model} hybrid drop@0.1 {det}', t, m23('hybrid', det, 'drop@0.1', model))
    gains = []
    for c in ('base_1', 'base_2', 'hybrid', 'hybrid__general2'):
        for model in ('dnn', 'lightgbm'):
            for det in T23['dnn'][1]:
                a0, a1 = m23(c, det, 'none', model), m23(c, det, 'drop@0.1', model)
                if a0 is not None and a1 is not None and a1 < a0:
                    gains.append((c, model, det, round(a0 - a1, 4)))
    r.claim('n23 QM9 drop@0.1: no hybrid gains; every gain 0.004 RMSE or less',
            (all(g[0] not in ('hybrid', 'hybrid__general2') and g[3] <= 0.0045 for g in gains)) if w23 else None,
            f'gains: {gains}')
    for f in sorted({x.get('fold') for x in rs} - {0}):
        g = [x for x in rs if x.get('fold') == f]
        parts = []
        for model in ('dnn', 'lightgbm'):
            h = [x for x in g if x['model'] == model and _B_cfg(x['model_input']) == 'hybrid']
            if h:
                b = min(h, key=lambda x: x['score_Dprime'])
                parts.append(f"{model} hybrid {b['score_D']:.4f} -> {b['score_Dprime']:.4f}")
        r.new_fold('qm9 L0 n07 hybrid best deletion', f, '; '.join(parts) or 'rows but no hybrid')


def _B_n18(ds, space, det, slot, model, cleanup):
    """n18 per_space.<space>.per_detector.<det>.<slot>.per_model.<model>.{no_cleaning_mean, <cleanup>.cleaned_mean}
    and shuffled_control.<det>.<slot>.<model>.<cleanup>.<seed>.shuffled_mean. Returns (nc, cleaned, {seed: shuffled}, folds_run)."""
    d = J('noise', 'n18_clean_the_real_labels', ds + '.json')
    if d is None:
        return None
    try:
        v = d['per_space'][space]
        m = v['per_detector'][det][slot]['per_model'][model]
        nc, cl = m.get('no_cleaning_mean'), (m.get(cleanup) or {}).get('cleaned_mean')
        sh = ((v.get('shuffled_control') or {}).get(det, {}).get(slot, {}).get(model, {}).get(cleanup, {}))
        return nc, cl, {s: x.get('shuffled_mean') for s, x in sh.items()}, d.get('folds_run')
    except Exception:
        return None


@check('N7-09', 'Labels as delivered, shuffled-score control (n18)')
def check_n7_09(r):
    """Subsection 'The labels as delivered', first paragraph. n18_clean_the_real_labels/<ds>.json,
    per_space.<config>.per_detector.<det>.<slot>.per_model.<model>.{no_cleaning_mean, <cleanup>.cleaned_mean};
    shuffled control per_space.<config>.shuffled_control.<det>.<slot>.<model>.<cleanup>.<seed>.shuffled_mean."""
    # (label, ds, space, detector, slot, model, cleanup, text untreated, text cleaned, shuffles: list of texts or ('min'/'max', text))
    SPEC = [
        ('herg_ki dnn base_1 neighbor_residual label_smooth', 'herg_ki', 'base_1', 'neighbor_residual', '0.2', 'dnn', 'label_smooth',
         '0.638', '0.578', ['0.6420', '0.6419', '0.6455']),
        ('herg_ki dnn hybrid cross_val label_smooth', 'herg_ki', 'hybrid', 'cross_val', '0.05', 'dnn', 'label_smooth',
         '0.5989', '0.5783', ['0.5998', '0.5877', '0.5960']),
        ('herg xgboost base_1 neighbor_residual label_smooth', 'herg', 'base_1', 'neighbor_residual', '0.01', 'xgboost', 'label_smooth',
         '0.8369', '0.8455', ('best', '0.8350')),
        ('herg xgboost hybrid__general2 neighbor_residual label_smooth', 'herg', 'hybrid__general2', 'neighbor_residual', '0.01', 'xgboost',
         'label_smooth', '0.8545', '0.8600', ('best', '0.8550')),
        ('caco2 dnn base_2 cross_val downweight', 'caco2', 'base_2', 'cross_val', '0.2', 'dnn', 'downweight', '0.2657', '0.2548', ('best', '0.2581')),
        ('logd dnn base_2 neighbor_residual downweight', 'logd', 'base_2', 'neighbor_residual', '0.05', 'dnn', 'downweight',
         '0.4212', '0.4143', ('best', '0.4172')),
    ]
    for lab, ds, sp, det, slot, model, cl, t_nc, t_cl, t_sh in SPEC:
        x = _B_n18(ds, sp, det, slot, model, cl)
        lab = f'{lab} slot {slot}'
        if x is None:
            r.missing(lab, f'n18_clean_the_real_labels/{ds}.json or key per_space.{sp}.per_detector.{det}.{slot}.per_model.{model} not found')
            continue
        nc, cv, sh, folds = x
        r.check(lab + ' no cleaning', t_nc, nc)
        r.check(lab + ' cleaned', t_cl, cv)
        lib = J('noise', 'n18_clean_the_real_labels', ds + '.json').get('lower_is_better', True)
        if isinstance(t_sh, list):
            for i, t in enumerate(t_sh):
                r.check(f'{lab} shuffled seed {i}', t, sh.get(str(i)))
        else:
            vals = [v for v in sh.values() if v is not None]
            r.check(f'{lab} best shuffle', t_sh[1], (min(vals) if lib else max(vals)) if vals else None)
        if folds and list(folds) != [0]:
            r.new_fold(lab, folds, f'folds_run now {folds}; the values above are means over them: {nc:.4f} -> {cv:.4f}')
    # New datasets: same cleanup grid, count of (config, detector, slot, model, cleanup) that beat
    # no cleaning and every shuffle.
    def desc(ds):
        d = J('noise', 'n18_clean_the_real_labels', ds + '.json')
        if not d or 'per_space' not in d:
            return _B_rows_desc('n18_clean_the_real_labels', ds)
        bet = _B_better(d.get('lower_is_better', True)); n = w = 0; best = None
        for sp, v in d['per_space'].items():
            for det, ww in (v.get('per_detector') or {}).items():
                for slot, x in ww.items():
                    for model, m in (x.get('per_model') or {}).items():
                        for cl, c in m.items():
                            if not isinstance(c, dict) or c.get('cleaned_mean') is None:
                                continue
                            sh = (v.get('shuffled_control') or {}).get(det, {}).get(slot, {}).get(model, {}).get(cl, {})
                            shv = [s.get('shuffled_mean') for s in sh.values() if s.get('shuffled_mean') is not None]
                            if not shv:
                                continue
                            n += 1
                            if bet(c['cleaned_mean'], m['no_cleaning_mean']) and all(bet(c['cleaned_mean'], s) for s in shv):
                                w += 1
        return f'{w} of {n} (configuration, detector, flag slot, model, cleanup) beat no cleaning and every shuffle; folds_run {d.get("folds_run")}'
    _B_new_datasets(r, 'n18_clean_the_real_labels', {'herg_ki', 'herg', 'caco2', 'logd'}, desc)


def _B_n24_arm(d, level, space, det, rate, arm, model):
    """n24 per_level[level][space][det][rate][arm][model] -> {fold: score} from keys l<L>d<d>f<f>."""
    try:
        x = d['per_level'][level][space][det][rate][arm][model]
    except Exception:
        return {}
    out = collections.defaultdict(list)
    for k, v in x.items():
        m = re.match(r'l[\d.]+d(\d+)f(\d+)', k)
        if m and v is not None:
            out[int(m.group(2))].append(v)
    return {f: mean(v) for f, v in out.items()}


@check('N7-10', 'hERG Ki label smoothing on three folds (n24)')
def check_n7_10(r):
    """Subsection 'The labels as delivered', paragraph 'The hERG Ki smoothing gain on the hybrid holds on
    three folds'. n24_delete_or_keep/herg_ki__outlier_p05.json (noise: fold 0; noise_full: folds 1, 2),
    per_level['0'][config][det][rate]['keep_rewrite/label_smooth'].dnn['l0d0f<f>'].
    Hybrid: detector cross_val@screen3, rate 0.05 (as the SOURCE says). Second single representation:
    the best label_smooth over every detector and screen on each fold (that is what the text's base_2
    numbers are; the cross_val@screen3 values differ on folds 0 and 2)."""
    E = {('hybrid', 0): ('0.5989', '0.5671'), ('hybrid', 1): ('0.5500', '0.5467'), ('hybrid', 2): ('0.5495', '0.5370'),
         ('base_2', 0): ('0.5892', '0.5965'), ('base_2', 1): ('0.6102', '0.6003'), ('base_2', 2): ('0.6452', '0.6291')}
    docs = [d for d in (J('noise', 'n24_delete_or_keep', 'herg_ki__outlier_p05.json'),
                        J('full', 'n24_delete_or_keep', 'herg_ki__outlier_p05.json')) if d]
    nc, hy, b2 = {}, {}, {}
    for d in docs:
        for sp in ('hybrid', 'base_2'):
            for f, v in _B_n24_arm(d, '0', sp, 'cross_val@screen3', '0.05', 'no_cleaning', 'dnn').items():
                nc[(sp, f)] = v
        hy.update(_B_n24_arm(d, '0', 'hybrid', 'cross_val@screen3', '0.05', 'keep_rewrite/label_smooth', 'dnn'))
        for det, ww in ((d.get('per_level') or {}).get('0', {}).get('base_2') or {}).items():
            for rate in ww:
                for f, v in _B_n24_arm(d, '0', 'base_2', det, rate, 'keep_rewrite/label_smooth', 'dnn').items():
                    if f not in b2 or v < b2[f][0]:
                        b2[f] = (v, det)
    for (sp, f), (t0, t1) in E.items():
        r.check(f'herg_ki L0 dnn {sp} fold {f} no cleaning', t0, nc.get((sp, f)))
        val = hy.get(f) if sp == 'hybrid' else (b2.get(f) or (None,))[0]
        note = 'cross_val@screen3 0.05' if sp == 'hybrid' else (b2.get(f) and f'best: {b2[f][1]}')
        r.check(f'herg_ki L0 dnn {sp} fold {f} label_smooth', t1, val, note=note or '')
    for f in sorted(set(hy) | set(b2) | {k[1] for k in nc}):
        if f in (0, 1, 2):
            continue
        r.new_fold('herg_ki L0 dnn label_smooth', f,
                   f"hybrid {nc.get(('hybrid', f), float('nan')):.4f} -> {hy.get(f, float('nan')):.4f}; "
                   f"base_2 {nc.get(('base_2', f), float('nan')):.4f} -> {(b2.get(f) or (float('nan'),))[0]:.4f}")

    def desc(ds):
        out = []
        for w in ('noise', 'full'):
            for fn in files(CTX.path(w, 'n24_delete_or_keep'), ds + '__*.json'):
                if 'fingerprint' in fn:
                    continue
                d = J(w, 'n24_delete_or_keep', fn)
                h = _B_n24_arm(d or {}, '0', 'hybrid', 'cross_val@screen3', '0.05', 'keep_rewrite/label_smooth', 'dnn')
                n0 = _B_n24_arm(d or {}, '0', 'hybrid', 'cross_val@screen3', '0.05', 'no_cleaning', 'dnn')
                for f in sorted(h):
                    out.append(f'{w}/{fn} f{f} hybrid {n0.get(f, float("nan")):.4f} -> {h[f]:.4f}')
        return ('dnn L0 label_smooth cross_val@screen3 0.05: ' + '; '.join(out)) if out else _B_rows_desc('n24_delete_or_keep', ds)
    # datasets with further folds in noise_full n24 (the paragraph is about extra folds)
    for ds in datasets_with_files('n24_delete_or_keep', where=('full',)):
        if ds != 'herg_ki':
            r.new_dataset(ds, '[n24 noise_full] ' + desc(ds))


@check('N7-11', 'Deletion against random deletion on labels as delivered (n24)')
def check_n7_11(r):
    """Subsection 'The labels as delivered', paragraph 'Deleting does not pass its control on the
    small-molecule labels as delivered'. n24_delete_or_keep/<ds>__<kind>.json per_level['0']: count of
    deletion settings (config x detector x flag rate x model, 96 per file) where delete/remove beats
    no cleaning and all five delete_at_random seeds (first kind file per dataset; level 0 is the same
    in every kind file). Buchwald-Hartwig: buchwald_hartwig__censoring.json per_level['0'].
    {hybrid,base_2}.neighbor_residual@screen1['0.2'].{remove, delete_at_random}.lightgbm; base_1
    delete_at_random/seed3 at cross_val 0.05."""
    COUNTS = {'caco2': '0', 'herg': '1', 'herg_ki': '0', 'logd': '0', 'adme_mdr1': '1', 'buchwald_hartwig': '31'}
    D = CTX.path('noise', 'n24_delete_or_keep')

    def count(ds):
        fl = [f for f in files(D, ds + '__*.json') if 'fingerprint' not in f and f.split('__')[0] == ds]
        if not fl:
            return None, None, None
        d = J('noise', 'n24_delete_or_keep', fl[0]) or {}
        pl = (d.get('per_level') or {}).get('0') or {}
        bet = _B_better(d.get('lower_is_better', True)); n = w = 0
        for sp, v in pl.items():
            for det, ww in v.items():
                for rate, x in ww.items():
                    if 'delete/remove' not in x:
                        continue
                    for m, ncv in x['no_cleaning'].items():
                        rm = (x['delete/remove'] or {}).get(m)
                        if not rm:
                            continue
                        a, b = mean(list(rm.values())), mean(list(ncv.values()))
                        rnd = [mean(list(x[k][m].values())) for k in x if k.startswith('delete_at_random') and m in x[k]]
                        n += 1
                        w += bet(a, b) and len(rnd) == 5 and all(bet(a, q) for q in rnd)
        return w, n, (fl[0], d.get('folds_run'))
    for ds, t in COUNTS.items():
        w, n, info = count(ds)
        r.check(f'{ds} L0 deletion settings beating no cleaning and all 5 random seeds', t, w,
                note=info and f'of {n}; {info[0]}, folds {info[1]}')
    # MDR1 (sentence added 2026-10-10): the one setting that passes, hybrid neighbor_residual@screen1 0.05, dnn
    dm = J('noise', 'n24_delete_or_keep', 'adme_mdr1__censoring.json') or {}
    xm = ((((dm.get('per_level') or {}).get('0') or {}).get('hybrid') or {}).get('neighbor_residual@screen1') or {}).get('0.05') or {}
    if xm:
        r.check('adme_mdr1 L0 dnn hybrid neighbor_residual@screen1 0.05 no cleaning', '0.466', mean(list(xm['no_cleaning']['dnn'].values())))
        r.check('adme_mdr1 L0 dnn hybrid neighbor_residual@screen1 0.05 delete', '0.446', mean(list(xm['delete/remove']['dnn'].values())))
        r.check('adme_mdr1 L0 dnn hybrid neighbor_residual@screen1 0.05 best random', '0.450',
                min(mean(list(xm[k]['dnn'].values())) for k in xm if k.startswith('delete_at_random')))
    else:
        r.missing('adme_mdr1 L0 n24', 'adme_mdr1__censoring.json per_level 0 hybrid neighbor_residual@screen1 0.05 not found')
    d = J('noise', 'n24_delete_or_keep', 'buchwald_hartwig__censoring.json')
    E = [('hybrid', 'neighbor_residual@screen1', '0.2', 'no_cleaning', '20.0823'), ('hybrid', 'neighbor_residual@screen1', '0.2', 'delete/remove', '18.0'),
         ('hybrid', 'neighbor_residual@screen1', '0.2', 'best random', '19.9'),
         ('base_2', 'neighbor_residual@screen1', '0.2', 'no_cleaning', '20.2242'), ('base_2', 'neighbor_residual@screen1', '0.2', 'delete/remove', '18.3284'),
         ('base_2', 'neighbor_residual@screen1', '0.2', 'best random', '19.9173'),
         ('base_1', 'cross_val', '0.05', 'no_cleaning', '19.4875'), ('base_1', 'cross_val', '0.05', 'delete_at_random/seed3', '18.8663')]
    for sp, det, rate, arm, t in E:
        lab = f'buchwald_hartwig L0 lightgbm {sp} {det} {rate} {arm} fold 0'
        if d is None:
            r.check(lab, t, None, note='buchwald_hartwig__censoring.json missing'); continue
        if arm == 'best random':
            vals = [(_B_n24_arm(d, '0', sp, det, rate, f'delete_at_random/seed{s}', 'lightgbm') or {}).get(0) for s in range(5)]
            vals = [v for v in vals if v is not None]
            r.check(lab, t, min(vals) if vals else None)
        else:
            r.check(lab, t, _B_n24_arm(d, '0', sp, det, rate, arm, 'lightgbm').get(0))
    if d:
        fs = set(_B_n24_arm(d, '0', 'hybrid', 'neighbor_residual@screen1', '0.2', 'no_cleaning', 'lightgbm')) - {0}
        for f in sorted(fs):
            g = lambda arm, sp='hybrid': _B_n24_arm(d, '0', sp, 'neighbor_residual@screen1', '0.2', arm, 'lightgbm').get(f, float('nan'))
            r.new_fold('buchwald_hartwig L0 lightgbm hybrid nr@screen1 0.2', f,
                       f"{g('no_cleaning'):.4f} -> remove {g('delete/remove'):.4f}, best random "
                       f"{min(g(f'delete_at_random/seed{s}') for s in range(5)):.4f}")
    _B_new_datasets(r, 'n24_delete_or_keep', set(COUNTS),
                    lambda ds: (lambda c: f'{c[0]} of {c[1]} deletion settings beat no cleaning and all 5 random seeds ({c[2][0]})'
                                if c[0] is not None else _B_rows_desc('n24_delete_or_keep', ds))(count(ds)))


@check('N7-12', 'Putting cliff molecules back (n25)')
def check_n7_12(r):
    """Subsection 'The labels as delivered', paragraph 'Putting activity-cliff molecules back after a
    cleanup can be tested on hERG Ki only'. n25_cliff_exemption/herg_ki__censoring__fold_aligned rows
    (noise: fold 0; noise_full: folds 1, 2), level 0, dnn:hybrid, method remove,
    detector=neighbor_residual|flag_rate=0.01|k=10|scheme=inverse_square, arms plain / cliff_exempt /
    random_exempt (one similarity setting per fold). Counts: <ds>__*.json per_pass_counts at level 0,
    fold 0, draw 0 (n_cliff_molecules_in_training, n_training_molecules, n_random_control_molecules)."""
    E = {(0, 'untreated'): '0.5972', (0, 'plain'): '0.5960', (0, 'cliff_exempt'): '0.579', (0, 'random_exempt'): '0.595',
         (2, 'cliff_exempt'): '0.5370', (2, 'random_exempt'): '0.5453', (1, 'cliff_exempt'): '0.5684', (1, 'random_exempt'): '0.5572'}
    SET = 'detector=neighbor_residual|flag_rate=0.01|k=10|scheme=inverse_square'
    rs = rows('n25_cliff_exemption', 'herg_ki__censoring*.rows.jsonl', where=('noise', 'full'), dataset='herg_ki',
              level=0.0, model='dnn', method='remove')
    rs = [x for x in rs if _B_cfg(x['model_input']) == 'hybrid' and x['method_setting'].startswith(SET + '|arm=')]
    by = collections.defaultdict(dict)
    for x in rs:
        arm = x['method_setting'].split('arm=')[-1]
        by[x['fold']][arm] = x['score_Dprime']
        by[x['fold']]['untreated'] = x['score_D']
        by[x['fold']].setdefault('_sim', set()).add(x['similarity_setting'])
    for (f, arm), t in E.items():
        r.check(f'herg_ki L0 dnn hybrid remove fold {f} {arm}', t, by.get(f, {}).get(arm))
    for f in sorted(by):
        if len(by[f]['_sim']) > 1:
            r.info(f'fold {f}: {len(by[f]["_sim"])} similarity settings for the cited setting; the last row read is used')
    # QM9 with no noise added (row added 2026-10-10 from the 10-10b results): censoring rows file, same setting
    qr = rows('n25_cliff_exemption', 'qm9__censoring*.rows.jsonl', dataset='qm9', level=0.0, model='dnn', method='remove')
    qb = {}
    for x in qr:
        if x.get('fold') == 0 and _B_cfg(x['model_input']) == 'hybrid' and x['method_setting'].startswith(SET + '|arm='):
            qb[x['method_setting'].split('arm=')[-1]] = x['score_Dprime']; qb['untreated'] = x['score_D']
    for arm, t in (('untreated', '0.349'), ('plain', '0.382'), ('cliff_exempt', '0.372'), ('random_exempt', '0.382')):
        r.check(f'qm9 L0 dnn hybrid remove fold 0 {arm}', t, qb.get(arm))
    for f in sorted(set(by) - {0, 1, 2}):
        b = by[f]
        r.new_fold('herg_ki L0 dnn hybrid remove cliff put-back', f,
                   f"untreated {b.get('untreated', float('nan')):.4f}, deleted {b.get('plain', float('nan')):.4f}, "
                   f"cliff back {b.get('cliff_exempt', float('nan')):.4f}, random back {b.get('random_exempt', float('nan')):.4f}")
    C_E = {'caco2': ('1,241', '1,726', '485'), 'herg': ('3,345', '6,206', '2,861'), 'logd': ('2,663', '4,022', '1,359'),
           'herg_ki': ('365', '1,129', '365')}

    def counts(ds):
        for w in ('noise', 'full'):
            for fn in files(CTX.path(w, 'n25_cliff_exemption'), ds + '__*.json'):
                if 'fingerprint' in fn or fn.split('__')[0] != ds:
                    continue
                d = J(w, 'n25_cliff_exemption', fn) or {}
                for p in d.get('per_pass_counts') or []:
                    if p.get('fold') == 0 and float(p.get('level', -1)) == 0 and p.get('draw', 0) == 0:
                        return p, fn
        return None, None
    shares = {}
    for ds, (tc, tn, tp) in C_E.items():
        p, fn = counts(ds)
        r.check(f'{ds} L0 fold 0 cliff molecules in training', tc, p and p.get('n_cliff_molecules_in_training'), note=fn or '')
        r.check(f'{ds} L0 fold 0 training molecules', tn, p and p.get('n_training_molecules'))
        r.check(f'{ds} L0 fold 0 random control pool', tp, p and p.get('n_random_control_molecules'))
        if p and ds != 'herg_ki':
            shares[ds] = 100 * p['n_cliff_molecules_in_training'] / p['n_training_molecules']
    r.check('caco2/herg/logd lowest cliff share (%)', '54', min(shares.values()) if shares else None)
    r.check('caco2/herg/logd highest cliff share (%)', '72', max(shares.values()) if shares else None)

    def desc(ds):
        p, fn = counts(ds)
        txt = (f'fold 0 L0 {p["n_cliff_molecules_in_training"]} cliff of {p["n_training_molecules"]} training molecules, '
               f'random pool {p["n_random_control_molecules"]}') if p else 'no per_pass_counts'
        return txt + '; ' + _B_rows_desc('n25_cliff_exemption', ds)
    _B_new_datasets(r, 'n25_cliff_exemption', set(C_E), desc)


# ---------------------------------------------------------------------------------------------
# N7-13 to N7-19: which detectors find the suspect labels (n07, n21, n12, n23, n32, n38, n41, n31)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Section "Results: Which Detectors Find the Suspect Labels" (N7-13 to N7-19)
# ---------------------------------------------------------------------------------------------
FLIP_KINDS = ('uniform', 'binary_asymmetric', 'class_dependent', 'class_imbalance',
              'confusion_directed', 'instance_noise')


def main_level_of(kind):
    """The damaged level the chapter reports for a damage kind (as a float)."""
    return 0.25 if kind == 'censoring' else (0.1 if kind in FLIP_KINDS else 1.0)


def _c_lvkey(x):
    return f'{float(x):g}'


def n07_detection(where, ds, kind):
    """Detection grading of n07 read from <where>/n07_remediation/<ds>__<kind>.units.jsonl
    (the run's per-pass units; same numbers as the final file's by_level[L].detection_by_detector
    .cross_val and detection_by_similarity.activity_cliff_aware, and present for partial runs).
    Returns {(fold, level): {'chance', 'resid', 'passes', 'cliff': {(space|scaling|distance|
    weighting, k): mean AP}}} for damaged levels, or None if the file is missing."""
    p = CTX.path(where, 'n07_remediation', f'{ds}__{kind}.units.jsonl')
    if p is None or PATIENT.search(p) or not os.path.exists(p):
        return None
    key = ('n07det', p)
    if key in CTX._json:
        return CTX._json[key]
    acc = collections.defaultdict(lambda: {'cv': [], 'sim': collections.defaultdict(list)})
    with open(p) as fh:
        for line in fh:
            try:
                u = json.loads(line)
            except Exception:
                continue
            ck = u.get('ckpt_key') or []
            if not ck or ck[0] != 'unit':
                continue
            fo, lv = ck[1], ck[2]
            if float(lv) == 0:
                continue
            v = u.get('value') or {}
            for m, bs in (v.get('detection_by_detector') or {}).items():
                for sp, bd in (bs or {}).items():
                    d = (bd or {}).get('cross_val')
                    if d:
                        acc[(fo, float(lv))]['cv'].append(d)
            for dn, bs in (v.get('sim_detection') or {}).items():
                for sn, bk in (bs or {}).items():
                    for k, rs in (bk or {}).items():
                        acc[(fo, float(lv))]['sim'][(sn, k)].extend(rs or [])
    out = {}
    for k, B in acc.items():
        cv = B['cv']
        sims = {sk: mean([x['detection_average_precision'] for x in rs]) for sk, rs in B['sim'].items() if rs}
        src = cv or [x for rs in B['sim'].values() for x in rs]
        if not src:
            continue
        out[k] = {'chance': mean([d['detection_n_corrupted'] / d['detection_n_training'] for d in src]),
                  'resid': mean([d['detection_average_precision'] for d in cv]) if cv else None,
                  'passes': len(cv), 'cliff': sims}
    CTX._json[key] = out
    return out


def _n07_kinds(where, ds):
    return sorted(f[len(ds) + 2:-len('.units.jsonl')] for f in files(CTX.path(where, 'n07_remediation'), f'{ds}__*.units.jsonl'))


def _n07_extra(r, cited_ds, quantity):
    """NEW FOLD (fold > 0 anywhere) and NEW DATASET lines for the n07 detection paragraphs."""
    for where in ('noise', 'full'):
        for ds in CH7_DATASETS:
            for kind in _n07_kinds(where, ds):
                det = n07_detection(where, ds, kind) or {}
                for (fo, lv), x in sorted(det.items()):
                    if fo != 0 and x['resid'] is not None:
                        r.new_fold(f'{ds} {kind} level {lv:g} [{where}]', fo, quantity(x))
    for ds in datasets_with_files('n07_remediation', where=('noise',)):
        if ds in cited_ds:
            continue
        parts = []
        for kind in _n07_kinds('noise', ds):
            det = n07_detection('noise', ds, kind) or {}
            x = det.get((0, main_level_of(kind)))
            if x and x['resid'] is not None:
                parts.append(f'{kind}@{main_level_of(kind):g} ' + quantity(x))
        r.new_dataset(ds, '; '.join(parts) if parts else 'n07 file, no graded detection at the main level')


@check('N7-13', 'residual detector AP against chance (n07)')
def n7_13(r):
    """Paragraph 1 of 'Which Detectors Find the Suspect Labels': the out-of-fold forest residual
    (n07 detection_by_detector.cross_val.detection_average_precision, mean over passes) against
    chance (n corrupted / n training), fold 0. Read from n07_remediation/<ds>__<kind>.units.jsonl,
    which also covers the logd partial runs."""
    EXPECTED = [  # (dataset, kind, level, residual AP text, chance text)
        ('caco2', 'outlier_p05', 1.0, '0.338', '0.051'),
        ('herg', 'uniform', 0.1, '0.3970', '0.0929'),
        ('buchwald_hartwig', 'censoring', 0.25, '0.1929', '0.2499'),
        ('logd', 'censoring', 0.25, '0.1957', '0.2369'),
        ('caco2', 'censoring', 0.25, '0.3065', '0.2503'),
    ]
    for ds, kind, lv, ap, ch in EXPECTED:
        det = n07_detection('noise', ds, kind)
        x = (det or {}).get((0, lv))
        if x is None:
            r.missing(f'{ds} {kind} level {lv:g} fold 0', 'n07 units file or level missing')
            continue
        r.check(f'{ds} {kind} level {lv:g} fold 0 residual AP', ap, x['resid'])
        r.check(f'{ds} {kind} level {lv:g} fold 0 chance', ch, x['chance'])
    # "well above chance on every dataset it ran on" for outliers, shifted labels and flips
    below = []; n = 0
    for ds in CH7_DATASETS:
        for kind in _n07_kinds('noise', ds):
            if kind not in ('outlier_p05', 'shift_only_p10') + FLIP_KINDS:
                continue
            x = (n07_detection('noise', ds, kind) or {}).get((0, main_level_of(kind)))
            if x and x['resid'] is not None:
                n += 1
                if x['resid'] <= x['chance']:
                    below.append(f'{ds} {kind}')
    r.claim('residual above chance for outliers, shift in 10%, flips (fold 0, main level)', not below if n else None,
            f'{n - len(below)} of {n} dataset x kind above chance' + (f'; not: {", ".join(below)}' if below else ''))
    _n07_extra(r, {'caco2', 'herg', 'buchwald_hartwig', 'logd'},
               lambda x: f'residual AP {x["resid"]:.4f} vs chance {x["chance"]:.4f} ({x["passes"]} passes)')
    # Antibody rows added 2026-10-10 to Table tab:noise:res:detect_residual (printed in the N7-14 block):
    # chance, residual AP, cliff-aware best and median over neighbour settings, fold 0, main level
    BLOCKS[r.id] = BLOCKS.get(r.id, '') + BLOCKS.get('N7-14', '')
    ROWS = [('qm9', 'shift_only_p10', 1.0, '0.101', '0.618', '0.724', '0.599'),  # 16 passes on 10-10b (8 before)
            ('ddg_combined', 'censoring', 0.25, '0.251', '0.289', '0.414', '0.328'),
            ('ddg_combined', 'outlier_p05', 1.0, '0.055', '0.397', '0.508', '0.390'),
            ('ddg_combined', 'shift_only_p10', 1.0, '0.098', '0.148', '0.295', '0.249'),
            ('ginkgo_titer', 'censoring', 0.25, '0.253', '0.362', '0.356', '0.311'),
            ('ginkgo_titer', 'outlier_p05', 1.0, '0.056', '0.378', '0.429', '0.378'),
            ('ginkgo_titer', 'shift_only_p10', 1.0, '0.104', '0.163', '0.193', '0.162'),
            ('ginkgo_pr_cho', 'censoring', 0.25, '0.250', '0.346', '0.428', '0.321'),
            ('ginkgo_pr_cho', 'outlier_p05', 1.0, '0.048', '0.380', '0.420', '0.358'),
            ('ginkgo_pr_cho', 'shift_only_p10', 1.0, '0.101', '0.201', '0.258', '0.193')]
    for ds, kind, lv, tch, tre, tbe, tme in ROWS:
        x = (n07_detection('noise', ds, kind) or {}).get((0, lv))
        cl = list((x or {}).get('cliff', {}).values())
        r.check(f'{ds} {kind} level {lv:g} fold 0 chance (table)', tch, x and x['chance'])
        r.check(f'{ds} {kind} level {lv:g} fold 0 residual AP (table)', tre, x and x['resid'])
        r.check(f'{ds} {kind} level {lv:g} fold 0 cliff-aware best setting', tbe, max(cl) if cl else None)
        r.check(f'{ds} {kind} level {lv:g} fold 0 cliff-aware median setting', tme, statistics.median(cl) if cl else None)


@check('N7-14', 'cliff-aware gap against the residual (n07)')
def n7_14(r):
    """Paragraph 2 of 'Which Detectors Find the Suspect Labels': activity_cliff_aware graded under
    66-132 neighbour settings (n07 detection_by_similarity, mean AP per setting over passes)
    against the residual, fold 0; median over settings and number of settings ahead."""
    det = (n07_detection('noise', 'caco2', 'shift_only_p10') or {}).get((0, 1.0))
    if det is None:
        r.missing('caco2 shift_only_p10 level 1', 'n07 units file or level missing')
    else:
        aps = sorted(det['cliff'].values())
        r.check('caco2 shift_only_p10 level 1 fold 0 cliff-aware median over settings', '0.269',
                statistics.median(aps) if aps else None)
        r.check('caco2 shift_only_p10 level 1 fold 0 residual AP', '0.202', det['resid'])
        r.check('caco2 shift_only_p10 level 1 fold 0 number of cliff settings', '120', len(aps))
        ahead = sum(a > det['resid'] for a in aps)
        r.claim('caco2 shift_only_p10: every cliff setting ahead of the residual', ahead == len(aps) and aps,
                f'{ahead} of {len(aps)} settings ahead')
    # under outliers only its best setting is ahead (caco2, the dataset of the example)
    x = (n07_detection('noise', 'caco2', 'outlier_p05') or {}).get((0, 1.0))
    if x:
        aps = sorted(x['cliff'].values())
        med_behind = statistics.median(aps) < x['resid']; best_ahead = max(aps) > x['resid']
        r.claim('caco2 outlier_p05 level 1: median cliff setting behind the residual, best ahead',
                med_behind and best_ahead,
                f'median {statistics.median(aps):.4f}, best {max(aps):.4f}, residual {x["resid"]:.4f}; '
                f'{sum(a > x["resid"] for a in aps)} of {len(aps)} settings ahead')
    # under censoring it trails the residual on caco2, herg_ki and logd at every setting
    for ds in ('caco2', 'herg_ki', 'logd'):
        x = (n07_detection('noise', ds, 'censoring') or {}).get((0, 0.25))
        if not x:
            r.claim(f'{ds} censoring 0.25: every cliff setting behind the residual', None, 'file or level missing')
            continue
        ahead = sum(a > x['resid'] for a in x['cliff'].values())
        r.claim(f'{ds} censoring 0.25: every cliff setting behind the residual', ahead == 0,
                f'{ahead} of {len(x["cliff"])} settings ahead; best {max(x["cliff"].values()):.4f} vs residual {x["resid"]:.4f}')

    def q(x):
        aps = sorted(x['cliff'].values())
        if not aps:
            return f'residual {x["resid"]:.4f}, no cliff grading'
        return (f'cliff median {statistics.median(aps):.4f}, best {aps[-1]:.4f}, residual {x["resid"]:.4f}, '
                f'{sum(a > x["resid"] for a in aps)} of {len(aps)} settings ahead')
    _n07_extra(r, {'caco2', 'herg_ki', 'logd'}, q)


# n21 -----------------------------------------------------------------------------------------
N21_FAMILY = {'plain': 'plain', 'importance_weighted': 'importance-weighted', 'forest_proximity': 'forest proximity',
              'hybrid': 'hybrid', 'hybrid_similarity': 'hybrid'}


def n21_families(d, lv):
    """Best AP (over settings) and best mean-over-settings AP per distance family at level lv of an
    n21 file: per_level[lv].per_space[space].detection_per_detector[det].average_precision.mean,
    family from spaces[space].family."""
    p = (d.get('per_level') or {}).get(lv)
    if not p:
        return None
    best, bestmean = {}, {}
    for sp, v in (p.get('per_space') or {}).items():
        fam = N21_FAMILY.get(((d.get('spaces') or {}).get(sp) or {}).get('family'), '?')
        for det, x in (v.get('detection_per_detector') or {}).items():
            ap = ((x or {}).get('average_precision') or {}).get('mean')
            if ap is not None and ap > best.get(fam, -1):
                best[fam] = ap
        m = v.get('detection_average_precision_mean')
        if m is not None and m > bestmean.get(fam, -1):
            bestmean[fam] = m
    ch = None
    if p.get('molecules_in_training_fold_mean'):
        ch = p['molecules_corrupted_mean'] / p['molecules_in_training_fold_mean']
    return {'best': best, 'mean': bestmean, 'chance': ch}


def _n21_files(where):
    """{(ds, kind): (level, family result)} for every damage-kind file in <where>/n21."""
    out = {}
    d0 = CTX.path(where, 'n21_neighbour_distance')
    for f in files(d0, '*__*.json'):
        if 'fingerprint' in f:
            continue
        ds, kind = f[:-5].split('__', 1)
        d = jload(os.path.join(d0, f))
        if not d or 'per_level' not in d:
            continue
        lv = _c_lvkey(main_level_of(kind))
        if lv not in d['per_level']:
            cands = [k for k in d['per_level'] if float(k) > 0]
            if not cands:
                continue
            lv = cands[0]
        res = n21_families(d, lv)
        if res:
            out[(ds, kind)] = (lv, res, d.get('folds_run'))
    return out


@check('N7-15', 'four families of distance for the neighbour detectors (n21)')
def n7_15(r):
    """Paragraph 3 of 'Which Detectors Find the Suspect Labels': n21_neighbour_distance, best AP
    in each distance family (max over spaces and detector settings) at the main level, fold 0.
    Read from the noise_view folder if given (it holds caco2 sparse_dprime_up_p10), else noise/."""
    where = 'view' if CTX.d('view') and os.path.isdir(CTX.path('view', 'n21_neighbour_distance')) else 'noise'
    F = _n21_files(where)
    x = F.get(('caco2', 'outlier_p05'))
    if x is None:
        r.missing('caco2 outlier_p05', 'n21 file missing')
    else:
        b = x[1]['best']
        for fam, t in [('plain', '0.3368'), ('importance-weighted', '0.3524'), ('forest proximity', '0.3358'), ('hybrid', '0.3352')]:
            r.check(f'caco2 outlier_p05 level 1 fold 0 best AP, {fam} features', t, b.get(fam))
        r.check('caco2 outlier_p05 level 1 fold 0 chance', '0.0512', x[1]['chance'])
    x = F.get(('ddg_combined', 'outlier_p05'))
    if x is None:
        r.missing('ddg_combined outlier_p05', 'n21 file missing')
    else:
        r.check('ddg_combined outlier_p05 level 1 fold 0 best AP, hybrid features', '0.5074', x[1]['best'].get('hybrid'))
        r.check('ddg_combined outlier_p05 level 1 fold 0 best AP, plain features', '0.4850', x[1]['best'].get('plain'))
    # winners per damage kind and per dataset (family winning most kinds)
    wins = collections.defaultdict(collections.Counter); hyb_kinds = []; meanwins = collections.Counter()
    for (ds, kind), (lv, res, folds) in sorted(F.items()):
        if ds not in CH7_DATASETS or not res['best']:
            continue
        w = max(res['best'], key=res['best'].get)
        wins[ds][w] += 1
        if w == 'hybrid':
            hyb_kinds.append(f'{ds} {kind}')
        if res['mean']:
            meanwins[max(res['mean'], key=res['mean'].get)] += 1
    ds_winner = {ds: c.most_common(1)[0][0] for ds, c in wins.items()
                 if len(c) == 1 or c.most_common(2)[0][1] > c.most_common(2)[1][1]}
    want = ['buchwald_hartwig', 'caco2', 'herg_ki', 'logd', 'adme_mdr1', 'ginkgo_colloidal']
    r.claim('importance-weighted features best family on BH, caco2, herg_ki, logd, adme_mdr1, colloidal',
            all(ds_winner.get(d) == 'importance-weighted' for d in want),
            'winner by damage kinds won: ' + ', '.join(f'{d} {ds_winner.get(d, "tie")}' for d in sorted(wins)))
    r.claim('hybrid features best family on no dataset overall', 'hybrid' not in ds_winner.values(),
            'datasets won by hybrid: ' + (', '.join(d for d, w in ds_winner.items() if w == 'hybrid') or 'none'))
    # 2026-10-10 (10-10b): caco2 sparse_dprime_up_p10 was set aside; qm9 and adme_mdr1 added
    r.claim('hybrid best on ddg outlier, colloidal censoring, qm9 outlier, adme_mdr1 shift (and only there)',
            sorted(hyb_kinds) == sorted(['ddg_combined outlier_p05', 'ginkgo_colloidal censoring', 'qm9 outlier_p05',
                                         'adme_mdr1 shift_only_p10']),
            'kinds won by hybrid at best setting: ' + ', '.join(hyb_kinds))
    r.info(f'mean over detector settings: hybrid wins {meanwins.get("hybrid", 0)} of {sum(meanwins.values())} damage kinds (author note: 8 of 23)')
    for (ds, kind), (lv, res, folds) in sorted(F.items()):
        for fo in (folds or []):
            if fo != 0:
                r.new_fold(f'{ds} {kind} level {lv}', fo, 'folds_run lists this fold; file stores fold-pooled means: '
                           + ' '.join(f'{k}={v:.4f}' for k, v in sorted(res['best'].items())))
    # table rows added 2026-10-10: QM9 and MDR1 (best AP per family and chance, fold 0)
    BLOCKS[r.id] = BLOCKS.get(r.id, '')
    TROWS = {('qm9', 'censoring'): ('0.250', '0.191', '0.192', '0.226', '0.195'),
             ('qm9', 'outlier_p05'): ('0.051', '0.403', '0.429', '0.416', '0.434'),
             ('qm9', 'shift_only_p10'): ('0.100', '0.508', '0.729', '0.700', '0.717'),
             ('adme_mdr1', 'censoring'): ('0.250', '0.686', '0.731', '0.211', '0.551'),
             ('adme_mdr1', 'outlier_p05'): ('0.050', '0.356', '0.364', '0.343', '0.357'),
             ('adme_mdr1', 'shift_only_p10'): ('0.101', '0.326', '0.343', '0.227', '0.378'),
             ('herg', 'confusion_directed'): ('0.100', '0.467', '0.484', '0.494', '0.482')}
    for (ds, kind), t in TROWS.items():
        x = F.get((ds, kind))
        b = x[1]['best'] if x else {}
        r.check(f'{ds} {kind} fold 0 chance (table)', t[0], x and x[1]['chance'])
        for fam, tv in zip(('plain', 'importance-weighted', 'forest proximity', 'hybrid'), t[1:]):
            r.check(f'{ds} {kind} fold 0 best AP, {fam} (table)', tv, b.get(fam))
    cited = {'buchwald_hartwig', 'caco2', 'herg_ki', 'logd', 'ginkgo_colloidal', 'ddg_combined', 'qm9', 'adme_mdr1'}
    for ds in sorted({k[0] for k in F} - cited):
        if ds not in CH7_DATASETS:
            continue
        parts = [f'{kind}: ' + ' '.join(f'{k}={v:.4f}' for k, v in sorted(res['best'].items()))
                 for (d2, kind), (lv, res, _) in sorted(F.items()) if d2 == ds]
        r.new_dataset(ds, 'best AP per family; ' + '; '.join(parts))


# n12 -----------------------------------------------------------------------------------------
def n12_mean_ap(d, lv):
    """Mean detection AP unscaled vs importance-scaled at level lv of an n12 file: per_sigma[lv]
    .detection[imp][cond][model][key].detection_average_precision; every <stem>_unweighted
    condition matched with each <stem>_imp* condition on the same model and key."""
    p = (d.get('per_sigma') or {}).get(lv)
    if not p:
        return None
    U, W = [], []; ch = None
    for imp, conds in (p.get('detection') or {}).items():
        for c, bym in conds.items():
            if not c.endswith('_unweighted'):
                continue
            stem = c[:-len('_unweighted')]
            for wc in [x for x in conds if x.startswith(stem + '_imp')]:
                for model, keys in bym.items():
                    for k, x in keys.items():
                        u = x.get('detection_average_precision')
                        y = ((conds[wc].get(model) or {}).get(k) or {}).get('detection_average_precision')
                        if u is None or y is None:
                            continue
                        ch = x['detection_n_corrupted'] / x['detection_n_training']
                        U.append(u); W.append(y)
    if not U:
        return None
    return {'unscaled': mean(U), 'scaled': mean(W), 'n': len(U), 'chance': ch}


def _n12_all(where):
    out = {}
    d0 = CTX.path(where, 'n12_importance_weighted_distance')
    for f in files(d0, '*__*.json'):
        if 'fingerprint' in f:
            continue
        ds, kind = f[:-5].split('__', 1)
        d = jload(os.path.join(d0, f))
        if not d or 'per_sigma' not in d:
            continue
        lv = _c_lvkey(main_level_of(kind))
        lv = lv if lv in d['per_sigma'] else (lv + '.0' if lv + '.0' in d['per_sigma'] else None)
        if lv is None:
            continue
        res = n12_mean_ap(d, lv)
        if res:
            out[(ds, kind)] = (res, d.get('folds_run'))
    return out


@check('N7-16', 'importance scaling before the neighbours are found (n12)')
def n7_16(r):
    """Paragraph 4 of 'Which Detectors Find the Suspect Labels': n12_importance_weighted_distance
    in the noise_view folder, per_sigma[main level].detection, mean AP over matched unscaled /
    importance-scaled comparisons, fold 0."""
    where = 'view' if CTX.d('view') else 'noise'
    A = _n12_all(where)
    for ds, kind, u, s in [('buchwald_hartwig', 'shift_only_p10', '0.465', '0.512'),
                           ('logd', 'shift_only_p10', '0.4123', '0.4423'),
                           ('ginkgo_colloidal', 'shift_only_p10', '0.3312', '0.3683'),
                           ('caco2', 'outlier_p05', '0.2576', '0.2586')]:
        x = A.get((ds, kind))
        r.check(f'{ds} {kind} fold 0 mean AP unscaled', u, x[0]['unscaled'] if x else None)
        r.check(f'{ds} {kind} fold 0 mean AP importance-scaled', s, x[0]['scaled'] if x else None)
    outl = [(ds, x[0]['scaled'] - x[0]['unscaled']) for (ds, k), x in A.items() if k == 'outlier_p05' and ds in CH7_DATASETS]
    r.check('outlier_p05: largest change scaled minus unscaled over datasets (absolute)', '0.0066',
            max(abs(v) for _, v in outl) if outl else None,
            note='; '.join(f'{d} {v:+.4f}' for d, v in sorted(outl)))
    cens = [(ds, x[0]['scaled'] - x[0]['unscaled']) for (ds, k), x in A.items() if k == 'censoring' and ds in CH7_DATASETS]
    r.claim('censoring: scaling lowered mean AP on every dataset', all(v < 0 for _, v in cens) if cens else None,
            '; '.join(f'{d} {v:+.4f}' for d, v in sorted(cens)))
    sh = [(ds, x[0]['scaled'] - x[0]['unscaled']) for (ds, k), x in A.items() if k == 'shift_only_p10' and ds in CH7_DATASETS]
    r.info('shift_only_p10 scaled minus unscaled: ' + '; '.join(f'{d} {v:+.4f}' for d, v in sorted(sh)))
    for (ds, kind), (res, folds) in sorted(A.items()):
        for fo in (folds or []):
            if fo != 0:
                r.new_fold(f'{ds} {kind}', fo, f'folds_run lists this fold; mean AP unscaled {res["unscaled"]:.4f} scaled {res["scaled"]:.4f}')
    # rows added 2026-10-10 (10-10b): QM9, MDR1, PR-CHO; QM9 shift is the largest gain cited in the text
    for ds, t in {'qm9': ('0.1682', '0.1678', '0.344', '0.349', '0.538', '0.634'),
                  'adme_mdr1': ('0.373', '0.356', '0.284', '0.286', '0.3046', '0.3052'),
                  'ginkgo_pr_cho': ('0.333', '0.330', '0.333', '0.334', '0.198', '0.197')}.items():
        for i, kind in enumerate(('censoring', 'outlier_p05', 'shift_only_p10')):
            x = A.get((ds, kind))
            r.check(f'{ds} {kind} fold 0 mean AP unscaled (table)', t[2 * i], x[0]['unscaled'] if x else None)
            r.check(f'{ds} {kind} fold 0 mean AP importance-scaled (table)', t[2 * i + 1], x[0]['scaled'] if x else None)
    r.claim('shift_only_p10: largest gain is QM9', (max(sh, key=lambda t: t[1])[0] == 'qm9') if sh else None,
            'largest: ' + (max(sh, key=lambda t: t[1])[0] if sh else 'none'))
    r.claim('shift_only_p10: scaling raised AP on every dataset except herg_ki, ginkgo_pr_cho (|change| <= 0.002 there)',
            all((v > 0) if d not in ('herg_ki', 'ginkgo_pr_cho') else abs(v) <= 0.002 for d, v in sh) if sh else None,
            '; '.join(f'{d} {v:+.4f}' for d, v in sorted(sh)))
    cited = {'buchwald_hartwig', 'logd', 'ginkgo_colloidal', 'caco2', 'qm9', 'adme_mdr1', 'ginkgo_pr_cho'}
    for ds in sorted({k[0] for k in A} - cited):
        if ds in CH7_DATASETS:
            r.new_dataset(ds, '; '.join(f'{k}: unscaled {x[0]["unscaled"]:.4f} scaled {x[0]["scaled"]:.4f}'
                                        for (d2, k), x in sorted(A.items()) if d2 == ds))


# n23 / n32 -----------------------------------------------------------------------------------
def n23_sources(d, lv):
    """Mean AP per source at level lv of an n23 file: detection[lv][space][model][source]
    .detection_average_precision, mean over the configurations."""
    det = (d.get('detection') or {}).get(lv)
    if not det:
        return None
    by = collections.defaultdict(list); ch = None
    for sp, bb in det.items():
        for m, c in (bb or {}).items():
            for src, v in (c or {}).items():
                ap = (v or {}).get('detection_average_precision')
                if ap is not None:
                    by[src].append(ap)
                    ch = v['detection_n_corrupted'] / v['detection_n_training']
    return {s: mean(v) for s, v in by.items()}, ch


def n32_detection(ds, kind):
    """n32 detection: final file detection[level][config][detector] if it has one, else the
    run's checkpoint state.det_acc (mean over passes). Returns ({level: {config: {detector:
    AP}}}, source) or (None, None)."""
    d0 = CTX.path('noise', 'n32_iterative_and_dual_selection')
    d = jload(os.path.join(d0, f'{ds}__{kind}.json')) if d0 else None
    if d and d.get('detection'):
        return ({l: {c: {k: v.get('detection_average_precision') for k, v in b.items()} for c, b in a.items()}
                 for l, a in d['detection'].items()}, 'final', d.get('folds_run'))
    c = jload(os.path.join(d0, '_checkpoints', f'{ds}__{kind}.json')) if d0 else None
    if c and (c.get('state') or {}).get('det_acc'):
        da = c['state']['det_acc']
        return ({l: {sk: {dk: mean([x.get('detection_average_precision') for x in rs]) for dk, rs in b.items()}
                     for sk, b in a.items()} for l, a in da.items()}, 'checkpoint',
                sorted({int(p.split('f')[-1]) for p in c.get('passes_done', []) if 'f' in p}))
    return None, None, None


def n32_family_means(det_level):
    """Mean AP per detector family (part before '@') over every config and setting."""
    by = collections.defaultdict(list)
    for sk, b in det_level.items():
        for dk, ap in b.items():
            if ap is not None:
                by[dk.split('@')[0]].append(ap)
    return {k: (mean(v), len(v)) for k, v in by.items()}


@check('N7-17', 'model-uncertainty detectors and detectors from outside cheminformatics (n23, n32)')
def n7_17(r):
    """Paragraph 5 of 'Which Detectors Find the Suspect Labels': n23_uncertainty_cleaning
    detection[level][space][model][source], mean over configurations (GP vs cliff-aware gap);
    sources_defined_under_this_condition for the three feature-only spreads; n32 detection
    (final or checkpoint det_acc) mean AP per detector family, fold 0."""
    d0 = CTX.path('noise', 'n23_uncertainty_cleaning')
    def n23(ds, kind):
        d = jload(os.path.join(d0, f'{ds}__{kind}.json')) if d0 else None
        if not d:
            return None, None
        return d, n23_sources(d, _c_lvkey(main_level_of(kind)))
    for ds, gp, ca in [('caco2', '0.3449', '0.3350'), ('logd', '0.4084', '0.3853')]:
        d, x = n23(ds, 'outlier_p05')
        r.check(f'{ds} outlier_p05 level 1 fold 0 GP (spread + residual) AP', gp, x[0].get('gaussian_process') if x else None)
        r.check(f'{ds} outlier_p05 level 1 fold 0 cliff-aware AP', ca, x[0].get('activity_cliff_aware') if x else None)
    # behind the cliff-aware gap under every shift and every hERG flip kind; undefined sources
    behind, total, undefined_ok, graded = [], 0, True, 0
    for f in files(d0, '*__*.json'):
        if 'fingerprint' in f:
            continue
        ds, kind = f[:-5].split('__', 1)
        if ds not in CH7_DATASETS:
            continue
        d, x = n23(ds, kind)
        if not x:
            continue
        graded += 1
        sd = d.get('sources_defined_under_this_condition') or {}
        if any(sd.get(s) for s in ('hetero_gp', 'ngboost', 'heteroscedastic_nn')):
            undefined_ok = False
        if kind in ('shift_only_p10', 'sparse_dprime_up_p10') or kind in FLIP_KINDS:
            total += 1
            if x[0].get('gaussian_process', 0) < x[0].get('activity_cliff_aware', 0):
                behind.append(f'{ds} {kind}')
    r.claim('GP behind the cliff-aware gap under every shift and every hERG flip kind', len(behind) == total and total,
            f'{len(behind)} of {total} dataset x kind')
    r.claim('hetero_gp, ngboost, heteroscedastic_nn undefined under every graded kind', undefined_ok if graded else None,
            f'{graded} graded files')
    for ds, kind, det, t, rt in [('buchwald_hartwig', 'censoring', 'knn_shapley', '0.391', '0.195'),
                                 ('ginkgo_colloidal', 'censoring', 'knn_shapley', '0.6402', '0.4905'),
                                 ('buchwald_hartwig', 'shift_only_p10', 'cleanlab', '0.8288', '0.2955')]:
        det_all, src, folds = n32_detection(ds, kind)
        lv = _c_lvkey(main_level_of(kind))
        fm = n32_family_means(det_all[lv]) if det_all and lv in det_all else {}
        r.check(f'{ds} {kind} level {lv} fold 0 n32 {det} mean AP', t, (fm.get(det) or (None,))[0], note=f'{src}, {(fm.get(det) or (0, 0))[1]} readings')
        r.check(f'{ds} {kind} level {lv} fold 0 n32 residual mean AP', rt, (fm.get('oof_residual') or (None,))[0], note=f'{(fm.get("oof_residual") or (0, 0))[1]} readings')
    # new folds / datasets
    for f in files(CTX.path('full', 'n23_uncertainty_cleaning'), '*.rows.jsonl'):
        fo = sorted({x.get('fold') for x in rows_file(os.path.join(CTX.path('full', 'n23_uncertainty_cleaning'), f))} - {0})
        if fo:
            r.new_fold(f'n23 {f.split("__fold")[0]}', ','.join(map(str, fo)), 'noise_full rows only (cleaning scores; detection AP is not in rows files)')
    for ds in sorted(set(datasets_with_files('n23_uncertainty_cleaning', where=('noise',))) - {'caco2', 'logd', 'herg', 'herg_ki'}):
        d, x = n23(ds, 'outlier_p05')
        r.new_dataset(ds, f'n23 outlier_p05: GP {x[0].get("gaussian_process", float("nan")):.4f} vs cliff {x[0].get("activity_cliff_aware", float("nan")):.4f}'
                      if x else 'n23 file, no outlier_p05 detection')
    n32ds = set()
    for w in ('noise',):
        for f in files(CTX.path(w, 'n32_iterative_and_dual_selection'), '*__*.json') + \
                files(os.path.join(CTX.path(w, 'n32_iterative_and_dual_selection') or '', '_checkpoints'), '*__*.json'):
            n32ds.add(f.split('__')[0])
    for ds in sorted((n32ds & set(CH7_DATASETS)) - {'buchwald_hartwig', 'ginkgo_colloidal'}):
        parts = []
        for kind in ('censoring', 'shift_only_p10'):
            det_all, src, _ = n32_detection(ds, kind)
            lv = _c_lvkey(main_level_of(kind))
            if det_all and lv in det_all:
                fm = n32_family_means(det_all[lv])
                parts.append(f'{kind}: knn_shapley {fm.get("knn_shapley", (float("nan"),))[0]:.4f} cleanlab '
                             f'{fm.get("cleanlab", (float("nan"),))[0]:.4f} residual {fm.get("oof_residual", (float("nan"),))[0]:.4f} [{src}]')
        r.new_dataset(ds, 'n32 ' + ('; '.join(parts) if parts else 'no graded censoring or shift_only_p10 detection'))


# n38 / n41 -----------------------------------------------------------------------------------
def n38_ranges(ds):
    """n38 checkpoint state.det_acc.real['1'] (d' = 1): mean AP per configuration/model over
    passes; returns (min, max, chance, passes_done) or None."""
    c = J('noise', 'n38_noise_relative_to_model_error', f'_checkpoints/{ds}__sparse_dprime_up_p10.json')
    st = (c or {}).get('state') or {}
    r1 = ((st.get('det_acc') or {}).get('real') or {}).get('1') or {}
    vals, ch = [], []
    for cfgm, lst in r1.items():
        aps = [x['scores']['detection_average_precision'] for x in lst if x.get('scores')]
        if aps:
            vals.append(mean(aps))
            ch += [x['scores']['detection_n_corrupted'] / x['scores']['detection_n_training'] for x in lst if x.get('scores')]
    if not vals:
        return None
    return min(vals), max(vals), mean(ch), c.get('passes_done', [])


def n41_bh_flags():
    """Every BH pass of n41: {calibration_q: [(molecules_flagged, ds__kind)]}, from the final
    file's bh.<lvl>.<cfg>.<cal>_q<q>.passes[] where it has one, else the run's checkpoint
    state.bh_acc."""
    out = collections.defaultdict(list)
    d0 = CTX.path('noise', 'n41_conformal_fdr_detection')
    names = {f[:-5] for f in files(d0, '*__*.json')} | {f[:-5] for f in files(os.path.join(d0 or '', '_checkpoints'), '*__*.json')}
    for nm in sorted(names):
        if nm.split('__')[0] not in CH7_DATASETS:
            continue
        d = jload(os.path.join(d0, nm + '.json'))
        src = d.get('bh') if d and d.get('bh') else None
        if src is None:
            c = jload(os.path.join(d0, '_checkpoints', nm + '.json'))
            src = ((c or {}).get('state') or {}).get('bh_acc')
        if not src:
            continue

        def walk(x, path):
            if isinstance(x, dict):
                if 'molecules_flagged' in x and path and not path[-1] == 'mean':
                    out[path[-1]].append((x['molecules_flagged'], nm, path[0]))
                    return
                for k, v in x.items():
                    if k == 'mean':
                        continue
                    walk(v, path + [k])
            elif isinstance(x, list):
                for v in x:
                    walk(v, path)
        walk(src, [])
    return out


@check('N7-18', 'residual under damage sized by model error; conformal FDR flags (n38, n41)')
def n7_18(r):
    """Paragraph 6 of 'Which Detectors Find the Suspect Labels': n38 checkpoints
    state.det_acc.real['1'] (d' = 1), range over configurations of the residual's AP, checkpoints
    only; n41 bh.<lvl>.<cfg>.<cal>_q<q>.passes[].molecules_flagged (final file or checkpoint
    bh_acc), fold 0."""
    # caco2 dropped 2026-10-10: its n38 d'=1 file was set aside (inputs changed) and the text cites titer only
    for ds, lo, hi, ch in [('ginkgo_titer', '0.20', '0.25', '0.106')]:
        x = n38_ranges(ds)
        r.check(f'{ds} d\'=1 residual AP, lowest configuration', lo, x[0] if x else None)
        r.check(f'{ds} d\'=1 residual AP, highest configuration', hi, x[1] if x else None)
        r.check(f'{ds} d\'=1 chance', ch, x[2] if x else None)
    above = []; allds = []
    for f in files(CTX.path('noise', 'n38_noise_relative_to_model_error') and os.path.join(CTX.path('noise', 'n38_noise_relative_to_model_error'), '_checkpoints'), '*__sparse_dprime_up_p10.json'):
        ds = f.split('__')[0]
        if ds not in CH7_DATASETS:
            continue
        x = n38_ranges(ds)
        if x:
            allds.append(ds)
            if x[0] > x[2]:
                above.append(ds)
            if ds not in ('ginkgo_titer', 'caco2'):
                r.new_dataset(ds, f"n38 d'=1 residual AP {x[0]:.3f} to {x[1]:.3f} vs chance {x[2]:.4f} ({len(x[3])} passes)")
            fo = sorted({int(p.split('f')[-1]) for p in x[3] if 'f' in p} - {0})
            if fo:
                r.new_fold(f'n38 {ds}', ','.join(map(str, fo)), 'checkpoint passes on this fold (AP pooled with fold 0)')
    r.claim("n38: residual above chance on every dataset (every configuration, d'=1)", len(above) == len(allds) and allds,
            f'{len(above)} of {len(allds)} datasets: {", ".join(allds)}')
    fl = n41_bh_flags()
    # Text rewritten 2026-10-10 for the 10-10b results (QM9 added to n41): measured datasets and QM9 apart.
    rq05 = [n for k, v in fl.items() if k.startswith('random') and k.endswith('_q0.05') for n, _, _ in v]
    rq1 = [n for k, v in fl.items() if k.startswith('random') and k.endswith('_q0.1') for n, _, _ in v]
    r.claim('n41: no molecule flagged at q = 0.05 or 0.1 with calibration from the noisy labels',
            (max(rq05 + rq1) == 0) if rq05 and rq1 else None, f'{len(rq05)} + {len(rq1)} passes')
    ocm = [(n, nm) for n, nm, _ in fl.get('oracle_clean_q0.1', []) if not nm.startswith('qm9')]
    ocq = [(n, nm) for n, nm, _ in fl.get('oracle_clean_q0.1', []) if nm.startswith('qm9')]
    oq5 = [(n, nm) for n, nm, _ in fl.get('oracle_clean_q0.05', []) if nm.startswith('qm9')]
    om5 = [n for n, nm, _ in fl.get('oracle_clean_q0.05', []) if not nm.startswith('qm9')]
    r.claim('n41 clean calibration q = 0.05: no molecule flagged outside QM9', (max(om5) == 0) if om5 else None, f'{len(om5)} passes')
    r.check('n41 clean calibration q = 0.1, measured datasets: passes', '120', len(ocm) or None)
    r.check('n41 clean calibration q = 0.1, measured datasets: passes that flag any', '1', sum(1 for n, _ in ocm if n) if ocm else None,
            note=', '.join(sorted({nm for n, nm in ocm if n})))
    r.check('n41 clean calibration q = 0.1, measured datasets: molecules flagged in that pass', '10',
            max(n for n, _ in ocm) if ocm else None)
    r.check('n41 clean calibration q = 0.1, QM9: passes', '28', len(ocq) or None)
    r.check('n41 clean calibration q = 0.1, QM9: passes that flag any', '21', sum(1 for n, _ in ocq if n) if ocq else None)
    r.check('n41 clean calibration q = 0.1, QM9: fewest molecules in a pass that flags any', '23',
            min(n for n, _ in ocq if n) if any(n for n, _ in ocq) else None)
    r.check('n41 clean calibration q = 0.1, QM9: most molecules in a pass', '485', max(n for n, _ in ocq) if ocq else None)
    r.check('n41 clean calibration q = 0.05, QM9: passes that flag any', '9', sum(1 for n, _ in oq5 if n) if oq5 else None)
    # "acting on those flags left the held-out score where it was in almost every pass"
    won = lost = tied = 0
    d0 = CTX.path('noise', 'n41_conformal_fdr_detection')
    for f in files(d0, '*__*.json'):
        d = jload(os.path.join(d0, f))
        for lvl, byrep in (((d or {}).get('summary') or {}).get('per_level') or {}).items():
            for rep, bym in byrep.items():
                for m, e in (bym or {}).items():
                    if not isinstance(e, dict):
                        continue
                    for arm, a in (e.get('per_arm') or {}).items():
                        if 'oracle_clean_q0.1' in arm and '/seed' not in arm:
                            vn = a.get('vs_no_cleaning') or {}
                            won += vn.get('passes_won', 0); lost += vn.get('passes_lost', 0); tied += vn.get('passes_tied', 0)
    r.info(f'n41 clean-calibration q = 0.1 arms vs no cleaning (final files): won {won}, lost {lost}, tied {tied} passes')


# n31 -----------------------------------------------------------------------------------------
def n31_cells(ds, cond, lv, share):
    """[(config, model, precision, {op: break_even})] at summary.per_condition[cond][lv][config]
    .per_model[m].synthetic[share<s>][op] of n31_regime_and_breakeven/<ds>.json."""
    d = J('noise', 'n31_regime_and_breakeven', f'{ds}.json')
    L = ((((d or {}).get('summary') or {}).get('per_condition') or {}).get(cond) or {}).get(lv)
    if not L:
        return None
    out = []
    for c, x in L.items():
        for m, pm in (x.get('per_model') or {}).items():
            ops = (pm.get('synthetic') or {}).get(share) or {}
            if not ops:
                continue
            prec = next(iter(ops.values())).get('detector_precision_at_this_share_mean')
            out.append((c, m, prec, {op: v.get('break_even_precision') for op, v in ops.items()}))
    return out


@check('N7-19', 'break-even precision (n31)')
def n7_19(r):
    """Paragraph 7 of 'Which Detectors Find the Suspect Labels': n31_regime_and_breakeven/<ds>.json
    summary.per_condition[cond][lv][config].per_model[m].synthetic.share<s>.<op>
    .{break_even_precision, detector_precision_at_this_share_mean}, fold 0.
    Reading used (the SOURCE line names no rule): outliers = network, top 5%, level 1, range of the
    precision over configurations, and range over operations of the median break-even over
    configurations x models; shifts = mean precision over configurations x models, median
    break-even of deletion over configurations x models."""
    for ds, plo, phi, blo, bhi in [('caco2', '0.62', '0.64', '0.13', '0.27'), ('herg_ki', '0.60', '0.62', '0.16', '0.42')]:
        cells = n31_cells(ds, 'outlier_only_p05', '1', 'share0.05')
        if not cells:
            r.missing(f'{ds} outlier_only_p05 level 1', 'n31 summary missing')
            continue
        pr = [p for c, m, p, _ in cells if m == 'dnn' and p is not None]
        r.check(f'{ds} outliers level 1 top 5% residual precision, network, lowest configuration', plo, min(pr) if pr else None)
        r.check(f'{ds} outliers level 1 top 5% residual precision, network, highest configuration', phi, max(pr) if pr else None)
        meds = {}
        for op in ('remove', 'downweight', 'rewrite', 'bayes_shrink'):
            v = [b[op] for _, _, _, b in cells if b.get(op) is not None]
            if v:
                meds[op] = statistics.median(v)
        r.check(f'{ds} outliers level 1 top 5% break-even, lowest operation (median over configurations x models)', blo,
                min(meds.values()) if meds else None, note=' '.join(f'{k} {v:.3f}' for k, v in meds.items()))
        r.check(f'{ds} outliers level 1 top 5% break-even, highest operation (median over configurations x models)', bhi,
                max(meds.values()) if meds else None)
    for ds, cond, share, pt, bt in [('caco2', 'sparse_dprime_up_p10', 'share0.05', '0.17', '0.62'),
                                    ('herg_ki', 'shift_only_p10', 'share0.1', '0.26', '0.68')]:
        cells = n31_cells(ds, cond, '1', share)
        pr = [p for _, _, p, _ in cells or [] if p is not None]
        be = [b['remove'] for _, _, _, b in cells or [] if b.get('remove') is not None]
        r.check(f'{ds} {cond} level 1 top {share[5:]} residual precision (mean over configurations x models)', pt, mean(pr) if pr else None,
                note=f'range {min(pr):.3f}-{max(pr):.3f}' if pr else '')
        r.check(f'{ds} {cond} level 1 top {share[5:]} deletion break-even (median over configurations x models)', bt,
                statistics.median(be) if be else None, note=' '.join(f'{x:.3f}' for x in sorted(be)))
    for ds in ('caco2', 'herg_ki'):
        d = J('noise', 'n31_regime_and_breakeven', f'{ds}.json')
        for fo in ((d or {}).get('folds_run') or []):
            if fo != 0:
                r.new_fold(f'n31 {ds}', fo, 'folds_run lists this fold (summary pools folds)')
    for ds in datasets_with_files('n31_regime_and_breakeven', where=('noise',)):
        if ds in ('caco2', 'herg_ki'):
            continue
        cells = n31_cells(ds, 'outlier_only_p05', '1', 'share0.05')
        if cells:
            pr = [p for _, _, p, _ in cells if p is not None]
            r.new_dataset(ds, f'n31 outliers level 1 top 5% precision {min(pr):.2f} to {max(pr):.2f}')
        else:
            r.new_dataset(ds, 'n31 rows file only, no summary with break-even precision' if not J('noise', 'n31_regime_and_breakeven', f'{ds}.json')
                          else 'n31 file without an outlier_only_p05 level-1 summary')


# ---------------------------------------------------------------------------------------------
# N7-20 to N7-29: by noise type (n30, n34, n39, n33)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# N7-20 to N7-29: clipped labels (n30), scaffold-family offsets (n34), several sources (n39),
# Huber teacher blend (n33).
# ---------------------------------------------------------------------------------------------
N30 = 'n30_censored_likelihood'
N30_ARMS = ['clean', 'clipped', 'constant_lift', 'truncated_normal_mean', 'censored_em', 'drop_clipped',
            'per_molecule_lift', 'per_molecule_lift_shuffled/seed0', 'per_molecule_lift_shuffled/seed1',
            'per_molecule_lift_shuffled/seed2']
SINGLE_CFGS, HYB_CFGS = ('base_1', 'base_2'), ('hybrid', 'hybrid__general2')


def _d_lvkey(level):
    return f'{float(level):g}'


def n30_cells(ds, sources=('noise_json', 'full_json', 'full_rows', 'noise_rows')):
    """{(level '0.25', cfg, model, arm, fold): value} for one dataset, from the n30 sources in the
    given priority order (first source that has a cell wins). Sources: the final json
    per_level[L][cfg][arm][model]['l{L}d0f{fold}'] in noise/ and noise_full/, and the
    .rows.jsonl files (clean = score_clean, clipped = score_D, arm = score_Dprime). Draw 0 only
    (the clip has no seed)."""
    out = {}
    for src in sources:
        where = 'noise' if src.startswith('noise') else 'full'
        cells = {}
        if src.endswith('json'):
            d = J(where, N30, f'{ds}.json')
            for L, byc in ((d or {}).get('per_level') or {}).items():
                for cfg, bya in byc.items():
                    for arm, bym in bya.items():
                        for m, byk in (bym or {}).items():
                            for k, v in (byk or {}).items():
                                mm = re.match(r'l([\d.]+)d(\d+)f(\d+)$', k)
                                if mm and mm.group(2) == '0' and v is not None:
                                    cells[(_d_lvkey(mm.group(1)), cfg, m, arm, int(mm.group(3)))] = v
        else:
            for r in rows(N30, f'{ds}__censoring*.rows.jsonl', where=(where,)):
                if r.get('draw', 0) != 0:
                    continue
                L, cfg, m, f = _d_lvkey(r['level']), r['model_input'].split('/')[0], r['model'], r['fold']
                arm = r['method_setting'] if r['method'] == 'per_molecule_lift_shuffled' else r['method']
                cells[(L, cfg, m, arm, f)] = r['score_Dprime']
                cells.setdefault((L, cfg, m, 'clean', f), r['score_clean'])
                cells.setdefault((L, cfg, m, 'clipped', f), r['score_D'])
        for k, v in cells.items():
            out.setdefault(k, v)
    return out


def n30_meta(ds):
    d = J('noise', N30, f'{ds}.json') or J('full', N30, f'{ds}.json') or {}
    return d


def n30_models(ds):
    return list(n30_meta(ds).get('models') or [])


def n30_net(ds):
    ms = n30_models(ds)
    return 'dnn' if 'dnn' in ms else (ms[0] if ms else None)


def n30_best(cells, cfgs, model, by_arm, lib, level='0.25', fold=0):
    """The configuration in cfgs whose by_arm score is better (lib: lower is better)."""
    c = [(cells[(level, cfg, model, by_arm, fold)], cfg) for cfg in cfgs if (level, cfg, model, by_arm, fold) in cells]
    if not c:
        return None
    return (min(c) if lib else max(c))[1]


def n30_folds(cells, level='0.25'):
    return sorted({k[4] for k in cells if k[0] == level})


def _n30_line(cells, cfg, model, arms, fold, level='0.25'):
    return ', '.join(f"{a} {cells[(level, cfg, model, a, fold)]:.4f}" if (level, cfg, model, a, fold) in cells
                     else f'{a} n/a' for a in arms)


def n30_has_results(ds):
    d = n30_meta(ds)
    return bool(d.get('per_level')) and d.get('result_status') != 'skipped'


def _n30_new_folds(r, ds, cited, arms, cited_folds=(0,)):
    """NEW FOLD lines: every (cfg, model) cited for ds, on folds other than cited_folds."""
    cells = n30_cells(ds)
    for f in n30_folds(cells):
        if f in cited_folds:
            continue
        for cfg, model in cited:
            r.new_fold(f'{ds} {model} {cfg} level 0.25', f, _n30_line(cells, cfg, model, arms, f))


def _n30_new_datasets(r, mentioned, arms=('clean', 'clipped', 'censored_em')):
    for ds in datasets_with_files(N30):
        if ds in mentioned:
            continue
        if not n30_has_results(ds):
            continue
        d = n30_meta(ds); lib = d.get('lower_is_better', True)
        cells = n30_cells(ds)
        parts = []
        for m in n30_models(ds):
            for name, cfgs in (('best single', SINGLE_CFGS), ('best hybrid', HYB_CFGS)):
                cfg = n30_best(cells, cfgs, m, 'censored_em', lib)
                if cfg:
                    parts.append(f'{m} {name} ({cfg}): ' + _n30_line(cells, cfg, m, arms, 0))
        r.new_dataset(ds, f'level 0.25, fold 0, {d.get("metric")}: ' + ('; '.join(parts) or 'no fold-0 cells'))


def recovery(clean, damaged, treated):
    if None in (clean, damaged, treated) or clean == damaged:
        return None
    return 100 * (treated - damaged) / (clean - damaged)


@check('N7-20', 'Clipped labels: the four steps on LogD, QM9, Caco-2 (fold 0)')
def n7_20(r):
    """Section 'Labels clipped at an assay limit', first paragraph. n30_censored_likelihood/<ds>.json
    per_level['0.25'][config][arm][model]['l0.25d0f0']. Best single = base_1/base_2 with the
    better censored_em score. Recovery = Equation eq:recovery, 100 x (EM - clipped)/(clean - clipped)."""
    EXPECTED = [  # (dataset, config or 'best single', model, arm, text)
        ('logd', 'best single', 'dnn', 'clean', '0.4314'), ('logd', 'best single', 'dnn', 'clipped', '0.6354'),
        ('logd', 'best single', 'dnn', 'censored_em', '0.4795'),
        ('logd', 'hybrid', 'dnn', 'clean', '0.4059'), ('logd', 'hybrid', 'dnn', 'clipped', '0.6049'),
        ('logd', 'hybrid', 'dnn', 'censored_em', '0.4610'),
        ('qm9', 'hybrid', 'dnn', 'clean', '0.3513'), ('qm9', 'hybrid', 'dnn', 'clipped', '0.5196'),
        ('qm9', 'hybrid', 'dnn', 'censored_em', '0.4406'),
        ('qm9', 'hybrid', 'lightgbm', 'clean', '0.3694'), ('qm9', 'hybrid', 'lightgbm', 'clipped', '0.5240'),
        ('qm9', 'hybrid', 'lightgbm', 'censored_em', '0.4301'),
        ('caco2', 'hybrid', 'dnn', 'clean', '0.2459'), ('caco2', 'hybrid', 'dnn', 'clipped', '0.3042'),
        ('caco2', 'hybrid', 'dnn', 'censored_em', '0.2740')]
    RECOVERY_TEXT = '76'   # LogD, network, best single, censored EM
    cited = collections.defaultdict(set)
    for ds, cfg, m, arm, text in EXPECTED:
        cells = n30_cells(ds)
        if not cells:
            r.missing(f'{ds} {cfg} {m} {arm}', f'no n30 file for {ds}'); continue
        c = n30_best(cells, SINGLE_CFGS, m, 'censored_em', True) if cfg == 'best single' else cfg
        cited[ds].add((c, m))
        r.check(f'{ds} {m} {cfg}{" (" + str(c) + ")" if cfg == "best single" else ""} {arm} level 0.25 fold 0',
                text, cells.get(('0.25', c, m, arm, 0)))
    cells = n30_cells('logd'); c = n30_best(cells, SINGLE_CFGS, 'dnn', 'censored_em', True)
    rec = recovery(*(cells.get(('0.25', c, 'dnn', a, 0)) for a in ('clean', 'clipped', 'censored_em')))
    r.check('logd dnn best single censored_em recovery %', RECOVERY_TEXT, rec)
    for ds, cm in cited.items():
        _n30_new_folds(r, ds, sorted(cm), ('clean', 'clipped', 'censored_em'))
    _n30_new_datasets(r, {'logd', 'qm9', 'caco2', 'herg_ki'})


@check('N7-21', 'Clipped labels: further folds on LogD and Caco-2')
def n7_21(r):
    """Section 'Labels clipped at an assay limit', second paragraph. noise_full/n30_censored_likelihood/
    {logd,caco2}__censoring__fold_aligned_censored_em_2026-10-06.rows.jsonl, level 0.25, dnn, hybrid:
    score_clean / score_D / score_Dprime (censored_em). Also re-checks that herg_ki and qm9 have
    fold 0 only."""
    EXPECTED = {
        'logd': {0: ('0.4040', '0.6128', '0.4798'), 1: ('0.4180', '0.5961', '0.4719'),
                 2: ('0.4114', '0.5720', '0.4702'), 3: ('0.4129', '0.5697', '0.4749')},
        'caco2': {1: (None, '0.3216', '0.2906'), 2: (None, '0.4004', '0.3484')}}
    for ds, byf in EXPECTED.items():
        cells = n30_cells(ds, sources=('full_rows', 'full_json'))
        if not cells:
            r.missing(f'{ds} noise_full', 'no noise_full n30 rows'); continue
        for f, texts in byf.items():
            for arm, text in zip(('clean', 'clipped', 'censored_em'), texts):
                if text is not None:
                    r.check(f'{ds} dnn hybrid {arm} level 0.25 fold {f} (noise_full)', text,
                            cells.get(('0.25', 'hybrid', 'dnn', arm, f)))
        allc = n30_cells(ds)
        for f in n30_folds(allc):
            if f not in byf and not (ds == 'caco2' and f == 0):
                r.new_fold(f'{ds} dnn hybrid level 0.25', f, _n30_line(allc, 'hybrid', 'dnn', ('clean', 'clipped', 'censored_em'), f))
    for ds in ('herg_ki', 'qm9'):
        cells = n30_cells(ds)
        fs = n30_folds(cells)
        r.info(f'{ds}: folds with level-0.25 results {fs} (text: fold 0 only)')
        for f in fs:
            if f != 0:
                for m in n30_models(ds):
                    r.new_fold(f'{ds} {m} hybrid level 0.25', f, _n30_line(cells, 'hybrid', m, ('clean', 'clipped', 'censored_em'), f))
    _n30_new_datasets(r, {'logd', 'caco2', 'herg_ki', 'qm9'})


@check('N7-22', 'Clipped labels: which move works best per dataset')
def n7_22(r):
    """Section 'Labels clipped at an assay limit', third paragraph. Same files, level 0.25, arms
    clean, clipped, constant_lift, truncated_normal_mean, censored_em."""
    EXPECTED = [
        ('herg_ki', 'base_2', 'dnn', 'clean', '0.5891'), ('herg_ki', 'base_2', 'dnn', 'clipped', '0.7485'),
        ('herg_ki', 'base_2', 'dnn', 'constant_lift', '0.666'), ('herg_ki', 'base_2', 'dnn', 'censored_em', '0.704'),
        ('buchwald_hartwig', 'hybrid', 'rf', 'censored_em', '17.585'), ('buchwald_hartwig', 'hybrid', 'rf', 'clipped', '18.691'),
        ('buchwald_hartwig', 'hybrid', 'rf', 'clean', '17.594'),
        ('flab_thermo_tresanco', 'hybrid', 'dnn', 'clean', '0.7487'), ('flab_thermo_tresanco', 'hybrid', 'dnn', 'clipped', '0.6097'),
        ('flab_thermo_tresanco', 'hybrid', 'dnn', 'constant_lift', '0.7164'),
        # rows added 2026-10-10 (10-10b) to tab:noise:res:censoring_moves_main and the sentence on MDR1
        ('adme_mdr1', 'hybrid', 'dnn', 'clean', '0.456'), ('adme_mdr1', 'hybrid', 'dnn', 'clipped', '0.511'),
        ('adme_mdr1', 'hybrid', 'dnn', 'constant_lift', '0.461'), ('adme_mdr1', 'hybrid', 'dnn', 'truncated_normal_mean', '0.483'),
        ('adme_mdr1', 'hybrid', 'dnn', 'censored_em', '0.466'),
        ('adme_mdr1', 'hybrid__general2', 'dnn', 'clean', '0.479'), ('adme_mdr1', 'hybrid__general2', 'dnn', 'constant_lift', '0.453'),
        ('ginkgo_titer', 'hybrid', 'dnn', 'clean', '0.348'), ('ginkgo_titer', 'hybrid', 'dnn', 'clipped', '0.3687'),
        ('ginkgo_titer', 'hybrid', 'dnn', 'constant_lift', '0.396'), ('ginkgo_titer', 'hybrid', 'dnn', 'truncated_normal_mean', '0.367'),
        ('ginkgo_titer', 'hybrid', 'dnn', 'censored_em', '0.3694'),
        ('ginkgo_acsins', 'hybrid', 'lightgbm', 'clean', '0.476'), ('ginkgo_acsins', 'hybrid', 'lightgbm', 'clipped', '0.464'),
        ('ginkgo_acsins', 'hybrid', 'lightgbm', 'constant_lift', '0.424'),
        ('ginkgo_acsins', 'hybrid', 'lightgbm', 'truncated_normal_mean', '0.364'),
        ('ginkgo_acsins', 'hybrid', 'lightgbm', 'censored_em', '0.407')]
    cited = collections.defaultdict(set)
    for ds, cfg, m, arm, text in EXPECTED:
        cells = n30_cells(ds); cited[ds].add((cfg, m))
        r.check(f'{ds} {m} {cfg} {arm} level 0.25 fold 0', text, cells.get(('0.25', cfg, m, arm, 0)))
    # Which of the three moves has the best mean over configurations and models, per dataset (no number in the text).
    # claims added 2026-10-10: best move per configuration and model (fold 0)
    def best_counts(ds):
        cells = n30_cells(ds); lib = n30_meta(ds).get('lower_is_better', True)
        cnt = collections.Counter(); lowered = 0; n = 0
        for (L, c, m, a, f), v in cells.items():
            if L != '0.25' or a != 'clean' or f != 0:
                continue
            arms = {x: cells.get(('0.25', c, m, x, 0)) for x in ('constant_lift', 'truncated_normal_mean', 'censored_em')}
            if None in arms.values():
                continue
            n += 1
            cnt[(min if lib else max)(arms, key=arms.get)] += 1
            cl = cells.get(('0.25', c, m, 'clipped', 0))
            lowered += (cl > v) if lib else (cl < v)
        return cnt, lowered, n
    cnt, _, n = best_counts('adme_mdr1')
    r.claim('adme_mdr1: constant lift best in most configuration-model pairs', cnt['constant_lift'] * 2 > n if n else None, f'{dict(cnt)} of {n}')
    cnt, low, n = best_counts('ginkgo_acsins')
    r.claim('ginkgo_acsins: clipping lowered Spearman in most pairs, censored EM best in most',
            (low * 2 > n and cnt['censored_em'] * 2 > n) if n else None, f'lowered {low} of {n}; best {dict(cnt)}')
    for ds in ['logd', 'qm9', 'herg_ki', 'caco2', 'buchwald_hartwig']:
        if not n30_has_results(ds):
            continue
        cells = n30_cells(ds); lib = n30_meta(ds).get('lower_is_better', True)
        means = {a: mean([v for k, v in cells.items() if k[0] == '0.25' and k[3] == a and k[4] == 0])
                 for a in ('constant_lift', 'truncated_normal_mean', 'censored_em')}
        means = {a: v for a, v in means.items() if v is not None}
        if means:
            best = (min if lib else max)(means, key=means.get)
            r.info(f'{ds}: mean over configurations and models, fold 0: ' + ', '.join(f'{a} {v:.4f}' for a, v in means.items())
                   + f' -> best {best}')
    for ds, cm in cited.items():
        _n30_new_folds(r, ds, sorted(cm), ('clean', 'clipped', 'constant_lift', 'truncated_normal_mean', 'censored_em'))
    _n30_new_datasets(r, {'logd', 'qm9', 'herg_ki', 'caco2', 'buchwald_hartwig', 'flab_thermo_tresanco', 'adme_mdr1',
                          'ginkgo_titer', 'ginkgo_acsins', 'ginkgo_colloidal', 'ddg_combined'},
                      arms=('clean', 'clipped', 'constant_lift', 'truncated_normal_mean', 'censored_em'))


@check('N7-23', 'Clipped labels: deleting the clipped molecules')
def n7_23(r):
    """Section 'Labels clipped at an assay limit', fourth paragraph. Same files, arms drop_clipped and
    clipped, level 0.25, the network."""
    EXPECTED = [('caco2', 'hybrid', 'drop_clipped', '0.3403'), ('caco2', 'hybrid', 'clipped', '0.3042'),
                ('herg_ki', 'base_2', 'drop_clipped', '0.8962'), ('herg_ki', 'base_2', 'clipped', '0.7485'),
                ('qm9', 'hybrid', 'drop_clipped', '0.5412'), ('qm9', 'hybrid', 'clipped', '0.5196'),
                ('logd', 'hybrid', 'drop_clipped', '0.5547'), ('logd', 'hybrid', 'clipped', '0.6049')]
    cited = collections.defaultdict(set)
    for ds, cfg, arm, text in EXPECTED:
        cells = n30_cells(ds); cited[ds].add((cfg, 'dnn'))
        r.check(f'{ds} dnn {cfg} {arm} level 0.25 fold 0', text, cells.get(('0.25', cfg, 'dnn', arm, 0)))
    for ds, cm in cited.items():
        _n30_new_folds(r, ds, sorted(cm), ('clipped', 'drop_clipped', 'constant_lift'))
    _n30_new_datasets(r, {'caco2', 'herg_ki', 'qm9', 'logd'}, arms=('clipped', 'drop_clipped', 'constant_lift'))


@check('N7-24', 'Clipped labels: per-molecule lift against shuffled lifts')
def n7_24(r):
    """Section 'Labels clipped at an assay limit', fifth paragraph. Same files, arms per_molecule_lift and
    per_molecule_lift_shuffled/seed0-2, level 0.25, the network."""
    SH = ['per_molecule_lift_shuffled/seed0', 'per_molecule_lift_shuffled/seed1', 'per_molecule_lift_shuffled/seed2']
    EXPECTED = [('logd', 'hybrid', ['0.499', '0.4714', '0.4673', '0.463']),
                ('herg_ki', 'base_2', ['0.7016', '0.6776', '0.6945', '0.6704']),
                ('caco2', 'hybrid', ['0.2878', '0.2750', '0.2734', '0.2708']),
                ('qm9', 'hybrid', ['0.4481', '0.4495', None, None]),
                # MDR1 rows added 2026-10-10 (10-10b)
                ('adme_mdr1', 'base_1', ['0.499', '0.485', '0.501', '0.489']),
                ('adme_mdr1', 'base_2', ['0.513', '0.515', '0.516', '0.503']),
                ('adme_mdr1', 'hybrid', ['0.481', '0.467', '0.468', '0.469']),
                ('adme_mdr1', 'hybrid__general2', ['0.480', '0.452', '0.469', '0.464'])]
    for ds, cfg, texts in EXPECTED:
        cells = n30_cells(ds)
        for arm, text in zip(['per_molecule_lift'] + SH, texts):
            v = cells.get(('0.25', cfg, 'dnn', arm, 0))
            if text is not None:
                r.check(f'{ds} dnn {cfg} {arm} level 0.25 fold 0', text, v)
        own = cells.get(('0.25', cfg, 'dnn', 'per_molecule_lift', 0))
        shf = [cells.get(('0.25', cfg, 'dnn', a, 0)) for a in SH]
        if own is not None and None not in shf:
            r.info(f'{ds}: shuffled lifts beating the own lift: {sum(s < own for s in shf)} of 3')
        _n30_new_folds(r, ds, [(cfg, 'dnn')], ['per_molecule_lift'] + SH)
    _n30_new_datasets(r, {'logd', 'herg_ki', 'caco2', 'qm9', 'adme_mdr1'}, arms=['per_molecule_lift'] + SH)


@check('N7-25', 'Clipped labels: two antibody sets did not need fixing')
def n7_25(r):
    """Section 'Labels clipped at an assay limit', sixth paragraph. n30 {ginkgo_colloidal,ddg_combined}.json
    level 0.25; the svm number is per_level['0.25']['base_1'][clean|clipped]['svm']."""
    EXPECTED = [('ddg_combined', 'base_1', 'svm', 'clean', '0.119'), ('ddg_combined', 'base_1', 'svm', 'clipped', '0.248')]
    for ds, cfg, m, arm, text in EXPECTED:
        r.check(f'{ds} {m} {cfg} {arm} level 0.25 fold 0', text, n30_cells(ds).get(('0.25', cfg, m, arm, 0)))
    for ds in ('ginkgo_colloidal', 'ddg_combined', 'ginkgo_titer'):  # titer added 2026-10-10 (EM raised it further)
        cells = n30_cells(ds)
        pairs = [(k[1], k[2]) for k in cells if k[0] == '0.25' and k[3] == 'clean' and k[4] == 0]
        if not pairs:
            r.missing(ds, 'no level-0.25 cells'); continue
        not_lower = sum(cells[('0.25', c, m, 'clipped', 0)] >= cells[('0.25', c, m, 'clean', 0)] for c, m in pairs)
        em_down = sum(cells.get(('0.25', c, m, 'censored_em', 0), 1e9) < cells[('0.25', c, m, 'clipped', 0)] for c, m in pairs)
        r.info(f'{ds}: clipping did not lower Spearman in {not_lower} of {len(pairs)} configuration-model pairs; '
               f'censored EM below clipped in {em_down} of {len(pairs)} (text: "most", "moved the score back down")')
        _n30_new_folds(r, ds, sorted(set(pairs)), ('clean', 'clipped', 'censored_em'))


@check('N7-26', 'Scaffold-family offsets under grouped-shifted noise')
def n7_26(r):
    """Section 'A shared offset per scaffold group', first paragraph. n34_group_random_effects/
    <ds>__grouped_shifted.rows.jsonl, fold 0, level 1.0, dnn; per draw: score_D (damaged),
    score_Dprime (method), score_clean."""
    EXPECTED = [  # (ds, cfg, method, draw, quantity, text)
        ('caco2', 'hybrid', 'random_intercept', 0, 'damaged', '0.321'), ('caco2', 'hybrid', 'random_intercept', 1, 'damaged', '0.3443'),
        ('caco2', 'hybrid', 'random_intercept', 2, 'damaged', '0.3612'),
        ('caco2', 'hybrid', 'random_intercept', 0, 'method', '0.280'), ('caco2', 'hybrid', 'random_intercept', 1, 'method', '0.2936'),
        ('caco2', 'hybrid', 'random_intercept', 2, 'method', '0.2668'),
        ('caco2', 'hybrid', 'random_intercept', 0, 'clean', '0.2385'), ('caco2', 'hybrid', 'random_intercept', 1, 'clean', '0.2423'),
        ('caco2', 'hybrid', 'random_intercept', 2, 'clean', '0.2314'),
        ('logd', 'hybrid', 'random_intercept', 0, 'damaged', '0.6837'), ('logd', 'hybrid', 'random_intercept', 1, 'damaged', '0.6899'),
        ('logd', 'hybrid', 'random_intercept', 0, 'method', '0.5457'), ('logd', 'hybrid', 'random_intercept', 1, 'method', '0.5725'),
        ('logd', 'hybrid', 'random_intercept', 0, 'clean', '0.3981'), ('logd', 'hybrid', 'random_intercept', 1, 'clean', '0.4017'),
        ('herg_ki', 'hybrid', 'group_center', 0, 'damaged', '0.8042'), ('herg_ki', 'hybrid', 'group_center', 0, 'method', '0.6914'),
        ('herg_ki', 'hybrid', 'random_intercept', 0, 'method', '0.6934')]
    # rows added 2026-10-10 (10-10b): QM9 draws 0-2, MDR1 draws 0-1
    for ds, draw, dmg, gc, ri, cl in (('qm9', 0, '0.909', '0.574', '0.507', '0.346'), ('qm9', 1, '0.844', '0.683', '0.571', '0.347'),
                                      ('qm9', 2, '1.295', '0.806', '0.638', '0.341'),
                                      ('adme_mdr1', 0, '0.536', '0.515', '0.513', '0.459'), ('adme_mdr1', 1, '0.511', '0.516', '0.499', '0.460')):
        EXPECTED += [(ds, 'hybrid', 'group_center', draw, 'damaged', dmg), (ds, 'hybrid', 'group_center', draw, 'method', gc),
                     (ds, 'hybrid', 'random_intercept', draw, 'method', ri), (ds, 'hybrid', 'random_intercept', draw, 'clean', cl)]
    Q = {'damaged': 'score_D', 'method': 'score_Dprime', 'clean': 'score_clean'}
    sub = 'n34_group_random_effects'
    for ds, cfg, meth, draw, q, text in EXPECTED:
        rr = rows(sub, f'{ds}__grouped_shifted*.rows.jsonl', model='dnn', level=1.0, fold=0, draw=draw,
                  model_input=cfg, method=meth)
        r.check(f'{ds} dnn {cfg} {meth} level 1.0 fold 0 draw {draw} {q}', text, rr[0][Q[q]] if rr else None)
    allr = rows(sub, '*__grouped_shifted*.rows.jsonl', where=('noise', 'full'), level=1.0, model_input='hybrid')
    for ds in sorted({x['dataset'] for x in allr}):
        dr = [x for x in allr if x['dataset'] == ds]
        for f in sorted({x['fold'] for x in dr}):
            fr = [x for x in dr if x['fold'] == f]
            line = '; '.join(
                f"{m} draw {d}: damaged {x['score_D']:.4f} -> {x['method']} {x['score_Dprime']:.4f} (clean {x['score_clean']:.4f})"
                for m in sorted({x['model'] for x in fr}) for d in sorted({x['draw'] for x in fr})
                for x in fr if x['model'] == m and x['draw'] == d and x['method'] == 'random_intercept')
            if ds not in ('caco2', 'logd', 'herg_ki', 'qm9', 'adme_mdr1'):
                r.new_dataset(ds, f'fold {f}, hybrid, level 1.0: {line}')
            elif f != 0:
                r.new_fold(f'{ds} hybrid level 1.0', f, line)
        drs = sorted({x['draw'] for x in dr if x['fold'] == 0 and x['model'] == 'dnn'})
        cited_draws = {'caco2': [0, 1, 2], 'logd': [0, 1, 2], 'herg_ki': [0, 1, 2], 'qm9': [0, 1, 2], 'adme_mdr1': [0, 1]}.get(ds)
        if cited_draws is not None and drs != cited_draws:
            r.info(f'{ds}: fold-0 dnn draws now {drs}, text cites {cited_draws}')
    for ds in datasets_with_files(sub):
        if ds in {x['dataset'] for x in allr}:
            continue
        d = J('noise', sub, f'{ds}__grouped_shifted.json')
        if d and d.get('has_computed_results'):
            r.new_dataset(ds, 'grouped_shifted .json with results but no rows file; read its summary by hand')


@check('N7-27', 'Scaffold-family offsets on the labels as delivered')
def n7_27(r):
    """Section 'A shared offset per scaffold group', second paragraph. Same rows files, level 0,
    dnn, hybrid, group_center: score_D -> score_Dprime."""
    EXPECTED = [('caco2', 'score_D', '0.233'), ('caco2', 'score_Dprime', '0.267'),
                ('logd', 'score_D', '0.3977'), ('logd', 'score_Dprime', '0.4872'),
                # rows added 2026-10-10 (10-10b)
                ('adme_mdr1', 'score_D', '0.466'), ('adme_mdr1', 'score_Dprime', '0.486'),
                ('qm9', 'score_D', '0.355'), ('qm9', 'score_Dprime', '0.400')]
    sub = 'n34_group_random_effects'
    for ds, q, text in EXPECTED:
        rr = rows(sub, f'{ds}__grouped_shifted*.rows.jsonl', model='dnn', level=0.0, fold=0, model_input='hybrid', method='group_center')
        r.check(f'{ds} dnn hybrid group_center level 0 fold 0 {"untreated" if q == "score_D" else "after"}', text, rr[0][q] if rr else None)
    for ds, text in (('adme_mdr1', '0.453'), ('qm9', '0.391')):
        rr = rows(sub, f'{ds}__grouped_shifted*.rows.jsonl', model='dnn', level=0.0, fold=0, model_input='hybrid', method='random_intercept')
        r.check(f'{ds} dnn hybrid random_intercept level 0 fold 0 after', text, rr[0]['score_Dprime'] if rr else None)
    allr = rows(sub, '*__grouped_shifted*.rows.jsonl', where=('noise', 'full'), level=0.0, method='group_center')
    for ds in sorted({x['dataset'] for x in allr}):
        for f in sorted({x['fold'] for x in allr if x['dataset'] == ds}):
            fr = [x for x in allr if x['dataset'] == ds and x['fold'] == f]
            line = '; '.join(f"{x['model']} {x['model_input']}: {x['score_D']:.4f} -> {x['score_Dprime']:.4f}"
                             for x in sorted(fr, key=lambda x: (x['model'], x['model_input'], x['_file'])))
            if ds not in ('caco2', 'logd', 'adme_mdr1', 'qm9'):
                r.new_dataset(ds, f'level 0, fold {f}, group_center ({fr[0]["_file"]}): {line}')
            elif f != 0:
                r.new_fold(f'{ds} level 0 group_center', f, line)


@check('N7-28', 'Several sources per molecule (simulated)')
def n7_28(r):
    """Section 'Several sources per molecule'. n39_correlated_sources/<ds>__sources.rows.jsonl, level 1.0,
    noise_condition 'herg_ic50|rho0|tau0.5' (correlation 0, offset SD 0.5), draw 0, fold 0,
    method offset_removed: score_D = plain mean, score_Dprime = offsets removed, score_clean."""
    EXPECTED = [('herg_ki', 'dnn', 'score_D', '0.742'), ('herg_ki', 'dnn', 'score_Dprime', '0.748'),
                ('herg_ki', 'dnn', 'score_clean', '0.5989'),
                ('buchwald_hartwig', 'rf', 'score_D', '21.315'), ('buchwald_hartwig', 'rf', 'score_Dprime', '21.080'),
                ('buchwald_hartwig', 'rf', 'score_clean', '17.797')]
    QN = {'score_D': 'plain mean', 'score_Dprime': 'offsets removed', 'score_clean': 'clean'}
    sub, cond = 'n39_correlated_sources', 'herg_ic50|rho0|tau0.5'
    for ds, m, q, text in EXPECTED:
        rr = rows(sub, f'{ds}__sources*.rows.jsonl', model=m, level=1.0, fold=0, draw=0, method='offset_removed',
                  noise_condition=cond, model_input=f'hybrid/{m}')
        r.check(f'{ds} {m} hybrid offset_removed level 1.0 rho0 tau0.5 fold 0 draw 0 {QN[q]}', text, rr[0][q] if rr else None)
    rr = rows(sub, 'buchwald_hartwig__sources*.rows.jsonl', method='offset_removed')
    if rr:
        won = sum(x['gain'] > 0 for x in rr)
        r.info(f'buchwald_hartwig offset_removed beats plain mean in {won} of {len(rr)} rows (text: "most settings")')
    allr = rows(sub, '*__sources*.rows.jsonl', where=('noise', 'full'), method='offset_removed', level=1.0,
                noise_condition=cond, draw=0)
    for ds in sorted({x['dataset'] for x in allr}):
        for f in sorted({x['fold'] for x in allr if x['dataset'] == ds}):
            fr = [x for x in allr if x['dataset'] == ds and x['fold'] == f and x['model_input'].startswith('hybrid/')]
            line = '; '.join(f"{x['model']} hybrid: plain mean {x['score_D']:.4f} -> offsets removed {x['score_Dprime']:.4f} "
                             f"(clean {x['score_clean']:.4f})" for x in fr)
            if ds not in ('herg_ki', 'buchwald_hartwig'):
                r.new_dataset(ds, f'level 1.0 rho0 tau0.5 draw 0 fold {f}: {line}')
            elif f != 0:
                r.new_fold(f'{ds} hybrid level 1.0 rho0 tau0.5 draw 0', f, line)


@check('N7-29', 'Huber teacher blend on hERG Ki')
def n7_29(r):
    """Section 'Gaussian noise, outliers and shifts in a few labels', first paragraph.
    n33_robust_training/herg_ki__<kind>.json summary.per_level['1'][config].per_model.dnn.per_arm.
    huber_teacher_blend.{noisy_mean, arm_mean, clean_mean, vs_noisy.passes_won}; 3 draws, fold 0.
    QM9: the text says no n33 QM9 file exists."""
    EXPECTED = [('gaussian', 'hybrid__general2', 'noisy_mean', '0.718'), ('gaussian', 'hybrid__general2', 'arm_mean', '0.681'),
                ('gaussian', 'hybrid__general2', 'clean_mean', '0.6075'),
                ('grouped_shifted', 'hybrid__general2', 'noisy_mean', '0.7788'), ('grouped_shifted', 'hybrid__general2', 'arm_mean', '0.7057'),
                ('outlier_p05', 'base_1', 'noisy_mean', '0.7413'), ('outlier_p05', 'base_1', 'arm_mean', '0.6980')]
    sub = 'n33_robust_training'

    def lvl(pl):
        for k in ('1', '1.0'):
            if k in pl:
                return pl[k]
        return None

    for kind, cfg, q, text in EXPECTED:
        d = J('noise', sub, f'herg_ki__{kind}.json')
        a = None
        try:
            a = lvl(d['summary']['per_level'])[cfg]['per_model']['dnn']['per_arm']['huber_teacher_blend']
        except Exception:
            pass
        r.check(f'herg_ki {kind} level 1.0 dnn {cfg} huber_teacher_blend {q} fold 0 (mean of draws)', text, a.get(q) if a else None)
    for kind, cfg in (('gaussian', 'hybrid__general2'), ('grouped_shifted', 'hybrid__general2'), ('outlier_p05', 'base_1')):
        d = J('noise', sub, f'herg_ki__{kind}.json')
        try:
            vs = lvl(d['summary']['per_level'])[cfg]['per_model']['dnn']['per_arm']['huber_teacher_blend']['vs_noisy']
            r.check(f'herg_ki {kind} dnn {cfg} draws won of 3', '3', vs['passes_won'], note=f"of {vs['n_passes_paired']} paired passes")
        except Exception:
            r.missing(f'herg_ki {kind} dnn {cfg} draws won', 'vs_noisy not found')
    # New folds / datasets / QM9 from the rows files (method huber_teacher_blend, main damaged level, mean over draws).
    allr = rows(sub, '*.rows.jsonl', where=('noise', 'full'), method='huber_teacher_blend')
    for ds in sorted({x['dataset'] for x in allr}):
        for kind in sorted({x['noise_condition'] for x in allr if x['dataset'] == ds}):
            kr = [x for x in allr if x['dataset'] == ds and x['noise_condition'] == kind]
            dmg = sorted({x['level'] for x in kr if x['level'] > 0})
            main = min(dmg, key=lambda L: abs(L - (0.25 if kind == 'censoring' else 1.0))) if dmg else None
            for f in sorted({x['fold'] for x in kr}):
                for L in sorted({x['level'] for x in kr if x['fold'] == f and x['level'] in (0, main)}):
                    g = collections.defaultdict(list)
                    for x in kr:
                        if x['fold'] == f and x['level'] == L:
                            g[(x['model'], x['model_input'])].append(x)
                    line = '; '.join(f"{m} {c}: {mean([x['score_D'] for x in v]):.4f} -> {mean([x['score_Dprime'] for x in v]):.4f}"
                                     f" (clean {mean([x['score_clean'] for x in v]):.4f}, {len(v)} draws)"
                                     for (m, c), v in sorted(g.items()))
                    lab = f'{kind} level {L:g} fold {f}'
                    if ds == 'qm9' and L == 0:
                        r.new_dataset(ds, f'QM9 with no noise added ({kind} file), fold {f}: {line}')
                    elif ds != 'herg_ki' and (L > 0 or f == 0):
                        if L > 0:
                            r.new_dataset(ds, f'{lab}: {line}')
                    elif ds == 'herg_ki' and f != 0 and L > 0:
                        r.new_fold(f'herg_ki {kind} level {L:g}', f, line)
    # QM9 with no noise added (rows added 2026-10-10 from 10-10b): gaussian file, level 0, fold 0, draw 0
    q = [x for x in rows('n33_robust_training', 'qm9__gaussian.rows.jsonl', dataset='qm9', level=0.0, fold=0, method='huber_teacher_blend')
         if x.get('draw') in (0, None)]
    for cfgn, tD, tP in (('hybrid__general2', '0.362', '0.368'), ('base_1', '0.396', '0.405')):
        g = [x for x in q if x['model'] == 'dnn' and str(x['model_input']).split(':')[-1].split('=')[0] == cfgn]
        r.check(f'qm9 L0 n33 dnn {cfgn} untreated', tD, g[0]['score_D'] if g else None)
        r.check(f'qm9 L0 n33 dnn {cfgn} huber blend', tP, g[0]['score_Dprime'] if g else None)
    qa = [x for x in rows('n33_robust_training', 'qm9__*.rows.jsonl', dataset='qm9', level=0.0, fold=0, method='huber_teacher_blend')]
    r.claim('qm9 L0 n33 huber blend worse in every configuration, both models', all(x['score_Dprime'] > x['score_D'] for x in qa) if qa else None,
            f'{sum(x["score_Dprime"] > x["score_D"] for x in qa)} of {len(qa)} rows worse')


# ---------------------------------------------------------------------------------------------
# N7-30 to N7-37: antibodies, hERG IC50, cleaning inside the feature search (n18, n02, n19, n29, n20, n15)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# N7-30 .. N7-37 (without N7-33): antibody datasets, hERG IC50, cleaning inside the feature search
# ---------------------------------------------------------------------------------------------
N18 = 'n18_clean_the_real_labels'
N20 = 'n20_clean_before_search'
N20_ARMS = ['search_asis__fit_asis', 'search_asis__fit_cleaned', 'search_cleaned__fit_cleaned',
            'search_cleaned__fit_asis']
ANTIBODY_DS = [d for d in CH7_DATASETS if DOMAIN.get(d) == 'antibodies']
_CFG_NAMES = ('base_1', 'base_2', 'hybrid', 'hybrid__general2', 'hybrid__distance1', 'hybrid_similarity')


def _cfg_any(model_input):
    """Configuration name from any model_input layout: 'dnn:base_1=prot_t5', 'base_1',
    'hybrid:rank_exp_d0.3/...' (n19, n29) -> the configuration name."""
    for tok in re.split(r'[:=]', str(model_input)):
        if tok in _CFG_NAMES:
            return tok
    return cfg(model_input)


def _n18_cell(ds, space, det, slot, model, cleanup):
    """n18 <ds>.json per_space.<space>.per_detector.<det>.<slot>.per_model.<model>:
    (no_cleaning_mean, <cleanup>.cleaned_mean, [shuffled_mean seed 0, 1, 2], folds_run).
    Any missing piece is None."""
    d = J('noise', N18, f'{ds}.json')
    if not d:
        return None, None, [], None
    sp = d.get('per_space', {}).get(space, {})
    m = sp.get('per_detector', {}).get(det, {}).get(slot, {}).get('per_model', {}).get(model, {})
    nc = m.get('no_cleaning_mean')
    cl = (m.get(cleanup) or {}).get('cleaned_mean')
    sh = sp.get('shuffled_control', {}).get(det, {}).get(slot, {}).get(model, {}).get(cleanup, {})
    shs = [sh[k].get('shuffled_mean') for k in sorted(sh, key=lambda s: int(s))] if sh else []
    return nc, cl, shs, d.get('folds_run')


def _n18_new_folds(r, ds, label, model, space, cleanup, det, slot):
    """Rows of n18 for folds other than 0 (noise and noise_full), same model, configuration,
    cleanup, detector and flag rate; plus a note if folds_run in the json grew."""
    d = J('noise', N18, f'{ds}.json')
    fr = (d or {}).get('folds_run') or [0]
    if any(f != 0 for f in fr):
        r.new_fold(label, ','.join(str(f) for f in fr if f != 0),
                   f'{ds}.json folds_run is now {fr}; the json means above now average these folds')
    rs = [x for x in rows(N18, f'{ds}*.rows.jsonl', where=('noise', 'full'), dataset=ds, model=model)
          if cfg(x.get('model_input')) == space and x.get('method') == cleanup and x.get('fold') not in (0, None)
          and f'detector={det}' in str(x.get('method_setting')) and str(x.get('method_setting')).endswith(f'flag_rate={slot}')]
    for f in sorted({x['fold'] for x in rs}):
        g = [x for x in rs if x['fold'] == f]
        r.new_fold(label, f, '; '.join(f"{x['method_setting']} [{x['similarity_setting']}] {x['score_D']:.4f} -> {x['score_Dprime']:.4f}" for x in g[:4]))


def _n18_other_antibodies(r, mentioned):
    """NEW DATASET lines: antibody datasets with an n18 file not mentioned in this paragraph
    (n18 small-molecule datasets are checked in N7-09). Quantity: per model and configuration,
    the cleanup x detector x flag rate with the largest gain over no cleaning, with the
    largest of its shuffled scores."""
    for ds in datasets_with_files(N18, where=('noise',)):
        if ds in mentioned or ds not in ANTIBODY_DS:
            continue
        d = J('noise', N18, f'{ds}.json')
        if not d or 'per_space' not in d:
            r.new_dataset(ds, 'n18 file has no per_space block')
            continue
        lib = d.get('lower_is_better')
        out = []
        for space, sp in d['per_space'].items():
            for model in d.get('models', []):
                best = None
                for det, w in sp.get('per_detector', {}).items():
                    for slot, x in w.items():
                        m = x.get('per_model', {}).get(model)
                        if not m or m.get('no_cleaning_mean') is None:
                            continue
                        for cl, c in m.items():
                            if not isinstance(c, dict) or c.get('cleaned_mean') is None:
                                continue
                            g = (m['no_cleaning_mean'] - c['cleaned_mean']) * (1 if lib else -1)
                            if best is None or g > best[0]:
                                sh = sp.get('shuffled_control', {}).get(det, {}).get(slot, {}).get(model, {}).get(cl, {})
                                shv = [v.get('shuffled_mean') for v in sh.values() if v.get('shuffled_mean') is not None]
                                best = (g, det, slot, cl, m['no_cleaning_mean'], c['cleaned_mean'],
                                        (min if lib else max)(shv) if shv else None)
                if best:
                    out.append(f"{model} {space} {best[3]} {best[1]}@{best[2]} {best[4]:.4f} -> {best[5]:.4f}"
                               + (f" (best shuffled {best[6]:.4f})" if best[6] is not None else ''))
        r.new_dataset(ds, f"n18 {d.get('metric')}, folds {d.get('folds_run')}; best cleanup per model and configuration: " + '; '.join(out))


@check('N7-30', 'SMAC colloidal stability, labels as delivered, n18')
def check_n7_30(r):
    """Section 'The antibody datasets', first paragraph (SMAC colloidal stability, deletion and
    out-of-fold move against shuffled detector scores). Key path: n18 ginkgo_colloidal.json
    per_space.<config>.per_detector.<det>.<flag rate slot>.per_model.<model>.{no_cleaning_mean,
    <cleanup>.cleaned_mean}; shuffled control per_space.<config>.shuffled_control.<det>.<slot>.
    <model>.<cleanup>.<seed>.shuffled_mean; fold_records.0.n_val. Fold 0.
    Deletion: svm, cross_val (forest residual), 20% flagged. Out-of-fold move: dnn, oof_smooth,
    base_1 with neighbor_residual at the 0.05 slot, hybrid with cross_val at the 0.01 slot
    (the slot each figure was read from)."""
    EXPECTED = [
        # (label stem, config, detector, slot, model, cleanup, text no cleaning, text cleaned, text shuffled list or max)
        ('svm base_1 remove cross_val@0.2', 'base_1', 'cross_val', '0.2', 'svm', 'remove', '0.274', '0.403', ['0.2207', '0.2345', '0.2782']),
        ('svm hybrid__general2 remove cross_val@0.2', 'hybrid__general2', 'cross_val', '0.2', 'svm', 'remove', '0.3108', '0.3782', ['0.3281', '0.2784', '0.3002']),
        ('dnn base_1 oof_smooth neighbor_residual@0.05', 'base_1', 'neighbor_residual', '0.05', 'dnn', 'oof_smooth', '0.0651', '0.3185', 'max 0.1327'),
        ('dnn hybrid oof_smooth cross_val@0.01', 'hybrid', 'cross_val', '0.01', 'dnn', 'oof_smooth', '0.2267', '0.4423', 'max 0.2228'),
    ]
    ds = 'ginkgo_colloidal'
    if J('noise', N18, f'{ds}.json') is None:
        r.missing(ds, f'noise/{N18}/{ds}.json not found')
        return
    for stem, sp, det, slot, model, cl, tnc, tcl, tsh in EXPECTED:
        nc, c, shs, _ = _n18_cell(ds, sp, det, slot, model, cl)
        r.check(f'{ds} {stem} no cleaning, fold 0', tnc, nc)
        r.check(f'{ds} {stem} cleaned, fold 0', tcl, c)
        if isinstance(tsh, list):
            for i, t in enumerate(tsh):
                r.check(f'{ds} {stem} shuffled seed {i}, fold 0', t, shs[i] if i < len(shs) else None)
        else:
            r.check(f'{ds} {stem} highest shuffled over seeds 0-2, fold 0', tsh.split()[1],
                    max(shs) if shs and None not in shs else None)
        _n18_new_folds(r, ds, f'{ds} {stem}', model, sp, cl, det, slot)
    fr = (J('noise', N18, f'{ds}.json') or {}).get('fold_records', {})
    f0 = fr.get('0') if isinstance(fr, dict) else None
    r.check(f'{ds} held-out antibodies, fold 0', '53', (f0 or {}).get('n_val'))
    _n18_other_antibodies(r, {ds, 'ddg_combined', 'ginkgo_titer'})


@check('N7-31', 'binding free energy and expression titer, labels as delivered, n18')
def check_n7_31(r):
    """Section 'The antibody datasets', second paragraph (binding free energy label smoothing,
    expression titer deletion, against shuffled scores). Key path as N7-30: n18 <ds>.json
    per_space.<config>.per_detector.cross_val.<slot>.per_model.<model>. ddg_combined: dnn
    label_smooth at the 0.2 slot (strength 1.0), base_1 and hybrid; ginkgo_titer: rf remove at the
    0.01 slot, base_2 and hybrid__general2. Fold 0. Also checks the closing claim (cleaned single
    representation above cleaned hybrid on both datasets) for these cells."""
    EXPECTED = [
        ('ddg_combined', 'dnn base_1 label_smooth cross_val@0.2', 'base_1', 'cross_val', '0.2', 'dnn', 'label_smooth', '0.341', '0.484', ['0.2772', '0.3520', '0.4595']),
        ('ddg_combined', 'dnn hybrid label_smooth cross_val@0.2', 'hybrid', 'cross_val', '0.2', 'dnn', 'label_smooth', '0.2589', '0.2866', 'max 0.2759'),
        ('ginkgo_titer', 'rf base_2 remove cross_val@0.01', 'base_2', 'cross_val', '0.01', 'rf', 'remove', '0.3445', '0.4068', 'max 0.4007'),
        ('ginkgo_titer', 'rf hybrid__general2 remove cross_val@0.01', 'hybrid__general2', 'cross_val', '0.01', 'rf', 'remove', '0.2244', '0.3168', 'max 0.2945'),
    ]
    cleaned = {}
    for ds, stem, sp, det, slot, model, cl, tnc, tcl, tsh in EXPECTED:
        if J('noise', N18, f'{ds}.json') is None:
            r.missing(f'{ds} {stem}', f'noise/{N18}/{ds}.json not found')
            continue
        nc, c, shs, _ = _n18_cell(ds, sp, det, slot, model, cl)
        cleaned[(ds, sp)] = c
        r.check(f'{ds} {stem} no cleaning, fold 0', tnc, nc)
        r.check(f'{ds} {stem} cleaned, fold 0', tcl, c)
        if isinstance(tsh, list):
            for i, t in enumerate(tsh):
                r.check(f'{ds} {stem} shuffled seed {i}, fold 0', t, shs[i] if i < len(shs) else None)
        else:
            r.check(f'{ds} {stem} highest shuffled over seeds 0-2, fold 0', tsh.split()[1],
                    max(shs) if shs and None not in shs else None)
        _n18_new_folds(r, ds, f'{ds} {stem}', model, sp, cl, det, slot)
    for ds, s, h in [('ddg_combined', 'base_1', 'hybrid'), ('ginkgo_titer', 'base_2', 'hybrid__general2')]:
        a, b = cleaned.get((ds, s)), cleaned.get((ds, h))
        r.claim(f'{ds} cleaned single above cleaned hybrid (cells above)', None if a is None or b is None else a > b,
                f'{s} {a if a is None else round(a, 4)} vs {h} {b if b is None else round(b, 4)}')


@check('N7-32', 'SMAC neighbor-agreement weights, n02')
def check_n7_32(r):
    """Section 'The antibody datasets', third paragraph (neighbor-based weights on SMAC, no
    control). Key path: n02 ginkgo_colloidal.rows.jsonl, method reliability_weight,
    method_setting k=10|scheme=rank|..., model dnn, level 0, fold 0, draw 0; score_D -> score_Dprime
    on base_1 and hybrid (the first hybrid)."""
    EXPECTED = [('base_1', '0.065', '0.209'), ('hybrid', '0.2267', '0.3703')]
    ds = 'ginkgo_colloidal'
    rs = rows('n02_sample_weighting', f'{ds}.rows.jsonl', where=('noise', 'full'), dataset=ds, model='dnn',
              method='reliability_weight')
    rs = [x for x in rs if str(x.get('method_setting', '')).startswith('k=10|scheme=rank|')]
    if not rs:
        r.missing(ds, 'no n02 dnn reliability_weight k=10 rank rows')
    for sp, tD, tP in EXPECTED:
        g = [x for x in rs if cfg(x.get('model_input')) == sp and x.get('fold') == 0 and x.get('draw') == 0
             and not x.get('level')]
        x = g[0] if g else {}
        r.check(f'{ds} dnn {sp} neighbor weights k10 rank, untreated, fold 0', tD, x.get('score_D'))
        r.check(f'{ds} dnn {sp} neighbor weights k10 rank, weighted, fold 0', tP, x.get('score_Dprime'))
        for f in sorted({x['fold'] for x in rs if cfg(x.get('model_input')) == sp and x.get('fold') not in (0, None)}):
            g = [x for x in rs if cfg(x.get('model_input')) == sp and x.get('fold') == f and not x.get('level')]
            r.new_fold(f'{ds} dnn {sp} neighbor weights k10 rank', f,
                       '; '.join(f"draw {x.get('draw')} {x['score_D']:.4f} -> {x['score_Dprime']:.4f}" for x in g))
    for o in datasets_with_files('n02_sample_weighting', where=('noise',)):
        if o == ds or o not in ANTIBODY_DS:
            continue
        g = rows('n02_sample_weighting', f'{o}.rows.jsonl', dataset=o, model='dnn', method='reliability_weight', fold=0)
        g = [x for x in g if str(x.get('method_setting', '')).startswith('k=10|scheme=rank|') and not x.get('level')]
        txt = '; '.join(f"dnn {cfg(x['model_input'])} {x['score_D']:.4f} -> {x['score_Dprime']:.4f}" for x in g
                        if cfg(x['model_input']) in ('base_1', 'hybrid'))
        r.new_dataset(o, ('n02 level 0, k=10 rank, fold 0: ' + txt) if txt else 'n02 file, no dnn k=10 rank rows at level 0')


@check('N7-34', 'hERG IC50 repeat measurements, n19 and n29')
def check_n7_34(r):
    """Section 'Repeat measurements on hERG IC50'. Key paths: n19 herg_ic50__per_fold_*.rows.jsonl,
    method_setting label_build=averaged, configuration hybrid (the first hybrid), fold 0, draws 0-2,
    score_D -> score_Dprime: count_weighted and spread_pooled_weighted (inverse error variance) for
    svm, count_weighted / count_feature / shrunk for dnn, xgboost_interval_aft against its point
    model (score_D) for model xgboost_aft. n29 herg_ic50.json scores.all.hybrid.{mean_label,
    per_record}.{svm,dnn}.f0 (one row per measurement against the mean label)."""
    EXPECTED = [
        ('svm', 'count_weighted', ['0.760', '0.7663', '0.7615'], ['0.728', '0.7218', '0.7068']),
        ('svm', 'spread_pooled_weighted', None, ['0.7351', '0.7377', '0.7382']),
        ('dnn', 'count_weighted', ['0.7119'], ['0.7647']),
        ('dnn', 'count_feature', ['0.7119', '0.7010', '0.6994'], ['0.6556', '0.6649', '0.6320']),
        ('dnn', 'shrunk', None, ['0.6733', '0.6677', '0.6589']),
        ('xgboost_aft', 'xgboost_interval_aft', ['0.7244'], ['0.7360']),
    ]
    _n7_34_curator(r)
    ds = 'herg_ic50'
    rs = [x for x in rows('n19_measurement_count', f'{ds}*.rows.jsonl', where=('noise', 'full'), dataset=ds)
          if x.get('method_setting') == 'label_build=averaged' and _cfg_any(x.get('model_input')) == 'hybrid']
    if not rs:
        r.missing(ds, 'no n19 herg_ic50 rows (label_build=averaged, hybrid)')
    for model, meth, tD, tP in EXPECTED:
        g = {x['draw']: x for x in rs if x.get('model') == model and x.get('method') == meth and x.get('fold') == 0}
        for i, t in enumerate(tD or []):
            r.check(f'n19 {model} hybrid {meth} untreated, fold 0 draw {i}', t, g.get(i, {}).get('score_D'))
        for i, t in enumerate(tP):
            r.check(f'n19 {model} hybrid {meth} treated, fold 0 draw {i}', t, g.get(i, {}).get('score_Dprime'))
        for f in sorted({x['fold'] for x in rs if x.get('model') == model and x.get('method') == meth and x.get('fold') not in (0, None)}):
            h = sorted((x for x in rs if x.get('model') == model and x.get('method') == meth and x.get('fold') == f), key=lambda x: x['draw'])
            r.new_fold(f'n19 {model} hybrid {meth}', f, ' '.join(f"d{x['draw']} {x['score_D']:.4f}->{x['score_Dprime']:.4f}" for x in h))
    d = J('noise', 'n29_document_bias', f'{ds}.json')
    if d is None:
        r.missing('n29 herg_ic50', f'noise/n29_document_bias/{ds}.json not found')
    else:
        h = d.get('scores', {}).get('all', {}).get('hybrid', {})
        for model, tm, tp in [('svm', '0.7668', '0.7652'), ('dnn', '0.7256', '0.7343')]:
            r.check(f'n29 {model} hybrid mean label, fold 0', tm, h.get('mean_label', {}).get(model, {}).get('f0'))
            r.check(f'n29 {model} hybrid one row per measurement, fold 0', tp, h.get('per_record', {}).get(model, {}).get('f0'))
            for fk in sorted(h.get('per_record', {}).get(model, {})):
                if fk != 'f0':
                    r.new_fold(f'n29 {model} hybrid', fk[1:], f"mean label {h['mean_label'][model].get(fk)} -> per record {h['per_record'][model][fk]}")
    r.info('herg_ic50 is the only dataset with repeat measurements; no NEW DATASET check for n19/n29 here.')


def _n20_sources():
    """n20 json files with per_regime, newest first: noise final, noise _partial, noise_full
    final, noise_full _partial. Returns [(where/relpath, name, dict)]."""
    out = []
    for w in ('noise', 'full'):
        base = CTX.path(w, N20)
        for sub in ('', '_partial'):
            dd = os.path.join(base, sub) if base else None
            for f in files(dd, '*.json'):
                if 'fingerprint' in f:
                    continue
                d = jload(os.path.join(dd, f))
                if d and 'per_regime' in d:
                    out.append((f"{w}/{sub + '/' if sub else ''}{f}", f[:-5], d))
    return out


def _n20_cell(d, regime, level, cleaned_in, cleanup, rate, mslot, key):
    """[arm1..arm4] at per_regime[regime][level][cleaned_in][cleanup][rate][mslot][arm][key]."""
    try:
        aa = d['per_regime'][regime][level][cleaned_in][cleanup][rate][mslot]
    except (KeyError, TypeError):
        return None
    return [aa.get(a, {}).get(key) for a in N20_ARMS]


def _n20_get(name, regime, level, cleaned_in, cleanup, rate, mslot, key, prefer=None):
    """First source (in _n20_sources order, or the named one) holding this cell."""
    for src, nm, d in _n20_sources():
        if nm != name or (prefer and not src.startswith(prefer)):
            continue
        v = _n20_cell(d, regime, level, cleaned_in, cleanup, rate, mslot, key)
        if v and any(x is not None for x in v):
            return src, v
    return None, None


def _n20_other_folds(r, label, name, regime, level, cleaned_in, cleanup, rate, mslot, cited_keys):
    """NEW FOLD: keys l{L}d{d}f{f} with f not in the cited keys, in any source."""
    seen = set()
    for src, nm, d in _n20_sources():
        if nm != name:
            continue
        try:
            aa = d['per_regime'][regime][level][cleaned_in][cleanup][rate][mslot]
        except (KeyError, TypeError):
            continue
        keys = set()
        for a in aa.values():
            keys |= set(a)
        for k in sorted(keys):
            m = re.match(r'l.*d(\d+)f(\d+)$', k)
            if not m or k in cited_keys or k in seen or m.group(2) == '0':
                continue
            seen.add(k)
            v = [aa.get(a, {}).get(k) for a in N20_ARMS]
            r.new_fold(label, m.group(2), f"draw {m.group(1)} [{src}] " + ' / '.join('NA' if x is None else f'{x:.4f}' for x in v)
                       + '  (search as given, fit as given / fit cleaned / both cleaned / search cleaned, fit as given)')


def _n20_new_datasets(r, mentioned):
    names = sorted({nm.split('__')[0] for _, nm, _ in _n20_sources()} |
                   set(datasets_with_files(N20, where=('noise', 'full'))))
    for ds in names:
        if ds in mentioned or ds not in CH7_DATASETS:
            continue
        kinds = sorted({nm.split('__', 1)[1] for _, nm, _ in _n20_sources() if nm.startswith(ds + '__')})
        r.new_dataset(ds, f'n20 json with per_regime for damage kinds {kinds}; the four arms per cleanup are in '
                          f'per_regime[regime][level][cleaned_in][cleanup][flag_rate][model__slot][arm][l<L>d<d>f<f>]')


def _n7_34_curator(r):
    """Sentence added 2026-10-10: n22_curator_flag/herg_ic50.json (10-10b), curator flags as ground truth."""
    d = J('noise', 'n22_curator_flag', 'herg_ic50.json')
    if not d:
        r.missing('herg_ic50 n22', 'n22_curator_flag/herg_ic50.json not found')
        return
    r.check('herg_ic50 n22 records flagged', '301', d.get('n_records_flagged'))
    r.check('herg_ic50 n22 records', '12,008', d.get('n_records'))
    aps, ch = [], []
    for sp, v in (d.get('per_space') or {}).items():
        for det in ('neighbor_residual', 'cross_val'):
            x = ((v or {}).get(det) or {}).get('records') or {}
            if x.get('detection_average_precision') is not None:
                aps.append(x['detection_average_precision']); ch.append(x.get('chance_average_precision'))
    r.check('herg_ic50 n22 neighbor gap / forest residual AP, lowest over spaces', '0.14', min(aps) if aps else None)
    r.check('herg_ic50 n22 neighbor gap / forest residual AP, highest over spaces', '0.23', max(aps) if aps else None)
    r.check('herg_ic50 n22 chance AP', '0.036', ch[0] if ch else None)
    r.claim('herg_ic50 n22 every AP above chance', all(a > c for a, c in zip(aps, ch)) if aps else None, f'{len(aps)} readings')


@check('N7-35', 'cleaning before the search: titer and SMAC censoring, n20')
def check_n7_35(r):
    """Section 'Cleaning Inside the Feature Search', first paragraph. Key path: n20
    <ds>__<kind>.json (final, else _partial) per_regime[regime][L][cleaned_in][cleanup][flag_rate]
    [model__slot][arm]['l{L}d{d}f{f}'], arms in the order search as given, fit as given / search as
    given, fit cleaned / both cleaned / search cleaned, fit as given. Fold 0, one draw.
    ginkgo_titer: labels as delivered (regime as_they_come, key lnoned0f0; same values under
    injected level 0), dnn__hybrid, remove at flag rate 0.2, cleaned in base_1.
    ginkgo_colloidal: censoring 0.25, dnn__hybrid, remove at 0.05, cleaned in base_1."""
    EXPECTED = [
        ('ginkgo_titer', 'ginkgo_titer__gaussian', [('as_they_come', 'none', 'lnoned0f0'), ('injected', '0', 'l0d0f0')],
         'base_1', 'remove', '0.2', 'dnn__hybrid', ['0.155', '0.3899', '0.562', '0.4231']),
        ('ginkgo_colloidal', 'ginkgo_colloidal__censoring', [('injected', '0.25', 'l0.25d0f0')],
         'base_1', 'remove', '0.05', 'dnn__hybrid', ['0.2369', '0.4165', '0.4889', '0.2214']),
    ]
    armname = ['search as given, fit as given', 'search as given, fit cleaned', 'both cleaned', 'search cleaned, fit as given']
    for ds, name, places, ci, cl, rate, ms, texts in EXPECTED:
        src = v = None
        for reg, lv, key in places:
            src, v = _n20_get(name, reg, lv, ci, cl, rate, ms, key)
            if v:
                break
        stem = f'{ds} {ms} {cl}@{rate} cleaned in {ci}, {reg} {lv}'
        if not v:
            r.missing(stem, f'{name}: cell not found in any n20 file')
            v = [None] * 4
        for i, t in enumerate(texts):
            r.check(f'{stem}: {armname[i]}, fold 0', t, v[i], note=src or '')
        _n20_other_folds(r, stem, name, reg, lv, ci, cl, rate, ms, {key})
    _n20_new_datasets(r, {'ginkgo_titer', 'ginkgo_colloidal'})


@check('N7-36', 'cleaning before the search: largest gain and reversals, n20')
def check_n7_36(r):
    """Section 'Cleaning Inside the Feature Search', second paragraph. Same key path as N7-35.
    SMAC labels as delivered (injected level 0 in ginkgo_colloidal__censoring.json): dnn__hybrid__general2
    js_shrink cleaned in base_2, search cleaned/fit as given (the same at every flag rate);
    dnn__hybrid remove@0.05 cleaned in base_1, both cleaned. hERG Ki Gaussian 1.0
    (_partial/herg_ki__gaussian.json): dnn__hybrid__general2 remove@0.05 cleaned in base_1, both
    cleaned, draws 0 and 1. hERG Ki labels as delivered fold 1 (noise_full _partial
    herg_ki__grouped_shifted.json, key l0d0f1): dnn__hybrid label_smooth cleaned in base_2, both
    cleaned, against no cleaning."""
    out = []
    # (label, file name, regime, level, cleaned_in, cleanup, rate, slot, key, arm index, text)
    E = [
        ('ginkgo_colloidal dnn hybrid__general2 js_shrink in base_2, level 0: as given, fold 0', 'ginkgo_colloidal__censoring', 'injected', '0', 'base_2', 'js_shrink', '0.05', 'dnn__hybrid__general2', 'l0d0f0', 0, '-0.014'),
        ('ginkgo_colloidal dnn hybrid__general2 js_shrink in base_2, level 0: search cleaned fit as given, fold 0', 'ginkgo_colloidal__censoring', 'injected', '0', 'base_2', 'js_shrink', '0.05', 'dnn__hybrid__general2', 'l0d0f0', 3, '0.560'),
        ('ginkgo_colloidal dnn hybrid remove@0.05 in base_1, level 0: as given, fold 0', 'ginkgo_colloidal__censoring', 'injected', '0', 'base_1', 'remove', '0.05', 'dnn__hybrid', 'l0d0f0', 0, '0.2267'),
        ('ginkgo_colloidal dnn hybrid remove@0.05 in base_1, level 0: both cleaned, fold 0', 'ginkgo_colloidal__censoring', 'injected', '0', 'base_1', 'remove', '0.05', 'dnn__hybrid', 'l0d0f0', 2, '0.0853'),
        ('herg_ki gaussian 1 dnn hybrid__general2 remove@0.05 in base_1: as given, fold 0 draw 0', 'herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', 'l1d0f0', 0, '0.7181'),
        ('herg_ki gaussian 1 dnn hybrid__general2 remove@0.05 in base_1: both cleaned, fold 0 draw 0', 'herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', 'l1d0f0', 2, '0.6488'),
        ('herg_ki gaussian 1 dnn hybrid__general2 remove@0.05 in base_1: as given, fold 0 draw 1', 'herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', 'l1d1f0', 0, '0.7583'),
        ('herg_ki gaussian 1 dnn hybrid__general2 remove@0.05 in base_1: both cleaned, fold 0 draw 1', 'herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', 'l1d1f0', 2, '0.7763'),
        ('herg_ki level 0 dnn hybrid label_smooth in base_2 (grouped_shifted file): both cleaned, fold 1', 'herg_ki__grouped_shifted', 'injected', '0', 'base_2', 'label_smooth', '0.05', 'dnn__hybrid', 'l0d0f1', 2, '0.5294'),
        ('herg_ki level 0 dnn hybrid label_smooth in base_2 (grouped_shifted file): no cleaning, fold 1', 'herg_ki__grouped_shifted', 'injected', '0', 'base_2', 'label_smooth', '0.05', 'dnn__hybrid', 'l0d0f1', 0, '0.5506'),
    ]
    for label, name, reg, lv, ci, cl, rate, ms, key, ai, text in E:
        src, v = _n20_get(name, reg, lv, ci, cl, rate, ms, key)
        r.check(label, text, v[ai] if v else None, note=src or 'not found')
    # same at every flag rate (js_shrink cell)
    vals = []
    for rate in ('0.05', '0.1', '0.2'):
        _, v = _n20_get('ginkgo_colloidal__censoring', 'injected', '0', 'base_2', 'js_shrink', rate, 'dnn__hybrid__general2', 'l0d0f0')
        vals.append(v[3] if v else None)
    r.claim('ginkgo_colloidal js_shrink search-cleaned arm same at flag rates 0.05/0.1/0.2', None if None in vals else len({round(x, 6) for x in vals}) == 1,
            ' / '.join('NA' if x is None else f'{x:.4f}' for x in vals))
    _n20_other_folds(r, 'ginkgo_colloidal dnn hybrid__general2 js_shrink in base_2, level 0', 'ginkgo_colloidal__censoring', 'injected', '0', 'base_2', 'js_shrink', '0.05', 'dnn__hybrid__general2', {'l0d0f0'})
    _n20_other_folds(r, 'herg_ki gaussian 1 dnn hybrid__general2 remove@0.05 in base_1', 'herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', {'l1d0f0', 'l1d1f0'})
    _n20_other_folds(r, 'herg_ki level 0 dnn hybrid label_smooth in base_2', 'herg_ki__grouped_shifted', 'injected', '0', 'base_2', 'label_smooth', '0.05', 'dnn__hybrid', {'l0d0f0', 'l0d0f1'})
    rf = [x for x in rows(N20, '*.rows.jsonl', where=('noise', 'full')) if 'random_flag' in str(x.get('method', '')) + str(x.get('method_setting', ''))]
    r.info(f'n20 random-flag control rows now: {len(rf)} (text: none has run)')
    _n20_new_datasets(r, {'ginkgo_colloidal', 'herg_ki'})


@check('N7-37', 'search on clean against damaged labels, n15')
def check_n7_37(r):
    """Section 'Cleaning Inside the Feature Search', third paragraph. Key path (noise_view):
    n15_clean_vs_noisy_hybrid/<ds>.json per_fold[f].draws['gaussian_L1_d<d>'].models[m].by_slot.
    hybrid__general2.{clean_arm_score, noisy_arm_score}, mean over the draws at level 1.0.
    caco2 dnn folds 0-4 (3 decimals), logd svm 'agree to the third decimal on two of three folds',
    herg_ki dnn fold 1 and its worst draw."""
    where = 'view' if CTX.d('view') else 'noise'
    if where == 'noise':
        r.info('no --noise-view given; reading n15 from <root>/noise')
    sub = 'n15_clean_vs_noisy_hybrid'

    def per_fold(ds, model):
        d = J(where, sub, f'{ds}.json')
        if not d:
            return None, {}
        out = {}
        for f, F in (d.get('per_fold') or {}).items():
            c, n = [], []
            for k, v in (F.get('draws') or {}).items():
                if not k.startswith('gaussian_L1_'):
                    continue
                x = (v.get('models', {}).get(model) or {}).get('by_slot', {}).get('hybrid__general2')
                if x and x.get('clean_arm_score') is not None and x.get('noisy_arm_score') is not None:
                    c.append(x['clean_arm_score']); n.append(x['noisy_arm_score'])
            if c:
                out[int(f)] = (mean(c), mean(n), n)
        return d, out

    d, pf = per_fold('caco2', 'dnn')
    if d is None:
        r.missing('caco2', f'{where}/{sub}/caco2.json not found')
    for f, (tc, tn) in enumerate([('0.298', '0.303'), ('0.328', '0.329'), ('0.328', '0.334'), ('0.368', '0.365'), ('0.346', '0.340')]):
        r.check(f'caco2 dnn hybrid__general2 searched on clean labels, fold {f}', tc, pf.get(f, (None,))[0])
        r.check(f'caco2 dnn hybrid__general2 searched on damaged labels, fold {f}', tn, pf.get(f, (None, None))[1])
    cited = {'caco2': range(5), 'ginkgo_colloidal': range(5), 'herg_ki': range(3), 'logd': range(3), 'buchwald_hartwig': range(3)}
    for f in sorted(set(pf) - set(cited['caco2'])):
        r.new_fold('caco2 dnn hybrid__general2 clean / damaged search', f, f'{pf[f][0]:.4f} / {pf[f][1]:.4f}')
    d, pf = per_fold('logd', 'svm')
    agree = [round(c, 3) == round(n, 3) for f, (c, n, _) in sorted(pf.items()) if f in cited['logd']]
    r.claim('logd svm hybrid__general2: arms agree to 3 decimals on 2 of 3 folds', None if len(agree) < 3 else sum(agree) == 2,
            ', '.join(f'fold {f} {c:.4f}/{n:.4f}' for f, (c, n, _) in sorted(pf.items())))
    for f in sorted(set(pf) - set(cited['logd'])):
        r.new_fold('logd svm hybrid__general2 clean / damaged search', f, f'{pf[f][0]:.4f} / {pf[f][1]:.4f}')
    d, pf = per_fold('herg_ki', 'dnn')
    r.check('herg_ki dnn hybrid__general2 searched on clean labels, fold 1', '0.669', pf.get(1, (None,))[0])
    r.check('herg_ki dnn hybrid__general2 searched on damaged labels, fold 1', '0.779', pf.get(1, (None, None))[1])
    r.check('herg_ki dnn hybrid__general2 damaged-label search, worst draw, fold 1', '0.979', max(pf[1][2]) if 1 in pf else None)
    for f in sorted(set(pf) - set(cited['herg_ki'])):
        r.new_fold('herg_ki dnn hybrid__general2 clean / damaged search', f, f'{pf[f][0]:.4f} / {pf[f][1]:.4f}')
    # SMAC: arms swap places from fold to fold (no number)
    d, pf = per_fold('ginkgo_colloidal', 'dnn')
    if pf:
        r.info('ginkgo_colloidal dnn hybrid__general2 clean / damaged search per fold: ' +
               ', '.join(f'f{f} {c:.4f}/{n:.4f}' for f, (c, n, _) in sorted(pf.items())))
    for f in sorted(set(pf) - set(cited['ginkgo_colloidal'])):
        r.new_fold('ginkgo_colloidal dnn hybrid__general2 clean / damaged search', f, f'{pf[f][0]:.4f} / {pf[f][1]:.4f}')
    # rows added 2026-10-10 (10-10b): DeltaDeltaG folds 0-4, QM9 folds 0-2
    NEWR = {('ddg_combined', 'dnn'): [('0.328', '0.225'), ('0.110', '0.141'), ('0.164', '0.095'), ('0.506', '0.516'), ('0.239', '0.005')],
            ('ddg_combined', 'svm'): [('0.06994', '0.06987'), ('0.178', '0.200'), ('0.202', '0.171'), ('0.038', '0.332'), ('0.131', '0.190')],
            ('qm9', 'dnn'): [('0.544', '0.553'), ('0.450', '0.460'), ('0.4654', '0.4651')],
            ('qm9', 'lightgbm'): [('0.563', '0.556'), ('0.471', '0.458'), ('0.500', '0.507')]}
    gaps = {}
    for (ds, m), vals in NEWR.items():
        _, pf = per_fold(ds, m)
        for f, (tc, tn) in enumerate(vals):
            r.check(f'{ds} {m} hybrid__general2 searched on clean labels, fold {f}', tc, pf.get(f, (None,))[0])
            r.check(f'{ds} {m} hybrid__general2 searched on damaged labels, fold {f}', tn, pf.get(f, (None, None))[1])
            if f in pf:
                gaps.setdefault(ds, []).append(abs(pf[f][0] - pf[f][1]))
    r.check('ddg_combined largest clean/damaged search gap over folds and models', '0.29', max(gaps['ddg_combined']) if gaps.get('ddg_combined') else None)
    r.claim('qm9 clean/damaged search within 0.013 RMSE', max(gaps['qm9']) <= 0.013 if gaps.get('qm9') else None,
            f"max {max(gaps['qm9']):.4f}" if gaps.get('qm9') else '')
    for ds in datasets_with_files(sub, where=(where,)):
        if ds in ('caco2', 'logd', 'herg_ki', 'ginkgo_colloidal', 'ddg_combined', 'qm9'):
            continue
        dd = J(where, sub, f'{ds}.json') or {}
        parts = []
        for m in dd.get('models', []):
            _, p = per_fold(ds, m)
            if p:
                parts.append(f'{m}: ' + ', '.join(f'f{f} {c:.4f}/{n:.4f}' for f, (c, n, _) in sorted(p.items())))
        r.new_dataset(ds, f"n15 {dd.get('metric')}, folds {dd.get('folds_run')}, hybrid__general2 clean / damaged search at Gaussian 1.0: " + ('; '.join(parts) or 'no scores'))


# ---------------------------------------------------------------------------------------------
# N7-38 to N7-47: changing the model, the expanded sweep, representation (n40, n33, n38, n32, n09, n13, n12)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# N7-38 to N7-45, N7-47: changing the model, the expanded published sweep, neighbour space
# ---------------------------------------------------------------------------------------------
N40 = 'n40_model_side_regularisation'


def _n40_scores(d, level, space, model):
    """{arm: score} for one configuration and model of an n40 file: 'fixed', each per_arm
    (score = fixed_mean - mean_gain, sign flipped when higher is better) and 'stack'."""
    lib = d.get('lower_is_better', True)
    try:
        sp = d['summary']['per_level'][level][space]
        pm = sp['per_model'][model]
    except (KeyError, TypeError):
        return None
    fx = pm['fixed_mean']
    out = {'fixed': fx}
    for arm, e in pm.get('per_arm', {}).items():
        g = (e.get('vs_fixed') or {}).get('mean_gain')
        if g is not None:
            out[arm] = fx - g if lib else fx + g
        if arm == 'seed_avg':
            out['_seed_avg_won'] = ((e.get('vs_fixed') or {}).get('passes_won'), (e.get('vs_fixed') or {}).get('n_passes_paired'))
        if 'clean_mean' in e and e['clean_mean'] is not None:
            out['_clean_mean'] = e['clean_mean']
    st = (sp.get('stack') or {}).get(model)
    if st and st.get('mean_gain') is not None:
        out['stack'] = fx - st['mean_gain'] if lib else fx + st['mean_gain']
    return out


def _n40_best(d, level, spaces, model):
    """Best score per arm over the given configurations (each arm picked on its own score),
    as the earlier extraction (ch7_sources/n40.py) did. {arm: (score, configuration)}."""
    lib = d.get('lower_is_better', True)
    best = {}
    for sp in spaces:
        s = _n40_scores(d, level, sp, model)
        if not s:
            continue
        for a, v in s.items():
            if a.startswith('_'):
                continue
            if a not in best or (v < best[a][0] if lib else v > best[a][0]):
                best[a] = (v, sp)
    return best


def _n40_new(r, cited):
    """NEW FOLD and NEW DATASET lines for n40."""
    for f in files(CTX.path('noise', N40), '*.json'):
        d = jload(os.path.join(CTX.path('noise', N40), f))
        if not d or 'summary' not in d:
            continue
        extra = [x for x in (d.get('folds_run') or []) if x != 0]
        if extra:
            r.new_fold(f'n40 {f[:-5]}', ','.join(map(str, extra)),
                       'summary now averages over folds ' + str(d.get('folds_run')) + '; per-fold gains in per_arm.*.vs_fixed.gain_over_folds')
    for ds in datasets_with_files(N40, where=('noise',)):
        if ds in cited:
            continue
        d = J('noise', N40, f'{ds}__gaussian.json')
        if not d or 'summary' not in d:
            d = next((x for x in (J('noise', N40, f) for f in files(CTX.path('noise', N40), ds + '__*.json'))
                      if x and 'summary' in x), None)
        if not d or 'summary' not in d:
            r.new_dataset(ds, 'n40 files present, none with scores (result_status of each: ' + ', '.join(sorted({str((J('noise', N40, f) or {}).get('result_status')) for f in files(CTX.path('noise', N40), ds + '*.json')})) + ')')
            continue
        pl = d['summary']['per_level']
        lv = '1' if '1' in pl else max(pl, key=float)
        parts = []
        for m in d.get('models', []):
            for grp, sps in (('single', ('base_1', 'base_2')), ('hybrid', ('hybrid', 'hybrid__general2'))):
                b = _n40_best(d, lv, sps, m)
                if 'fixed' in b:
                    parts.append(f'{m} {grp} fixed {b["fixed"][0]:.4f}' + ''.join(
                        f', {a} {b[a][0]:.4f}' for a in ('seed_avg', 'stack', 'retuned', 'tuned_clean_reused') if a in b))
        r.new_dataset(ds, f'n40 {d.get("noise_condition")} level {lv}: ' + '; '.join(parts))


@check('N7-38', 'Changing the model: hERG Ki Gaussian 1.0, network second hybrid')
def n7_38(r):
    """Section 'Changing the Model Instead of the Labels', first paragraph. n40
    herg_ki__gaussian.json summary.per_level['1'].hybrid__general2.per_model.dnn; method score =
    fixed_mean - per_arm[arm].vs_fixed.mean_gain; stack from ...hybrid__general2.stack.dnn;
    clean from per_arm.retuned.clean_mean; fold 0, draws 0-2."""
    EXPECTED = {'fixed': '0.718', 'seed_avg': '0.656', 'stack': '0.6713', 'retuned': '0.6911', 'clean': '0.6075'}
    d = J('noise', N40, 'herg_ki__gaussian.json')
    if not d:
        r.missing('herg_ki gaussian', 'n40 herg_ki__gaussian.json missing'); return
    s = _n40_scores(d, '1', 'hybrid__general2', 'dnn') or {}
    for a, t in EXPECTED.items():
        now = s.get('_clean_mean') if a == 'clean' else s.get(a)
        r.check(f'herg_ki gaussian 1.0 dnn hybrid__general2 {a}', t, now)
    h1 = _n40_scores(d, '1', 'hybrid', 'dnn') or {}
    if 'stack' in h1:
        r.info(f'stack on the first hybrid (hybrid) is {h1["stack"]:.4f}')
    w = s.get('_seed_avg_won')
    r.claim('herg_ki gaussian 1.0 dnn hybrid__general2 seed_avg won every draw', None if not w else w[0] == w[1] and w[1] == 3,
            f'won {w[0]} of {w[1]} draws' if w else 'no seed_avg arm')
    # QM9 Gaussian 1.0 rows added 2026-10-10 (10-10b) to tab:noise:res:change_model (block of N7-40)
    BLOCKS[r.id] = BLOCKS.get(r.id, '') + BLOCKS.get('N7-40', '')
    dq = J('noise', N40, 'qm9__gaussian.json')
    QT = {'dnn': {'base_1': ('0.595', '0.535', '0.528', '0.560', '0.579', '0.400'), 'base_2': ('0.641', '0.597', '0.514', '0.626', '0.641', '0.487'),
                  'hybrid': ('0.521', '0.479', '0.494', '0.515', '0.518', '0.357'), 'hybrid__general2': ('0.547', '0.476', '0.505', '0.511', '0.529', '0.358')},
          'lightgbm': {'base_1': ('0.582', '0.533', '0.528', '0.542', '0.550', '0.412'), 'base_2': ('0.565', '0.540', '0.514', '0.574', '0.703', '0.459'),
                       'hybrid': ('0.564', '0.504', '0.494', '0.506', '0.611', '0.378'), 'hybrid__general2': ('0.563', '0.506', '0.505', '0.499', '0.567', '0.382')}}
    for m, per in QT.items():
        for sp, t in per.items():
            q = _n40_scores(dq, '1', sp, m) if dq else None
            for a, tv in zip(('fixed', 'seed_avg', 'stack', 'retuned', 'tuned_clean_reused', '_clean_mean'), t):
                r.check(f'qm9 gaussian 1.0 {m} {sp} {a}', tv, (q or {}).get(a))
    _n40_new(r, {'herg_ki', 'logd', 'qm9'})


@check('N7-39', 'Changing the model on the labels as delivered (LogD, hERG Ki)')
def n7_39(r):
    """Section 'Changing the Model', second paragraph. n40 <ds>__gaussian.json level '0'; best
    per arm over base_1/base_2 (single) and hybrid/hybrid__general2 (hybrid), as
    ch7_sources/n40.py; stack against the support vector regressor = best svm single fixed vs
    best single stack."""
    dl = J('noise', N40, 'logd__gaussian.json')
    dh = J('noise', N40, 'herg_ki__gaussian.json')
    if not dl:
        r.missing('logd', 'n40 logd__gaussian.json missing')
    else:
        s = _n40_best(dl, '0', ('base_1', 'base_2'), 'dnn')
        h = _n40_best(dl, '0', ('hybrid', 'hybrid__general2'), 'dnn')
        r.check('logd L0 dnn best single fixed', '0.4167', s.get('fixed', (None,))[0])
        r.check('logd L0 dnn best single seed_avg', '0.4049', s.get('seed_avg', (None,))[0])
        r.check('logd L0 dnn best hybrid fixed', '0.3935', h.get('fixed', (None,))[0])
        r.check('logd L0 dnn best hybrid seed_avg', '0.3756', h.get('seed_avg', (None,))[0])
        sv = _n40_best(dl, '0', ('base_1', 'base_2'), 'svm')
        r.check('logd L0 svm best single fixed', '0.475', sv.get('fixed', (None,))[0])
        r.check('logd L0 best single stack', '0.408', sv.get('stack', (None,))[0])
        dc = J('noise', N40, 'logd__censoring.json')
        if dc:
            r.check('logd L0 dnn base_2 fixed in the censoring file', '0.4212',
                    (_n40_scores(dc, '0', 'base_2', 'dnn') or {}).get('fixed'))
    if not dh:
        r.missing('herg_ki', 'n40 herg_ki__gaussian.json missing'); return
    s = _n40_best(dh, '0', ('base_1', 'base_2'), 'dnn')
    if 'fixed' in s:
        fx = s['fixed'][0]
        improved = [a for a in ('seed_avg', 'stack', 'retuned') if a in s and round(s[a][0], 3) < round(fx, 3)]
        r.claim('herg_ki L0 dnn best single: none of seed_avg, stack, retuned improves', not improved,
                f'fixed {fx:.4f}; ' + ', '.join(f'{a} {s[a][0]:.4f}' for a in ('seed_avg', 'stack', 'retuned') if a in s))
    h = _n40_best(dh, '0', ('hybrid', 'hybrid__general2'), 'dnn')
    r.check('herg_ki L0 dnn best hybrid fixed', '0.5972', h.get('fixed', (None,))[0])
    r.check('herg_ki L0 dnn best hybrid retuned', '0.6229', h.get('retuned', (None,))[0])


@check('N7-40', 'Retuning on damaged labels against settings tuned on clean labels')
def n7_40(r):
    """Section 'Changing the Model', third paragraph. n40 herg_ki__gaussian.json level '1'
    hybrid__general2 dnn: retuned vs tuned_clean_reused; herg_ki__censoring.json level '0.25':
    retuned beats fixed in how many of the 8 model x configuration pairs."""
    d = J('noise', N40, 'herg_ki__gaussian.json')
    s = _n40_scores(d, '1', 'hybrid__general2', 'dnn') if d else None
    r.check('herg_ki gaussian 1.0 dnn hybrid__general2 retuned', '0.691', (s or {}).get('retuned'))
    r.check('herg_ki gaussian 1.0 dnn hybrid__general2 tuned_clean_reused', '0.703', (s or {}).get('tuned_clean_reused'))
    dc = J('noise', N40, 'herg_ki__censoring.json')
    if not dc:
        r.missing('herg_ki censoring 0.25', 'n40 herg_ki__censoring.json missing'); return
    won = n = 0
    for sp in ('base_1', 'base_2', 'hybrid', 'hybrid__general2'):
        for m in dc.get('models', []):
            x = _n40_scores(dc, '0.25', sp, m)
            if x and 'retuned' in x:
                n += 1; won += round(x['retuned'], 3) < round(x['fixed'], 3)
    r.check('herg_ki censoring 0.25 pairs where retuned beat fixed', '1', won if n else None, note=f'of {n} pairs')
    r.claim('herg_ki censoring 0.25 number of model x configuration pairs is eight', n == 8 if n else None, f'{n} pairs')
    r.info('n40 has no QM9 file' if not [f for f in files(CTX.path('noise', N40)) if f.startswith('qm9')]
           else 'n40 QM9 file(s) now exist: ' + ', '.join(f for f in files(CTX.path('noise', N40)) if f.startswith('qm9')))


@check('N7-41', 'Other noise-robust training: Huber network, robust loss, two-learner selection')
def n7_41(r):
    """Section 'Changing the Model', fourth paragraph. n33 herg_ki__<kind>.json
    summary.per_level['1'][space].model_swaps.dnn_huber_c*; n38
    caco2__sparse_dprime_up_p10.rows.jsonl method robust_loss, label_base=real, dnn base_1,
    level 1.0, draw 0; n32 buchwald_hartwig__shift_only_p10.rows.jsonl rf base_1 level 1.0,
    two_learner_share0.1 against its random seeds 0-2, means over 3 draws; fold 0."""
    N33 = 'n33_robust_training'
    d = J('noise', N33, 'herg_ki__outlier_p05.json')
    sw = (((d or {}).get('summary') or {}).get('per_level', {}).get('1', {}).get('base_1', {}) or {}).get('model_swaps', {})
    hub = {k: v for k, v in sw.items() if k.startswith('dnn_huber_c') and v.get('arm_mean') is not None}
    r.check('n33 herg_ki outlier 1.0 dnn base_1 no Huber', '0.7413', next(iter(hub.values()))['noisy_mean'] if hub else None)
    r.check('n33 herg_ki outlier 1.0 dnn base_1 Huber, best threshold', '0.7269', min(v['arm_mean'] for v in hub.values()) if hub else None)
    dg = J('noise', N33, 'herg_ki__gaussian.json')
    if dg:
        worse = tot = 0
        for sp, x in dg['summary']['per_level'].get('1', {}).items():
            if not isinstance(x, dict):
                continue
            for k, v in (x.get('model_swaps') or {}).items():
                if k.startswith('dnn_huber_c') and v.get('arm_mean') is not None:
                    tot += 1; worse += v['arm_mean'] > v['noisy_mean']
        r.claim('n33 herg_ki gaussian 1.0: Huber network worse at every threshold, every configuration',
                worse == tot if tot else None, f'worse in {worse} of {tot}')
    else:
        r.missing('n33 herg_ki gaussian', 'herg_ki__gaussian.json missing')
    # n38
    # 2026-10-10: the caco2 d'=1 file was set aside (inputs changed); the example is now LogD's first hybrid, draw 0
    rr = rows('n38_noise_relative_to_model_error', 'logd__sparse_dprime_up_p10.rows.jsonl', method='robust_loss',
              method_setting='label_base=real', model='dnn', model_input='hybrid', level=1.0)
    r0 = [x for x in rr if x.get('fold') == 0 and x.get('draw') == 0]
    r.check('n38 logd sparse_dprime 1.0 dnn hybrid noisy', '0.404', r0[0]['score_D'] if r0 else None)
    r.check('n38 logd sparse_dprime 1.0 dnn hybrid robust loss', '0.398', r0[0]['score_Dprime'] if r0 else None)
    r.check('n38 logd sparse_dprime 1.0 dnn hybrid clean', '0.392', r0[0]['score_clean'] if r0 else None)
    for x in rr:
        if x.get('fold') != 0:
            r.new_fold('n38 logd sparse_dprime 1.0 dnn hybrid robust_loss', x['fold'],
                       f'draw {x.get("draw")}: {x["score_D"]:.4f} -> {x["score_Dprime"]:.4f}, clean {x["score_clean"]:.4f}')
    # n32
    N32 = 'n32_iterative_and_dual_selection'
    rb = rows(N32, 'buchwald_hartwig__shift_only_p10.rows.jsonl', model='rf', model_input='base_1', level=1.0)
    by = collections.defaultdict(list)
    for x in rb:
        by[(x['method'], x['fold'])].append(x)
    tl = by.get(('two_learner_share0.1', 0), [])
    r.check('n32 buchwald shift_only 1.0 rf base_1 no cleaning (3 draws)', '20.3', mean(x['score_D'] for x in tl))
    r.check('n32 buchwald shift_only 1.0 rf base_1 two_learner share 0.1', '18.6', mean(x['score_Dprime'] for x in tl))
    for s, t in zip((0, 1, 2), ('20.2822', '20.5441', '20.7828')):
        g = by.get((f'two_learner_share0.1_random/seed{s}', 0), [])
        r.check(f'n32 buchwald shift_only 1.0 rf base_1 random deletion seed {s}', t, mean(x['score_Dprime'] for x in g))
    for (m, f), g in sorted(by.items()):
        if f != 0 and m == 'two_learner_share0.1':
            r.new_fold('n32 buchwald shift_only 1.0 rf base_1 two_learner 0.1', f,
                       f'{mean(x["score_D"] for x in g):.4f} -> {mean(x["score_Dprime"] for x in g):.4f}; random ' +
                       ', '.join(f'{mean(y["score_Dprime"] for y in by.get((f"two_learner_share0.1_random/seed{s}", f), [])) or float("nan"):.4f}' for s in (0, 1, 2)))


# ---- n09 sweep -------------------------------------------------------------------------------
N09 = 'n09_lnl_survey'
N09_SPACES = ('base_1', 'base_2', 'hybrid', 'hybrid__general2')
N09_DOM = {'qm9': 'small', 'caco2': 'small', 'herg_ki': 'small', 'logd': 'small', 'herg': 'small',
           'herg_ic50': 'small', 'adme_mdr1': 'small', 'buchwald_hartwig': 'reaction'}


def _n09_dir(r):
    if CTX.path('view', N09) and os.path.isdir(CTX.path('view', N09)):
        return CTX.path('view', N09)
    r.info('no --noise-view folder: n09 read from <root>/noise instead of the noise view')
    return CTX.path('noise', N09)


def _n09_files(r):
    """[(ds, kind, data)] for every in-scope n09 file with result_status ok."""
    D = _n09_dir(r)
    out = []
    for f in files(D, '*.json'):
        if 'fingerprint' in f or '__' not in f:
            continue
        ds, kind = f[:-5].split('__', 1)
        if ds not in CH7_DATASETS:
            continue
        d = jload(os.path.join(D, f))
        if d and d.get('result_status') == 'ok' and 'vs_plain_dnn_same_labels' in d:
            out.append((ds, kind, d))
    return out


def _n09_verdicts(r, mode='after'):
    """Rebuild of ch7_sources/n09_b.py + n09_c.py. One row per (dataset, kind, damaged level)
    plus one level-0 row per dataset (gains averaged over the kind files). Per method: the
    better of base_1/base_2 and the better of the two hybrids, picked on the method's own score
    (mode 'after', the rule that reproduces the table) or the first of each (mode 'rank').
    Returns {(ds, label): {method: (gain_single, gain_hybrid, row_single, row_hybrid)}}."""
    G = collections.defaultdict(dict)
    for ds, kind, d in _n09_files(r):
        lib = d['lower_is_better']
        for lv, c in d['vs_plain_dnn_same_labels'].items():
            for sp in N09_SPACES:
                for m, x in (c.get(sp + '_lnl') or {}).items():
                    G[(ds, kind, float(lv), m)][sp] = dict(plain=x.get('plain_dnn_mean'), meth=x.get('method_mean'),
                                                           gain=x.get('gain_mean'), lib=lib, sp=sp)
    ok = lambda v: v is not None and v == v

    def pick(spd, cands):
        if mode == 'rank':
            return spd.get(cands[0])
        c = [spd[s] for s in cands if s in spd and ok(spd[s]['meth'])]
        if not c:
            return None
        return (min if c[0]['lib'] else max)(c, key=lambda x: x['meth'])
    verd = collections.defaultdict(dict)
    for (ds, kind, lv, m), spd in G.items():
        s = pick(spd, ['base_1', 'base_2']); h = pick(spd, ['hybrid', 'hybrid__general2'])
        if not s or not h or not ok(s['gain']) or not ok(h['gain']):
            continue
        if lv > 0:
            verd[(ds, f'{kind}@{lv}')][m] = (s['gain'], h['gain'], s, h)
        else:
            verd[(ds, 'L0')].setdefault(m, []).append((s['gain'], h['gain'], s, h))
    for k, v in verd.items():
        if k[1] == 'L0':
            for m, lst in list(v.items()):
                v[m] = (mean(a[0] for a in lst), mean(a[1] for a in lst), lst[0][2], lst[0][3])
    return verd


N09_TABLE = [  # (method key, table row label, small, reaction, antibodies), from tab:noise:res:sweep (10-10b, MDR1 counted as small)
    ('small_loss', 'Small-loss selection', '2', '5', '12'),
    ('confident_learning', 'Confident learning', '17', '1', '12'),
    ('distance_selection', 'Distance-based selection', '10', '4', '10'),
    ('multistage_cleaning', 'Multistage cleaning', '1', '2', '5'),
    ('early_learning', 'Early learning', '16', '2', '11'),
    ('self_paced_learning', 'Self-paced learning', '15', '3', '14'),  # renamed from uncertainty_curriculum 2026-10-29; counts are from the old runs
    ('dividemix', 'DivideMix', '8', '2', '14'),
    ('meta_weight_net', 'Meta-Weight-Net', '12', '3', '11'),
    ('mentornet', 'MentorNet', '14', '4', '12'),
    ('contrast_to_divide', 'Contrast-to-divide', '10', '2', '12'),
    ('heteroscedastic_gp', 'Heteroscedastic GP', '18', '2', '14'),
    ('evidential_kernel', 'Evidential network', '15', '4', '13'),
]


@check('N7-42', 'Expanded published sweep: table tab:noise:res:sweep and its paragraph')
def n7_42(r):
    """Section 'The expanded published sweep', first paragraph and table tab:noise:res:sweep.
    noise_view n09_lnl_survey/<ds>__<kind>.json vs_plain_dnn_same_labels[L][<space>_lnl][method]
    .{plain_dnn_mean, method_mean, gain_mean}; a condition = dataset x kind x damaged level, plus
    one level-0 row per dataset; 'helped' = gain > 0 on the better single representation and on
    the better hybrid, each picked on the method's own score."""
    verd = _n09_verdicts(r, 'after')
    dom = lambda ds: N09_DOM.get(ds, 'antibody')
    tab = collections.defaultdict(lambda: [0, 0])
    conds = collections.Counter()
    for (ds, lab), v in verd.items():
        conds[dom(ds)] += 1
        for m, t in v.items():
            tab[(m, dom(ds))][0] += (t[0] > 0 and t[1] > 0)
            tab[(m, dom(ds))][1] += 1
    r.check('conditions, small molecules', '37', conds.get('small'))
    r.check('conditions, Buchwald-Hartwig', '5', conds.get('reaction'))
    r.check('conditions, antibodies', '25', conds.get('antibody'))
    r.check('conditions, small molecules, regression-only methods', '30', tab[('heteroscedastic_gp', 'small')][1] or None)
    for m, lab, a, b, c in N09_TABLE:
        for dm, t in (('small', a), ('reaction', b), ('antibody', c)):
            x = tab.get((m, dm))
            r.check(f'{lab}, {dm}', t, x[0] if x else None, note=f'of {x[1]}' if x else '')
    # NEW FOLD: any file run on more than fold 0
    for ds, kind, d in _n09_files(r):
        fr = [f for f in (d.get('folds_run') or []) if f != 0]
        if fr:
            r.new_fold(f'n09 {ds} {kind}', ','.join(map(str, fr)), 'file now covers folds ' + str(d.get('folds_run')) + ' (means are over folds)')
    counted = {ds for ds, _ in verd}
    have = set(datasets_with_files(N09, where=('view',) if CTX.path('view', N09) else ('noise',)))
    for ds in sorted(have - counted):
        r.new_dataset(ds, 'has n09 files but no condition counted (no ok status or no scores)')
    r.info('datasets counted: ' + ', '.join(sorted(counted)))


@check('N7-43', 'Sweep against the earlier study: heteroscedastic GP on ddg, distance selection on hERG')
def n7_43(r):
    """Section 'The expanded published sweep', second paragraph. noise_view n09
    ddg_combined__<kind>.json level '0.0' heteroscedastic_gp (same in every kind file; the
    gaussian file is read); herg__binary_asymmetric.json level '0.1' distance_selection, mean over
    3 draws; 'only method that helps' counted with the table rule (better single and better
    hybrid picked on the method's score) and, for hERG, also on base_1 and hybrid."""
    D = _n09_dir(r)
    d = jload(os.path.join(D, 'ddg_combined__gaussian.json'))
    c = ((d or {}).get('vs_plain_dnn_same_labels') or {}).get('0.0', {})
    g = lambda sp, m, k: ((c.get(sp + '_lnl') or {}).get(m) or {}).get(k)
    r.check('ddg L0 base_1 plain', '0.341', g('base_1', 'heteroscedastic_gp', 'plain_dnn_mean'))
    r.check('ddg L0 base_1 heteroscedastic_gp', '0.433', g('base_1', 'heteroscedastic_gp', 'method_mean'))
    r.check('ddg L0 hybrid__general2 plain', '0.2498', g('hybrid__general2', 'heteroscedastic_gp', 'plain_dnn_mean'))
    r.check('ddg L0 hybrid__general2 heteroscedastic_gp', '0.364', g('hybrid__general2', 'heteroscedastic_gp', 'method_mean'))
    r.check('ddg L0 hybrid plain', '0.3048', g('hybrid', 'heteroscedastic_gp', 'plain_dnn_mean'))
    r.check('ddg L0 hybrid heteroscedastic_gp', '0.2880', g('hybrid', 'heteroscedastic_gp', 'method_mean'))
    verd = _n09_verdicts(r, 'after')
    v = verd.get(('ddg_combined', 'L0'), {})
    helped = sorted(m for m, t in v.items() if t[0] > 0 and t[1] > 0)
    r.claim('ddg L0: heteroscedastic GP is the only method that helps', helped == ['heteroscedastic_gp'] if v else None,
            'methods that help: ' + (', '.join(helped) or 'none'))
    d = jload(os.path.join(D, 'herg__binary_asymmetric.json'))
    c = ((d or {}).get('vs_plain_dnn_same_labels') or {}).get('0.1', {})
    r.check('herg binary_asymmetric 0.1 base_1 plain', '0.8151', g('base_1', 'distance_selection', 'plain_dnn_mean'))
    r.check('herg binary_asymmetric 0.1 base_1 distance_selection', '0.8166', g('base_1', 'distance_selection', 'method_mean'))
    r.check('herg binary_asymmetric 0.1 hybrid plain', '0.8214', g('hybrid', 'distance_selection', 'plain_dnn_mean'))
    r.check('herg binary_asymmetric 0.1 hybrid distance_selection', '0.8245', g('hybrid', 'distance_selection', 'method_mean'))
    for mode in ('after', 'rank'):
        v = _n09_verdicts(r, mode).get(('herg', 'binary_asymmetric@0.1'), {})
        helped = sorted(m for m, t in v.items() if t[0] > 0 and t[1] > 0)
        r.claim(f'herg binary_asymmetric 0.1: distance selection the only one of twelve that helps both '
                f'({"better single and better hybrid" if mode == "after" else "base_1 and hybrid"})',
                helped == ['distance_selection'] if v else None, 'methods that help: ' + (', '.join(helped) or 'none'))


@check('N7-44', 'QM9 with no noise added: the twelve sweep methods, and the second run inside n13')
def n7_44(r):
    """Section 'The expanded published sweep', third paragraph. noise_view n09
    qm9__censoring.json vs_plain_dnn_same_labels['0.0']: plain dnn base_1 and hybrid, DivideMix
    and confident learning on the hybrid; noise_view n13 qm9__censoring.json
    per_noise_type.censoring.per_strength['0.0'].track_a_vs_plain_dnn_same_labels: plain
    hybrid__general2, methods that beat it, contrast-to-divide."""
    D = _n09_dir(r)
    d = jload(os.path.join(D, 'qm9__censoring.json'))
    c = ((d or {}).get('vs_plain_dnn_same_labels') or {}).get('0.0', {})
    hy = {m: x['method_mean'] for m, x in (c.get('hybrid_lnl') or {}).items() if x.get('method_mean') is not None}
    pl = lambda sp: next((x.get('plain_dnn_mean') for x in (c.get(sp + '_lnl') or {}).values()), None)
    r.check('n09 qm9 L0 plain dnn base_1', '0.3962', pl('base_1'))
    r.check('n09 qm9 L0 plain dnn hybrid', '0.355', pl('hybrid'))
    r.check('n09 qm9 L0 hybrid DivideMix', '0.452', hy.get('dividemix'))
    r.check('n09 qm9 L0 hybrid worst method (max)', '0.452', max(hy.values()) if hy else None,
            note=f'worst is {max(hy, key=hy.get)}' if hy else '')
    r.check('n09 qm9 L0 hybrid distance-based selection', '0.3677', hy.get('distance_selection'))
    r.check('n09 qm9 L0 hybrid least bad method (min)', '0.3677', min(hy.values()) if hy else None,
            note=f'least bad is {min(hy, key=hy.get)}' if hy else '')
    v = _n09_verdicts(r, 'after').get(('qm9', 'L0'), {})
    both = sorted(m for m, t in v.items() if t[0] < 0 and t[1] < 0)
    r.claim('n09 qm9 L0: every one of the twelve worse on best single and best hybrid',
            (len(both) == 12) if v else None, f'{len(both)} of {len(v)} worse on both' + (
                '' if len(both) == len(v) else '; not worse on both: ' + ', '.join(sorted(set(v) - set(both)))))
    D13 = CTX.path('view', 'n13_noise_distributions') if CTX.path('view', 'n13_noise_distributions') else CTX.path('noise', 'n13_noise_distributions')
    d = jload(os.path.join(D13, 'qm9__censoring.json'))
    try:
        t = d['per_noise_type']['censoring']['per_strength']['0.0']['track_a_vs_plain_dnn_same_labels']
    except (KeyError, TypeError):
        r.missing('n13 qm9 L0', 'n13 qm9__censoring.json track_a_vs_plain_dnn_same_labels missing'); return
    g2 = {m: x for m, x in (t.get('hybrid__general2') or {}).items() if isinstance(x, dict) and x.get('method_mean') is not None}
    plain = next((x['plain_dnn_mean'] for x in g2.values()), None)
    r.check('n13 qm9 L0 plain dnn hybrid__general2', '0.3638', plain)
    beat = sorted(m for m, x in g2.items() if x['method_mean'] < x['plain_dnn_mean'])
    r.check('n13 qm9 L0 methods beating plain on hybrid__general2', '6', len(beat) if g2 else None, note=', '.join(beat))
    r.check('n13 qm9 L0 hybrid__general2 MentorNet', '0.3433', (g2.get('mentornet') or {}).get('method_mean'))
    r.check('n13 qm9 L0 hybrid__general2 best method (min)', '0.3433', min(x['method_mean'] for x in g2.values()) if g2 else None,
            note=(min(g2, key=lambda m: g2[m]['method_mean']) if g2 else ''))


@check('N7-45', 'Top method on the hybrid against the top method on both single representations')
def n7_45(r):
    """Section 'The expanded published sweep', fourth paragraph. noise_view n09 files; per
    file x level, the method with the best method_mean on hybrid, base_1 and base_2; count
    combinations and those where the first hybrid's top method is also top on both singles."""
    EXPECTED_EXC = {('caco2', 'shift_only_p10', 'distance_selection'), ('herg', 'class_dependent', 'confident_learning'),
                    ('herg', 'instance_noise', 'distance_selection')}
    n = 0; exc = set()
    for ds, kind, d in _n09_files(r):
        lib = d['lower_is_better']
        for lv, c in d['vs_plain_dnn_same_labels'].items():
            top = {}
            for sp in ('base_1', 'base_2', 'hybrid'):
                v = {m: x['method_mean'] for m, x in (c.get(sp + '_lnl') or {}).items() if x.get('method_mean') is not None}
                if v:
                    top[sp] = (min if lib else max)(v, key=v.get)
            if len(top) < 3:
                continue
            n += 1
            if top['hybrid'] == top['base_1'] == top['base_2']:
                exc.add((ds, kind, top['hybrid']))
    r.check('file x level combinations', '110', n or None)
    r.claim('three exceptions on the first hybrid', (len(exc) == 3) if n else None, f'{len(exc)} exceptions')
    r.claim('the exceptions are caco2 shift_only_p10 (distance selection), herg class_dependent (confident '
            'learning), herg instance_noise (distance selection)', exc == EXPECTED_EXC if n else None,
            'raw: ' + '; '.join(f'{a} {b} ({c})' for a, b, c in sorted(exc)))


# ---- n12: neighbours in the own representation or in the hybrid --------------------------------
N12 = 'n12_importance_weighted_distance'
N12_COUNTED = ['caco2', 'herg_ki', 'logd', 'herg', 'buchwald_hartwig', 'ginkgo_colloidal', 'flab_thermo_tresanco',
               'ddg_combined', 'ginkgo_titer']


def _n12_units(datasets):
    """Rebuild of ch7_sources/n12cmp.py: per n12 file, damaged level, importance method and
    key model@base_X|estimator|scheme: base_X_unweighted vs hybrid_unweighted on the passes both
    ran. Returns list of dicts."""
    D = CTX.path('noise', N12)
    units = []
    for f in files(D, '*.json'):
        if 'fingerprint' in f:
            continue
        ds = f.split('__')[0].split('.')[0]
        if ds not in datasets:
            continue
        d = jload(os.path.join(D, f))
        if not d:
            continue
        lib = d.get('lower_is_better')
        for sg, p in (d.get('per_sigma') or {}).items():
            if float(sg) == 0:
                continue
            for imp, q in (p.get('per_importance_method') or {}).items():
                for bx in ('base_1', 'base_2'):
                    A = q.get(bx + '_unweighted', {}); B = q.get('hybrid_unweighted', {})
                    for key, ea in A.items():
                        if '@' + bx + '|' not in key or key not in B:
                            continue
                        eb = B[key]
                        pa = {tuple(x): (y, z) for x, y, z in zip(ea['passes'], ea['denoised_per_pass'], ea['noisy_per_pass'])}
                        pb = {tuple(x): (y, z) for x, y, z in zip(eb['passes'], eb['denoised_per_pass'], eb['noisy_per_pass'])}
                        com = sorted(set(pa) & set(pb))
                        if not com:
                            continue
                        units.append(dict(ds=ds, file=f, level=sg, imp=imp, key=key, mc=key.split('|')[0], lib=lib, passes=com,
                                          noisy=mean(pa[c][1] for c in com), own=mean(pa[c][0] for c in com),
                                          hyb=mean(pb[c][0] for c in com), pp={c: (pa[c][1], pa[c][0], pb[c][0]) for c in com}))
    return units


@check('N7-47', 'Neighbours found in the own representation or in the hybrid (n12)')
def n7_47(r):
    """Section 'Model and Representation', second paragraph. n12 *.json (results root, not the
    view), conditions base_X_unweighted vs hybrid_unweighted matched on key
    model@base_X|estimator|scheme and passes; fold 0; pairs = (file, level, model@config), mean
    over matched keys; examples are single (draw) passes."""
    units = _n12_units(N12_COUNTED)
    if not units:
        r.missing('n12', 'no n12 units'); return
    g = collections.defaultdict(list)
    for u in units:
        g[(u['ds'], u['file'], u['level'], u['mc'])].append(u)
    better = lambda x, n, lib: x < n if lib else x > n
    own = hyb = 0
    for k, us in g.items():
        lib = us[0]['lib']; n = mean(u['noisy'] for u in us)
        own += better(mean(u['own'] for u in us), n, lib); hyb += better(mean(u['hyb'] for u in us), n, lib)
    r.check('pairs (dataset x file x level x model@configuration)', '160', len(g))
    r.check('pairs improved with neighbours in the own representation', '87', own)
    r.check('pairs improved with neighbours in the hybrid', '98', hyb)
    EX = [('logd', 'logd__grouped_shifted.json', 'dnn@base_1|neighbor_residual_k20|rank', (0, 0), ('0.705', '0.7535', '0.663')),
          ('flab_thermo_tresanco', 'flab_thermo_tresanco__outlier_p05.json', 'dnn@base_2|local_variance_ratio_k20|rank', (0, 2), ('0.3336', '0.3344', '0.5183')),
          ('ginkgo_titer', 'ginkgo_titer__gaussian.json', 'dnn@base_2|neighbor_residual_k10|rank', (0, 2), ('-0.0504', '0.1499', '-0.0117'))]
    for ds, f, key, ps, texts in EX:
        u = next((u for u in units if u['file'] == f and u['key'] == key and float(u['level']) == 1.0 and ps in u['pp']), None)
        vals = u['pp'][ps] if u else (None, None, None)
        for what, t, v in zip(('untreated', 'own neighbours', 'hybrid neighbours'), texts, vals):
            r.check(f'{ds} {f.split("__")[1][:-5]} 1.0 {key} draw {ps[1]} {what}', t, v)
    for u in units:
        for (fo, dr), (n, o, h) in u['pp'].items():
            if fo != 0:
                r.new_fold(f'n12 {u["file"][:-5]} {u["level"]} {u["imp"]} {u["key"]}', fo, f'draw {dr}: {n:.4f} -> own {o:.4f}, hybrid {h:.4f}')
    others = [ds for ds in datasets_with_files(N12, where=('noise',)) if ds not in N12_COUNTED]
    if others:
        ou = _n12_units(others)
        for ds in others:
            gg = collections.defaultdict(list)
            for u in ou:
                if u['ds'] == ds:
                    gg[(u['file'], u['level'], u['mc'])].append(u)
            if not gg:
                r.new_dataset(ds, 'n12 file(s) but no matched own/hybrid pairs'); continue
            lib = next(iter(gg.values()))[0]['lib']
            o = sum(better(mean(u['own'] for u in us), mean(u['noisy'] for u in us), lib) for us in gg.values())
            h = sum(better(mean(u['hyb'] for u in us), mean(u['noisy'] for u in us), lib) for us in gg.values())
            r.new_dataset(ds, f'{len(gg)} pairs, improved with own neighbours {o}, with hybrid neighbours {h}')



@check('N7-01', 'Standing caveats: the network scores QM9 differently in each experiment')
def n7_01(r):
    """Section 'Results: Does Noise Handling Work at All?', first paragraph. The untreated
    (score_D) QM9 score of the network on the second hybrid (hybrid__general2), level 0, fold 0,
    draw 0, in each experiment's rows file; and LightGBM's, which should be one value."""
    # experiment dir, file pattern, text value (from the SOURCE line)
    EXPECTED = [
        ('n02_sample_weighting', 'qm9.rows.jsonl', '0.362'),
        ('n07_remediation', 'qm9__gaussian__*.rows.jsonl', '0.362'),
        ('n07_remediation', 'qm9__grouped_shifted__*.rows.jsonl', '0.362'),
        ('n07_remediation', 'qm9__outlier_p05__*.rows.jsonl', '0.362'),
        ('n07_remediation', 'qm9__shift_only_p10__*.rows.jsonl', '0.362'),
        ('n09_lnl_survey', 'qm9__censoring.rows.jsonl', '0.362'),
        ('n09_lnl_survey', 'qm9__gaussian.rows.jsonl', '0.362'),
        ('n09_lnl_survey', 'qm9__outlier_p05.rows.jsonl', '0.362'),
        ('n09_lnl_survey', 'qm9__shift_only_p10.rows.jsonl', '0.362'),
        ('n12_importance_weighted_distance', 'qm9__gaussian__*.rows.jsonl', '0.362'),
        ('n12_importance_weighted_distance', 'qm9__shift_only_p10__*.rows.jsonl', '0.362'),
        # n21 holds two fits on 2026-10-10b (0.3568 and 0.3618); the smaller is read; inside the 0.348-0.362 range of the text
        ('n21_neighbour_distance', 'qm9__*.rows.jsonl', '0.3568'),
        ('n30_censored_likelihood', 'qm9__*.rows.jsonl', '0.362'),
        ('n07_remediation', 'qm9__censoring__*.rows.jsonl', '0.3568'),
        ('n12_importance_weighted_distance', 'qm9__censoring__*.rows.jsonl', '0.3568'),
        ('n12_importance_weighted_distance', 'qm9__grouped_shifted__*.rows.jsonl', '0.3568'),
        ('n12_importance_weighted_distance', 'qm9__outlier_p05__*.rows.jsonl', '0.3568'),
        ('n24_delete_or_keep', 'qm9__*.rows.jsonl', '0.3568'),
        ('n09_lnl_survey', 'qm9__grouped_shifted.rows.jsonl', '0.348'),
    ]
    lgb_all, dnn_all = set(), set()
    for sub, pat, text in EXPECTED:
        rs = [x for x in rows(sub, pat, level=0.0, fold=0, dataset='qm9')
              if x.get('draw') in (0, None) and cfg(x.get('model_input')) == 'hybrid__general2']
        dv = sorted({round(x['score_D'], 4) for x in rs if x.get('model') == 'dnn' and x.get('score_D') is not None})
        lgb_all |= {round(x['score_D'], 4) for x in rs if x.get('model') == 'lightgbm' and x.get('score_D') is not None}
        dnn_all |= set(dv)
        lab = f'{sub} {pat} dnn hybrid__general2 level 0 fold 0 draw 0 score_D'
        if not dv:
            r.check(lab, text, None, 'no rows')
        else:
            r.check(lab, text, dv[0], f'{len(dv)} distinct values in file: {dv}' if len(dv) > 1 else '')
    r.check('largest dnn untreated score over these files', '0.362', max(dnn_all) if dnn_all else None)
    r.check('smallest dnn untreated score over these files', '0.348', min(dnn_all) if dnn_all else None)
    r.claim('LightGBM gives one score in every experiment', (len(lgb_all) == 1) if lgb_all else None,
            f'lightgbm hybrid__general2 untreated scores seen: {sorted(lgb_all)}')
    # Further folds of the same quantity (any experiment, noise and noise_full)
    extra = collections.defaultdict(set)
    for sub in sorted({e[0] for e in EXPECTED}):
        for x in rows(sub, 'qm9*.rows.jsonl', where=('noise', 'full'), level=0.0, dataset='qm9'):
            if x.get('fold') not in (0, None) and x.get('model') in ('dnn', 'lightgbm') \
                    and cfg(x.get('model_input')) == 'hybrid__general2' and x.get('score_D') is not None:
                extra[(x['fold'], x['model'])].add((sub, round(x['score_D'], 4)))
    for (fold, model), v in sorted(extra.items()):
        r.new_fold(f'{model} hybrid__general2 untreated QM9 score', fold,
                   '; '.join(f'{s} {val:.4f}' for s, val in sorted(v)))


_SIBLING = {}


def _sibling_value(nid, parts):
    """Raw value of the one label of checker nid whose label contains every string in parts."""
    if nid not in _SIBLING:
        f = next((f for i, _, f in CHECKS if i == nid), None)
        rep = Report(nid, '')
        if f:
            try:
                f(rep)
            except Exception:
                pass
        _SIBLING[nid] = rep.values
    hits = [k for k in _SIBLING[nid] if all(p in k for p in parts)]
    return (_SIBLING[nid][hits[0]], hits[0]) if len(hits) == 1 else (None, f'{len(hits)} labels of {nid} match {parts}')


@check('N7-02', 'Noise mitigation works: the three opening examples')
def n7_02(r):
    """Section 'Results: Does Noise Handling Work at All?', second paragraph. The three examples
    repeat numbers of later paragraphs, so each is read through that paragraph's checker:
    LogD clipping (N7-20, n30), hERG flips (N7-03, n24 neighbor_residual@screen1) and hERG Ki
    label smoothing (N7-09, n18; 'stays at 0.6419 or above' = the smallest shuffled seed)."""
    EXPECTED = [
        ('logd dnn best single, clipped', 'N7-20', ['logd dnn best single', 'clipped level 0.25 fold 0'], '0.6354'),
        ('logd dnn best single, censored EM', 'N7-20', ['logd dnn best single', 'censored_em level 0.25 fold 0'], '0.4795'),
        ('logd dnn best single, before clipping', 'N7-20', ['logd dnn best single', 'clean level 0.25 fold 0'], '0.4314'),
        ('herg xgboost hybrid untreated', 'N7-03', ['n07 hybrid untreated'], '0.8028'),
        ('herg xgboost hybrid delete flagged', 'N7-03', ['n24 hybrid neighbor_residual@screen1 0.1 delete'], '0.8375'),
        ('herg xgboost hybrid best random deletion', 'N7-03', ['n24 hybrid neighbor_residual@screen1 0.1 best random'], '0.8179'),
        ('herg_ki dnn base_1 untreated', 'N7-09', ['herg_ki dnn base_1 neighbor_residual label_smooth', 'no cleaning'], '0.6380'),
        ('herg_ki dnn base_1 label smoothing', 'N7-09', ['herg_ki dnn base_1 neighbor_residual label_smooth', 'cleaned'], '0.5775'),
    ]
    for label, nid, parts, text in EXPECTED:
        v, note = _sibling_value(nid, parts)
        r.check(label, text, v, f'via {nid}: {note}')
    shuf = [_sibling_value('N7-09', ['herg_ki dnn base_1 neighbor_residual label_smooth', f'shuffled seed {s}'])[0] for s in (0, 1, 2)]
    shuf = [x for x in shuf if x is not None]
    r.check('herg_ki dnn base_1 label smoothing, smallest shuffled score', '0.6419', min(shuf) if shuf else None,
            f'shuffled seeds: {", ".join(f"{x:.4f}" for x in shuf)}')
    r.info('The table this paragraph points to (tab:noise:res:bygroup) is in section 5 (group table, table A).')


def _group_table_C():
    """{(domain, group, labels): [rows, yesyes, noyes, yesno, nono, tie]} from the rebuilt table C."""
    if not GT.get('md'):
        return None
    out = {}
    sec = None
    for line in GT['md'].split('\n'):
        if line.startswith('## '):
            sec = line[3:4]
        if sec == 'C' and line.startswith('| ') and not line.startswith('| domain'):
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            if len(c) == 9 and c[3].isdigit():
                out[(c[0], c[1], c[2])] = [int(x) for x in c[3:]]
    return out


def _check_table_C(r, dom, expected):
    C_ = _group_table_C()
    if C_ is None:
        r.missing(f'{dom} table C', 'group table not rebuilt (--skip-group-table or it failed)')
        return
    row = C_.get((dom, 'mitigation total', 'both'))
    names = ['rows with both scores', 'hybrid better before and after', 'worse before, better after',
             'better before, worse after', 'worse before and after', 'ties']
    for i, (n, t) in enumerate(zip(names, expected)):
        if t is None:
            continue
        r.check(f'{dom}, mitigation, {n}', t, row[i] if row else None)


@check('N7-33', 'Antibodies: hybrid against single representation before and after cleaning')
def n7_33(r):
    """Section 'The antibody datasets', paragraph 'Two things about these datasets...'. Counts of
    table C of the group table (antibodies, mitigation total, both labels), rebuilt from the raw
    files in section 5; held-out sizes from n18 fold_records.0.n_val."""
    _check_table_C(r, 'antibodies', ['734', '305', '84', '70', '258', '17'])
    sizes = {}
    for ds in CH7_DATASETS:
        if DOMAIN.get(ds) != 'antibodies':
            continue
        d = J('noise', 'n18_clean_the_real_labels', ds + '.json')
        fr = (d or {}).get('fold_records') or {}
        if isinstance(fr, dict) and '0' in fr and fr['0'].get('n_val'):
            sizes[ds] = fr['0']['n_val']
    r.check('largest antibody held-out fold 0 (n18 fold_records n_val)', '135', max(sizes.values()) if sizes else None,
            ', '.join(f'{k} {v}' for k, v in sorted(sizes.items())))
    r.check('ginkgo_colloidal held-out fold 0 antibodies', '53', sizes.get('ginkgo_colloidal'))
    r.info('"44" (smallest held-out fold) comes from PATTERNS.md §4.5, not a raw key read here; '
           'the n18 sizes above do not include ginkgo_pr_cho, which has no n18 file.')


@check('N7-50', 'Clean-label test: QM9 rows of tab:noise:res:shuffled (n18) and tab:noise:res:change_model_delivered (n40)')
def check_n7_50(r):
    """Subsection 'The clean-label test' (sec:noise:res:qm9test), added 2026-10-10. Its numbers sit in
    the QM9 rows of two tables printed in other blocks (the shuffled-score table under N7-10, the
    model-change table under N7-40), so those blocks are searched for the text as well."""
    BLOCKS[r.id] = BLOCKS.get(r.id, '') + BLOCKS.get('N7-10', '') + BLOCKS.get('N7-40', '')
    # n18 QM9: (label, space, detector, slot, model, cleanup, text untreated, text cleaned, three shuffles)
    SPEC = [
        ('qm9 dnn base_1 neighbor_residual label_smooth', 'base_1', 'neighbor_residual', '0.2', 'dnn', 'label_smooth',
         '0.3999', '0.3927', ['0.3996', '0.3967', '0.4005']),
        ('qm9 dnn hybrid cross_val label_smooth', 'hybrid', 'cross_val', '0.05', 'dnn', 'label_smooth',
         '0.3491', '0.3487', ['0.3568', '0.3502', '0.3456']),
        ('qm9 dnn base_1 cross_val remove', 'base_1', 'cross_val', '0.2', 'dnn', 'remove',
         '0.400', '0.461', ['0.433', '0.429', '0.412']),
        ('qm9 lightgbm base_1 neighbor_residual label_smooth', 'base_1', 'neighbor_residual', '0.2', 'lightgbm', 'label_smooth',
         '0.412', '0.414', ['0.417', '0.418', '0.417']),
    ]
    for lab, sp, det, slot, model, cl, t_nc, t_cl, t_sh in SPEC:
        x = _B_n18('qm9', sp, det, slot, model, cl)
        lab = f'{lab} slot {slot}'
        if x is None:
            r.missing(lab, f'n18_clean_the_real_labels/qm9.json key per_space.{sp}.per_detector.{det}.{slot}.per_model.{model} not found')
            continue
        nc, cv, sh, folds = x
        r.check(lab + ' no cleaning', t_nc, nc)
        r.check(lab + ' cleaned', t_cl, cv)
        for i, tx in enumerate(t_sh):
            r.check(f'{lab} shuffled seed {i}', tx, sh.get(str(i)))
        if folds and list(folds) != [0]:
            r.new_fold(lab, folds, f'folds_run now {folds}')
    # n40 QM9, level 0: plain, seed averaging, stacking, retuned per model and configuration
    d = J('noise', 'n40_model_side_regularisation', 'qm9__gaussian.json')
    N40 = {
        'dnn': {'base_1': ('0.400', '0.391', '0.375', '0.389'), 'base_2': ('0.487', '0.463', '0.410', '0.487'),
                'hybrid': ('0.3491', '0.3493', '0.331', '0.356'), 'hybrid__general2': ('0.357', '0.346', '0.340', '0.358')},
        'lightgbm': {'base_1': ('0.412', '0.403', '0.375', '0.403'), 'base_2': ('0.459', '0.458', '0.410', '0.455'),
                     'hybrid': ('0.372', '0.371', '0.331', '0.3705'), 'hybrid__general2': ('0.376', '0.370', '0.340', '0.370')},
    }
    for model, per in N40.items():
        for sp, texts in per.items():
            lab = f'qm9 L0 n40 {model} {sp}'
            s = _n40_scores(d, '0', sp, model) if d else None
            if not s:
                r.missing(lab, 'n40_model_side_regularisation/qm9__gaussian.json level 0 not found')
                continue
            for arm, tx in zip(('fixed', 'seed_avg', 'stack', 'retuned'), texts):
                r.check(f'{lab} {arm}', tx, s.get(arm))
    if d and d.get('folds_run') and list(d['folds_run']) != [0]:
        r.new_fold('qm9 n40', d['folds_run'], 'n40 QM9 has more folds now')


@check('N7-51', 'Labels as delivered: MDR1, nanobody thermostability and AC-SINS rows of tab:noise:res:shuffled (n18)')
def check_n7_51(r):
    """Rows added 2026-10-10 from the 10-10b results to the shuffled-score table (printed in the N7-10
    block) and the sentence on them in the antibody section (N7-33 block). Key path as N7-30:
    n18 <ds>.json per_space.<config>.per_detector.<det>.<slot>.per_model.<model>.{no_cleaning_mean,
    <cleanup>.cleaned_mean}; shuffled_control ... <seed>.shuffled_mean. Fold 0."""
    BLOCKS[r.id] = BLOCKS.get(r.id, '') + BLOCKS.get('N7-10', '') + BLOCKS.get('N7-33', '')
    SPEC = [
        ('adme_mdr1', 'base_1', 'neighbor_residual', '0.2', 'dnn', 'label_smooth', '0.4828', '0.467', ['0.4831', '0.484', '0.481']),
        ('adme_mdr1', 'hybrid', 'neighbor_residual', '0.2', 'dnn', 'downweight', '0.466', '0.445', ['0.4530', '0.4534', '0.463']),
        ('flab_thermo_tresanco', 'base_1', 'neighbor_residual', '0.05', 'dnn', 'oof_smooth', '0.770', '0.775', ['0.773', '0.769', '0.762']),
        ('flab_thermo_tresanco', 'base_2', 'cross_val', '0.2', 'dnn', 'remove', '0.548', '0.691', ['0.561', '0.560', '0.530']),
        ('flab_thermo_tresanco', 'base_2', 'cross_val', '0.2', 'svm', 'remove', '0.581', '0.737', ['0.606', '0.671', '0.524']),
        ('flab_thermo_tresanco', 'hybrid', 'cross_val', '0.2', 'dnn', 'remove', '0.724', '0.761', ['0.635', '0.780', '0.737']),
        ('ginkgo_acsins', 'base_1', 'neighbor_residual', '0.05', 'lightgbm', 'remove', '0.437', '0.539', ['0.419', '0.422', '0.522']),
        ('ginkgo_acsins', 'base_1', 'cross_val', '0.2', 'gp', 'remove', '0.522', '0.540', ['0.608', '0.505', '0.607']),
        ('ginkgo_acsins', 'base_2', 'cross_val', '0.2', 'gp', 'remove', '0.429', '0.598', ['0.398', '0.543', '0.459']),
        ('ginkgo_acsins', 'hybrid__general2', 'neighbor_residual', '0.01', 'lightgbm', 'remove', '0.509', '0.590', ['0.566', '0.572', '0.531']),
    ]
    for ds, sp, det, slot, model, cl, tnc, tcl, tsh in SPEC:
        lab = f'{ds} {model} {sp} {cl} {det}@{slot}'
        if J('noise', N18, f'{ds}.json') is None:
            r.missing(lab, f'noise/{N18}/{ds}.json not found')
            continue
        nc, c, shs, _ = _n18_cell(ds, sp, det, slot, model, cl)
        r.check(f'{lab} no cleaning, fold 0', tnc, nc)
        r.check(f'{lab} cleaned, fold 0', tcl, c)
        for i, t in enumerate(tsh):
            r.check(f'{lab} shuffled seed {i}, fold 0', t, shs[i] if i < len(shs) else None)
        _n18_new_folds(r, ds, lab, model, sp, cl, det, slot)


@check('N7-46', 'Cleaning keeps the hybrid ahead on small molecules')
def n7_46(r):
    """Section 'Results: Model and Representation', first paragraph. Table C of the group table:
    small molecules, antibodies and reaction, mitigation total, both labels. The LogD example
    (0.5560 against 0.5915) is the n21 number checked under N7-07."""
    _check_table_C(r, 'small molecules', ['1,792', '1,624', '72', '46', '41', None])
    _check_table_C(r, 'antibodies', ['734', '305', '84', '70', '258', None])
    _check_table_C(r, 'reaction', ['289', '148', '34', '10', '97', None])


NOT_CHECKABLE['N7-48'] = ('the model counts in its SOURCE line come from PATTERNS.md §4.3, a parse of the analysis '
                          'page NOISE_BY_GROUP.md; the paragraph names no number')
NOT_CHECKABLE['N7-49'] = 'an outline list (THESIS_OUTLINE.md 7.12 and 7.14); no number'


# =============================================================================================
# Section 3: QM9 with no noise added
# =============================================================================================
QM9_FLAGGED = ['n40', 'n20', 'n33', 'n23', 'n25', 'n31', 'n34', 'n35', 'n39', 'n11']


def _exp_dirs():
    names = set()
    for w in ('noise', 'full', 'view'):
        for f in files(CTX.d(w)):
            if re.match(r'n\d+[a-z]?_', f) and os.path.isdir(CTX.path(w, f)):
                names.add(f)
    return sorted(names, key=lambda s: (int(re.match(r'n(\d+)', s).group(1)), s))


def _json_has_level0(d):
    if not isinstance(d, dict):
        return False
    lv = d.get('noise_levels') or d.get('levels')
    if isinstance(lv, list) and any(float(x) == 0 for x in lv if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()):
        return True
    for k in ('per_level', 'per_sigma', 'by_level', 'vs_plain_dnn_same_labels'):
        v = d.get(k)
        if isinstance(v, dict) and any(str(x) in ('0', '0.0') for x in v):
            return True
    pr = d.get('per_regime')
    if isinstance(pr, dict):
        for v in pr.values():
            if isinstance(v, dict) and any(str(x) in ('0', '0.0') for x in v):
                return True
    return False


def _best_before_after(rs, cfgs, lib):
    """Over all settings of one method on the given configurations: the row with the best treated
    score -> (cfg, before, after)."""
    c = [x for x in rs if cfg(x.get('model_input')) in cfgs and x.get('score_Dprime') is not None]
    if not c:
        return None
    # one treated value per (cfg, setting): mean over draws
    g = collections.defaultdict(list)
    for x in c:
        g[(cfg(x['model_input']), x.get('method_setting'), x.get('similarity_setting'))].append(x)
    best = None
    for k, xs in g.items():
        a, b = mean([x['score_D'] for x in xs]), mean([x['score_Dprime'] for x in xs])
        if b is None or a is None:
            continue
        if best is None or (b < best[2] if lib else b > best[2]):
            best = (k[0], a, b)
    return best


def qm9_section(out):
    out.append(('head', '3. QM9 with no noise added (level 0): which experiments have it now'))
    out.append(('text', 'Per experiment: files for qm9 (final json, rows, checkpoints) and whether a level-0 result is in them. '
                        'Where rows exist: per model and method, the best single representation (base_1/base_2) and the best '
                        'hybrid (hybrid/hybrid__general2), each the setting with the best treated score, untreated -> treated, '
                        'fold 0, mean over draws. Flagged: experiments with no QM9 level-0 result on 2026-10-09.'))
    avail = []
    for sub in _exp_dirs():
        short = sub.split('_')[0]
        flag = ' [FLAGGED: missing on 2026-10-09]' if short in QM9_FLAGGED else ''
        finals, rowsf, ckpts = [], [], []
        for w in ('noise', 'full', 'view'):
            d = CTX.path(w, sub)
            for f in files(d, 'qm9*'):
                if 'fingerprint' in f:
                    continue
                (rowsf if f.endswith('.rows.jsonl') else finals if f.endswith('.json') else []).append((w, f))
            for part in ('_checkpoints', '_checkpoint', '_partial'):
                for f in files(os.path.join(d, part) if d else None, 'qm9*.json'):
                    ckpts.append((w, part, f))
        level0 = False
        for w, f in finals:
            if _json_has_level0(jload(CTX.path(w, sub, f))):
                level0 = True
        rs0 = []
        for w, f in rowsf:
            rs0 += [x for x in rows_file(CTX.path(w, sub, f)) if x.get('fold') in (0, None) and (
                x.get('level') in (0, 0.0) or (x.get('level') is None and x.get('noise_condition') in ('none', 'as_they_come')))]
        if rs0:
            level0 = True
        ck_txt = []
        for w, part, f in ckpts:
            d = jload(CTX.path(w, sub, part, f)) or {}
            pdone = d.get('passes_done')
            ck_txt.append(f'{w}/{part}/{f} passes done {len(pdone) if isinstance(pdone, list) else "?"}')
            if part == '_partial' and _json_has_level0(d):
                level0 = True
        if not (finals or rowsf or ckpts):
            if flag:
                out.append(('QM9 MISSING', f'{sub}: no qm9 file{flag}'))
            continue
        skipped = [str((jload(CTX.path(w, sub, f)) or {}).get('skipped') or (jload(CTX.path(w, sub, f)) or {}).get('reason'))[:60]
                   for w, f in finals if (jload(CTX.path(w, sub, f)) or {}).get('result_status') == 'skipped']
        kind = 'QM9 AVAILABLE' if level0 else 'QM9 NO LEVEL 0'
        if not level0 and finals and len(skipped) == len(finals) and not rowsf:
            kind = 'QM9 SKIPPED'
            finals = [(w, f + f' (skipped: {why})') for (w, f), why in zip(finals, skipped)]
        if level0 and flag:
            kind = 'QM9 NOW AVAILABLE'
            avail.append(short)
        out.append((kind, f'{sub}{flag}: final {[w + "/" + f for w, f in finals]} rows {[w + "/" + f for w, f in rowsf]}'
                          + (f' checkpoints {ck_txt}' if ck_txt else '')))
        if rs0:
            lib = rs0[0].get('lower_is_better', True)
            models = sorted({x.get('model') for x in rs0})
            show = [m for m in models if m in ('dnn', 'lightgbm')] or models
            for m in show:
                for meth in sorted({x.get('method') for x in rs0 if x.get('model') == m}):
                    rm = [x for x in rs0 if x.get('model') == m and x.get('method') == meth]
                    s = _best_before_after(rm, ('base_1', 'base_2'), lib)
                    h = _best_before_after(rm, ('hybrid', 'hybrid__general2'), lib)
                    ft = lambda t: 'n/a' if t is None else f'{t[0]} {t[1]:.4f} -> {t[2]:.4f}'
                    out.append(('', f'    {m:9s} {meth:28s} single {ft(s)} | hybrid {ft(h)}'))
        elif level0 and GT.get('rows'):
            q = [x for x in GT['rows'] if x['ds'] == 'qm9' and x['noise'].startswith('none added') and x['exp'] == short]
            for x in q:
                ft = lambda t: 'n/a' if t is None else f'{t[0]:.4f} -> {t[1]:.4f}'
                out.append(('', f'    {x["model"]:9s} {x["method"]:28s} single {ft(x["base"])} | hybrid {ft(x["hyb"])} (group-table page)'))
    for short in QM9_FLAGGED:
        if not any(s.split('_')[0] == short for s in _exp_dirs()):
            out.append(('QM9 MISSING', f'{short}: no experiment folder [FLAGGED: missing on 2026-10-09]'))
    return avail


# =============================================================================================
# Section 4: pending result slots
# =============================================================================================
def pending_section(out):
    out.append(('head', '4. Pending result slots'))
    now = []
    # n30b real censoring on ginkgo_pr_cho, and the QM9 arm
    for label, pat in [('n30b real censoring, ginkgo_pr_cho (layout real_lower_floor_*)', 'ginkgo_pr_cho*.json'),
                       ('n30b QM9 arm with injected clipping (--inject-censoring)', 'qm9__injected*.json')]:
        found = []
        for w in ('noise', 'full', 'view'):
            for sub in files(CTX.d(w), 'n30b*'):
                for f in files(CTX.path(w, sub), pat):
                    d = jload(CTX.path(w, sub, f)) or {}
                    lay = d.get('result_layout', '?')
                    folds = d.get('folds')
                    nf = sorted(folds) if isinstance(folds, (dict, list)) else folds
                    found.append(f'{w}/{sub}/{f} layout {lay} folds {nf}')
        if found:
            now.append(label)
            out.append(('PENDING NOW AVAILABLE', f'{label}: ' + '; '.join(found)))
        else:
            out.append(('PENDING', f'{label}: no file'))
    # n22 curator flag on herg_ic50
    found = []
    for w in ('noise', 'full', 'view'):
        for f in files(CTX.path(w, 'n22_curator_flag'), 'herg_ic50*'):
            found.append(f'{w}/n22_curator_flag/{f}')
    if found:
        now.append('n22 herg_ic50')
        out.append(('PENDING NOW AVAILABLE', 'n22 curator-flag detection on herg_ic50: ' + '; '.join(found)))
    else:
        out.append(('PENDING', 'n22 curator-flag detection on herg_ic50: no file'))
    return now


# =============================================================================================
# Section 5: the group table, rebuilt read-only
# =============================================================================================
GT = {}
GT_SCRIPTS = ['/Volumes/seagate/claude_analysis_2026-10-08/answers/_scripts',
              '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/f55db1bf-8fce-4a29-ab8b-783ed5b81253/scratchpad/group_table/answers/_scripts']
GT_LIB = '/Volumes/seagate/claude_analysis_2026-10-08/_lib.py'
GT_DIR = os.path.join(HERE, 'ch7_sources', 'group_table')


def rebuild_group_table(view_root, results_root, log):
    """Copy meeting_methods.py / meeting_readable.py / _common.py / method_groups.json and _lib.py
    into a temp dir, run them there under guard_run.py (patient files hidden and refused) on
    view_root, then parse.py and build.py. Nothing is written outside the temp dir."""
    src = next((d for d in GT_SCRIPTS if os.path.exists(os.path.join(d, 'meeting_methods.py'))), None)
    if src is None or not os.path.exists(GT_LIB):
        log(f'group table: scripts not found ({GT_SCRIPTS[0]}, {GT_LIB})')
        return
    tmp = tempfile.mkdtemp(prefix='ch7_group_table_')
    sd = os.path.join(tmp, 'answers', '_scripts')
    os.makedirs(sd)
    for f in ('meeting_methods.py', 'meeting_readable.py', '_common.py', 'method_groups.json'):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), sd)
    shutil.copy(GT_LIB, tmp)
    mm = os.path.join(sd, 'meeting_methods.py')
    s = open(mm).read().replace("sys.path.insert(0, '/Volumes/seagate/claude_analysis_2026-10-08')",
                                f"sys.path.insert(0, {tmp!r})")
    open(mm, 'w').write(s)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', KIRBY_RESULTS=results_root)
    guard = os.path.join(GT_DIR, 'guard_run.py')
    steps = [[sys.executable, guard, mm, view_root],
             [sys.executable, guard, os.path.join(sd, 'meeting_readable.py')],
             [sys.executable, os.path.join(GT_DIR, 'parse.py'), os.path.join(tmp, 'answers', 'MEETING_METHODS.md'),
              os.path.join(tmp, 'rows.json')],
             [sys.executable, os.path.join(GT_DIR, 'build.py'), os.path.join(tmp, 'rows.json')]]
    for cmd in steps:
        p = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True)
        refused = re.findall(r'REFUSED opens: (\d+)', p.stdout)
        if p.returncode != 0:
            log(f'group table: step {os.path.basename(cmd[2] if len(cmd) > 2 and cmd[1] == guard else cmd[1])} failed: '
                + (p.stderr.strip().split('\n')[-1] if p.stderr.strip() else ''))
            GT['tmp'] = tmp
            return
        if refused and refused[-1] != '0':
            log(f'group table: {refused[-1]} patient-data opens refused')
    GT['tmp'] = tmp
    GT['md'] = open(os.path.join(tmp, 'GROUP_TABLE.md')).read()
    GT['rows'] = json.load(open(os.path.join(tmp, 'rows.json')))


def _md_tables(md):
    """{(section letter / subsection, first cells): cells} for every table row of GROUP_TABLE.md."""
    out, sec = collections.OrderedDict(), ''
    for line in md.split('\n'):
        if line.startswith('## '):
            sec = line[3:].split('.')[0]
        elif line.startswith('### '):
            sec = sec[:1] + ' ' + line[4:]
        elif line.startswith('QM9, labels as delivered'):
            sec = sec[:1] + ' QM9 clean'
        elif line.startswith('| ') and not line.startswith('| group |') and not line.startswith('| domain |') \
                and not line.startswith('|---'):
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            nkey = 3 if sec.startswith('C') else (3 if sec.startswith('D') else 2)
            out[(sec, tuple(c[:nkey]))] = c[nkey:]
    return out


def group_table_section(out):
    out.append(('head', '5. Group-table counts rebuilt from the raw files, next to GROUP_TABLE.md'))
    cur_p = os.path.join(GT_DIR, 'GROUP_TABLE.md')
    if not GT.get('md'):
        out.append(('MISSING', 'group table not rebuilt' + (f' (temp dir {GT["tmp"]})' if GT.get('tmp') else '')))
        return 0
    out.append(('text', f'Rebuilt in {GT["tmp"]}. Columns of A/B: rows, yes, yes hybrid better after, yes single better after, '
                        'yes tie, single only, hybrid only, neither, no score. Columns of C: rows with both scores, hybrid better '
                        'before and after, worse before/better after, better before/worse after, worse both, ties. '
                        'D: rows, yes, yes as delivered, yes added damage, QM9 clean, datasets.'))
    cur = _md_tables(open(cur_p).read()) if os.path.exists(cur_p) else {}
    new = _md_tables(GT['md'])
    nchg = 0
    for k, v in new.items():
        if not k[0][:1] in 'ABC' and not k[0].startswith('D'):
            continue
        old = cur.get(k)
        name = f'{k[0]} | ' + ' | '.join(k[1])
        if old is None:
            out.append(('GT NEW ROW', f'{name}: {" ".join(v)}'))
            nchg += 1
        elif old != v:
            out.append(('GT CHANGED', f'{name}: now {" ".join(v)}  (GROUP_TABLE.md {" ".join(old)})'))
            nchg += 1
        else:
            out.append(('GT same', f'{name}: {" ".join(v)}'))
    for k in cur:
        if k not in new and k[0][:1] in 'ABCD':
            out.append(('GT ROW GONE', f'{k[0]} | ' + ' | '.join(k[1])))
            nchg += 1
    dropped = [l for l in GT['md'].split('\n') if l.startswith('- Rows on datasets outside')]
    for l in dropped:
        out.append(('text', l[2:]))
    return nchg


# =============================================================================================
# Section 5b: datasets with noise results that are not in Chapter 7 and not cut
# =============================================================================================
def unknown_datasets_section(out):
    out.append(('head', '5b. Datasets with noise results that are neither on the Chapter 7 list nor cut'))
    seen = collections.defaultdict(set)
    for w in ('noise', 'full', 'view'):
        for sub in files(CTX.d(w)):
            if not re.match(r'n\d+', sub):
                continue
            for f in files(CTX.path(w, sub)):
                if f.startswith('_') or 'fingerprint' in f:
                    continue
                ds = f.split('__')[0].split('.')[0]
                if ds and ds not in CH7_DATASETS and not is_cut(ds):
                    seen[ds].add(sub.split('_')[0])
    for ds, exps in sorted(seen.items()):
        out.append(('NOT ON LIST', f'{ds}: noise results in {len(exps)} experiments ({", ".join(sorted(exps, key=lambda s: int(re.sub(r"[^0-9]", "", s) or 0)))})'))
    if not seen:
        out.append(('text', 'none'))
    return sorted(seen)


# =============================================================================================
# Main
# =============================================================================================
def main():
    global CTX, BLOCKS, BASE
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='/Volumes/seagate/kirby_results_2026-10-09')
    ap.add_argument('--noise-view', default=None,
                    help='folder holding noise/ with the newest view of each file; default <results-root>_noise_view if it exists')
    ap.add_argument('--only', default=None, help='comma-separated N7 IDs')
    ap.add_argument('--markdown', default=None, help='also save the report here')
    ap.add_argument('--skip-group-table', action='store_true', help='do not rebuild the group table (saves ~1 min)')
    ap.add_argument('--write-baseline', action='store_true',
                    help=f'write the raw values of this run to {os.path.basename(BASELINE)} (only against the 2026-10-09 folders)')
    a = ap.parse_args()
    t0 = time.time()
    view = a.noise_view
    if view is None and os.path.isdir(a.results_root.rstrip('/') + '_noise_view'):
        view = a.results_root.rstrip('/') + '_noise_view'
    CTX = Ctx(a.results_root, view)
    BLOCKS = thesis_blocks()
    _add_plan_cuts()
    BASE = {} if a.write_baseline else (jload(BASELINE) or {})
    only = set(a.only.split(',')) if a.only else None
    out = []          # (kind, text)
    log = lambda s: print(s, file=sys.stderr, flush=True)
    out.append(('head', f'Chapter 7 numbers against {CTX.root}' + (f' and {CTX.view}' if CTX.view else ' (no noise view)')))
    out.append(('text', f'thesis.tex blocks found: {len(BLOCKS)}; baseline: '
                        f'{"loaded" if BASE else "none (every mismatch is reported as TEXT DIFFERS FROM RAW)"}'))
    need_gt = not a.skip_group_table and (only is None or only & {'N7-33', 'N7-46'})
    if need_gt:
        log('rebuilding the group table from raw files (about a minute) ...')
        tg = time.time()
        rebuild_group_table(CTX.view or CTX.root, CTX.root, log)
        log(f'  done in {time.time() - tg:.0f}s')
    counts = collections.Counter()
    baseline = {'_datasets': {}}
    ids_seen = set()
    for nid, label, f in sorted(CHECKS, key=lambda c: int(c[0].split('-')[1])):
        if only and nid not in only:
            continue
        ids_seen.add(nid)
        log(f'{nid} ...')
        r = Report(nid, label)
        t = time.time()
        try:
            f(r)
        except Exception as e:
            import traceback
            r.add('ERROR', f'{type(e).__name__}: {e} | ' + traceback.format_exc().strip().split('\n')[-3].strip())
        out.append(('id', f'{nid}  {label}  ({time.time() - t:.1f}s)'))
        if f.__doc__:
            out.append(('doc', ' '.join(f.__doc__.split())))
        for k, s in r.lines:
            out.append((k, s))
            counts[k] += 1
        baseline[nid] = r.values
        baseline['_datasets'][nid] = sorted(r.new_ds)
        counts['NEW DATASET (new)'] += sum(1 for k, s in r.lines if k == 'NEW DATASET' and '(new)' in s.split(':')[0])
    for nid, why in sorted(NOT_CHECKABLE.items(), key=lambda t: int(t[0].split('-')[1])):
        if only and nid not in only:
            continue
        out.append(('id', f'{nid}  not checkable automatically'))
        out.append(('NOT CHECKABLE', why))
    all_ids = {f'N7-{i:02d}' for i in range(1, 50)}
    unhandled = sorted(all_ids - {c[0] for c in CHECKS} - set(NOT_CHECKABLE), key=lambda s: int(s[3:]))
    if not only and unhandled:
        out.append(('id', 'IDs with no checker'))
        out.append(('NOT CHECKABLE', ', '.join(unhandled)))
    qm9_now, pend_now, gt_chg, unknown = [], [], None, []
    if not only:
        qm9_now = qm9_section(out)
        pend_now = pending_section(out)
        if need_gt:
            gt_chg = group_table_section(out)
        unknown = unknown_datasets_section(out)
        out.append(('head', '6. Clipped-label figure (Figure fig:noise:censoring_ladder): re-run with'))
        out.append(('text', f'python tests/analysis/thesis_figures.py --figure f9 --results-root {CTX.view or CTX.root} '
                            '--f9-datasets caco2,herg_ki,logd,qm9 --f9-correction censored_em'))
    # summary
    summ = [('head', 'SUMMARY'),
            ('text', f'IDs checked: {len(ids_seen)}; numbers OK: {counts["OK"]}'),
            ('text', f'CHANGED: {counts["CHANGED"]}'),
            ('text', f'TEXT DIFFERS FROM RAW: {counts["TEXT DIFFERS FROM RAW"]}'),
            ('text', f'MISSING: {counts["MISSING"]}; ERROR: {counts["ERROR"]}; TEXT EDITED?: {counts["TEXT EDITED?"]}'),
            ('text', f'NEW FOLD lines: {counts["NEW FOLD"]}'),
            ('text', f'NEW DATASET lines: {counts["NEW DATASET"]} (of which with no file on 2026-10-09: {counts["NEW DATASET (new)"]})')]
    if not only:
        summ += [('text', f'QM9 level-0 now available for flagged experiments: {", ".join(qm9_now) or "none"}'),
                 ('text', f'Pending slots now available: {", ".join(pend_now) or "none"}'),
                 ('text', f'Group-table rows changed against GROUP_TABLE.md: {gt_chg if gt_chg is not None else "not rebuilt"}'),
                 ('text', f'Datasets with noise results, not on the Chapter 7 list, not cut: {len(unknown)}')]
    tdr = [(k, s) for k, s in out if k in ('TEXT DIFFERS FROM RAW', 'CHANGED')]
    summ.append(('text', f'Not checkable: {", ".join(sorted(NOT_CHECKABLE, key=lambda s: int(s[3:])))}'))
    summ.append(('text', f'Run time: {time.time() - t0:.0f}s'))
    out += summ
    # print
    txt, md = [], []
    for k, s in out:
        if k == 'head':
            txt += ['', '=' * 100, s, '=' * 100]; md += ['', f'## {s}', '']
        elif k == 'id':
            txt += ['', f'--- {s}']; md += ['', f'### {s}', '']
        elif k == 'doc':
            txt.append(f'    ({s})'); md.append(f'_{s}_  ')
        elif k in ('text', ''):
            txt.append(s if k == 'text' else s); md.append(s + '  ')
        else:
            txt.append(f'  {k:<22s} {s}'); md.append(f'- **{k}** {s}')
    print('\n'.join(txt))
    if a.markdown:
        open(a.markdown, 'w').write('\n'.join(md) + '\n')
        log(f'wrote {a.markdown}')
    if a.write_baseline:
        if only:
            old = jload(BASELINE) or {}
            old.update({k: v for k, v in baseline.items() if k != '_datasets'})
            old.setdefault('_datasets', {}).update(baseline['_datasets'])
            baseline = old
        baseline['_written'] = f'{time.strftime("%Y-%m-%d %H:%M")} from {CTX.root}'
        json.dump(baseline, open(BASELINE, 'w'), indent=1, sort_keys=True)
        log(f'wrote {BASELINE}')


if __name__ == '__main__':
    main()
