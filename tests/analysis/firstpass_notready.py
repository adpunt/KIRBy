#!/usr/bin/env python3
"""One-shot reader for ALL complexity / similarity / noise experiment results.

Run on the server (where the real results live):
    git pull
    python tests/analysis/firstpass_notready.py            # defaults to tests/results
    python tests/analysis/firstpass_notready.py <RESULTS_ROOT>
    python tests/analysis/firstpass_notready.py <RESULTS_ROOT> c05 s02 n09   # only some

Design rules (locked — do not relitigate):
  * NO metric averages as a headline. Per-dataset rows first; win/tie/loss COUNTS
    and median/range describe a spread, never a mean of the metric.
  * We care about BEATING THE TOP existing thing: every core-claim reader compares
    the hybrid against the single best existing (model, rep), same model both sides.
  * Metric DIRECTION corrected everywhere: lower_is_better (RMSE/MAE) for molecular
    & reaction regression, higher-better otherwise. Preferred source = the file's
    own stored lower_is_better; else the dataset registry (populated as a SIDE
    EFFECT of importing the dataset-policy module, so we import it first); else a
    hardcoded fallback. Similarity/MI/robustness axes (SAC, mi_density, auc_norm,
    contamination, coherence, retrieval) have their OWN fixed direction, noted per
    reader — the dataset metric direction does NOT apply to them.
  * Read-only, stdlib only. Every reader is wrapped so one bad schema can't kill
    the run. Skipped / degenerate / error stubs are reported, not silently dropped.

Each experiment prints ONE section. Sections are grouped complexity -> similarity
-> noise so you can walk them one at a time.
"""
import json, os, sys, math, ast
from collections import defaultdict

_raw = sys.argv[1:]
def _is_expid(a):
    return len(a) == 3 and a[0] in "csn" and a[1:].isdigit()   # c02 / s13 / n09
OUT = "tests/analysis/latest_report.txt"   # full report is ALSO written here every run
QUIET = False                              # --quiet => file only, nothing to terminal
_pos = []
_i = 0
while _i < len(_raw):
    a = _raw[_i]
    if a in ("--out", "-o") and _i + 1 < len(_raw):
        OUT = _raw[_i + 1]; _i += 2; continue
    if a == "--no-save":
        OUT = None; _i += 1; continue
    if a == "--quiet":
        QUIET = True; _i += 1; continue
    _pos.append(a); _i += 1
ROOT = next((p for p in _pos if not _is_expid(p)), "tests/results")
ONLY = {p for p in _pos if _is_expid(p)}   # optional experiment-id filter e.g. {'c05','s02'}

TIE = 0.01                      # |Δ| <= TIE counts as a tie, not a win/loss
T95_DF4 = 2.776                 # two-sided t, 95%, df = n_folds-1 = 4


# ============================================================ direction / domain
def _load_directions():
    for p in ("tests", "."):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib
    for m in ("experiments.dataset_policy", "tests.experiments.dataset_policy"):
        try:
            importlib.import_module(m)
        except Exception:
            pass
    try:
        from kirby.registry import DATASET_REGISTRY as reg
        # Read the field, do not default it — see quick_hybrid_summary.
        # Fixed 2026-09-10 (metric-direction audit).
        return {k: bool(v.lower_is_better) for k, v in reg.items()}
    except Exception:
        return {}

_DIR = _load_directions()
_LOWER_FALLBACK = {           # molecular / reaction REGRESSION (metric = rmse/mae)
    "caco2", "herg_ki", "logd", "qm9", "drd2", "esol",
    "adme_hlm", "adme_rlm", "adme_mdr1", "adme_solubility",
    "bh_doyle_plates", "buchwald_hartwig", "doyle_hte_full_02",
    "heck_mizoroki_doyle", "suzuki_miyaura",
}

def lower_is_better(ds, d=None):
    """Prefer the file's own stored flag (describes the dataset's metric), then the
    registry, then the fallback set. `d` = the loaded result dict, if available."""
    if isinstance(d, dict):
        v = d.get("lower_is_better")
        if isinstance(v, bool):
            return v
    if ds in _DIR:
        return _DIR[ds]
    return ds in _LOWER_FALLBACK

def improvement(base, hyb, ds, d=None):
    lo = lower_is_better(ds, d)
    return (base - hyb) if lo else (hyb - base)

_MOLECULAR = {
    "caco2", "herg", "herg_ki", "logd", "qm9", "drd2", "esol",
    "adme_hlm", "adme_rlm", "adme_mdr1", "adme_solubility",
}
_REACTION = ("bh_", "buchwald", "doyle", "suzuki", "heck")
def domain(ds):
    if ds in _MOLECULAR:
        return "molecular"
    if ds.startswith(_REACTION):
        return "reaction"
    if ds.startswith("tcga_"):
        return "tcga"
    if ds.startswith(("nt_", "gb_")):
        return "genomic"
    return "antibody"

_DOMAIN_ORDER = ["molecular", "reaction", "antibody", "tcga", "genomic"]


# ============================================================ io helpers
def _fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)          # tolerant of NaN/Infinity tokens by default
    except Exception as e:
        return {"_error": str(e)}

_DIRCACHE = None
def _all_dirs():
    global _DIRCACHE
    if _DIRCACHE is None:
        _DIRCACHE = {}
        for dp, _dirs, _files in os.walk(ROOT):
            _DIRCACHE.setdefault(os.path.basename(dp), dp)
    return _DIRCACHE

def exp_dir(expname):
    """Locate an experiment's output dir by basename, wherever it landed."""
    return _all_dirs().get(expname)

def ds_files(expname):
    """[(dataset, path)] for a per-dataset experiment, sorted, domain-bucketable."""
    d = exp_dir(expname)
    if not d:
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            out.append((fn[:-5], os.path.join(d, fn)))
    return out

def combined_file(expname):
    """The single JSON of a combined ('multi'/'results') experiment, or None."""
    d = exp_dir(expname)
    if not d:
        return None
    js = [f for f in sorted(os.listdir(d)) if f.endswith(".json")]
    return os.path.join(d, js[0]) if js else None

def _ci(std):
    return T95_DF4 * std / math.sqrt(5) if _fin(std) else None

def f7(x, w=7, p=3):
    return f"{x:>{w}.{p}f}" if _fin(x) else f"{'--':>{w}}"

def sgn(x, w=8, p=3):
    return f"{x:>+{w}.{p}f}" if _fin(x) else f"{'--':>{w}}"

def iw(x, w=7):
    return f"{int(x):>{w}}" if _fin(x) else f"{'--':>{w}}"

def hdr(exp, title):
    print("\n" + "=" * 92)
    print(f"[{exp}]  {title}")
    print("=" * 92)

def wtl(deltas):
    """win/tie/loss counts from direction-corrected deltas (+ = hybrid better)."""
    w = sum(1 for x in deltas if _fin(x) and x > TIE)
    l = sum(1 for x in deltas if _fin(x) and x < -TIE)
    t = sum(1 for x in deltas if _fin(x) and -TIE <= x <= TIE)
    return w, t, l

def med_range(xs):
    xs = sorted(x for x in xs if _fin(x))
    if not xs:
        return None, None, None
    return xs[len(xs) // 2], xs[0], xs[-1]

def want(exp):
    return (not ONLY) or (exp in ONLY)

def by_domain(items):
    """items: [(ds, ...)]. yields (domain, [rows]) in canonical order."""
    buckets = defaultdict(list)
    for row in items:
        buckets[domain(row[0])].append(row)
    for dom in _DOMAIN_ORDER:
        if buckets.get(dom):
            yield dom, buckets[dom]
    for dom in buckets:                     # any unexpected domain
        if dom not in _DOMAIN_ORDER:
            yield dom, buckets[dom]

def safe(fn):
    """Run a reader, swallow schema surprises so the whole run survives."""
    try:
        fn()
    except Exception as e:
        print(f"  !! reader crashed: {type(e).__name__}: {e}")


# ============================================================ honest LOFO helper
# Experiments now persist a per-model, per-fold "fold_reselection" block (shape in
# tests/experiments/_shared.fold_reselection). Reading that block and calling the
# shared helper replaces the old read-time SELECTION (argmax over models / reps /
# grid cells) with a leave-one-fold-out re-score that taxes the hybrid arm AND
# the best-single arm identically: per held-out fold the winner is re-picked on
# the OTHER folds. The import is OPTIONAL — if experiments._shared can't load, or
# a JSON predates the block, every reader falls back to its historical view
# rather than crashing. sys.path was already extended by _load_directions().
try:
    from experiments._shared import (fold_reselection as _fold_reselection,
                                     hr_direction as _hr_direction,
                                     hr_exclude_folds as _hr_exclude,
                                     build_fold_reselection_input as _build_hr)  # noqa: F401
except Exception:
    _fold_reselection = None
    _hr_direction = None
    _build_hr = None
    # A file too old to carry the key reads as fold-0 pruned; see
    # experiments._shared.hr_exclude_folds for why.
    def _hr_exclude(block):
        if isinstance(block, dict) and 'exclude_folds' in block:
            return [int(f) for f in (block.get('exclude_folds') or [])]
        return [0]

def _reselection_block(d, *keys):
    """Fetch a persisted fold_reselection block by key, at top level or nested
    under 'results' (s10 and friends wrap their payload). Default key
    'fold_reselection'. Returns the block dict or None."""
    keys = keys or ("fold_reselection",)
    inner = d.get("results") if isinstance(d, dict) else None
    for src in (d, inner if isinstance(inner, dict) else None):
        if not isinstance(src, dict):
            continue
        for k in keys:
            b = src.get(k)
            if isinstance(b, dict):
                return b
    return None

def _run_reselection(block):
    """Call the shared fold_reselection on a stored block. Returns {model: {...}}
    or None when the helper is unavailable or the block is unusable (old file)."""
    if _fold_reselection is None or not isinstance(block, dict):
        return None
    cbm = block.get("candidates_by_model")
    bbm = block.get("baselines_by_model")
    if not isinstance(cbm, dict) and not isinstance(bbm, dict):
        return None

    # Direction is resolved OUTSIDE the try, and a failure is shouted, not
    # swallowed. bool(block.get("lower_is_better")) used to turn a missing key
    # into False = higher-is-better, which silently inverts every rmse/mae
    # delta. A block we cannot orient is dropped LOUDLY — a silent drop looks
    # identical to "this dataset had no result".
    try:
        lower = _hr_direction(block)
    except Exception as e:
        print(f"  [DIRECTION] refusing to score a block with no trustworthy "
              f"metric direction: {e}", file=sys.stderr)
        return None

    try:
        res = _fold_reselection(cbm or {}, bbm or {}, lower,
                              metric_name=block.get("metric_name"),
                              # Fold 0 filtered the recipe list of any search
                              # that produced this block, so its held-out term
                              # is dropped from BOTH arms. A file written
                              # before the key existed reads as pruned.
                              exclude_folds=_hr_exclude(block))
    except Exception:
        return None
    return res or None

def _opt(r):
    """single_fold => the arm has only one fold, so the score is the plain best,
    NOT a held-out honest number (flag it as optimistic)."""
    return " OPTIMISTIC(1fold)" if isinstance(r, dict) and r.get("single_fold") else ""

def _reselected_single(res, model=None):
    """Pull one entry from an fold_reselection result (for single-pseudo-model
    axes like SAC/kNN/kmeans). Prefers `model`, else the first sorted key."""
    if not isinstance(res, dict) or not res:
        return None
    if model and model in res:
        return res[model]
    return res[sorted(res)[0]]

def _print_reselected_models(res, indent="    "):
    """Print per-model honest rows (best single vs hybrid, held-out) under the
    already-printed dataset name. Returns the list of finite deltas so the caller
    can roll them into win/tie/loss counts. + = hybrid better (direction already
    corrected by the helper)."""
    deltas = []
    for m in sorted(res):
        r = res[m]
        bs = r.get("best_single_reselected"); hyb = r.get("hybrid_reselected")
        dl = r.get("delta")
        if _fin(dl):
            deltas.append(dl)
        verdict = "--"
        if _fin(dl):
            verdict = "WIN" if dl > TIE else ("LOSS" if dl < -TIE else "tie")
        print(f"{indent}{str(m)[:9]:<10}{f7(bs,8)}{f7(hyb,8)}{sgn(dl,9)}  "
              f"{verdict:<5}{iw(r.get('n_folds_used'),4)}f{_opt(r)}")
    return deltas


# ============================================================ C02 — core claim
# The honest first-pass view: per model, hybrid vs that model's best single rep on
# the SAME model, per dataset. When the c02 file carries an fold_reselection block
# the winner is re-picked per held-out fold (same tax both arms); the old cross-
# model BEST-ANY argmax and top-3 baseline-combo cherry-pick are DROPPED.

def _c02_path(ds):
    hits = []
    for dp, _dirs, files in os.walk(ROOT):
        if "c02" in os.path.basename(dp) and ds + ".json" in files:
            hits.append(os.path.join(dp, ds + ".json"))
    if not hits:
        return None
    def rank(p):
        try:
            n = len(load(p).get("configs", []))
        except Exception:
            n = -1
        return ("matching_grid" in p, n)
    return max(hits, key=rank)

def _c02_legacy_models(d, ds):
    """Fallback for pre-fold_reselection c02 files: per-model same-fold hybrid vs
    that model's OWN best single rep. No cross-model argmax, no top-3 combo
    cherry-pick (both dropped). Returns [(model, single, hybrid, delta)]."""
    base = d.get("baselines", {}) if isinstance(d.get("baselines"), dict) else {}
    bym = d.get("best_by_model", {}) if isinstance(d.get("best_by_model"), dict) else {}
    rows = []
    for m in sorted(set(base) | set(bym)):
        b = base.get(m) if isinstance(base.get(m), dict) else {}
        h = bym.get(m) if isinstance(bym.get(m), dict) else {}
        single = b.get("best_single_score"); hyb = h.get("score_mean")
        dl = improvement(single, hyb, ds, d) if (_fin(single) and _fin(hyb)) else None
        rows.append((m, single, hyb, dl))
    return rows

def read_c02():
    hdr("c02", "complexity grid — hybrid vs best single rep, SAME model, honest "
        "leave-one-fold-out (core claim)")
    # discover every dataset that has a c02 file, wherever it landed
    seen = {}
    for dp, _dirs, files in os.walk(ROOT):
        if "c02" in os.path.basename(dp):
            for fn in files:
                if fn.endswith(".json"):
                    seen.setdefault(fn[:-5], True)
    if not seen:
        print("  (no c02 files)"); return
    print("  Per model: held-out honest hybrid vs best single rep (winner re-picked per")
    print("  held-out fold on the OTHER folds; SAME tax both arms). No cross-model argmax,")
    print("  no top-3 combo cherry-pick. + = hybrid better, tie |Δ|<=%.2f." % TIE)
    print("  OPTIMISTIC(1fold) => only one fold available, NOT an honest number.\n")
    print(f"  {'dataset / model':<24}{'single':>8}{'hybrid':>8}{'Δ':>9}  "
          f"{'verdict':<6}{'nf':>4}")
    print("  " + "-" * 72)
    domwtl = defaultdict(list)
    n_legacy = 0
    for ds in sorted(seen):
        path = _c02_path(ds)
        if not path:
            continue
        d = load(path)
        if "_error" in d:
            print(f"  {str(ds)[:23]:<24}  (error: {str(d.get('_error'))[:40]})"); continue
        res = _run_reselection(_reselection_block(d))
        if res:
            print(f"  {str(ds)[:23]:<24}")
            for dl in _print_reselected_models(res):
                domwtl[domain(ds)].append(dl)
            continue
        # ---- fallback: pre-fold_reselection file (same-fold per-model view) ----
        rows = _c02_legacy_models(d, ds)
        if not any(r[0] for r in rows):
            print(f"  {str(ds)[:23]:<24}  --  no baselines/best_by_model"); continue
        n_legacy += 1
        print(f"  {str(ds)[:23]:<24}  (legacy same-fold — re-run c02 for held-out)")
        for m, single, hyb, dl in rows:
            if dl is not None:
                domwtl[domain(ds)].append(dl)
            verdict = "--" if dl is None else (
                "WIN" if dl > TIE else ("LOSS" if dl < -TIE else "tie"))
            print(f"    {str(m)[:9]:<10}{f7(single,8)}{f7(hyb,8)}{sgn(dl,9)}  {verdict:<6}")
    print("  " + "-" * 72)
    for dom, dd in domwtl.items():
        w, t, l = wtl(dd)
        print(f"  [{dom}] per-model honest: +{w} hybrid-better  ={t} tie  -{l} worse "
              f"({len(dd)} model-cases)")
    if n_legacy:
        print(f"  ({n_legacy} dataset(s) had no fold_reselection block — legacy same-fold "
              f"view shown, optimistic.)")


# ============================================================ C05 — core claim
def read_c05():
    hdr("c05", "learning curves — does the hybrid advantage GROW or SHRINK with n?")
    files = ds_files("c05_learning_curves")
    if not files:
        print("  (no c05 files)"); return
    print("  Per model (no deployed-model pick): Δ = hybrid − best single per n,")
    print("  direction-corrected (+ = hybrid better). fold_reselection_by_n gives the")
    print("  held-out Δ (winner re-picked per fold), averaged across seeds; older files")
    print("  fall back to the stored same-fold curves. Δ at smallest & largest n.\n")
    print(f"  {'dataset':<24}{'model':<9}{'n_small':>8}{'Δsmall':>9}"
          f"{'n_large':>9}{'Δlarge':>9}  trend      win@large")
    print("  " + "-" * 92)
    rows = []
    n_legacy = 0
    for ds, path in files:
        d = load(path)
        if "_error" in d or d.get("skipped"):
            continue
        hrbn = d.get("fold_reselection_by_n")
        if _fold_reselection is not None and isinstance(hrbn, dict) and hrbn:
            # model -> {n: mean held-out delta over seeds}
            per_model = defaultdict(dict)
            for nkey, blocks in hrbn.items():
                try:
                    n = int(float(nkey))
                except (TypeError, ValueError):
                    continue
                if not isinstance(blocks, list):
                    continue
                acc = defaultdict(list)
                for blk in blocks:
                    res = _run_reselection(blk)
                    if not res:
                        continue
                    for m, r in res.items():
                        if _fin(r.get("delta")):
                            acc[m].append(r["delta"])
                for m, dl_list in acc.items():
                    per_model[m][n] = sum(dl_list) / len(dl_list)
            for m in sorted(per_model):
                pts = sorted((n, v) for n, v in per_model[m].items() if _fin(v))
                if not pts:
                    continue
                n_s, d_s = pts[0]; n_l, d_l = pts[-1]
                trend = "1pt" if len(pts) < 2 else (
                    "grows" if d_l - d_s > TIE else
                    ("shrinks" if d_l - d_s < -TIE else "flat"))
                rows.append((ds, m, n_s, d_s, n_l, d_l, trend, d_l))
            continue
        # ---- fallback: pre-honest file, per-model same-fold curves (no pick) ----
        if not d.get("models"):
            continue
        n_legacy += 1
        sizes = d.get("sample_sizes") or []
        for m, mv in d["models"].items():
            hy, bs = mv.get("hybrid") or [], mv.get("best_single") or []
            pts = [(sizes[i] if i < len(sizes) else i,
                    improvement(bs[i], hy[i], ds, d))
                   for i in range(min(len(hy), len(bs)))
                   if _fin(hy[i]) and _fin(bs[i])]
            if not pts:
                continue
            n_s, d_s = pts[0]; n_l, d_l = pts[-1]
            trend = "1pt" if len(pts) < 2 else (
                "grows" if d_l - d_s > TIE else
                ("shrinks" if d_l - d_s < -TIE else "flat"))
            rows.append((ds, m, n_s, d_s, n_l, d_l, trend, d_l))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, m, ns, dsm, nl, dl, tr, dlarge in rr:
            win = "yes" if dlarge > TIE else ("no" if dlarge < -TIE else "tie")
            print(f"  {str(ds)[:23]:<24}{str(m)[:8]:<9}{iw(ns,8)}{sgn(dsm,9)}{iw(nl,9)}{sgn(dl,9)}"
                  f"  {tr:<9}  {win}")
    grows = sum(1 for r in rows if r[6] == "grows")
    shr = sum(1 for r in rows if r[6] == "shrinks")
    winl = sum(1 for r in rows if r[7] > TIE)
    print("  " + "-" * 92)
    print(f"  advantage grows with n: {grows}   shrinks: {shr}   "
          f"hybrid wins at largest n: {winl}/{len(rows)} model-curves"
          + (f"  ({n_legacy} legacy file(s))" if n_legacy else ""))


# ============================================================ C11 — core claim
def _c11_recipe(d):
    """Where this c11 file's hybrid recipe was chosen.

    Files written before the per-fold search existed carry no marker; their
    recipe came from a whole-dataset search, so they read as ALL ROWS.
    """
    v = d.get("hybrid_recipe_search")
    if v == "per_fold":
        return "fold"
    return "ALL ROWS"


def read_c11():
    hdr("c11", "ensemble_vs_hybrid — hybrid vs best single rep vs PCA-concat "
        "(same model; NOTE ensemble arm removed)")
    files = ds_files("c11_ensemble_vs_hybrid")
    if not files:
        print("  (no c11 files)"); return
    print("  Per model (no deployed-model pick). Held-out hybrid vs best single")
    print("  rep (winner re-picked per fold, same tax both arms); pca_cat shown from the")
    print("  stored means for context. + = hybrid better. Older files: same-fold view.")
    print("  The recipe column says where the hybrid's allocation method, importance")
    print("  method, evaluator and budget were chosen: 'fold' inside each outer training")
    print("  fold, 'ALL ROWS' over every row including the ones it is scored on.\n")
    print(f"  {'dataset':<24}{'model':<9}{'single':>8}{'pca_cat':>9}{'hybrid':>8}"
          f"{'Δvs_single':>11}{'recipe':>9}  verdict")
    print("  " + "-" * 92)
    rows = []
    all_deltas = []
    n_legacy = 0
    for ds, path in files:
        d = load(path)
        rs = d.get("results")
        if "_error" in d:
            continue
        lo = lower_is_better(ds, d)
        # pca_matched context per model, from the stored means (best per model).
        pca_by_model = defaultdict(lambda: None)
        if isinstance(rs, list):
            for r in rs:
                if r.get("method") != "pca_matched":
                    continue
                m, s = r.get("model"), r.get("score_mean")
                if not _fin(s):
                    continue
                cur = pca_by_model[m]
                if cur is None or (s < cur if lo else s > cur):
                    pca_by_model[m] = s
        res = _run_reselection(_reselection_block(d))
        if res:
            for m in sorted(res):
                r = res[m]
                dl = r.get("delta")
                if _fin(dl):
                    all_deltas.append(dl)
                v = "--" if not _fin(dl) else ("WIN" if dl > TIE else
                                               ("LOSS" if dl < -TIE else "tie"))
                rows.append((ds, m, r.get("best_single_reselected"), pca_by_model.get(m),
                             r.get("hybrid_reselected"), dl, _c11_recipe(d),
                             v + _opt(r)))
            continue
        # ---- fallback: per-model same-fold (no deployed-model argmax) ----
        if not isinstance(rs, list):
            continue
        n_legacy += 1
        best = defaultdict(dict)   # model -> method -> score
        for r in rs:
            m, meth, s = r.get("model"), r.get("method"), r.get("score_mean")
            if not _fin(s):
                continue
            cur = best[m].get(meth)
            if cur is None or (s < cur if lo else s > cur):
                best[m][meth] = s
        for m in sorted(best):
            mv = best[m]
            single, pca, hyb = mv.get("best_single"), mv.get("pca_matched"), mv.get("hybrid")
            delta = improvement(single, hyb, ds, d) if (_fin(single) and _fin(hyb)) else None
            if _fin(delta):
                all_deltas.append(delta)
            verdict = "--" if delta is None else ("WIN" if delta > TIE else
                                                  ("LOSS" if delta < -TIE else "tie"))
            rows.append((ds, m, single, pca, hyb, delta, _c11_recipe(d), verdict))
    n_all_rows = 0
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, m, s, p, h, dl, rec, v in rr:
            if rec != "fold":
                n_all_rows += 1
            print(f"  {str(ds)[:23]:<24}{str(m)[:8]:<9}{f7(s,8)}{f7(p,9)}{f7(h,8)}"
                  f"{sgn(dl,11)}{rec:>9}  {v}")
    print("  " + "-" * 92)
    w, t, l = wtl(all_deltas)
    print(f"  hybrid vs best single rep: +{w} better  ={t} tie  -{l} worse "
          f"({len(all_deltas)} model-cases)"
          + (f"  ({n_legacy} legacy file(s))" if n_legacy else ""))
    if n_all_rows:
        print(f"  {n_all_rows} row(s) marked ALL ROWS: the hybrid's recipe was "
              f"chosen over every row,")
        print(f"  including the rows its column is scored on, so that column "
              f"runs high. Re-run")
        print(f"  those datasets with --c11-recipe-search per_fold.")


# ============================================================ C13 — core thesis
def read_c13():
    hdr("c13", "forward compatibility — does a NEW rep improve the hybrid? "
        "(debiased gain; core thesis)")
    files = ds_files("c13_forward_compatibility")
    if not files:
        print("  (no c13 files)"); return
    print("  Δ already sign-normalized (+ = improvement). D−Drandom = debiased new-rep")
    print("  gain (the honest number); D−B = hybrid-with-new-rep beats new-rep-alone.")
    print("  (D−C is a selection-biased upper bound — deliberately NOT shown.)\n")
    print(f"  {'dataset':<22}{'new rep':<20}{'D-Drandom':>11}{'D-B':>9}  helps?")
    print("  " + "-" * 74)
    tot_pos = tot = 0
    for ds, path in files:
        d = load(path)
        if "error" in d and "results" not in d:
            print(f"  {str(ds)[:21]:<22}  (stub: {d.get('error')})"); continue
        res = d.get("results", {})
        per = res.get("per_rep", {})
        if not per:
            continue
        print(f"  --- {ds} ({domain(ds)}) ---")
        ds_has_delta = False
        for rep, rv in per.items():
            dl = rv.get("deltas", {})
            ddr = dl.get("D_minus_D_random"); dmb = dl.get("D_minus_B")
            if _fin(ddr) or _fin(dmb):
                ds_has_delta = True
            helps = "yes" if _fin(ddr) and ddr > TIE else (
                "no" if _fin(ddr) and ddr < -TIE else "~")
            if _fin(ddr):
                tot += 1; tot_pos += 1 if ddr > TIE else 0
            print(f"  {'':<22}{str(rep)[:19]:<20}{sgn(ddr,11)}{sgn(dmb,9)}  {helps}")
        if not ds_has_delta:                   # per_rep populated but no debiased-delta keys
            print(f"  {'':<22}(stale schema: no D_minus_D_random / D_minus_B — re-run c13, "
                  f"not a real null result)")
        s = res.get("summary", {})
        print(f"  {'':<22}win_rate D>Drandom={s.get('win_rate_D_vs_D_random')}  "
              f"D>B={s.get('win_rate_D_vs_B')}")
    print("  " + "-" * 74)
    print(f"  new reps with a real (debiased) gain: {tot_pos}/{tot}")


# ============================================================ C16 — core-ish
def read_c16():
    hdr("c16", "feature-selection-as-regularization — is the hybrid's edge just the "
        "budget CAP?")
    files = ds_files("c16_feature_selection_as_regularization")
    if not files:
        print("  (no c16 files)"); return
    print("  Per model family: simple_hybrid (selection only, default reg) vs")
    print("  best_tuned_full_dim (best reg, no selection). hybrid_wins => selection adds")
    print("  something beyond regularization.\n")
    print(f"  {'dataset':<24}{'family':<8}{'hybrid':>9}{'tuned_full':>11}  wins?")
    print("  " + "-" * 62)
    wins = tot = 0
    for ds, path in files:
        d = load(path)
        fams = d.get("families", {})
        if "_error" in d or not fams:
            continue
        print(f"  --- {ds} ({domain(ds)}) ---")
        for fam, fv in fams.items():
            comp = fv.get("comparison", {})
            sh = comp.get("simple_hybrid", {}).get("score")
            bt = comp.get("best_tuned_full_dim", {}).get("score")
            hw = comp.get("hybrid_wins")
            if hw is not None:
                tot += 1; wins += 1 if hw else 0
            print(f"  {'':<24}{str(fam)[:7]:<8}{f7(sh,9)}{f7(bt,11)}  {hw}")
    print("  " + "-" * 62)
    print(f"  selection beats regularization-only (hybrid_wins): {wins}/{tot} family-cases")


# ============================================================ C18 — core-ish
def read_c18():
    hdr("c18", "diversity vs depth — wide-diverse vs deep-single-family hybrids")
    files = ds_files("c18_diversity_vs_depth")
    if not files:
        print("  (no c18 files)"); return
    print("  Best-over-budgets score per condition. Direction from `metric`.")
    print("  Δ = diverse(uniform_all) − best depth (+ = diversity wins).\n")
    print(f"  {'dataset':<24}{'depth_best':>11}{'uniform':>9}{'natural':>9}"
          f"{'Δdiv-depth':>11}  winner")
    print("  " + "-" * 78)
    rows = []
    for ds, path in files:
        d = load(path)
        pb = d.get("per_budget", {})
        if "_error" in d or not pb:
            continue
        lo = lower_is_better(ds, d)
        best = {}
        for _bud, cv in pb.items():
            for cond, cc in (cv or {}).items():
                s = cc.get("score") if isinstance(cc, dict) else None
                if not _fin(s):
                    continue
                cur = best.get(cond)
                if cur is None or (s < cur if lo else s > cur):
                    best[cond] = s
        depth_vals = [best[k] for k in ("depth_1", "depth_2", "depth_3") if k in best]
        depth_best = (min if lo else max)(depth_vals) if depth_vals else None
        uni = best.get("uniform_all"); nat = best.get("natural")
        delta = improvement(depth_best, uni, ds, d) if (_fin(depth_best) and _fin(uni)) else None
        winner = "--"
        if delta is not None:
            winner = "diverse" if delta > TIE else ("depth" if delta < -TIE else "tie")
        rows.append((ds, depth_best, uni, nat, delta, winner))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, db, u, n, dl, w in rr:
            print(f"  {str(ds)[:23]:<24}{f7(db,11)}{f7(u,9)}{f7(n,9)}{sgn(dl,11)}  {w}")
    div = sum(1 for r in rows if r[5] == "diverse")
    dep = sum(1 for r in rows if r[5] == "depth")
    print("  " + "-" * 78)
    print(f"  diversity wins: {div}   depth wins: {dep}   ({len(rows)} datasets)")


# ============================================================ C10 — robustness
def read_c10():
    hdr("c10", "noise robustness — best hybrid vs best single rep across label-noise "
        "sigmas (auc_norm)")
    files = ds_files("c10_noise_robustness")
    if not files:
        print("  (no c10 files)"); return
    print("  auc_norm = direction-normalized robustness (HIGHER = more robust, any "
          "metric).")
    print("  Prefers a per-model held-out fold_reselection block when present; else the")
    print("  stored auc_norm robustness summary (single RF model; auc_norm is an over-")
    print("  sigma aggregate, so no per-fold per-fold reselection exists for this arm).\n")
    print(f"  {'dataset':<24}{'rep_auc':>9}{'hyb_auc':>9}{'Δ':>9}  more robust?")
    print("  " + "-" * 66)
    rows = []
    n_reselected = 0
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:23]:<24}  (skipped: {d.get('reason')})"); continue
        if d.get("error"):                     # e.g. noiseInject not installed — say why, don't drop
            print(f"  {str(ds)[:23]:<24}  (skipped: {d.get('error')})"); continue
        # Forward-compat: if c10 ever writes a per-model fold_reselection block,
        # report it per model rather than the read-time cross-cell auc_norm max.
        res = _run_reselection(_reselection_block(d))
        if res:
            n_reselected += 1
            print(f"  {str(ds)[:23]:<24}")
            for dl in _print_reselected_models(res):
                rows.append((ds, None, None, dl))
            continue
        ir = d.get("individual_reps", {}); hy = d.get("hybrids", {})
        if "_error" in d or not ir or not hy:
            continue
        # NOTE: c10 was not updated with per-fold auc_norm, so the best-hybrid vs
        # best-rep contrast here is still a read-time max over cells (single RF
        # model). Left as-is until the experiment emits per-fold robustness.
        def best_auc(block):
            b = None
            for _k, strat in block.items():
                for _s, cell in (strat or {}).items():
                    a = cell.get("auc_norm") if isinstance(cell, dict) else None
                    if _fin(a) and (b is None or a > b):
                        b = a
            return b
        ra, ha = best_auc(ir), best_auc(hy)
        delta = (ha - ra) if (_fin(ra) and _fin(ha)) else None
        rows.append((ds, ra, ha, delta))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, ra, ha, dl in rr:
            tag = "yes" if _fin(dl) and dl > TIE else ("no" if _fin(dl) and dl < -TIE else "~")
            if ra is None and ha is None:      # honest-block row already printed above
                continue
            print(f"  {str(ds)[:23]:<24}{f7(ra,9)}{f7(ha,9)}{sgn(dl,9)}  {tag}")
    w, t, l = wtl([r[3] for r in rows])
    print("  " + "-" * 66)
    print(f"  hybrid more robust: +{w}  ~{t}  -{l}  ({len(rows)} cases"
          + (f", {n_reselected} honest" if n_reselected else "") + ")")


# ============================================================ C12 — modality
def read_c12():
    hdr("c12", "modality ablation — does combining modalities beat the best single "
        "modality?")
    files = ds_files("c12_modality_ablation")
    if not files:
        print("  (no c12 files)"); return
    print("  Per model: best single modality vs best COMBINED, HELD-OUT (+ = combining")
    print("  helps). Winner re-picked per left-out fold, same tax both arms. The old")
    print("  cross-model/cross-rep max delta is DROPPED. Legacy files fall back to the")
    print("  stored same-fold all-combined delta.\n")
    print(f"    {'model':<10}{'single':>8}{'hybrid':>8}{'Δ':>9}  verdict  folds")
    print("  " + "-" * 58)
    wins = ties = losses = 0
    for ds, path in files:
        d = load(path)
        if "_error" in d:
            continue
        res = _run_reselection(_reselection_block(d))
        print(f"  {str(ds)[:60]}:")
        if res:
            for dl in _print_reselected_models(res):
                if dl > TIE:
                    wins += 1
                elif dl < -TIE:
                    losses += 1
                else:
                    ties += 1
        else:
            ac = d.get("all_combined")
            allc = ac.get("delta_vs_best_single") if isinstance(ac, dict) else None
            print(f"    (legacy same-fold) all-combined Δ={sgn(allc,0)}")
    print("  " + "-" * 58)
    print(f"  combining beats best single modality (per model, held-out): "
          f"+{wins} better  ={ties} tie  -{losses} worse")


# ============================================================ C01 — descriptive
def read_c01():
    hdr("c01", "rep characterization — intrinsic properties per rep (descriptive)")
    files = ds_files("c01_rep_characterization")
    if not files:
        print("  (no c01 files)"); return
    print("  Per dataset: strongest baseline rep (dir-aware) with its effective rank &")
    print("  mutual information — 'is the best rep intrinsically rich or compact?'\n")
    print(f"  {'dataset':<24}{'#reps':>6}  {'best rep':<18}{'score':>8}{'erank':>8}{'MI':>8}")
    print("  " + "-" * 72)
    for ds, path in files:
        d = load(path)
        reps = {k: v for k, v in d.items()
                if isinstance(v, dict) and "baselines" in v and k != "_meta"}
        if not reps:
            continue
        lo = lower_is_better(ds, d)
        best = None
        for rep, rv in reps.items():
            bl = rv.get("baselines", {})
            sc = [b.get("score_mean") for b in bl.values() if _fin(b.get("score_mean"))]
            if not sc:
                continue
            s = (min if lo else max)(sc)
            if best is None or (s < best[1] if lo else s > best[1]):
                best = (rep, s, rv.get("effective_rank"), rv.get("mutual_info"))
        if best:
            print(f"  {str(ds)[:23]:<24}{len(reps):>6}  {str(best[0])[:17]:<18}"
                  f"{f7(best[1],8)}{f7(best[2],8)}{f7(best[3],8)}")


# ============================================================ C03 — combined
def read_c03():
    hdr("c03", "cross-dataset transfer — native-optimal config + does complexity "
        "predict budget? (combined file)")
    path = combined_file("c03_cross_dataset_transfer")
    if not path:
        print("  (no c03 file — single combined job, depends on c02)"); return
    d = load(path)
    if "_error" in d:
        print(f"  ERROR {d['_error'][:60]}"); return
    print(f"  {'dataset':<24}{'model':<9}{'importance':<13}{'budget':>7}{'dim':>6}{'score':>9}")
    print("  " + "-" * 70)
    for ds, r in d.items():
        if ds.startswith("_") or not isinstance(r, dict) or "error" in r:
            continue
        print(f"  {str(ds)[:23]:<24}{str(r.get('best_model'))[:8]:<9}"
              f"{str(r.get('best_importance'))[:12]:<13}{iw(r.get('best_budget'))}"
              f"{iw(r.get('best_dim'),6)}{f7(r.get('best_score'),9,4)}")
    corr = d.get("_complexity_correlation", {})
    if corr:
        print("\n  complexity -> optimal budget (Spearman rho; want large |rho|, low p):")
        for k, c in corr.items():
            print(f"    {k:<18} budget rho={sgn(c.get('rho_vs_budget'),0)} "
                  f"(p={f7(c.get('p_vs_budget'),0)})  dim rho={sgn(c.get('rho_vs_dim'),0)}")


# ============================================================ C04 — importance metric
def read_c04():
    hdr("c04", "importance as a metric — importance-weighted kNN / retrieval vs "
        "unweighted")
    files = ds_files("c04_importance_as_metric")
    if not files:
        print("  (no c04 files)"); return
    print("  Δ = weighted − unweighted (+ = importance weighting helps). clf skipped.\n")
    print(f"  {'dataset':<24}{'knn method':<15}{'knnΔ':>8}   {'sim method':<15}{'simΔ':>8}")
    print("  " + "-" * 78)
    kpos = kneg = 0
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:23]:<24}  (clf skipped)"); continue
        methods = d.get("methods", {})
        if "_error" in d or not methods:
            continue
        bk, bs = d.get("best_knn_method"), d.get("best_sim_method")
        kd = methods.get(bk, {}).get("knn_delta_k10")
        sd = methods.get(bs, {}).get("sim_delta_r2")
        if _fin(kd):
            kpos += kd > TIE; kneg += kd < -TIE
        print(f"  {str(ds)[:23]:<24}{str(bk)[:14]:<15}{sgn(kd)}   {str(bs)[:14]:<15}{sgn(sd)}")
    print("  " + "-" * 78)
    print(f"  importance-weighted kNN helps: +{kpos}  hurts: -{kneg}")


# ============================================================ C06 / C07 / C08 / C09
def read_c06():
    hdr("c06", "within-family gradient — capacity↔optimal-budget correlation "
        "(regression only)")
    files = ds_files("c06_within_family_gradient")
    if not files:
        print("  (no c06 files)"); return
    print("  Per dataset: model families with a MEASURED capacity/budget rho, and the")
    print("  rho range across families (higher-capacity model -> larger optimal budget?).\n")
    print(f"  {'dataset':<24}{'#fam':>5}{'#measured':>10}{'med rho':>9}{'rho range':>18}")
    print("  " + "-" * 66)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:23]:<24}  (skipped: {d.get('reason')})"); continue
        fams = d.get("families", {})
        if not fams:
            continue
        rhos = []
        for fv in fams.values():
            c = fv.get("capacity_budget_correlation", {})
            if c.get("status") == "measured" and _fin(c.get("rho")):
                rhos.append(c["rho"])
        m, lo_, hi_ = med_range(rhos)
        rng = f"[{sgn(lo_,0)},{sgn(hi_,0)}]" if rhos else "--"
        print(f"  {str(ds)[:23]:<24}{len(fams):>5}{len(rhos):>10}{sgn(m,9)}{rng:>18}")

def read_c07():
    hdr("c07", "data-complexity partitions — best hybrid on EASY vs HARD subpopulations")
    files = ds_files("c07_data_complexity_partitions")
    if not files:
        print("  (no c07 files)"); return
    print("  Per model: HELD-OUT hybrid score on the EASY vs HARD subpopulation (winner")
    print("  re-picked per left-out fold). The old single cross-model best cell is DROPPED.")
    print("  config_differs = the tuned hybrid config changes between easy & hard.\n")
    print(f"  {'dataset':<20}{'partition':<16}{'model':<10}{'easy':>8}{'hard':>8}  cfg_differs")
    print("  " + "-" * 74)
    for ds, path in files:
        d = load(path)
        parts = d.get("partitions", {})
        if "_error" in d or not parts:
            continue
        for pn, pv in parts.items():
            eh = _run_reselection(_reselection_block(pv.get("easy") or {}))
            hh = _run_reselection(_reselection_block(pv.get("hard") or {}))
            if eh or hh:
                for m in sorted(set(eh or {}) | set(hh or {})):
                    es = (eh.get(m, {}) if eh else {}).get("hybrid_reselected")
                    hs = (hh.get(m, {}) if hh else {}).get("hybrid_reselected")
                    print(f"  {str(ds)[:19]:<20}{str(pn)[:15]:<16}{str(m)[:9]:<10}"
                          f"{f7(es,8)}{f7(hs,8)}  {pv.get('config_differs')}")
            else:
                e = (pv.get("easy") or {}).get("best") or {}
                h = (pv.get("hard") or {}).get("best") or {}
                print(f"  {str(ds)[:19]:<20}{str(pn)[:15]:<16}{'(legacy)':<10}"
                      f"{f7(e.get('score'),8)}{f7(h.get('score'),8)}  {pv.get('config_differs')}")

def read_c08():
    hdr("c08", "calibration — does the 'optimal' hybrid stay calibrated?")
    files = ds_files("c08_calibration")
    if not files:
        print("  (no c08 files)"); return
    print("  cal_error = |PICP − 0.90| (lower = better calibrated). PICP target 0.90.")
    print("  conditions_distinct=False => budgets collapsed, contrast meaningless.\n")
    print(f"  {'dataset':<24}{'PICP':>8}{'cal_err':>9}  distinct?")
    print("  " + "-" * 54)
    for ds, path in files:
        d = load(path)
        conds = d.get("conditions", {})
        opt = (conds.get("optimal") or {}).get("avg", {})
        if not opt:
            continue
        print(f"  {str(ds)[:23]:<24}{f7(opt.get('picp'),8)}{f7(opt.get('cal_error'),9)}"
              f"  {d.get('conditions_distinct')}")

def read_c09():
    hdr("c09", "formal analysis — ANOVA variance decomposition of the hybrid lift")
    files = ds_files("c09_formal_analysis")
    if not files:
        print("  (no c09 files)"); return
    print("  eta2 = % of score variance explained by model / budget / their interaction")
    print("  eta_resid = % UNexplained (usually the largest term — shown so the others")
    print("  are interpretable); needs c02 results; absent otherwise.\n")
    print(f"  {'dataset':<24}{'eta_model':>10}{'eta_budget':>11}{'eta_inter':>10}{'eta_resid':>11}")
    print("  " + "-" * 69)
    for ds, path in files:
        d = load(path)
        an = d.get("anova")
        if not isinstance(an, dict) or "error" in an or "eta2" not in an:
            print(f"  {str(ds)[:23]:<24}  (no anova{': '+an.get('error') if isinstance(an,dict) and an.get('error') else ''})")
            continue
        e = an["eta2"]
        resid = e.get("residual")
        if not _fin(resid):                    # derive it if the file didn't store it
            parts = [e.get(k) for k in ("model", "budget", "interaction")]
            resid = 100.0 - sum(p for p in parts if _fin(p)) if any(_fin(p) for p in parts) else None
        print(f"  {str(ds)[:23]:<24}{f7(e.get('model'),10,1)}{f7(e.get('budget'),11,1)}"
              f"{f7(e.get('interaction'),10,1)}{f7(resid,11,1)}")


# ============================================================ C15 / C17 / C19
def read_c15():
    hdr("c15", "ML-theory depth — double-descent instability (regression only)")
    files = ds_files("c15_ml_theory_depth")
    if not files:
        print("  (no c15 files)"); return
    print("  instability_ratio: hybrid vs PCA (lower = more stable across budgets).\n")
    print(f"  {'dataset':<24}{'inst_hybrid':>12}{'inst_pca':>10}  sub-analyses")
    print("  " + "-" * 62)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:23]:<24}  (skipped: {d.get('reason')})"); continue
        dd = d.get("double_descent", {})
        subs = [k for k in ("distance_concentration", "kernel_analysis",
                            "double_descent", "approximation_theory") if k in d]
        ih = dd.get("instability_ratio_hybrid"); ip = dd.get("instability_ratio_pca")
        nbud = len(dd.get("budgets") or [])
        # far-bucket-empty degeneracy: narrow budget grid -> variance/floor blows the ratio
        # up to millions; flag it instead of printing it as if it were a normal quantity.
        degen = (_fin(ih) and abs(ih) > 100) or (_fin(ip) and abs(ip) > 100)
        note = f"!! DEGENERATE (far bucket empty? nbud={nbud})" if degen else ",".join(subs)
        print(f"  {str(ds)[:23]:<24}{f7(ih,12)}{f7(ip,10)}  {note}")

def read_c17():
    hdr("c17", "information-theoretic — does the hybrid add INFORMATION or just "
        "dimensions? (reg only)")
    files = ds_files("c17_information_theoretic_decomposition")
    if not files:
        print("  (no c17 files)"); return
    print("  winning strategy across {hybrid,pca,mrmr,rfe,random}, one budget at a time.")
    print("  mi_density is comparable between strategies at one budget, never across")
    print("  budgets, so no average is taken. hyb/rnd densities below are the MEDIAN")
    print("  over budgets and are a spread, not a score.\n")
    print(f"  {'dataset':<24}{'wins/#bud':<12}{'hyb_dens':>10}{'rnd_dens':>10}"
          f"{'hyb range':>18}{'#bud':>6}")
    print("  " + "-" * 84)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:23]:<24}  (skipped: {d.get('reason')})"); continue
        pb = d.get("per_budget", {})
        if not pb:
            continue
        hd, rd = [], []
        for _b, cell in pb.items():
            h = (cell.get("hybrid") or {}).get("mi_density") if isinstance(cell, dict) else None
            r = (cell.get("random") or {}).get("mi_density") if isinstance(cell, dict) else None
            if _fin(h):
                hd.append(h)
            if _fin(r):
                rd.append(r)
        bpb = d.get("summary", {}).get("best_strategy_per_budget") or {}
        won = [v for v in bpb.values() if v]
        top = max(set(won), key=won.count) if won else None
        bs = f"{top}{won.count(top)}/{len(won)}" if top else "--"
        mh, lo_h, hi_h = med_range(hd); mr, _, _ = med_range(rd)
        rng = f"[{f7(lo_h,0,3)},{f7(hi_h,0,3)}]" if hd else "--"
        print(f"  {str(ds)[:23]:<24}{str(bs)[:11]:<12}{f7(mh,10)}{f7(mr,10)}{rng:>18}{len(hd):>6}")

def read_c19():
    hdr("c19", "importance-weighted training — does weighting beat unweighted on the "
        "hybrid?")
    files = ds_files("c19_importance_weighted_training")
    if not files:
        print("  (no c19 files)"); return
    print("  Per dataset: winning condition, and #models where best != 'unweighted'.\n")
    print(f"  {'dataset':<24}{'most_winning':<20}{'models_weighted_wins':>22}")
    print("  " + "-" * 68)
    for ds, path in files:
        d = load(path)
        ps = (d.get("per_space") or {}).get("hybrid", {})
        if "_error" in d or not ps:
            print(f"  {str(ds)[:23]:<24}  (no hybrid space)"); continue
        nweighted = sum(1 for mv in ps.values()
                        if isinstance(mv, dict) and mv.get("best_condition") not in
                        (None, "unweighted"))
        summ = d.get("summary", {})
        mw = summ.get("most_winning_condition") or summ.get("best_overall")
        if mw is None:                         # last resort: argmax of the win tally it stored
            cw = summ.get("condition_wins")
            if isinstance(cw, dict) and cw:
                mw = max(cw, key=lambda k: cw[k])
        print(f"  {str(ds)[:23]:<24}{str(mw)[:19]:<20}{f'{nweighted}/{len(ps)}':>22}")


# ============================================================ C21 — often dead
def read_c21():
    hdr("c21", "single-rep augmentation — base rep vs base+augmentation (known often "
        "degenerate)")
    files = ds_files("c21_single_rep_augmentation")
    if not files:
        print("  (no c21 files)"); return
    print("  Only aug_status=='ok' rows count. improved/hurt from summary (sign-applied).")
    print("  all_skipped / error => dead run (flagged, not counted as 'no improvement').\n")
    print(f"  {'dataset':<24}{'state':<14}{'improved':>9}{'hurt':>6}{'best_rel':>10}")
    print("  " + "-" * 66)
    for ds, path in files:
        d = load(path)
        if "error" in d and "rep_results" not in d:
            print(f"  {str(ds)[:23]:<24}{'ERROR':<14}  {d.get('error')}"); continue
        s = d.get("summary", {})
        if s.get("all_skipped"):
            print(f"  {str(ds)[:23]:<24}{'all_skipped':<14}  (dead)"); continue
        oks = [r for r in d.get("rep_results", [])
               if r.get("aug_status") == "ok" and _fin(r.get("relative_improvement"))]
        best_rel = max((r["relative_improvement"] for r in oks), default=None)
        state = "ok" if oks else "no-ok(dead)"   # zero countable tests => dead, not "no gain"
        print(f"  {str(ds)[:23]:<24}{state:<14}{iw(s.get('improved'),9)}{iw(s.get('hurt'),6)}"
              f"{sgn(best_rel,10)}")


# ============================================================ C22 / C23 / C24
def read_c22():
    hdr("c22", "explainability — per-rep SHAP contribution of the winning hybrid "
        "(descriptive)")
    files = ds_files("c22_explainability")
    if not files:
        print("  (no c22 files)"); return
    print("  Top rep by SHAP share of the hybrid, + the score the recipe was\n"
          "  SELECTED by over every row. That column is not held-out\n"
          "  performance — c22's per-fold scores are in fold_results.\n")
    print(f"  {'dataset':<24}{'top rep (SHAP)':<20}{'shap%':>8}{'sel':>9}")
    print("  " + "-" * 64)
    for ds, path in files:
        d = load(path)
        if "error" in d:
            print(f"  {str(ds)[:23]:<24}  (stub: {d.get('error')})"); continue
        # shap_share_by_rep is each fold's own share, averaged — matched to
        # that fold's feature map before averaging. shap_by_rep is the older
        # total; keep reading it for files written before the change.
        share = d.get("shap_share_by_rep") or {}
        shap = d.get("shap_by_rep", {})
        if share:
            top = max(share.items(),
                      key=lambda kv: abs(kv[1]) if _fin(kv[1]) else -1)
            pct = 100 * abs(top[1]) if _fin(top[1]) else None
        elif shap:
            tot = sum(abs(v) for v in shap.values() if _fin(v)) or 1.0
            top = max(shap.items(), key=lambda kv: abs(kv[1]) if _fin(kv[1]) else -1)
            pct = 100 * abs(top[1]) / tot if _fin(top[1]) else None
        if share or shap:
            print(f"  {str(ds)[:23]:<24}{str(top[0])[:19]:<20}{f7(pct,8,1)}{f7(d.get('recipe_selection_score', d.get('search_score')),9)}")
        else:
            print(f"  {str(ds)[:23]:<24}{'(no SHAP)':<20}{'--':>8}{f7(d.get('recipe_selection_score', d.get('search_score')),9)}")

def read_c23():
    hdr("c23", "evaluator-model matching — does matching the internal evaluator to "
        "the deployed model help?")
    files = ds_files("c23_evaluator_model_matching")
    if not files:
        print("  (no c23 files)"); return
    print("  delta_vs_matched (+ = the MATCHED evaluator is better). Median across models")
    print("  for slightly-off (B) and strongly-off (C) evaluators.\n")
    print(f"  {'dataset':<24}{'medΔ B(slight)':>15}{'medΔ C(strong)':>16}  matters?")
    print("  " + "-" * 66)
    for ds, path in files:
        d = load(path)
        rs = d.get("results")
        if not isinstance(rs, list):
            continue
        bd, cd = [], []
        for r in rs:
            if r.get("condition_type") != "ABC":
                continue
            sc = r.get("scores", {})
            b = (sc.get("B_slight_off") or {}).get("delta_vs_matched")
            c = (sc.get("C_strong_off") or {}).get("delta_vs_matched")
            if _fin(b):
                bd.append(b)
            if _fin(c):
                cd.append(c)
        mb, _, _ = med_range(bd); mc, _, _ = med_range(cd)
        matters = "yes" if (_fin(mc) and mc > TIE) else "~"
        print(f"  {str(ds)[:23]:<24}{sgn(mb,15)}{sgn(mc,16)}  {matters}")

def read_c24():
    hdr("c24", "blueprint transfer — does a hybrid blueprint from other folds transfer?")
    files = ds_files("c24_blueprint_transfer")
    if not files:
        print("  (no c24 files)"); return
    print("  Δ (+ = OWN/native blueprint better; ~0 = transfers well). mean_abs_delta")
    print("  = typical transfer penalty magnitude.\n")
    print(f"  {'dataset':<24}{'mean_delta':>11}{'mean|delta|':>12}{'max|delta|':>12}  note")
    print("  " + "-" * 70)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:23]:<24}  (skipped: {d.get('reason')})"); continue
        agg = d.get("aggregate", {})
        if not agg:
            continue
        mad = agg.get("mean_abs_delta")
        if not _fin(mad):                      # all-NaN fold scores propagated in -> degenerate
            note = "degenerate (all-NaN)"
        elif mad < TIE:
            note = "transfers"
        else:
            note = "penalty"
        print(f"  {str(ds)[:23]:<24}{sgn(agg.get('mean_delta'),11)}"
              f"{f7(mad,12)}{f7(agg.get('max_abs_delta'),12)}  {note}")


# ============================================================ SIMILARITY
def read_s01():
    hdr("s01", "similarity baseline — best-rep similarity-activity correlation (SAC) "
        "per dataset")
    files = ds_files("s01_similarity_baseline")
    if not files:
        print("  (no s01 files)"); return
    print("  SAC = Spearman(|Δrep|, |Δactivity|); HIGHER = better (own direction).\n")
    print(f"  {'dataset':<26}{'best rep':<18}{'SAC':>8}{'sig?':>6}{'viol':>8}")
    print("  " + "-" * 68)
    for ds, path in files:
        d = load(path)
        best = (None, 0.0, None, None)
        for rep, st in d.items():
            if not isinstance(st, dict) or "sac_rho" not in st:
                continue
            rho = st.get("sac_rho")
            if _fin(rho) and abs(rho) > abs(best[1]):
                best = (rep, rho, st.get("sac_significant"), st.get("violation_rate"))
        if best[0]:
            print(f"  {str(ds)[:25]:<26}{str(best[0])[:17]:<18}{f7(best[1],8)}"
                  f"{str(best[2]):>6}{f7(best[3],8)}")

def read_s02():
    hdr("s02", "hybrid similarity — hybrid distance SAC vs best single rep (core claim, "
        "similarity side)")
    files = ds_files("s02_hybrid_similarity")
    if not files:
        print("  (no s02 files)"); return
    print("  SAC higher = better. Prefers the held-out CV fold_reselection block (single")
    print("  kNN pseudo-model; winner re-picked per fold, same tax both arms); older")
    print("  files fall back to the stored whole-data best_sac. lift = hybrid − best rep.\n")
    print(f"  {'dataset':<26}{'best indiv rep':<18}{'indiv':>8}{'hybrid':>8}{'lift':>8}  win")
    print("  " + "-" * 78)
    rows = []
    for ds, path in files:
        d = load(path)
        if "_error" in d:
            continue
        r = _reselected_single(_run_reselection(_reselection_block(d)))
        if r:
            bw = r.get("baseline_winners_per_fold") or []
            iname = (bw[0] + _opt(r)) if bw else ("(held-out)" + _opt(r))
            rows.append((ds, iname, r.get("best_single_reselected"),
                         r.get("hybrid_reselected"), r.get("delta")))
            continue
        # ---- fallback: pre-honest whole-data SAC ----
        cands = []
        for key in ("best_individual_rep", "best_individual_rep_weighted"):
            bi = d.get(key)
            if isinstance(bi, dict) and _fin(bi.get("sac_rho")):
                cands.append((bi.get("name"), bi["sac_rho"]))
        cands.sort(key=lambda c: -c[1])
        iname, isac = cands[0] if cands else (None, None)
        hs = d.get("best_sac")
        lift = (hs - isac) if (_fin(hs) and _fin(isac)) else None
        rows.append((ds, iname, isac, hs, lift))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, nm, isac, hs, lift in rr:
            win = "yes" if _fin(lift) and lift > TIE else ("no" if _fin(lift) and lift < -TIE else "~")
            print(f"  {str(ds)[:25]:<26}{str(nm)[:17]:<18}{f7(isac,8)}{f7(hs,8)}{sgn(lift)}  {win}")
    w, t, l = wtl([r[4] for r in rows])
    print("  " + "-" * 78)
    print(f"  hybrid distance beats best single rep: +{w}  ~{t}  -{l}  ({len(rows)} datasets)")

def read_s03():
    hdr("s03", "feature concentration — is the signal concentrated after selection? "
        "(mi_density_ratio)")
    files = ds_files("s03_feature_concentration")
    if not files:
        print("  (no s03 files)"); return
    print("  mi_density_ratio > 1 = selected features are denser in signal than the full")
    print("  rep. Median & range across (model,budget)×rep cells (NO mean headline).")
    print("  DESCRIPTIVE ONLY — this is a signal-concentration diagnostic, not a hybrid")
    print("  win/loss claim (the MI arm is redesigned as descriptive).\n")
    print(f"  {'dataset':<26}{'#cells':>7}{'med ratio':>11}{'ratio range':>20}")
    print("  " + "-" * 66)
    caveat_shown = False
    for ds, path in files:
        d = load(path)
        if d.get("descriptive_only") and not caveat_shown:
            cav = d.get("_caveat")
            if cav:
                print(f"  (caveat: {str(cav)[:80]})")
            caveat_shown = True
        per = d.get("per_rep", {})
        ratios = []
        for _mb, repmap in per.items():
            if not isinstance(repmap, dict) or "error" in repmap:
                continue
            for _rep, cell in repmap.items():
                if isinstance(cell, dict) and _fin(cell.get("mi_density_ratio")):
                    ratios.append(cell["mi_density_ratio"])
        m, lo_, hi_ = med_range(ratios)
        rng = f"[{f7(lo_,0,2)},{f7(hi_,0,2)}]" if ratios else "--"
        print(f"  {str(ds)[:25]:<26}{len(ratios):>7}{f7(m,11,2)}{rng:>20}")

def read_s04():
    hdr("s04", "activity cliffs — hybrid vs best rep on cliff compounds (molecular only)")
    files = ds_files("s04_activity_cliffs")
    if not files:
        print("  (no s04 files)"); return
    print("  Non-molecular / no-ecfp4 datasets are skipped by design. Prefers the")
    print("  held-out fold_reselection block: per model (knn hybrid recipes; rf baseline-")
    print("  only -> NaN Δ) held-out hybrid vs best single rep on the overall metric.")
    print("  Older files fall back to the stored whole-data cliff score.\n")
    print(f"  {'dataset / model':<26}{'single':>10}{'hybrid':>10}{'Δ':>9}  verdict")
    print("  " + "-" * 74)
    for ds, path in files:
        d = load(path)
        if d.get("skipped") or "error" in d:
            print(f"  {str(ds)[:25]:<26}{('skip: '+str(d.get('reason') or d.get('error')))[:25]:<26}")
            continue
        res = _run_reselection(_reselection_block(d))
        if res:
            print(f"  {str(ds)[:25]:<26}")
            _print_reselected_models(res)
            continue
        # ---- fallback: pre-honest whole-data cliff score (single best rep) ----
        hy = (d.get("hybrid") or {}).get("unweighted", {})
        hcliff = hy.get("rmse_cliff", hy.get("score_cliff"))
        rep_cliffs = []
        for rep, rv in (d.get("per_rep") or {}).items():
            v = rv.get("rmse_cliff", rv.get("score_cliff")) if isinstance(rv, dict) else None
            if _fin(v):
                rep_cliffs.append(v)
        lo = lower_is_better(ds, d)
        best_rep = (min if lo else max)(rep_cliffs) if rep_cliffs else None
        dl = improvement(best_rep, hcliff, ds, d) if (_fin(best_rep) and _fin(hcliff)) else None
        v = "--" if not _fin(dl) else ("WIN" if dl > TIE else ("LOSS" if dl < -TIE else "tie"))
        print(f"  {str(ds)[:25]:<26}{f7(best_rep,10)}{f7(hcliff,10)}{sgn(dl,9)}  {v} (legacy)")

def read_s05():
    hdr("s05", "kNN prediction — hybrid-space kNN vs best single-rep kNN (core claim)")
    files = ds_files("s05_knn_prediction")
    if not files:
        print("  (no s05 files)"); return
    print("  Uses fold_reselection_result.knn.delta_mean_of_folds: for each held-out")
    print("  fold both arms are re-picked on the OTHER folds, so neither the hybrid")
    print("  cell nor the baseline representation is chosen on the fold it scores.")
    print("  summary.fixed_ref is NOT used: its baseline is the best representation")
    print("  at its best k on these same folds, picked after the hybrid was pinned.")
    print("  posthoc_max deliberately NOT used (multiple-comparison inflated).\n")
    print(f"  {'dataset':<26}{'best rep kNN':>13}{'hyb kNN':>9}{'Δ(signed)':>11}  win")
    print("  " + "-" * 70)
    rows = []
    for ds, path in files:
        d = load(path)
        if "_error" in d:
            continue
        hr = (d.get("fold_reselection_result") or {}).get("knn") or {}
        delta = hr.get("delta_mean_of_folds")
        if not _fin(delta):
            continue
        rows.append((ds, hr.get("best_single_reselected"),
                     hr.get("hybrid_reselected"), delta))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, br, hk, dl in rr:
            win = "yes" if _fin(dl) and dl > TIE else ("no" if _fin(dl) and dl < -TIE else "~")
            print(f"  {str(ds)[:25]:<26}{f7(br,13)}{f7(hk,9)}{sgn(dl,11)}  {win}")
    w, t, l = wtl([r[3] for r in rows])
    print("  " + "-" * 70)
    print(f"  hybrid kNN beats best single-rep kNN: +{w}  ~{t}  -{l}  ({len(rows)} datasets)")

def read_s06():
    hdr("s06", "applicability domain — does distance-to-train track error better for "
        "the hybrid?")
    files = ds_files("s06_applicability_domain")
    if not files:
        print("  (no s06 files)"); return
    print("  Spearman(distance, error) at k=5, neutral-model (RF-on-concat errors, the")
    print("  un-tautological view). Higher = distance better flags unreliable preds.\n")
    print(f"  {'dataset':<26}{'concat':>9}{'hybrid':>9}{'Δ':>9}")
    print("  " + "-" * 56)
    for ds, path in files:
        d = load(path)
        if "error" in d and "distance_error_correlation" not in d:
            print(f"  {str(ds)[:25]:<26}  (stub: {d.get('error')})"); continue
        dec = d.get("distance_error_correlation", {})
        k5 = dec.get("k5", {})
        neutral = k5.get("neutral_model", {}) if isinstance(k5, dict) else {}
        c = neutral.get("concat", k5.get("concat"))
        h = neutral.get("hybrid", k5.get("hybrid"))
        delta = (h - c) if (_fin(h) and _fin(c)) else None
        print(f"  {str(ds)[:25]:<26}{f7(c,9)}{f7(h,9)}{sgn(delta,9)}")

def read_s08():
    hdr("s08", "optimal dimension — the budget/dimension sweet spot")
    files = ds_files("s08_optimal_dimension")
    if not files:
        print("  (no s08 files)"); return
    print("  Best budget per objective; optima_differ = the SAC-optimal and RF-optimal")
    print("  budgets disagree (only trustworthy with >=4 distinct dims). sacΔ = held-out")
    print("  honest SAC of the budget-swept hybrid vs best single rep (+ = hybrid better,")
    print("  SAC pseudo-model; from the redesigned CV block).\n")
    print(f"  {'dataset':<26}{'best_rf_bud':>12}{'best_sac_bud':>13}{'#dims':>7}"
          f"{'sacΔ(honest)':>13}  optima_differ")
    print("  " + "-" * 90)
    for ds, path in files:
        d = load(path)
        opt = d.get("optimal", {})
        if "_error" in d or not opt:
            continue
        pareto = d.get("pareto", {}).get("tradeoff", {})
        differ = pareto.get("optima_differ")
        if d.get("insufficient_budgets"):
            differ = f"{differ} (untrustworthy)"
        # Held-out honest SAC: prefer the stored sac_reselected, else re-run the block.
        sac = d.get("sac_reselected") if isinstance(d.get("sac_reselected"), dict) else None
        if not sac or "delta" not in sac:
            sac = _reselected_single(_run_reselection(_reselection_block(d)), model="sac")
        sdelta = sac.get("delta") if isinstance(sac, dict) else None
        stag = _opt(sac) if isinstance(sac, dict) else ""
        print(f"  {str(ds)[:25]:<26}{iw(opt.get('best_rf_budget'),12)}"
              f"{iw(opt.get('best_sac_budget'),13)}{iw(d.get('n_distinct_dims'),7)}"
              f"{sgn(sdelta,13)}  {differ}{stag}")

def read_s09():
    hdr("s09", "cross-dataset summary — per-dataset deltas (combined file; NO mean "
        "headline)")
    path = combined_file("s09_cross_dataset_summary")
    if not path:
        print("  (no s09 file)"); return
    d = load(path)
    per = d.get("per_dataset", {})
    if "_error" in d or not per:
        print("  (empty)"); return
    print("  Per-dataset values pulled from the roll-up (we IGNORE its means as a")
    print("  headline). sac_delta / knn_delta / ad_corr_delta: + = hybrid better.\n")
    print(f"  {'dataset':<26}{'sac_delta':>10}{'knn_delta':>10}{'ad_delta':>10}"
          f"{'cliff_red%':>11}")
    print("  " + "-" * 70)
    for ds in sorted(per):
        r = per[ds] if isinstance(per[ds], dict) else {}
        print(f"  {str(ds)[:25]:<26}{sgn(r.get('sac_delta'),10)}{sgn(r.get('knn_delta'),10)}"
              f"{sgn(r.get('ad_corr_delta'),10)}{sgn(r.get('cliff_rmse_reduction_pct'),11,1)}")

def read_s10():
    hdr("s10", "importance downstream — does importance weighting help the downstream "
        "hybrid?")
    files = ds_files("s10_importance_downstream")
    if not files:
        print("  (no s10 files)"); return
    print("  Prefers the pooled held-out fold_reselection block: per DISTANCE-pool model")
    print("  (winner re-picked per fold), held-out hybrid vs best single rep — no single-")
    print("  best-rep / single-best-model collapse. Older files fall back to the stored")
    print("  kNN / GP / SVR / qRASAR weighting deltas.\n")
    print(f"  {'dataset / model':<24}{'single':>8}{'hybrid':>8}{'Δ':>9}  verdict")
    print("  " + "-" * 62)
    for ds, path in files:
        d = load(path)
        res_h = _run_reselection(_reselection_block(d))
        if res_h:
            print(f"  {str(ds)[:23]:<24}")
            _print_reselected_models(res_h)
            continue
        # ---- fallback: pre-honest weighted-vs-unweighted deltas per arm ----
        res = d.get("results", d)
        if not res.get("hybrid_evaluated", True) and "weighted_knn" not in res:
            print(f"  {str(ds)[:23]:<24}  (hybrid not evaluated)"); continue
        def hdelta(block, key="delta"):
            b = res.get(block, {})
            if isinstance(b, dict) and not b.get("skipped"):
                hy = b.get("hybrid", {})
                if isinstance(hy, dict):
                    if key in hy:
                        return hy.get(key)
                    # knn: nested by k
                    for kk in ("k10", "k5", "k20"):
                        if isinstance(hy.get(kk), dict) and "delta" in hy[kk]:
                            return hy[kk]["delta"]
            return None
        print(f"  {str(ds)[:23]:<24}  (legacy weighting Δ) "
              f"knn={sgn(hdelta('weighted_knn'),0)} gp={sgn(hdelta('weighted_gp'),0)} "
              f"svr={sgn(hdelta('weighted_svr'),0)} qrasar={sgn(hdelta('qrasar','delta_w'),0)}")

def read_s11():
    hdr("s11", "bayesian optimization — (known skipped on ~all datasets)")
    files = ds_files("s11_bayesian_optimization")
    if not files:
        print("  (no s11 files)"); return
    print(f"  {'dataset':<26}state")
    print("  " + "-" * 60)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:25]:<26}skipped: {d.get('reason')}")
        elif "error" in d and "results" not in d:
            print(f"  {str(ds)[:25]:<26}error: {d.get('error')}")
        elif d.get("results"):
            conds = list((d["results"].get("conditions") or {}).keys())
            print(f"  {str(ds)[:25]:<26}RAN: conditions={conds}")
        else:
            print(f"  {str(ds)[:25]:<26}(unrecognized)")

def read_s12():
    hdr("s12", "cross-domain cliffs — hybrid vs best rep on activity cliffs")
    files = ds_files("s12_cross_domain_cliffs")
    if not files:
        print("  (no s12 files)"); return
    print("  Per model (no single-best-model collapse). Prefers the held-out honest")
    print("  cliff re-score (winner re-picked per fold, same tax both arms); + = hybrid")
    print("  better on cliffs. Older files: per-model RAW cliff delta (dir-corrected).\n")
    print(f"  {'dataset / model':<26}{'single':>10}{'hybrid':>10}{'Δ':>9}  verdict")
    print("  " + "-" * 74)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {str(ds)[:25]:<26}{('skip: '+str(d.get('reason')))[:19]:<20}"); continue
        # Prefer the stored per-model honest result; else re-run the block.
        res = d.get("hybrid_cliff_delta_reselected")
        if not (isinstance(res, dict) and res and all(isinstance(v, dict) for v in res.values())):
            res = _run_reselection(_reselection_block(d))
        if res:
            print(f"  {str(ds)[:25]:<26}")
            _print_reselected_models(res)
            continue
        # ---- fallback: pre-honest raw per-model cliff delta (report ALL) ----
        hcd = d.get("hybrid_cliff_delta", {})
        lo = lower_is_better(ds, d)
        if not isinstance(hcd, dict) or not hcd:
            print(f"  {str(ds)[:25]:<26}  (no cliff deltas)"); continue
        print(f"  {str(ds)[:25]:<26}")
        for m in sorted(hcd):
            v = hcd[m]
            if not _fin(v):
                continue
            dl = (-v if lo else v)   # raw = hybrid_cliff - best_rep_cliff; flip so + = hybrid
            verdict = "WIN" if dl > TIE else ("LOSS" if dl < -TIE else "tie")
            print(f"    {str(m)[:9]:<10}{'--':>10}{'--':>10}{sgn(dl,9)}  {verdict} (raw)")

def read_s13():
    hdr("s13", "clustering / retrieval — does the hybrid space cluster & retrieve better?")
    files = ds_files("s13_clustering_retrieval")
    if not files:
        print("  (no s13 files)"); return
    print("  Held-out per-fold reselection of both axes (winner re-picked per fold, same tax")
    print("  both arms): clustering coherence (kmeans, higher better), retrieval Δy (knn,")
    print("  lower better). Both Δ direction-corrected so + = hybrid better. Older files")
    print("  fall back to the stored win booleans.\n")
    print(f"  {'dataset':<26}{'clustΔ':>10}{'retrΔ':>10}{'clust win':>11}{'retr win':>10}")
    print("  " + "-" * 70)
    cw = rw = tot = 0
    for ds, path in files:
        d = load(path)
        rc = _reselected_single(_run_reselection(_reselection_block(d, "fold_reselection_clustering")))
        rr = _reselected_single(_run_reselection(_reselection_block(d, "fold_reselection_retrieval")))
        if rc or rr:
            tot += 1
            cd = rc.get("delta") if isinstance(rc, dict) else None
            rd = rr.get("delta") if isinstance(rr, dict) else None
            cwin = "yes" if _fin(cd) and cd > TIE else ("no" if _fin(cd) and cd < -TIE else "~")
            rwin = "yes" if _fin(rd) and rd > TIE else ("no" if _fin(rd) and rd < -TIE else "~")
            cw += 1 if (_fin(cd) and cd > TIE) else 0
            rw += 1 if (_fin(rd) and rd > TIE) else 0
            opt = _opt(rc) or _opt(rr)
            print(f"  {str(ds)[:25]:<26}{sgn(cd,10)}{sgn(rd,10)}{cwin:>11}{rwin:>10}{opt}")
            continue
        # ---- fallback: pre-honest stored win booleans ----
        s = d.get("summary")
        if not isinstance(s, dict):
            print(f"  {str(ds)[:25]:<26}  (no hybrid/summary)"); continue
        tot += 1
        cw += 1 if s.get("hybrid_clustering_wins") else 0
        rw += 1 if s.get("hybrid_retrieval_wins") else 0
        print(f"  {str(ds)[:25]:<26}{'--':>10}{'--':>10}"
              f"{str(s.get('hybrid_clustering_wins')):>11}{str(s.get('hybrid_retrieval_wins')):>10}"
              f"  (legacy)")
    print("  " + "-" * 70)
    print(f"  hybrid wins clustering: {cw}/{tot}   retrieval: {rw}/{tot}")


# ============================================================ NOISE (thin)
def read_n01():
    hdr("n01", "activity-cliff framework — hybrid cliff/non-cliff gap vs best rep gap")
    files = ds_files("n01_activity_cliff_framework")
    if not files:
        print("  (no n01 files)"); return
    print("  gap = cliff vs non-cliff error, sign-normalized (SMALLER = better). Compares")
    print("  hybrid_gap vs the smallest single-rep gap, per model.\n")
    print(f"  {'dataset':<24}{'model':<9}{'hyb gap':>9}{'best rep gap':>14}  hyb smaller?")
    print("  " + "-" * 72)
    for ds, path in files:
        d = load(path)
        strat = (d.get("results", {}) or {}).get("stratified", {})
        if isinstance(strat, dict) and strat.get("error"):
            print(f"  {str(ds)[:23]:<24}  (stub: {strat.get('error')})"); continue
        bym = (strat or {}).get("stratified_by_model", {})
        for model, mv in bym.items():
            comp = mv.get("_comparison", {}) if isinstance(mv, dict) else {}
            hg = comp.get("hybrid_gap")
            top3 = comp.get("top3_smallest_gap") or []
            best_rep_gap = top3[0][1] if top3 and len(top3[0]) > 1 else None
            sm = "yes" if (_fin(hg) and _fin(best_rep_gap) and hg < best_rep_gap) else (
                "" if not _fin(hg) else "no")
            print(f"  {str(ds)[:23]:<24}{str(model)[:8]:<9}{f7(hg,9)}{f7(best_rep_gap,14)}  {sm}")

def read_n02():
    hdr("n02", "sample weighting — does reliability weighting help base reps or the "
        "hybrid more?")
    files = ds_files("n02_sample_weighting")
    if not files:
        print("  (no n02 files)"); return
    print("  hybrid delta = weighted − unweighted, dir-corrected here (+ = weighting "
          "helps the hybrid).\n")
    print(f"  {'dataset':<24}{'model':<9}{'hybrid Δ(dir)':>14}  helps?")
    print("  " + "-" * 58)
    for ds, path in files:
        d = load(path)
        res = d.get("results", {})
        if "error" in d and not res:
            print(f"  {str(ds)[:23]:<24}  (stub: {d.get('error')})"); continue
        hy = res.get("hybrid", {})
        lo = lower_is_better(ds, d)
        for model, mv in hy.items():
            raw = mv.get("delta") if isinstance(mv, dict) else None
            dl = (-raw if lo else raw) if _fin(raw) else None
            tag = "yes" if _fin(dl) and dl > TIE else ("no" if _fin(dl) and dl < -TIE else "~")
            print(f"  {str(ds)[:23]:<24}{str(model)[:8]:<9}{sgn(dl,14)}  {tag}")

def read_n06():
    hdr("n06", "BindingDB ground truth — do noise scores agree with real replicate "
        "disagreement?")
    files = ds_files("n06_bindingdb_ground_truth")
    if not files:
        print("  (no n06 files)"); return
    print("  Spearman(noise score, replicate disagreement); HIGHER = better. DESCRIPTIVE:")
    print("  we report the full estimator×space grid per feature space (median & range")
    print("  across estimators), NOT the single best (estimator,space) cell — that")
    print("  post-hoc argmax is multiple-comparison inflated.\n")
    print(f"  {'dataset':<22}{'space':<14}{'#est':>5}{'med corr':>10}{'corr range':>18}")
    print("  " + "-" * 72)
    for ds, path in files:
        d = load(path)
        # n06 was rebuilt on 2026-09-11 around ground-truth PROVIDERS: the
        # grid moved from results['estimators'][space][est]['ground_truth_
        # correlation'] to results['ground_truth'][provider]['per_space']
        # [space][est]['value'], and 'has_ground_truth' is gone — a dataset
        # where no provider found repeats now writes a skip marker instead.
        # Both shapes are read, so files written before that date still print.
        if d.get("skipped"):
            print(f"  {str(ds)[:21]:<22}  (skipped: {d.get('reason')})")
            continue
        by_space = {}
        gt_all = d.get("ground_truth")
        if isinstance(gt_all, dict) and gt_all:
            provider = (d.get("transfer_measured_against")
                        or sorted(gt_all)[0])
            block = gt_all.get(provider) or {}
            for space, space_res in (block.get("per_space") or {}).items():
                if isinstance(space_res, dict):
                    by_space[space] = [er.get("value") for er in
                                       space_res.values()
                                       if isinstance(er, dict)
                                       and _fin(er.get("value"))]
        else:
            est_grid = d.get("estimators")
            if not d.get("has_ground_truth") or not isinstance(est_grid, dict):
                print(f"  {str(ds)[:21]:<22}  (no ground truth)"); continue
            for space, space_res in est_grid.items():
                if isinstance(space_res, dict):
                    by_space[space] = [
                        er.get("ground_truth_correlation")
                        for er in space_res.values()
                        if isinstance(er, dict)
                        and _fin(er.get("ground_truth_correlation"))]
        printed = False
        for space in sorted(by_space):
            corrs = by_space[space]
            if not corrs:
                continue
            m, lo_, hi_ = med_range(corrs)
            rng = f"[{sgn(lo_,0)},{sgn(hi_,0)}]"
            print(f"  {(ds if not printed else '')[:21]:<22}{str(space)[:13]:<14}"
                  f"{len(corrs):>5}{f7(m,10)}{rng:>18}")
            printed = True
        if not printed:
            print(f"  {str(ds)[:21]:<22}  (no valid correlations)")

def read_n07():
    hdr("n07", "remediation — best strategy recovering the clean→noisy gap (thin, "
        "synthetic noise)")
    files = ds_files("n07_remediation")
    if not files:
        print("  (no n07 files)"); return
    print("  Prefers the held-out fold_reselection block: per downstream model, held-out")
    print("  hybrid vs best single rep (winner re-picked per fold, same tax both arms),")
    print("  + = hybrid better. No post-hoc max over the model×strategy grid. Older files")
    print("  fall back to per-model recovery% (median & range across strategies, no max).\n")
    print(f"  {'dataset / model':<24}{'single':>8}{'hybrid':>8}{'Δ':>9}  verdict")
    print("  " + "-" * 62)
    for ds, path in files:
        d = load(path)
        if "error" in d and "summary" not in d:
            print(f"  {str(ds)[:21]:<22}  (stub: {d.get('error')})"); continue
        res = _run_reselection(_reselection_block(d))
        if res:
            print(f"  {str(ds)[:23]:<24}")
            _print_reselected_models(res)
            continue
        # ---- fallback: per-model recovery% median/range (NO max-cell argmax) ----
        s = d.get("summary", {})
        pmr = s.get("per_model_recovery_pct", {})
        if not isinstance(pmr, dict) or not pmr:
            print(f"  {str(ds)[:23]:<24}  (no per-model recovery)"); continue
        print(f"  {str(ds)[:23]:<24}  (legacy recovery% median[min,max] across strategies)")
        for m in sorted(pmr):
            pcts = [pct for pct in (pmr[m] or {}).values() if _fin(pct)]
            med, lo_, hi_ = med_range(pcts)
            print(f"    {str(m)[:9]:<10}{f7(med,8,1)}  [{f7(lo_,0,1)},{f7(hi_,0,1)}]")

def read_n09():
    hdr("n09", "learning-with-noisy-labels survey — does adding the hybrid re-order the "
        "method ranking?")
    files = ds_files("n09_lnl_survey")
    if not files:
        print("  (no n09 files)"); return
    print("  ranking_changed = the winning LNL method changes when the hybrid is added.")
    print("  improves/harms counted at the highest sigma (direction-aware).\n")
    print(f"  {'dataset':<22}{'ranking_changed':>16}{'hyb improves':>14}{'hyb harms':>11}")
    print("  " + "-" * 66)
    for ds, path in files:
        d = load(path)
        if "error" in d and "summary" not in d:
            print(f"  {str(ds)[:21]:<22}  (stub: {d.get('error')})"); continue
        s = d.get("summary", {})
        print(f"  {str(ds)[:21]:<22}{str(s.get('ranking_changed')):>16}"
              f"{iw(s.get('n_hybrid_improves'),14)}{iw(s.get('n_hybrid_harms'),11)}")

def read_n11():
    hdr("n11", "estimation×remediation grid — best recovery config (thin, oracle-biased "
        "space compare)")
    files = ds_files("n11_estimation_remediation_grid")
    if not files:
        print("  (no n11 files)"); return
    print("  Two recovery numbers per sigma.")
    print("    search%  = the largest recovery over ~4,800 configs, scored on the")
    print("               folds that chose it. A search result, not a measured")
    print("               recovery.")
    print("    heldout% = for each fold the winner is re-picked from the other")
    print("               folds and read on this one, then averaged. 'same' says")
    print("               whether one config won on every fold.")
    print("  (space effect is oracle-biased — ignore that arm.)\n")
    print(f"  {'dataset':<20}{'sigma':<8}{'best config (space/est/method)':<34}"
          f"{'search%':>9}{'heldout%':>10}{'same':>6}")
    print("  " + "-" * 92)
    for ds, path in files:
        d = load(path)
        if "error" in d and "summary" not in d:
            print(f"  {str(ds)[:19]:<20}  (stub: {d.get('error')})"); continue
        for skey, sv in (d.get("summary", {}) or {}).items():
            if not skey.startswith("sigma_") or not isinstance(sv, dict):
                continue
            # best_config_search_max is the current name; best_config is the
            # pre-2026-09-04 one, kept so older files still read.
            bc = sv.get("best_config_search_max") or sv.get("best_config", {}) or {}
            desc = f"{bc.get('space')}/{bc.get('estimator')}/{bc.get('method')}@{bc.get('alpha')}"
            ho = sv.get("heldout_recovery") or {}
            same = ho.get("same_config_every_fold")
            print(f"  {str(ds)[:19]:<20}{skey.replace('sigma_',''):<8}{desc[:33]:<34}"
                  f"{f7(bc.get('recovery'),9,1)}{f7(ho.get('mean_recovery'),10,1)}"
                  f"{('' if same is None else str(same)[:5]):>6}")

def read_n12():
    hdr("n12", "importance-weighted distance — does weighting the kNN distance help "
        "(base / hybrid)?")
    files = ds_files("n12_importance_weighted_distance")
    if not files:
        print("  (no n12 files)"); return
    print("  Boolean flags (NOT significance tests — spread often overlaps).\n")
    print(f"  {'dataset':<20}{'sigma':<8}{'helps_base':>12}{'helps_hybrid':>14}"
          f"{'already_wt':>12}")
    print("  " + "-" * 68)
    for ds, path in files:
        d = load(path)
        if "error" in d and "summary" not in d:
            print(f"  {str(ds)[:19]:<20}  (stub: {d.get('error')})"); continue
        for skey, sv in (d.get("summary", {}) or {}).items():
            if not skey.startswith("sigma_") or not isinstance(sv, dict):
                continue
            print(f"  {str(ds)[:19]:<20}{skey.replace('sigma_',''):<8}"
                  f"{str(sv.get('weighting_helps_base')):>12}"
                  f"{str(sv.get('weighting_helps_hybrid')):>14}"
                  f"{str(sv.get('hybrid_already_weighted')):>12}")

def read_n13():
    hdr("n13", "noise distributions — does the hybrid advantage persist across noise "
        "shapes?")
    files = ds_files("n13_noise_distributions")
    if not files:
        print("  (no n13 files)"); return
    print("  hybrid_advantage_persists + win rate per noise type (sensitivity study, "
          "NOT absolute robustness).\n")
    print(f"  {'dataset':<20}{'persists':>10}{'gauss_transfers':>16}  win rate by noise type")
    print("  " + "-" * 78)
    for ds, path in files:
        d = load(path)
        if "error" in d and "summary" not in d:
            print(f"  {str(ds)[:19]:<20}  (stub: {d.get('error')})"); continue
        s = d.get("summary", {})
        wr = s.get("hybrid_win_rate_by_noise_type", {})
        wrs = " ".join(f"{k}={f7(v,0,2)}" for k, v in wr.items()) if isinstance(wr, dict) else ""
        print(f"  {str(ds)[:19]:<20}{str(s.get('hybrid_advantage_persists')):>10}"
              f"{str(s.get('gaussian_rank_transfers')):>16}  {wrs}")


# ============================================================ registry / main
COMPLEXITY = [
    ("c02", read_c02), ("c05", read_c05), ("c11", read_c11), ("c13", read_c13),
    ("c16", read_c16), ("c18", read_c18), ("c10", read_c10), ("c12", read_c12),
    ("c01", read_c01), ("c03", read_c03), ("c04", read_c04), ("c06", read_c06),
    ("c07", read_c07), ("c08", read_c08), ("c09", read_c09), ("c15", read_c15),
    ("c17", read_c17), ("c19", read_c19), ("c21", read_c21), ("c22", read_c22),
    ("c23", read_c23), ("c24", read_c24),
]
SIMILARITY = [
    ("s02", read_s02), ("s05", read_s05), ("s13", read_s13), ("s01", read_s01),
    ("s03", read_s03), ("s04", read_s04), ("s06", read_s06),
    ("s08", read_s08), ("s09", read_s09), ("s10", read_s10), ("s11", read_s11),
    ("s12", read_s12),
]
NOISE = [
    ("n01", read_n01), ("n02", read_n02), ("n06", read_n06), ("n07", read_n07),
    ("n09", read_n09), ("n11", read_n11), ("n12", read_n12), ("n13", read_n13),
]

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

def _run():
    print(f"ROOT = {os.path.abspath(ROOT)}")
    print(f"direction registry: {len(_DIR)} datasets"
          f"{' (EMPTY — using fallback set)' if not _DIR else ''}")
    if ONLY:
        print(f"filter: {sorted(ONLY)}")
    print("\nRULES: no metric averages as a headline; win/tie/loss COUNTS + median/range")
    print("for spreads; core-claim readers report PER MODEL (no read-time argmax over")
    print("models/reps/grid cells), hybrid vs best single rep on the SAME model; when the")
    print("experiment persists an fold_reselection block the winner is re-picked per held-")
    print("out fold and the SAME tax hits both arms (OPTIMISTIC(1fold) = only one fold, not")
    print("honest); older files fall back to the stored same-fold view. Δ direction-")
    print("corrected (+ = hybrid better); tie band |Δ|<=%.2f." % TIE)
    print("Core-claim experiments first in each block; diagnostics after.")

    for block, name in ((COMPLEXITY, "COMPLEXITY"), (SIMILARITY, "SIMILARITY"),
                        (NOISE, "NOISE")):
        if ONLY and not any(exp in ONLY for exp, _ in block):
            continue
        print("\n\n" + "#" * 92 + f"\n#  {name}\n" + "#" * 92)
        for exp, fn in block:
            if want(exp):
                safe(fn)

def main():
    fh = open(OUT, "w") if OUT else None
    old = sys.stdout
    if fh:
        sys.stdout = fh if QUIET else _Tee(old, fh)
    try:
        _run()
    finally:
        if fh:
            sys.stdout = old
            fh.close()
            print(f"\n(full report written to {OUT} — commit+push it, or paste it here)")

if __name__ == "__main__":
    main()
