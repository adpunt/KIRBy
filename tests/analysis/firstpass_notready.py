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
        return {k: bool(getattr(v, "lower_is_better", False)) for k, v in reg.items()}
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


# ============================================================ C02 — core claim
# The validated first-pass view: best existing (model, rep) vs the hybrid on the
# SAME model, per dataset, with a 5-fold CI on the hybrid and a beat-the-top line.
TOP_N = 3

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

def _rep_dim(d, rep):
    rd = d.get("_rep_dims", {})
    per = rd.get("per_rep", {}) if isinstance(rd, dict) else {}
    v = per.get(rep) if isinstance(per, dict) else None
    if isinstance(v, dict):
        v = v.get("dim") or v.get("total_dim") or v.get("width")
    return v if _fin(v) else None

def _c02_top_combos(d, ds, n=TOP_N):
    base = d.get("baselines", {})
    lo = lower_is_better(ds, d)
    scored = [(m, b.get("best_single_rep"), b.get("best_single_score"))
              for m, b in base.items()
              if _fin(b.get("best_single_score")) and b.get("best_single_rep")]
    scored.sort(key=lambda t: t[2], reverse=not lo)
    return scored[:n]

def _c02_best_hybrid_any(d, ds):
    lo = lower_is_better(ds, d)
    best = None
    for m, h in d.get("best_by_model", {}).items():
        s = h.get("score_mean")
        if not _fin(s):
            continue
        if best is None or (s < best[0] if lo else s > best[0]):
            best = (s, m, h.get("score_std"))
    return best

def read_c02():
    hdr("c02", "complexity grid — best existing (model,rep) vs hybrid, SAME model "
        "(core claim)")
    # discover every dataset that has a c02 file, wherever it landed
    seen = {}
    for dp, _dirs, files in os.walk(ROOT):
        if "c02" in os.path.basename(dp):
            for fn in files:
                if fn.endswith(".json"):
                    seen.setdefault(fn[:-5], True)
    if not seen:
        print("  (no c02 files)"); return
    print("  Δ direction-corrected (+ = hybrid beats that existing combo). tie |Δ|<=%.2f."
          % TIE)
    print("  WIN* = existing combo OUTSIDE the hybrid's 95%% CI (5 folds) = 'probably real'.")
    print("  BEST-ANY row = best hybrid on ANY model vs the #1 existing combo.\n")
    print(f"  {'dataset':<24}{'#':>2} {'model':<9}{'existing rep':<15}"
          f"{'base':>7}{'hybrid':>8}{'±95CI':>7}{'Δ':>8}  {'verdict':<7}"
          f"{'repdim':>7}{'hybdim':>7} smaller?")
    print("  " + "-" * 104)
    domwtl = defaultdict(list)
    for ds in sorted(seen):
        path = _c02_path(ds)
        if not path:
            continue
        d = load(path)
        if "_error" in d or not d.get("baselines") or not d.get("best_by_model"):
            print(f"  {ds:<24}  --  no baselines/best_by_model"); continue
        combos = _c02_top_combos(d, ds)
        if not combos:
            print(f"  {ds:<24}  --  no scored baselines"); continue
        for rank, (model, rep, base_score) in enumerate(combos, 1):
            h = d["best_by_model"].get(model, {})
            hyb = h.get("score_mean"); ci = _ci(h.get("score_std"))
            delta = improvement(base_score, hyb, ds, d) if _fin(hyb) else None
            if rank == 1 and delta is not None:
                domwtl[domain(ds)].append(delta)
            beats = None
            if _fin(hyb) and ci is not None:
                beats = base_score > (hyb + ci) if lower_is_better(ds, d) \
                    else base_score < (hyb - ci)
            verdict = "--"
            if delta is not None:
                verdict = ("WIN" + ("*" if beats else "")) if delta > TIE \
                    else ("LOSS" if delta < -TIE else "tie")
            rdim = _rep_dim(d, rep); hdim = h.get("n_features")
            sm = "yes" if (_fin(hdim) and _fin(rdim) and hdim < rdim) else (
                "" if rdim is None else "no")
            tag = "*" if rank == 1 else " "
            print(f"  {ds if rank==1 else '':<24}{rank:>2}{tag}{model:<8}"
                  f"{str(rep)[:14]:<15}{f7(base_score)}{f7(hyb,8)}{f7(ci,7)}"
                  f"{sgn(delta)}  {verdict:<7}{iw(rdim)}{iw(hdim)}  {sm}")
        bany = _c02_best_hybrid_any(d, ds)
        if bany and combos:
            _, _, top_score = combos[0]
            hyb_any, any_model, any_std = bany
            dany = improvement(top_score, hyb_any, ds, d)
            print(f"  {'':<24}   BEST-ANY ({any_model})".ljust(50)
                  + f"{f7(top_score)}{f7(hyb_any,8)}{f7(_ci(any_std),7)}{sgn(dany)}"
                  + "   <- beats #1 existing?")
    print("  " + "-" * 104)
    for dom, ds in domwtl.items():
        w, t, l = wtl(ds)
        print(f"  [{dom}] deployed model: +{w} hybrid-better  ={t} tie  -{l} worse "
              f"({len(ds)} datasets)")


# ============================================================ C05 — core claim
def read_c05():
    hdr("c05", "learning curves — does the hybrid advantage GROW or SHRINK with n?")
    files = ds_files("c05_learning_curves")
    if not files:
        print("  (no c05 files)"); return
    print("  For the deployed model (best final single-rep), Δ = hybrid−single per n,")
    print("  direction-corrected (+ = hybrid better). Reports Δ at smallest & largest n.\n")
    print(f"  {'dataset':<24}{'model':<9}{'n_small':>8}{'Δsmall':>9}"
          f"{'n_large':>9}{'Δlarge':>9}  trend      win@large")
    print("  " + "-" * 92)
    rows = []
    for ds, path in files:
        d = load(path)
        if "_error" in d or d.get("skipped") or not d.get("models"):
            continue
        sizes = d.get("sample_sizes") or []
        lo = lower_is_better(ds, d)
        # deployed model = best single at the largest n with a finite value
        best_model, best_final = None, None
        for m, mv in d["models"].items():
            bs = mv.get("best_single") or []
            fin = [x for x in bs if _fin(x)]
            if not fin:
                continue
            cand = fin[-1]
            if best_final is None or (cand < best_final if lo else cand > best_final):
                best_final, best_model = cand, m
        if not best_model:
            continue
        mv = d["models"][best_model]
        hy, bs = mv.get("hybrid") or [], mv.get("best_single") or []
        pts = [(sizes[i] if i < len(sizes) else i,
                improvement(bs[i], hy[i], ds, d))
               for i in range(min(len(hy), len(bs)))
               if _fin(hy[i]) and _fin(bs[i])]
        if not pts:
            continue
        n_s, d_s = pts[0]; n_l, d_l = pts[-1]
        trend = "grows" if d_l - d_s > TIE else ("shrinks" if d_l - d_s < -TIE else "flat")
        rows.append((ds, best_model, n_s, d_s, n_l, d_l, trend, d_l))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, m, ns, dsm, nl, dl, tr, dlarge in rr:
            win = "yes" if dlarge > TIE else ("no" if dlarge < -TIE else "tie")
            print(f"  {ds:<24}{m:<9}{iw(ns,8)}{sgn(dsm,9)}{iw(nl,9)}{sgn(dl,9)}"
                  f"  {tr:<9}  {win}")
    grows = sum(1 for r in rows if r[6] == "grows")
    shr = sum(1 for r in rows if r[6] == "shrinks")
    winl = sum(1 for r in rows if r[7] > TIE)
    print("  " + "-" * 92)
    print(f"  advantage grows with n: {grows}   shrinks: {shr}   "
          f"hybrid wins at largest n: {winl}/{len(rows)}")


# ============================================================ C11 — core claim
def read_c11():
    hdr("c11", "ensemble_vs_hybrid — hybrid vs best single rep vs PCA-concat "
        "(same model; NOTE ensemble arm removed)")
    files = ds_files("c11_ensemble_vs_hybrid")
    if not files:
        print("  (no c11 files)"); return
    print("  Deployed model = best single-rep score. Δ = hybrid−single (dir-corrected).\n")
    print(f"  {'dataset':<24}{'model':<9}{'single':>8}{'pca_cat':>9}{'hybrid':>8}"
          f"{'Δvs_single':>11}  verdict")
    print("  " + "-" * 82)
    rows = []
    for ds, path in files:
        d = load(path)
        rs = d.get("results")
        if "_error" in d or not isinstance(rs, list):
            continue
        lo = lower_is_better(ds, d)
        # collect best per (model, method)
        best = defaultdict(dict)   # model -> method -> score
        for r in rs:
            m, meth, s = r.get("model"), r.get("method"), r.get("score_mean")
            if not _fin(s):
                continue
            cur = best[m].get(meth)
            if cur is None or (s < cur if lo else s > cur):
                best[m][meth] = s
        # deployed model = best best_single
        dep, dep_s = None, None
        for m, mv in best.items():
            s = mv.get("best_single")
            if _fin(s) and (dep_s is None or (s < dep_s if lo else s > dep_s)):
                dep_s, dep = s, m
        if not dep:
            continue
        mv = best[dep]
        single, pca, hyb = mv.get("best_single"), mv.get("pca_matched"), mv.get("hybrid")
        delta = improvement(single, hyb, ds, d) if _fin(hyb) else None
        verdict = "--" if delta is None else ("WIN" if delta > TIE else
                                              ("LOSS" if delta < -TIE else "tie"))
        rows.append((ds, dep, single, pca, hyb, delta, verdict))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, m, s, p, h, dl, v in rr:
            print(f"  {ds:<24}{m:<9}{f7(s,8)}{f7(p,9)}{f7(h,8)}{sgn(dl,11)}  {v}")
    print("  " + "-" * 82)
    w, t, l = wtl([r[5] for r in rows])
    print(f"  hybrid vs best single rep: +{w} better  ={t} tie  -{l} worse "
          f"({len(rows)} datasets)")


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
            print(f"  {ds:<22}  (stub: {d.get('error')})"); continue
        res = d.get("results", {})
        per = res.get("per_rep", {})
        if not per:
            continue
        print(f"  --- {ds} ({domain(ds)}) ---")
        for rep, rv in per.items():
            dl = rv.get("deltas", {})
            ddr = dl.get("D_minus_D_random"); dmb = dl.get("D_minus_B")
            helps = "yes" if _fin(ddr) and ddr > TIE else (
                "no" if _fin(ddr) and ddr < -TIE else "~")
            if _fin(ddr):
                tot += 1; tot_pos += 1 if ddr > TIE else 0
            print(f"  {'':<22}{str(rep)[:19]:<20}{sgn(ddr,11)}{sgn(dmb,9)}  {helps}")
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
            print(f"  {ds:<24}{f7(db,11)}{f7(u,9)}{f7(n,9)}{sgn(dl,11)}  {w}")
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
    print("  Δ = best hybrid auc_norm − best single-rep auc_norm (+ = hybrid tougher).\n")
    print(f"  {'dataset':<24}{'rep_auc':>9}{'hyb_auc':>9}{'Δ':>9}  more robust?")
    print("  " + "-" * 66)
    rows = []
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {ds:<24}  (skipped: {d.get('reason')})"); continue
        ir = d.get("individual_reps", {}); hy = d.get("hybrids", {})
        if "_error" in d or not ir or not hy:
            continue
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
            print(f"  {ds:<24}{f7(ra,9)}{f7(ha,9)}{sgn(dl,9)}  {tag}")
    w, t, l = wtl([r[3] for r in rows])
    print("  " + "-" * 66)
    print(f"  hybrid more robust: +{w}  ~{t}  -{l}  ({len(rows)} datasets)")


# ============================================================ C12 — modality
def read_c12():
    hdr("c12", "modality ablation — does combining modalities beat the best single "
        "modality?")
    files = ds_files("c12_modality_ablation")
    if not files:
        print("  (no c12 files)"); return
    print("  Scores here are always HIGHER-better (r2/auroc/f1). delta_vs_best_single")
    print("  is stored (+ = combining helps). Shows all-combined + best pairwise.\n")
    print(f"  {'dataset':<24}{'all_comb Δ':>11}{'best_pair Δ':>12}  helps?")
    print("  " + "-" * 58)
    rows = []
    for ds, path in files:
        d = load(path)
        if "_error" in d:
            continue
        ac = d.get("all_combined")
        allc = ac.get("delta_vs_best_single") if isinstance(ac, dict) else None
        bestp = None
        for _pair, pv in (d.get("pairwise") or {}).items():
            dv = pv.get("delta_vs_best_single") if isinstance(pv, dict) else None
            if _fin(dv) and (bestp is None or dv > bestp):
                bestp = dv
        rows.append((ds, allc, bestp))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, a, b in rr:
            tag = "yes" if _fin(a) and a > TIE else ("no" if _fin(a) and a < -TIE else "~")
            print(f"  {ds:<24}{sgn(a,11)}{sgn(b,12)}  {tag}")
    print("  " + "-" * 58)


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
            print(f"  {ds:<24}{len(reps):>6}  {str(best[0])[:17]:<18}"
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
        print(f"  {ds:<24}{str(r.get('best_model'))[:8]:<9}"
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
            print(f"  {ds:<24}  (clf skipped)"); continue
        methods = d.get("methods", {})
        if "_error" in d or not methods:
            continue
        bk, bs = d.get("best_knn_method"), d.get("best_sim_method")
        kd = methods.get(bk, {}).get("knn_delta_k10")
        sd = methods.get(bs, {}).get("sim_delta_r2")
        if _fin(kd):
            kpos += kd > TIE; kneg += kd < -TIE
        print(f"  {ds:<24}{str(bk)[:14]:<15}{sgn(kd)}   {str(bs)[:14]:<15}{sgn(sd)}")
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
    print(f"  {'dataset':<24}{'#fam':>5}{'#measured':>10}{'rho range':>18}")
    print("  " + "-" * 60)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {ds:<24}  (skipped: {d.get('reason')})"); continue
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
        print(f"  {ds:<24}{len(fams):>5}{len(rhos):>10}{rng:>18}")

def read_c07():
    hdr("c07", "data-complexity partitions — best hybrid on EASY vs HARD subpopulations")
    files = ds_files("c07_data_complexity_partitions")
    if not files:
        print("  (no c07 files)"); return
    print("  Raw metric scores (dir per dataset). config_differs = the tuned hybrid")
    print("  config changes between easy & hard partitions.\n")
    print(f"  {'dataset':<22}{'partition':<20}{'easy':>8}{'hard':>8}  cfg_differs")
    print("  " + "-" * 70)
    for ds, path in files:
        d = load(path)
        parts = d.get("partitions", {})
        if "_error" in d or not parts:
            continue
        for pn, pv in parts.items():
            e = (pv.get("easy") or {}).get("best") or {}
            h = (pv.get("hard") or {}).get("best") or {}
            print(f"  {ds:<22}{str(pn)[:19]:<20}{f7(e.get('score'),8)}"
                  f"{f7(h.get('score'),8)}  {pv.get('config_differs')}")

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
        print(f"  {ds:<24}{f7(opt.get('picp'),8)}{f7(opt.get('cal_error'),9)}"
              f"  {d.get('conditions_distinct')}")

def read_c09():
    hdr("c09", "formal analysis — ANOVA variance decomposition of the hybrid lift")
    files = ds_files("c09_formal_analysis")
    if not files:
        print("  (no c09 files)"); return
    print("  eta2 = % of score variance explained by model / budget / their interaction")
    print("  (needs c02 results; absent otherwise).\n")
    print(f"  {'dataset':<24}{'eta_model':>10}{'eta_budget':>11}{'eta_inter':>10}")
    print("  " + "-" * 58)
    for ds, path in files:
        d = load(path)
        an = d.get("anova")
        if not isinstance(an, dict) or "error" in an or "eta2" not in an:
            print(f"  {ds:<24}  (no anova{': '+an.get('error') if isinstance(an,dict) and an.get('error') else ''})")
            continue
        e = an["eta2"]
        print(f"  {ds:<24}{f7(e.get('model'),10,1)}{f7(e.get('budget'),11,1)}"
              f"{f7(e.get('interaction'),10,1)}")


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
            print(f"  {ds:<24}  (skipped: {d.get('reason')})"); continue
        dd = d.get("double_descent", {})
        subs = [k for k in ("distance_concentration", "kernel_analysis",
                            "double_descent", "approximation_theory") if k in d]
        print(f"  {ds:<24}{f7(dd.get('instability_ratio_hybrid'),12)}"
              f"{f7(dd.get('instability_ratio_pca'),10)}  {','.join(subs)}")

def read_c17():
    hdr("c17", "information-theoretic — does the hybrid add INFORMATION or just "
        "dimensions? (reg only)")
    files = ds_files("c17_information_theoretic_decomposition")
    if not files:
        print("  (no c17 files)"); return
    print("  best_strategy across {hybrid,pca,mrmr,rfe,random}; hybrid vs random")
    print("  mi_density (higher = denser info per dim). Averaged over budgets -> range.\n")
    print(f"  {'dataset':<24}{'best_strat':<12}{'hyb_dens':>10}{'rnd_dens':>10}")
    print("  " + "-" * 60)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {ds:<24}  (skipped: {d.get('reason')})"); continue
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
        bs = d.get("summary", {}).get("best_strategy")
        mh, _, _ = med_range(hd); mr, _, _ = med_range(rd)
        print(f"  {ds:<24}{str(bs)[:11]:<12}{f7(mh,10)}{f7(mr,10)}")

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
            print(f"  {ds:<24}  (no hybrid space)"); continue
        nweighted = sum(1 for mv in ps.values()
                        if isinstance(mv, dict) and mv.get("best_condition") not in
                        (None, "unweighted"))
        mw = d.get("summary", {}).get("most_winning_condition")
        print(f"  {ds:<24}{str(mw)[:19]:<20}{f'{nweighted}/{len(ps)}':>22}")


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
            print(f"  {ds:<24}{'ERROR':<14}  {d.get('error')}"); continue
        s = d.get("summary", {})
        if s.get("all_skipped"):
            print(f"  {ds:<24}{'all_skipped':<14}  (dead)"); continue
        oks = [r for r in d.get("rep_results", [])
               if r.get("aug_status") == "ok" and _fin(r.get("relative_improvement"))]
        best_rel = max((r["relative_improvement"] for r in oks), default=None)
        print(f"  {ds:<24}{'ok':<14}{iw(s.get('improved'),9)}{iw(s.get('hurt'),6)}"
              f"{sgn(best_rel,10)}")


# ============================================================ C22 / C23 / C24
def read_c22():
    hdr("c22", "explainability — per-rep SHAP contribution of the winning hybrid "
        "(descriptive)")
    files = ds_files("c22_explainability")
    if not files:
        print("  (no c22 files)"); return
    print("  Top rep by SHAP share of the hybrid, + search score.\n")
    print(f"  {'dataset':<24}{'top rep (SHAP)':<20}{'shap%':>8}{'search':>9}")
    print("  " + "-" * 64)
    for ds, path in files:
        d = load(path)
        if "error" in d:
            print(f"  {ds:<24}  (stub: {d.get('error')})"); continue
        shap = d.get("shap_by_rep", {})
        if shap:
            tot = sum(abs(v) for v in shap.values() if _fin(v)) or 1.0
            top = max(shap.items(), key=lambda kv: abs(kv[1]) if _fin(kv[1]) else -1)
            pct = 100 * abs(top[1]) / tot if _fin(top[1]) else None
            print(f"  {ds:<24}{str(top[0])[:19]:<20}{f7(pct,8,1)}{f7(d.get('search_score'),9)}")
        else:
            print(f"  {ds:<24}{'(no SHAP)':<20}{'--':>8}{f7(d.get('search_score'),9)}")

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
        print(f"  {ds:<24}{sgn(mb,15)}{sgn(mc,16)}  {matters}")

def read_c24():
    hdr("c24", "blueprint transfer — does a hybrid blueprint from other folds transfer?")
    files = ds_files("c24_blueprint_transfer")
    if not files:
        print("  (no c24 files)"); return
    print("  Δ (+ = OWN/native blueprint better; ~0 = transfers well). mean_abs_delta")
    print("  = typical transfer penalty magnitude.\n")
    print(f"  {'dataset':<24}{'mean_delta':>11}{'mean|delta|':>12}  note")
    print("  " + "-" * 58)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {ds:<24}  (skipped: {d.get('reason')})"); continue
        agg = d.get("aggregate", {})
        if not agg:
            continue
        note = "transfers" if _fin(agg.get("mean_abs_delta")) and \
            agg["mean_abs_delta"] < TIE else "penalty"
        print(f"  {ds:<24}{sgn(agg.get('mean_delta'),11)}"
              f"{f7(agg.get('mean_abs_delta'),12)}  {note}")


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
            print(f"  {ds:<26}{str(best[0])[:17]:<18}{f7(best[1],8)}"
                  f"{str(best[2]):>6}{f7(best[3],8)}")

def read_s02():
    hdr("s02", "hybrid similarity — hybrid distance SAC vs best single rep (core claim, "
        "similarity side)")
    files = ds_files("s02_hybrid_similarity")
    if not files:
        print("  (no s02 files)"); return
    print("  SAC higher = better. lift = hybrid best_sac − best individual rep SAC.\n")
    print(f"  {'dataset':<26}{'best indiv rep':<18}{'indiv':>8}{'hybrid':>8}{'lift':>8}  win")
    print("  " + "-" * 78)
    rows = []
    for ds, path in files:
        d = load(path)
        if "_error" in d:
            continue
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
            print(f"  {ds:<26}{str(nm)[:17]:<18}{f7(isac,8)}{f7(hs,8)}{sgn(lift)}  {win}")
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
    print("  rep. Median & range across (model,budget)×rep cells (NO mean headline).\n")
    print(f"  {'dataset':<26}{'#cells':>7}{'med ratio':>11}{'ratio range':>20}")
    print("  " + "-" * 66)
    for ds, path in files:
        d = load(path)
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
        print(f"  {ds:<26}{len(ratios):>7}{f7(m,11,2)}{rng:>20}")

def read_s04():
    hdr("s04", "activity cliffs — hybrid vs best rep on cliff compounds (molecular only)")
    files = ds_files("s04_activity_cliffs")
    if not files:
        print("  (no s04 files)"); return
    print("  Non-molecular / no-ecfp4 datasets are skipped by design.\n")
    print(f"  {'dataset':<26}{'state':<26}{'hyb cliff':>10}{'best rep cliff':>15}")
    print("  " + "-" * 80)
    for ds, path in files:
        d = load(path)
        if d.get("skipped") or "error" in d:
            print(f"  {ds:<26}{('skip: '+str(d.get('reason') or d.get('error')))[:25]:<26}")
            continue
        hy = (d.get("hybrid") or {}).get("unweighted", {})
        hcliff = hy.get("rmse_cliff", hy.get("score_cliff"))
        rep_cliffs = []
        for rep, rv in (d.get("per_rep") or {}).items():
            v = rv.get("rmse_cliff", rv.get("score_cliff")) if isinstance(rv, dict) else None
            if _fin(v):
                rep_cliffs.append(v)
        lo = lower_is_better(ds, d)
        best_rep = (min if lo else max)(rep_cliffs) if rep_cliffs else None
        print(f"  {ds:<26}{'ok':<26}{f7(hcliff,10)}{f7(best_rep,15)}")

def read_s05():
    hdr("s05", "kNN prediction — hybrid-space kNN vs best single-rep kNN (core claim)")
    files = ds_files("s05_knn_prediction")
    if not files:
        print("  (no s05 files)"); return
    print("  Uses summary.fixed_ref.delta_vs_best_rep (already sign-corrected, + = win).")
    print("  posthoc_max deliberately NOT used (multiple-comparison inflated).\n")
    print(f"  {'dataset':<26}{'best rep kNN':>13}{'hyb kNN':>9}{'Δ(signed)':>11}  win")
    print("  " + "-" * 70)
    rows = []
    for ds, path in files:
        d = load(path)
        s = d.get("summary", {})
        fr = s.get("fixed_ref", {})
        if "_error" in d or not fr:
            continue
        delta = fr.get("delta_vs_best_rep")
        rows.append((ds, s.get("best_rep_score"), fr.get("score_mean"), delta))
    for dom, rr in by_domain(rows):
        print(f"  --- {dom} ---")
        for ds, br, hk, dl in rr:
            win = "yes" if _fin(dl) and dl > TIE else ("no" if _fin(dl) and dl < -TIE else "~")
            print(f"  {ds:<26}{f7(br,13)}{f7(hk,9)}{sgn(dl,11)}  {win}")
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
            print(f"  {ds:<26}  (stub: {d.get('error')})"); continue
        dec = d.get("distance_error_correlation", {})
        k5 = dec.get("k5", {})
        neutral = k5.get("neutral_model", {}) if isinstance(k5, dict) else {}
        c = neutral.get("concat", k5.get("concat"))
        h = neutral.get("hybrid", k5.get("hybrid"))
        delta = (h - c) if (_fin(h) and _fin(c)) else None
        print(f"  {ds:<26}{f7(c,9)}{f7(h,9)}{sgn(delta,9)}")

def read_s07():
    hdr("s07", "noise bridge — do hybrid distances flag fewer false 'noise' cliffs than "
        "the best rep?")
    files = ds_files("s07_noise_bridge")
    if not files:
        print("  (no s07 files)"); return
    print("  Cliff-contamination rate: LOWER is better. delta = hybrid − best rep, so a")
    print("  NEGATIVE delta = hybrid improvement.\n")
    print(f"  {'dataset':<26}{'best rep':>9}{'hybrid':>9}{'Δ(neg=win)':>12}  win")
    print("  " + "-" * 68)
    rows = []
    for ds, path in files:
        d = load(path)
        s = d.get("summary", {})
        if "_error" in d or s.get("hybrid_failed"):
            print(f"  {ds:<26}  (hybrid failed / no cliffs)"); continue
        delta = s.get("delta_vs_best_rep")
        hy = (d.get("hybrid_unweighted") or {})
        rows.append((ds, s.get("best_rep_cliff_contamination_rate"),
                     hy.get("cliff_contamination_rate"), delta))
    for ds, br, h, dl in rows:
        win = "yes" if _fin(dl) and dl < -TIE else ("no" if _fin(dl) and dl > TIE else "~")
        print(f"  {ds:<26}{f7(br,9)}{f7(h,9)}{sgn(dl,12)}  {win}")

def read_s08():
    hdr("s08", "optimal dimension — the budget/dimension sweet spot")
    files = ds_files("s08_optimal_dimension")
    if not files:
        print("  (no s08 files)"); return
    print("  Best budget per objective; optima_differ = the SAC-optimal and RF-optimal")
    print("  budgets disagree (only trustworthy with >=4 distinct dims).\n")
    print(f"  {'dataset':<26}{'best_rf_bud':>12}{'best_sac_bud':>13}{'#dims':>7}  optima_differ")
    print("  " + "-" * 76)
    for ds, path in files:
        d = load(path)
        opt = d.get("optimal", {})
        if "_error" in d or not opt:
            continue
        pareto = d.get("pareto", {}).get("tradeoff", {})
        differ = pareto.get("optima_differ")
        if d.get("insufficient_budgets"):
            differ = f"{differ} (untrustworthy)"
        print(f"  {ds:<26}{iw(opt.get('best_rf_budget'),12)}"
              f"{iw(opt.get('best_sac_budget'),13)}{iw(d.get('n_distinct_dims'),7)}  {differ}")

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
        print(f"  {ds:<26}{sgn(r.get('sac_delta'),10)}{sgn(r.get('knn_delta'),10)}"
              f"{sgn(r.get('ad_corr_delta'),10)}{sgn(r.get('cliff_rmse_reduction_pct'),11,1)}")

def read_s10():
    hdr("s10", "importance downstream — does importance weighting help the downstream "
        "hybrid?")
    files = ds_files("s10_importance_downstream")
    if not files:
        print("  (no s10 files)"); return
    print("  Payload under `results`; deltas ORIENTED (+ = improvement). Shows the hybrid")
    print("  kNN / GP / SVR / qRASAR weighting deltas.\n")
    print(f"  {'dataset':<24}{'knnΔ':>8}{'gpΔ':>8}{'svrΔ':>8}{'qrasarΔ':>9}")
    print("  " + "-" * 60)
    for ds, path in files:
        d = load(path)
        res = d.get("results", d)
        if not res.get("hybrid_evaluated", True) and "weighted_knn" not in res:
            print(f"  {ds:<24}  (hybrid not evaluated)"); continue
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
        print(f"  {ds:<24}{sgn(hdelta('weighted_knn'),8)}{sgn(hdelta('weighted_gp'),8)}"
              f"{sgn(hdelta('weighted_svr'),8)}{sgn(hdelta('qrasar','delta_w'),9)}")

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
            print(f"  {ds:<26}skipped: {d.get('reason')}")
        elif "error" in d and "results" not in d:
            print(f"  {ds:<26}error: {d.get('error')}")
        elif d.get("results"):
            conds = list((d["results"].get("conditions") or {}).keys())
            print(f"  {ds:<26}RAN: conditions={conds}")
        else:
            print(f"  {ds:<26}(unrecognized)")

def read_s12():
    hdr("s12", "cross-domain cliffs — hybrid vs best rep on activity cliffs")
    files = ds_files("s12_cross_domain_cliffs")
    if not files:
        print("  (no s12 files)"); return
    print("  hybrid_cliff_delta is RAW (dir-corrected here). + = hybrid better on cliffs.\n")
    print(f"  {'dataset':<26}{'state':<20}{'best model Δ':>13}")
    print("  " + "-" * 62)
    for ds, path in files:
        d = load(path)
        if d.get("skipped"):
            print(f"  {ds:<26}{('skip: '+str(d.get('reason')))[:19]:<20}"); continue
        hcd = d.get("hybrid_cliff_delta", {})
        lo = lower_is_better(ds, d)
        # raw delta = hybrid_cliff - best_rep_cliff; flip so + = hybrid better
        vals = [(-v if lo else v) for v in hcd.values() if _fin(v)]
        best = (max(vals) if vals else None)
        print(f"  {ds:<26}{'ok':<20}{sgn(best,13)}")

def read_s13():
    hdr("s13", "clustering / retrieval — does the hybrid space cluster & retrieve better?")
    files = ds_files("s13_clustering_retrieval")
    if not files:
        print("  (no s13 files)"); return
    print("  clustering coherence: higher better; retrieval Δy: lower better. Win flags")
    print("  are direction-correct booleans from the file.\n")
    print(f"  {'dataset':<26}{'clust win':>10}{'retr win':>10}  agree?")
    print("  " + "-" * 58)
    cw = rw = tot = 0
    for ds, path in files:
        d = load(path)
        s = d.get("summary")
        if not isinstance(s, dict):
            print(f"  {ds:<26}  (no hybrid/summary)"); continue
        tot += 1
        cw += 1 if s.get("hybrid_clustering_wins") else 0
        rw += 1 if s.get("hybrid_retrieval_wins") else 0
        print(f"  {ds:<26}{str(s.get('hybrid_clustering_wins')):>10}"
              f"{str(s.get('hybrid_retrieval_wins')):>10}  {s.get('metrics_agree')}")
    print("  " + "-" * 58)
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
            print(f"  {ds:<24}  (stub: {strat.get('error')})"); continue
        bym = (strat or {}).get("stratified_by_model", {})
        for model, mv in bym.items():
            comp = mv.get("_comparison", {}) if isinstance(mv, dict) else {}
            hg = comp.get("hybrid_gap")
            top3 = comp.get("top3_smallest_gap") or []
            best_rep_gap = top3[0][1] if top3 and len(top3[0]) > 1 else None
            sm = "yes" if (_fin(hg) and _fin(best_rep_gap) and hg < best_rep_gap) else (
                "" if not _fin(hg) else "no")
            print(f"  {ds:<24}{str(model)[:8]:<9}{f7(hg,9)}{f7(best_rep_gap,14)}  {sm}")

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
            print(f"  {ds:<24}  (stub: {d.get('error')})"); continue
        hy = res.get("hybrid", {})
        lo = lower_is_better(ds, d)
        for model, mv in hy.items():
            raw = mv.get("delta") if isinstance(mv, dict) else None
            dl = (-raw if lo else raw) if _fin(raw) else None
            tag = "yes" if _fin(dl) and dl > TIE else ("no" if _fin(dl) and dl < -TIE else "~")
            print(f"  {ds:<24}{str(model)[:8]:<9}{sgn(dl,14)}  {tag}")

def read_n06():
    hdr("n06", "BindingDB ground truth — do noise scores agree with real replicate "
        "disagreement?")
    files = ds_files("n06_bindingdb_ground_truth")
    if not files:
        print("  (no n06 files)"); return
    print("  Spearman(noise score, replicate disagreement); HIGHER = better. concat vs "
          "hybrid space.\n")
    print(f"  {'dataset':<22}{'best est':<20}{'best space':<12}{'best corr':>10}")
    print("  " + "-" * 68)
    for ds, path in files:
        d = load(path)
        s = d.get("summary")
        if not d.get("has_ground_truth") or not isinstance(s, dict):
            print(f"  {ds:<22}  (no ground truth)"); continue
        print(f"  {ds:<22}{str(s.get('best_estimator'))[:19]:<20}"
              f"{str(s.get('best_space'))[:11]:<12}{f7(s.get('best_correlation'),10)}")

def read_n07():
    hdr("n07", "remediation — best strategy recovering the clean→noisy gap (thin, "
        "synthetic noise)")
    files = ds_files("n07_remediation")
    if not files:
        print("  (no n07 files)"); return
    print("  recovery% = fraction of the clean→noisy gap recovered (higher better).\n")
    print(f"  {'dataset':<22}{'best hybrid strat':<20}{'best recovery%':>15}")
    print("  " + "-" * 60)
    for ds, path in files:
        d = load(path)
        if "error" in d and "summary" not in d:
            print(f"  {ds:<22}  (stub: {d.get('error')})"); continue
        s = d.get("summary", {})
        pmr = s.get("per_model_recovery_pct", {})
        best = None
        for _m, sv in pmr.items():
            for _st, pct in (sv or {}).items():
                if _fin(pct) and (best is None or pct > best):
                    best = pct
        print(f"  {ds:<22}{str(s.get('best_hybrid_strategy'))[:19]:<20}{f7(best,15,1)}")

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
            print(f"  {ds:<22}  (stub: {d.get('error')})"); continue
        s = d.get("summary", {})
        print(f"  {ds:<22}{str(s.get('ranking_changed')):>16}"
              f"{iw(s.get('n_hybrid_improves'),14)}{iw(s.get('n_hybrid_harms'),11)}")

def read_n11():
    hdr("n11", "estimation×remediation grid — best recovery config (thin, oracle-biased "
        "space compare)")
    files = ds_files("n11_estimation_remediation_grid")
    if not files:
        print("  (no n11 files)"); return
    print("  Best recovery% config per sigma. (space effect is oracle-biased — ignore "
          "that arm.)\n")
    print(f"  {'dataset':<20}{'sigma':<8}{'best config (space/est/method)':<38}{'recov%':>8}")
    print("  " + "-" * 78)
    for ds, path in files:
        d = load(path)
        if "error" in d and "summary" not in d:
            print(f"  {ds:<20}  (stub: {d.get('error')})"); continue
        for skey, sv in (d.get("summary", {}) or {}).items():
            if not skey.startswith("sigma_") or not isinstance(sv, dict):
                continue
            bc = sv.get("best_config", {})
            desc = f"{bc.get('space')}/{bc.get('estimator')}/{bc.get('method')}@{bc.get('alpha')}"
            print(f"  {ds:<20}{skey.replace('sigma_',''):<8}{desc[:37]:<38}"
                  f"{f7(bc.get('recovery'),8,1)}")

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
            print(f"  {ds:<20}  (stub: {d.get('error')})"); continue
        for skey, sv in (d.get("summary", {}) or {}).items():
            if not skey.startswith("sigma_") or not isinstance(sv, dict):
                continue
            print(f"  {ds:<20}{skey.replace('sigma_',''):<8}"
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
            print(f"  {ds:<20}  (stub: {d.get('error')})"); continue
        s = d.get("summary", {})
        wr = s.get("hybrid_win_rate_by_noise_type", {})
        wrs = " ".join(f"{k}={f7(v,0,2)}" for k, v in wr.items()) if isinstance(wr, dict) else ""
        print(f"  {ds:<20}{str(s.get('hybrid_advantage_persists')):>10}"
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
    ("s03", read_s03), ("s04", read_s04), ("s06", read_s06), ("s07", read_s07),
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
    print("for spreads; every core-claim reader compares against the single BEST existing")
    print("(model,rep), same model both sides; Δ direction-corrected (+ = hybrid better);")
    print("tie band |Δ|<=%.2f; WIN* = existing combo outside the hybrid's 5-fold 95%% CI." % TIE)
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
