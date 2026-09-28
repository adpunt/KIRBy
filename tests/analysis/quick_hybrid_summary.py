#!/usr/bin/env python3
"""Read-only direction-correct summary of c02 / similarity results.

Usage:  python tests/analysis/quick_hybrid_summary.py [RESULTS_ROOT]
RESULTS_ROOT defaults to tests/results (must contain complexity/ similarity/).

Reports:
  * C02 TOP-2 MODELS / DATASET: for the 2 models with the best single-rep score
    (the ones you'd deploy), compare that model's best single rep against that
    model's best hybrid — SAME model on both sides. Δ is direction-corrected
    (positive = hybrid better for both higher- and lower-better metrics).
  * S01/S02 similarity baselines.
Prints to stdout only. Touches nothing. Pure stdlib.
"""
import json, glob, os, sys, math
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "tests/results"
MODELS = ["rf", "xgboost", "ridge", "svm", "mlp", "lightgbm"]

# ---- metric DIRECTION (the recurring sign bug) -------------------------------
# c02/etc JSON does NOT persist lower_is_better, so we must re-derive it.
# Try the authoritative registry first (exact on the server); fall back to a
# hardcoded set of molecular/reaction REGRESSION datasets (metric=rmse/mae).
_LOWER_BETTER_FALLBACK = {
    "caco2", "herg_ki", "herg", "logd", "qm9", "drd2", "esol",
    "adme_hlm", "adme_rlm", "adme_mdr1", "adme_solubility",
    "buchwald_hartwig", "doyle_hte_full_02", "heck_mizoroki_doyle",
}
_REGISTRY = None
def _load_registry():
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    _REGISTRY = {}
    for p in (os.path.join(os.getcwd(), "tests"), os.getcwd()):
        if p not in sys.path:
            sys.path.insert(0, p)
    for modpath in ("experiments.dataset_policy", "tests.experiments.dataset_policy",
                    "kirby.registry"):
        try:
            import importlib
            mod = importlib.import_module(modpath)
            reg = getattr(mod, "DATASET_REGISTRY")
            # Every DATASET_REGISTRY entry declares lower_is_better (it is a
            # dataclass field), so read it. A defaulted False here would turn a
            # malformed entry into "higher is better" without a word.
            # Fixed 2026-09-10 (metric-direction audit).
            _REGISTRY = {k: bool(v.lower_is_better) for k, v in reg.items()}
            if _REGISTRY:
                break
            # An importable-but-empty registry is a failed import, not an empty
            # world. Keep trying the other module paths instead of locking in
            # {}, which would send every dataset to the hardcoded fallback.
            # Fixed 2026-09-10 (metric-direction audit).
        except Exception:
            continue
    if not _REGISTRY:
        import sys as _sys
        print("  !! quick_hybrid_summary: DATASET_REGISTRY could not be loaded; "
              "falling back to the hardcoded name list, which covers only the "
              "rmse molecular sets.", file=_sys.stderr)
    return _REGISTRY

def lower_is_better(ds):
    reg = _load_registry()
    if ds in reg:
        return reg[ds]
    if _LOWER_BETTER_FALLBACK and ds not in _LOWER_BETTER_FALLBACK and reg:
        # The registry loaded and does not know this dataset. Guessing
        # higher-is-better from a name list is how an rmse set reads backwards.
        raise ValueError(
            f"{ds!r} is in neither DATASET_REGISTRY nor the fallback list, so "
            f"the direction of its scores is unknown. Register it.")
    return ds in _LOWER_BETTER_FALLBACK

def improvement(base, hyb, ds):
    """Signed so POSITIVE always means hybrid is better, regardless of metric."""
    return (base - hyb) if lower_is_better(ds) else (hyb - base)

def load(f):
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception as e:
        return {"_error": str(e)}

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def dsname(f):
    return os.path.basename(f)[:-5]

def expdirs(cat):
    p = os.path.join(ROOT, cat)
    return sorted(glob.glob(os.path.join(p, "*"))) if os.path.isdir(p) else []

def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)

# ---------------------------------------------------------------- C02 top-2 models
_CONDLBL = {
    'no_aug': 'general', 'uncapped': 'gen+aug', 'capped': 'top+aug',
    'top_tier_only': 'top-only', 'frozen_plus_ft': 'fine-tuned',
    'per_model_top_rep_dim': 'dim-match',
}
def analyze_c02_top2():
    """For the 2 models you'd actually pick (best single-rep score), compare the
    best single rep against the best hybrid -- SAME model on both sides."""
    banner("C02  TOP-2 MODELS / DATASET: best single rep vs best hybrid (same model)")
    d = os.path.join(ROOT, "complexity", "c02_complexity_matching_grid")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no c02 files)"); return
    print("  model #1/#2 = best/2nd-best single-rep score (what you'd deploy).")
    print("  best_rep/rep_score = best single rep for THAT model; hybrid = best hybrid")
    print("  for THAT model (hybrid_used = which); Δ = hybrid - rep (same model).\n")
    print(f"  {'dataset':<27}{'dir':<6}{'#':<2}{'model':<9}{'best_rep':<15}"
          f"{'rep':>8}{'hybrid':>8}{'Δ':>8}  hybrid_used")
    print("  " + "-" * 96)
    pos = neg = tie = 0          # over the #1 (deployed) model only
    pos2 = neg2 = 0             # over both top-2 cells
    for f in files:
        ds = dsname(f); j = load(f)
        if "_error" in j: continue
        lib = lower_is_better(ds); dirlbl = "lower" if lib else "high"
        base = j.get("baselines", {})
        across = j.get("best_per_model_across_conditions", {})
        ranked = [(m, base[m].get("best_single_score"), base[m].get("best_single_rep"))
                  for m in MODELS
                  if m in base and finite(base[m].get("best_single_score"))]
        ranked.sort(key=lambda t: t[1], reverse=not lib)   # best first
        top2 = ranked[:2]
        if not top2: continue
        for i, (m, b, rep) in enumerate(top2):
            entry = across.get(m, {})
            h = entry.get("score_mean"); used = entry.get("condition", "")
            d_ = improvement(b, h, ds) if finite(h) else float("nan")
            if finite(d_):
                if d_ > 0.005: pos2 += 1
                elif d_ < -0.005: neg2 += 1
                if i == 0:
                    if d_ > 0.005: pos += 1
                    elif d_ < -0.005: neg += 1
                    else: tie += 1
            dsl = ds if i == 0 else ""
            dl = dirlbl if i == 0 else ""
            repl = (str(rep)[:13]) if rep else "?"
            hs = f"{h:>8.3f}" if finite(h) else f"{'--':>8}"
            ds_ = f"{d_:>+8.3f}" if finite(d_) else f"{'--':>8}"
            print(f"  {dsl:<27}{dl:<6}{i+1:<2}{m:<9}{repl:<15}"
                  f"{b:>8.3f}{hs}{ds_}  {_CONDLBL.get(used, used)}")
    print("  " + "-" * 96)
    print(f"  deployed (#1) model:      +{pos} hybrid-better  ={tie} tie  -{neg} worse")
    print(f"  both top-2 models:        +{pos2} hybrid-better  -{neg2} worse")

# ---------------------------------------------------------------- C02 by-model
def analyze_c02_by_model():
    """Does the hybrid help certain models more? Every dataset, every model:
    Δ = best hybrid - best single rep (same model)."""
    banner("C02  HYBRID LIFT BY MODEL × DOMAIN  (best single rep -> best hybrid, same model)")
    d = os.path.join(ROOT, "complexity", "c02_complexity_matching_grid")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no c02 files)"); return
    print("  segmented by domain — NO cross-domain pooling. Δ direction-corrected.")
    print("  win/tie/loss counts are the signal; medΔ/range describe the spread.\n")
    # domain -> model -> [(delta, n_features)]
    by = defaultdict(lambda: defaultdict(list))
    for f in files:
        ds = dsname(f); j = load(f)
        if "_error" in j: continue
        base = j.get("baselines", {})
        across = j.get("best_per_model_across_conditions", {})
        dom = _domain(ds)
        for m in MODELS:
            b = base.get(m, {}).get("best_single_score")
            e = across.get(m, {})
            h = e.get("score_mean"); nf = e.get("n_features")
            if not finite(b) or not finite(h): continue
            by[dom][m].append((improvement(b, h, ds), nf if finite(nf) else None))
    for dom in ("molecular", "antibody", "tcga", "genomic"):
        models = by.get(dom)
        if not models: continue
        ndatasets = max((len(v) for v in models.values()), default=0)
        print(f"  [{dom}]  ({ndatasets} datasets)")
        print(f"    {'model':<10}{'n':>4}{'+':>4}{'=':>4}{'-':>4}"
              f"{'medΔ':>9}{'Δ range':>16}{'med_size':>10}")
        for m in MODELS:
            rows = models.get(m, [])
            if not rows: continue
            deltas = sorted(r[0] for r in rows)
            sizes = sorted(r[1] for r in rows if r[1] is not None)
            nbet = sum(1 for x in deltas if x > 0.005)
            nwor = sum(1 for x in deltas if x < -0.005)
            ntie = len(deltas) - nbet - nwor
            med = deltas[len(deltas) // 2]
            rng = f"[{deltas[0]:+.3f},{deltas[-1]:+.3f}]"
            ms = f"{int(sizes[len(sizes)//2]):>10}" if sizes else f"{'--':>10}"
            print(f"    {m:<10}{len(rows):>4}{nbet:>4}{ntie:>4}{nwor:>4}"
                  f"{med:>+9.3f}{rng:>16}{ms}")
        print()

# ---------------------------------------------------------------- C02 hybrid size
_MOLECULAR = {
    "caco2", "herg_ki", "herg", "logd", "qm9", "drd2", "esol",
    "adme_hlm", "adme_rlm", "adme_mdr1", "adme_solubility",
    "buchwald_hartwig", "doyle_hte_full_02", "heck_mizoroki_doyle",
}
def _domain(ds):
    """Coarse domain bucket so by-model lift is never pooled across unrelated
    tasks. molecular = the user's QM/ADMET domain."""
    if ds in _MOLECULAR: return "molecular"
    if ds.startswith("tcga_"): return "tcga"
    if ds.startswith(("gb_", "nt_")): return "genomic"
    return "antibody"  # abagym_/abcov_/flab_/ginkgo_/humanness_/tz_/ddg_

def _repdim(rep_dims, name):
    # c02 stamps _rep_dims as {'per_rep': {name: dim}, 'total_dim':.., 'n_reps':..}
    # (rep_dim_summary). Older/other shapes use a flat {name: dim}. Handle both.
    if isinstance(rep_dims, dict) and isinstance(rep_dims.get("per_rep"), dict):
        rep_dims = rep_dims["per_rep"]
    v = rep_dims.get(name)
    if isinstance(v, (int, float)) and math.isfinite(v): return int(v)
    if isinstance(v, dict):
        for k in ("dim", "total_dim", "n_features", "width"):
            if finite(v.get(k)): return int(v[k])
    return None
def analyze_c02_hybrid_size():
    """How big are the winning hybrids, vs the single rep they beat?"""
    banner("C02  HYBRID SIZE vs SINGLE-REP SIZE  (deployed model per dataset)")
    d = os.path.join(ROOT, "complexity", "c02_complexity_matching_grid")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no c02 files)"); return
    print("  deployed model = best single-rep score. rep_dim = that rep's full width;")
    print("  hyb_size = #features in the chosen hybrid; budget = size cap it searched.")
    print("  smaller? = hybrid uses FEWER features than the single rep it beats.\n")
    print(f"  {'dataset':<27}{'model':<9}{'rep':<14}{'rep_dim':>8}{'hyb_size':>9}"
          f"{'budget':>8}{'Δ':>8}  smaller?")
    print("  " + "-" * 92)
    sizes = []; smaller_and_better = 0; comparable = 0
    for f in files:
        ds = dsname(f); j = load(f)
        if "_error" in j: continue
        lib = lower_is_better(ds)
        base = j.get("baselines", {})
        across = j.get("best_per_model_across_conditions", {})
        rep_dims = j.get("_rep_dims", {}) or {}
        ranked = [(m, base[m].get("best_single_score"), base[m].get("best_single_rep"))
                  for m in MODELS
                  if m in base and finite(base[m].get("best_single_score"))]
        ranked.sort(key=lambda t: t[1], reverse=not lib)
        if not ranked: continue
        m, b, rep = ranked[0]
        e = across.get(m, {})
        h = e.get("score_mean"); nf = e.get("n_features"); bud = e.get("budget")
        if not finite(b) or not finite(h): continue
        rd = _repdim(rep_dims, rep)
        d_ = improvement(b, h, ds)
        sm = ""
        if rd is not None and finite(nf):
            comparable += 1
            sm = "yes" if nf < rd else "no"
            if nf < rd and d_ > 0.005: smaller_and_better += 1
        if finite(nf): sizes.append(int(nf))
        rds = f"{rd:>8}" if rd is not None else f"{'--':>8}"
        nfs = f"{int(nf):>9}" if finite(nf) else f"{'--':>9}"
        buds = f"{int(bud):>8}" if finite(bud) else f"{'--':>8}"
        print(f"  {ds:<27}{m:<9}{(str(rep)[:13]):<14}{rds}{nfs}{buds}{d_:>+8.3f}  {sm}")
    print("  " + "-" * 92)
    if sizes:
        ss = sorted(sizes)
        print(f"  hybrid size:  min {ss[0]}   median {ss[len(ss)//2]}   max {ss[-1]}")
    if comparable:
        print(f"  hybrids SMALLER than the single rep they beat AND still better: "
              f"{smaller_and_better}/{comparable}")

# ---------------------------------------------------------------- C04 importance-as-metric
def analyze_c04():
    """Per dataset: does using feature importance as a distance/retrieval metric
    beat the unweighted version? Δ = weighted r2 - unweighted r2 (clf skipped)."""
    banner("C04  IMPORTANCE AS A METRIC  (importance-weighted kNN / retrieval vs unweighted)")
    d = os.path.join(ROOT, "complexity", "c04_importance_as_metric")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no c04 files)"); return
    print(f"  {len(files)} files. Δ = weighted r2 - unweighted r2 (higher = importance")
    print("  weighting helps). best method shown per use. classification = skipped.\n")
    print(f"  {'dataset':<27}{'knn_method':<16}{'knnΔ':>8}   {'sim_method':<16}{'simΔ':>8}")
    print("  " + "-" * 80)
    knn_pos = knn_neg = real = skipped = 0
    for f in files:
        ds = dsname(f); j = load(f)
        if "_error" in j: continue
        if j.get("skipped"):
            print(f"  {ds:<27}{'(classification — skipped)':<40}"); skipped += 1; continue
        if not j.get("methods"):
            # A regression file whose importance weighting failed. It used to be
            # printed and counted as a skipped classification dataset.
            why = j.get("methods_failed") or j.get("importance_failed") or "no reason saved"
            print(f"  {ds:<27}FAILED: {str(why)[:60]}"); continue
        real += 1
        methods = j.get("methods", {})
        bk = j.get("best_knn_method"); bs = j.get("best_sim_method")
        kd = methods.get(bk, {}).get("knn_delta_k10")
        sd = methods.get(bs, {}).get("sim_delta_r2")
        if finite(kd):
            if kd > 0.005: knn_pos += 1
            elif kd < -0.005: knn_neg += 1
        kds = f"{kd:>+8.3f}" if finite(kd) else f"{'--':>8}"
        sds = f"{sd:>+8.3f}" if finite(sd) else f"{'--':>8}"
        print(f"  {ds:<27}{str(bk)[:15]:<16}{kds}   {str(bs)[:15]:<16}{sds}")
    print("  " + "-" * 80)
    print(f"  {real} regression datasets ({skipped} clf skipped).  "
          f"importance-weighted kNN helps: +{knn_pos}  hurts: -{knn_neg}")

# ---------------------------------------------------------------- C03 cross-dataset config
def analyze_c03():
    """Single combined multi.json: native-optimal hybrid config per dataset, and
    whether intrinsic complexity (erank) predicts the optimal hybrid budget."""
    banner("C03  CROSS-DATASET CONFIG  (one multi.json; does complexity predict hybrid budget)")
    d = os.path.join(ROOT, "complexity", "c03_cross_dataset_transfer")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no c03 results — single combined job, depends on c02; not run yet)"); return
    j = load(files[0])
    if "_error" in j:
        print(f"  ERROR {j['_error'][:60]}"); return
    print("  per dataset: native-optimal hybrid (chosen model / importance / budget / dim).\n")
    print(f"  {'dataset':<27}{'model':<10}{'importance':<14}{'budget':>8}{'dim':>7}{'score':>9}")
    print("  " + "-" * 76)
    for ds, r in j.items():
        if ds.startswith("_") or not isinstance(r, dict) or "error" in r: continue
        bm = r.get("best_model"); imp = r.get("best_importance")
        bud = r.get("best_budget"); dim = r.get("best_dim"); sc = r.get("best_score")
        if bm is None: continue
        buds = f"{int(bud):>8}" if finite(bud) else f"{'--':>8}"
        dims = f"{int(dim):>7}" if finite(dim) else f"{'--':>7}"
        scs = f"{sc:>9.4f}" if finite(sc) else f"{'--':>9}"
        print(f"  {ds:<27}{str(bm)[:9]:<10}{str(imp)[:13]:<14}{buds}{dims}{scs}")
    corr = j.get("_complexity_correlation", {})
    if corr:
        print("\n  complexity -> optimal budget (Spearman rho, want |rho| large + low p):")
        for key, c in corr.items():
            rb = c.get("rho_vs_budget"); pb = c.get("p_vs_budget")
            rd = c.get("rho_vs_dim")
            if finite(rb):
                print(f"    {key:<20} budget rho={rb:>+.3f} (p={pb:.3f})   "
                      f"dim rho={rd:>+.3f}" if finite(rd) else
                      f"    {key:<20} budget rho={rb:>+.3f} (p={pb:.3f})")

# ---------------------------------------------------------------- S01/S02 similarity
def analyze_similarity():
    banner("S01  SIMILARITY-ACTIVITY CORRELATION (best rep |SAC| per dataset)")
    d = os.path.join(ROOT, "similarity", "s01_similarity_baseline")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no s01 files)")
    else:
        print(f"  {len(files)} datasets\n")
        print(f"  {'dataset':<34}{'best_rep':<18}{'SAC_rho':>9}{'sig?':>6}{'viol':>8}")
        print("  " + "-" * 74)
        for f in files:
            ds = dsname(f); j = load(f)
            if "_error" in j: continue
            best = (None, 0.0, None, None)
            for rep, st in j.items():
                if not isinstance(st, dict): continue
                rho = st.get("sac_rho")
                if finite(rho) and abs(rho) > abs(best[1]):
                    best = (rep, rho, st.get("sac_significant"), st.get("violation_rate"))
            if best[0]:
                viol = f"{best[3]:>8.3f}" if finite(best[3]) else f"{'--':>8}"
                print(f"  {ds:<34}{best[0]:<18}{best[1]:>9.3f}{str(best[2]):>6}{viol}")
    # S02 hybrid improvement
    banner("S02  HYBRID DISTANCE vs BEST INDIVIDUAL REP (SAC improvement)")
    d2 = os.path.join(ROOT, "similarity", "s02_hybrid_similarity")
    f2 = sorted(glob.glob(os.path.join(d2, "*.json")))
    if not f2:
        print("  (no s02 files)"); return
    print(f"  {len(f2)} datasets   (hybrid SAC vs best individual rep's SAC)\n")
    print(f"  {'dataset':<34}{'indiv_rep':<16}{'indiv_sac':>10}{'hyb_sac':>9}{'lift':>8}")
    print("  " + "-" * 78)
    for f in f2:
        ds = dsname(f); j = load(f)
        if "_error" in j: continue
        # FAIR baseline = stronger of {unweighted, weighted} best single rep,
        # since the hybrid distance is itself importance-weighted.
        cands = []
        for key in ("best_individual_rep", "best_individual_rep_weighted"):
            bi = j.get(key)
            if isinstance(bi, dict) and finite(bi.get("sac_rho")):
                cands.append((bi.get("name"), bi["sac_rho"], "w" if "weighted" in key else "u"))
        cands.sort(key=lambda c: -c[1])
        iname, isac, which = cands[0] if cands else (None, None, "")
        hs = j.get("best_sac")
        isacs = f"{isac:>10.3f}" if finite(isac) else f"{'--':>10}"
        hss = f"{hs:>9.3f}" if finite(hs) else f"{'--':>9}"
        lift = f"{hs - isac:>+8.3f}" if finite(hs) and finite(isac) else f"{'--':>8}"
        print(f"  {ds:<34}{(str(iname)[:13]+'('+which+')'):<16}{isacs}{hss}{lift}")

# ---------------------------------------------------------------- schema recon
def schema_recon():
    banner("SCHEMA RECON  (top-level keys of one sample file per experiment)")
    for cat in ["complexity", "similarity", "noise"]:
        print(f"\n[{cat}]")
        for ed in expdirs(cat):
            files = sorted(glob.glob(os.path.join(ed, "*.json")))
            if not files: continue
            j = load(files[0])
            keys = [k for k in j if k != "timestamp"] if isinstance(j, dict) else ["<list>"]
            print(f"  {os.path.basename(ed):<42} n={len(files):<3} keys={keys[:14]}")

if __name__ == "__main__":
    print(f"ROOT = {os.path.abspath(ROOT)}")
    analyze_c02_top2()
    analyze_c02_by_model()
    analyze_c02_hybrid_size()
    analyze_c04()
    analyze_c03()
    analyze_similarity()
    schema_recon()
