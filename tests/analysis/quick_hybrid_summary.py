#!/usr/bin/env python3
"""Read-only direction-correct + FAIR summary of c02 / similarity results.

Usage:  python tests/analysis/quick_hybrid_summary.py [RESULTS_ROOT]
RESULTS_ROOT defaults to tests/results (must contain complexity/ similarity/).

Reports:
  * C02 per-cell, direction-corrected (impr>0 == hybrid better for higher- AND
    lower-better/RMSE metrics). impr_na = general pool no-aug; impr = best across
    conditions (augmentation + selection, optimistic).
  * C02 FAIR TOP-LINE: best hybrid vs best single rep with the BEST MODEL on
    BOTH sides — does mixing beat the best single-rep approach, without
    crediting the hybrid for rescuing a weak model.
  * S02 SAC lift vs the STRONGER of {unweighted, weighted} single rep (the
    hybrid distance is itself weighted, so the weighted rep is the fair opponent).
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
            _REGISTRY = {k: getattr(v, "lower_is_better", False) for k, v in reg.items()}
            break
        except Exception:
            continue
    return _REGISTRY

def lower_is_better(ds):
    reg = _load_registry()
    if ds in reg:
        return reg[ds]
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

# ---------------------------------------------------------------- C02 core
def analyze_c02():
    banner("C02  HYBRID vs BEST-SINGLE-REP  (the core KIRBy signal)")
    d = os.path.join(ROOT, "complexity", "c02_complexity_matching_grid")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no c02 files)"); return
    print(f"  {len(files)} datasets   (dir=lower means RMSE/MAE: improvement = baseline-hybrid)\n")
    hdr = f"  {'dataset':<30}{'dir':<6}{'model':<9}{'base':>8}{'noaug':>8}{'best':>8}{'impr_na':>9}{'impr':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    wins = losses = ties = 0
    per_ds_best = {}
    for f in files:
        ds = dsname(f); j = load(f)
        if "_error" in j:
            print(f"  {ds:<30}  ERROR {j['_error'][:40]}"); continue
        base = j.get("baselines", {})
        across = j.get("best_per_model_across_conditions", {})
        noaug = j.get("conditions", {}).get("no_aug", {})
        lib = lower_is_better(ds); dirlbl = "lower" if lib else "high"
        ds_max_impr = -9e9
        for m in MODELS:
            b = base.get(m, {}).get("best_single_score")
            hb = across.get(m, {}).get("score_mean")
            hn = noaug.get(m, {}).get("score_mean")
            if not finite(b) or not finite(hb):
                continue
            impr = improvement(b, hb, ds)            # best-condition improvement (signed: + = better)
            impr_na = improvement(b, hn, ds) if finite(hn) else float("nan")  # general pool, no_aug
            ds_max_impr = max(ds_max_impr, impr)
            mark = "+" if impr > 0.005 else ("-" if impr < -0.005 else "=")
            if mark == "+": wins += 1
            elif mark == "-": losses += 1
            else: ties += 1
            hns = f"{hn:>8.3f}" if finite(hn) else f"{'--':>8}"
            ina = f"{impr_na:>+9.3f}" if finite(impr_na) else f"{'--':>9}"
            print(f"  {ds:<30}{dirlbl:<6}{m:<9}{b:>8.3f}{hns}{hb:>8.3f}{ina}{impr:>+8.3f}")
        if ds_max_impr > -9e8:
            per_ds_best[ds] = (ds_max_impr, dirlbl)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  per-(dataset,model) cells: +{wins} hybrid-better  -{losses} worse  ={ties} tie (|impr|<0.005)")
    print("  (impr is direction-corrected: positive = hybrid better for BOTH higher- and lower-better metrics)")
    # FAIR TOP-LINE: best-single-rep (best MODEL too) vs best-hybrid (best model
    # & condition). This answers "does mixing beat the best you can do with ONE
    # rep" without crediting the hybrid for rescuing a weak model.
    banner("C02  FAIR TOP-LINE: best hybrid vs best single rep (best model in BOTH)")
    print("  noaug = general pool, NO aug (one config); any = best of ALL conditions (win col = which)")
    print("  sorted by Δany (descending)\n")
    # Human labels for the raw condition keys (see c02_complexity_grid.py).
    CONDLBL = {
        'no_aug': 'general', 'uncapped': 'gen+aug', 'capped': 'top+aug',
        'top_tier_only': 'top-only', 'frozen_plus_ft': 'fine-tuned',
        'per_model_top_rep_dim': 'dim-match',
    }
    print(f"  {'dataset':<30}{'dir':<6}{'base':>8}{'noaug':>8}{'any':>8}{'Δnoaug':>9}{'Δany':>7}  win")
    print("  " + "-" * 86)
    rows = []
    win_tally = defaultdict(int)
    for f in files:
        ds = dsname(f); j = load(f)
        if "_error" in j: continue
        lib = lower_is_better(ds); dirlbl = "lower" if lib else "high"
        pick = min if lib else max
        base = j.get("baselines", {}); conds = j.get("conditions", {})
        bvals = [base[m]["best_single_score"] for m in MODELS
                 if m in base and finite(base[m].get("best_single_score"))]
        if not bvals or not isinstance(conds, dict) or not conds: continue
        bb = pick(bvals)
        # best MODEL score within each condition (direction-aware), then the
        # winning CONDITION over all of them -> recovers which lever wins.
        cond_best = {}
        for cname, cdict in conds.items():
            if not isinstance(cdict, dict): continue
            vals = [cdict[m]["score_mean"] for m in MODELS
                    if m in cdict and finite(cdict[m].get("score_mean"))]
            if vals: cond_best[cname] = pick(vals)
        if not cond_best: continue
        hn = cond_best.get('no_aug', float("nan"))
        win_cname = pick(cond_best, key=cond_best.get)   # arg-best condition
        ha = cond_best[win_cname]
        dh = improvement(bb, hn, ds) if finite(hn) else float("nan")
        da = improvement(bb, ha, ds)
        win_tally[win_cname] += 1
        rows.append((ds, dirlbl, bb, hn, ha, dh, da, win_cname))
    for ds, dirlbl, bb, hn, ha, dh, da, win_cname in sorted(rows, key=lambda r: -r[6]):
        hns = f"{hn:>8.3f}" if finite(hn) else f"{'--':>8}"
        dhs = f"{dh:>+9.3f}" if finite(dh) else f"{'--':>9}"
        print(f"  {ds:<30}{dirlbl:<6}{bb:>8.3f}{hns}{ha:>8.3f}{dhs}{da:>+7.3f}  {CONDLBL.get(win_cname, win_cname)}")
    # aggregate over Δany
    das = [r[6] for r in rows]
    if das:
        npos = sum(1 for x in das if x > 0.005); nneg = sum(1 for x in das if x < -0.005)
        ntie = len(das) - npos - nneg
        mean_da = sum(das) / len(das); med_da = sorted(das)[len(das) // 2]
        print("  " + "-" * 86)
        print(f"  Δany over {len(das)} datasets:  mean {mean_da:+.3f}   median {med_da:+.3f}   "
              f"+{npos} win  ={ntie} tie  -{nneg} loss")
        print("  which condition wins:  " + "   ".join(
            f"{CONDLBL.get(k, k)}={v}" for k, v in sorted(win_tally.items(), key=lambda kv: -kv[1])))

# ---------------------------------------------------------------- C02 top-2 models
_CONDLBL = {
    'no_aug': 'general', 'uncapped': 'gen+aug', 'capped': 'top+aug',
    'top_tier_only': 'top-only', 'frozen_plus_ft': 'fine-tuned',
    'per_model_top_rep_dim': 'dim-match',
}
def analyze_c02_top2():
    """For the 2 models you'd actually pick (best single-rep score), show what
    the best hybrid does to EACH of them -- SAME model on both sides, so no
    credit for the hybrid letting a different model shine."""
    banner("C02  TOP-2 MODELS / DATASET: best rep vs best hybrid (SAME model)")
    d = os.path.join(ROOT, "complexity", "c02_complexity_matching_grid")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print("  (no c02 files)"); return
    print("  model #1/#2 = best/2nd-best single-rep score (what you'd deploy).")
    print("  base=best single rep for THAT model; noaug=general-pool mix; any=best")
    print("  condition (win col). Δ are same-model (hybrid[model] - base[model]).\n")
    print(f"  {'dataset':<27}{'dir':<6}{'#':<2}{'model':<9}{'best_rep':<15}"
          f"{'base':>7}{'noaug':>7}{'any':>7}{'Δnoaug':>8}{'Δany':>7}  win")
    print("  " + "-" * 104)
    pos = neg = tie = 0          # over the #1 (deployed) model only
    pos2 = neg2 = 0             # over both top-2 cells
    for f in files:
        ds = dsname(f); j = load(f)
        if "_error" in j: continue
        lib = lower_is_better(ds); dirlbl = "lower" if lib else "high"
        base = j.get("baselines", {})
        noaug = j.get("conditions", {}).get("no_aug", {})
        across = j.get("best_per_model_across_conditions", {})
        ranked = [(m, base[m].get("best_single_score"), base[m].get("best_single_rep"))
                  for m in MODELS
                  if m in base and finite(base[m].get("best_single_score"))]
        ranked.sort(key=lambda t: t[1], reverse=not lib)   # best first
        top2 = ranked[:2]
        if not top2: continue
        for i, (m, b, rep) in enumerate(top2):
            hn = noaug.get(m, {}).get("score_mean")
            entry = across.get(m, {})
            ha = entry.get("score_mean"); win = entry.get("condition", "")
            dn = improvement(b, hn, ds) if finite(hn) else float("nan")
            da = improvement(b, ha, ds) if finite(ha) else float("nan")
            if finite(da):
                if da > 0.005: pos2 += 1
                elif da < -0.005: neg2 += 1
                if i == 0:
                    if da > 0.005: pos += 1
                    elif da < -0.005: neg += 1
                    else: tie += 1
            dsl = ds if i == 0 else ""
            dl = dirlbl if i == 0 else ""
            repl = (str(rep)[:13]) if rep else "?"
            hns = f"{hn:>7.3f}" if finite(hn) else f"{'--':>7}"
            has = f"{ha:>7.3f}" if finite(ha) else f"{'--':>7}"
            dns = f"{dn:>+8.3f}" if finite(dn) else f"{'--':>8}"
            das = f"{da:>+7.3f}" if finite(da) else f"{'--':>7}"
            print(f"  {dsl:<27}{dl:<6}{i+1:<2}{m:<9}{repl:<15}"
                  f"{b:>7.3f}{hns}{has}{dns}{das}  {_CONDLBL.get(win, win)}")
    print("  " + "-" * 104)
    print(f"  best (deployed) model only:  +{pos} hybrid-better  ={tie} tie  -{neg} worse")
    print(f"  across both top-2 models:    +{pos2} hybrid-better  -{neg2} worse")

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
    analyze_c02()
    analyze_similarity()
    schema_recon()
