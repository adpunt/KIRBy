#!/usr/bin/env python3
"""First-pass 'does the hybrid work?' reader for the NOT-READY group.

Run on the server (where the real results live):
    git pull
    python tests/analysis/firstpass_notready.py            # defaults to tests/results
    python tests/analysis/firstpass_notready.py <RESULTS_ROOT>

Reports, per dataset, for the DEPLOYED model (best single-rep score, direction
aware): best single rep score vs best hybrid score (SAME model), direction-
corrected Δ, and the hybrid's 95% CI across its 5 CV folds. Verdict uses a 0.01
tie band. Domain-segmented, NO cross-dataset averages. Read-only, stdlib only.

Handles the two quirks that silently corrupt this analysis:
  * metric DIRECTION is not stored in the JSON -> re-derived from the registry,
    which is only populated as a SIDE EFFECT of importing the dataset-policy
    module, so we import it first.
  * the c02 JSON lives in different dirs across runs -> we search for it.
"""
import json, glob, os, sys, math

ROOT = sys.argv[1] if len(sys.argv) > 1 else "tests/results"
TIE = 0.01                      # |Δ| <= TIE counts as a tie, not a win/loss
T95_DF4 = 2.776                 # two-sided t, 95%, df = n_folds-1 = 4

GROUPS = {
    "molecular": ["caco2", "herg", "herg_ki", "adme_mdr1", "adme_hlm",
                  "adme_rlm", "logd", "qm9"],
    "reaction":  ["bh_doyle_plates"],
    "tcga":      ["tcga_brca_pam50", "tcga_luad_vs_lusc"],
    "nucleotide":["nt_H3K27me3", "nt_H3K4me3", "nt_H3K9ac", "nt_H3K9me3",
                  "nt_promoter_tata", "gb_human_enhancers_cohn"],
}

# ---- direction: populate the registry (side-effect import), then read it -----
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
# hardcoded fallback if the registry can't be imported on the run host
_LOWER_FALLBACK = {"caco2", "herg_ki", "adme_mdr1", "adme_hlm", "adme_rlm",
                   "logd", "qm9", "bh_doyle_plates"}   # herg = clf (higher better)

def lower_is_better(ds):
    if ds in _DIR:
        return _DIR[ds]
    return ds in _LOWER_FALLBACK

def improvement(base, hyb, ds):
    return (base - hyb) if lower_is_better(ds) else (hyb - base)

# ---- locate the c02 result file for a dataset, wherever it landed -----------
def find_c02(ds):
    hits = []
    for dirpath, _dirs, files in os.walk(ROOT):
        base = os.path.basename(dirpath)
        if "c02" not in base:
            continue
        fn = ds + ".json"
        if fn in files:
            hits.append(os.path.join(dirpath, fn))
    if not hits:
        return None
    # prefer the canonical 'matching_grid' dir, then the fullest by #configs
    def rank(p):
        canonical = "matching_grid" in p
        try:
            n = len(json.load(open(p)).get("configs", []))
        except Exception:
            n = -1
        return (canonical, n)
    return max(hits, key=rank)

TOP_N = 3   # how many top existing (model, rep) combos to compare per dataset

def top_combos(d, ds, n=TOP_N):
    """The top-N EXISTING (model, rep) combinations — the strongest things you'd
    deploy today — ranked by best single-rep score (direction-aware)."""
    base = d.get("baselines", {})
    lo = lower_is_better(ds)
    scored = [(m, b.get("best_single_rep"), b.get("best_single_score"))
              for m, b in base.items()
              if isinstance(b.get("best_single_score"), (int, float))]
    scored.sort(key=lambda t: t[2], reverse=not lo)
    return scored[:n]

def best_hybrid_any(d, ds):
    """Best hybrid across ALL models (score, model, std) — best achievable."""
    lo = lower_is_better(ds)
    best = None
    for m, h in d.get("best_by_model", {}).items():
        s = h.get("score_mean")
        if not isinstance(s, (int, float)):
            continue
        if best is None or (s < best[0] if lo else s > best[0]):
            best = (s, m, h.get("score_std"), h.get("budget"), h.get("n_features"))
    return best

def _ci(std):
    return T95_DF4 * std / math.sqrt(5) if isinstance(std, (int, float)) else None

def _beats_ci(base, hyb, ci, ds):
    if ci is None:
        return None
    return base > (hyb + ci) if lower_is_better(ds) else base < (hyb - ci)

def _verdict(delta, beats_ci):
    if delta > TIE:
        return "WIN" + ("*" if beats_ci else "")
    if delta < -TIE:
        return "LOSS"
    return "tie"

def analyze_one(ds):
    path = find_c02(ds)
    if path is None:
        return {"ds": ds, "status": "MISSING (no c02 file yet)"}
    try:
        d = json.load(open(path))
    except Exception as e:
        return {"ds": ds, "status": f"UNREADABLE ({e})"}
    d["_dataset"] = ds
    if isinstance(d.get("status"), str) and d["status"]:
        return {"ds": ds, "status": f"stub: {d['status']}", "path": path}
    if not d.get("baselines") or not d.get("best_by_model"):
        return {"ds": ds, "status": "no baselines/best_by_model", "path": path}
    n_cfg = len(d.get("configs", []))

    combos = top_combos(d, ds)                 # top-N existing (model, rep, score)
    if not combos:
        return {"ds": ds, "status": "no scored baselines", "path": path}

    rows = []
    for rank, (model, base_rep, base_score) in enumerate(combos, 1):
        # hybrid on the SAME model as this existing combo (controlled comparison)
        h = d["best_by_model"].get(model, {})
        hyb = h.get("score_mean")
        ci = _ci(h.get("score_std"))
        delta = improvement(base_score, hyb, ds) if hyb is not None else None
        verdict = _verdict(delta, _beats_ci(base_score, hyb, ci, ds)) \
            if delta is not None else "--"
        rows.append(dict(rank=rank, model=model, base_rep=base_rep, base=base_score,
                         hyb=hyb, ci=ci, delta=delta, verdict=verdict))

    # best hybrid on ANY model vs the #1 existing combo (sharpest "beat the top")
    top_model, _top_rep, top_score = combos[0]
    bany = best_hybrid_any(d, ds)
    if bany:
        hyb_any, any_model, any_std = bany[0], bany[1], bany[2]
        ci_any = _ci(any_std)
        delta_any = improvement(top_score, hyb_any, ds)
        verdict_any = _verdict(delta_any, _beats_ci(top_score, hyb_any, ci_any, ds))
    else:
        hyb_any = any_model = ci_any = delta_any = verdict_any = None

    return {"ds": ds, "status": "ok", "path": path, "n_cfg": n_cfg, "rows": rows,
            "top_score": top_score, "hyb_any": hyb_any, "any_model": any_model,
            "ci_any": ci_any, "delta_any": delta_any, "verdict_any": verdict_any}

def _f(x, w=7, p=3):
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}"

def main():
    print(f"ROOT = {os.path.abspath(ROOT)}")
    print(f"direction registry loaded: {len(_DIR)} datasets"
          f"{' (EMPTY - using fallback)' if not _DIR else ''}")
    print(f"Per dataset: top-{TOP_N} EXISTING (model, best rep) combos vs the "
          f"hybrid on that SAME model.")
    print("Δ direction-corrected: POSITIVE = hybrid beats that existing combo. "
          "tie band |Δ|<=%.2f." % TIE)
    print("WIN* = existing combo falls OUTSIDE the hybrid's 95%% CI (5 folds) — "
          "the 'probably real' cases.")
    print("BEST-ANY row = best hybrid on ANY model vs the #1 existing combo "
          "(sharpest 'did we beat the top thing you'd deploy').\n")

    for dom, dss in GROUPS.items():
        print("=" * 84)
        print(f"[{dom}]  (↓=lower-better / ↑=higher-better: "
              + ", ".join(f"{x}{'↓' if lower_is_better(x) else '↑'}" for x in dss) + ")")
        print("=" * 84)
        print(f"  {'dataset':<26}{'#':>2} {'model':<10}{'existing rep':<16}"
              f"{'base':>7}{'hybrid':>8}{'±95CI':>8}{'Δ':>8}  verdict")
        print("  " + "-" * 82)
        for ds in dss:
            r = analyze_one(ds)
            if r["status"] != "ok":
                print(f"  {ds:<26}  --  {r['status']}")
                continue
            for row in r["rows"]:
                tag = "*" if row["rank"] == 1 else " "   # mark the #1 existing combo
                dcol = f"{row['delta']:>+8.3f}" if row["delta"] is not None else f"{'--':>8}"
                print(f"  {ds if row['rank']==1 else '':<26}{row['rank']:>2}{tag}"
                      f"{row['model']:<9}{str(row['base_rep'])[:15]:<16}"
                      f"{_f(row['base'])}{_f(row['hyb'],8)}{_f(row['ci'],8)}"
                      f"{dcol}  {row['verdict']}")
            # sharpest read: best hybrid on ANY model vs the #1 existing combo
            if r.get("hyb_any") is not None:
                dany = f"{r['delta_any']:>+8.3f}" if r["delta_any"] is not None else f"{'--':>8}"
                print(f"  {'':<26}  {'BEST-ANY':<9}{('('+str(r['any_model'])+')')[:15]:<16}"
                      f"{_f(r['top_score'])}{_f(r['hyb_any'],8)}{_f(r['ci_any'],8)}"
                      f"{dany}  {r['verdict_any']}   <- beats #1 existing combo?")
        print()

if __name__ == "__main__":
    main()
