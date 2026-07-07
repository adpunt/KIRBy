#!/usr/bin/env python3
"""Trace the BEST hybrid's features back to base reps -> chemical meaning.

SERVER-SIDE (needs the data + precompute cache, like general_hybrid_vs_base.py).

For each dataset it:
  1. loads the precomputed search and picks the single best hybrid (best model x
     budget, direction-aware);
  2. re-fits that blueprint on full data to recover the CONCRETE per-rep selected
     column indices (the cache stores only the recipe, not the indices);
  3. maps every hybrid feature -> (base rep, within-rep original column) and, where
     the rep is chemically interpretable, to an ACTUAL name / SMARTS:
        pdv  -> RDKit physchem descriptor name   (DEFAULT_DESCRIPTOR_LIST)
        mordred -> mordred descriptor name        (Calculator(descriptors))
        maccs   -> MACCS key SMARTS               (MACCSkeys.smartsPatts)
        ecfp4/morgan -> substructure SMARTS        (decoded from example SMILES)
        avalon / hashed FPs / learned embeddings -> opaque (rep + dim index only)
     (drug_/reactant_/product_/diff_ prefixes are stripped before matching.)

Outputs, per dataset:
  * console: blueprint recipe, per-rep contribution breakdown, top-K features named;
  * tests/analysis/blueprints/<dataset>_features.csv  — EVERY feature, fully traced;
  * tests/analysis/blueprints/<dataset>_blueprint.json — recipe + per-rep counts + map.

Usage (on the server, after precompute):
    python tests/analysis/blueprint_trace.py                 # all DATASET_REPS
    python tests/analysis/blueprint_trace.py caco2 herg_ki   # specific
    python tests/analysis/blueprint_trace.py --domain molecular
    python tests/analysis/blueprint_trace.py --top 40 --aug uncapped
Read-only w.r.t. results; only writes under tests/analysis/blueprints/.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys, json, csv, math, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTDIR = Path(__file__).resolve().parent / "blueprints"


# ---------------------------------------------------------------- domain (matches precompute)
def get_domain(ds):
    if ds.startswith(("tcga_", "luad", "brca")):
        return "tcga"
    if ds.startswith(("davis", "kiba", "bindingdb")):
        return "dta"
    if ds.startswith(("buchwald", "doyle", "suzuki", "bh_", "heck")):
        return "reaction"
    if any(ds.startswith(p) for p in ("flab_", "abagym_", "abcov_", "ddg_", "tz_",
                                      "humanness", "ginkgo_", "graphinity_")):
        return "antibody"
    if ds.startswith(("nt_", "gb_")):
        return "nucleotide"
    return "molecular"


# ---------------------------------------------------------------- static chemical names
_NAMECACHE = {}
_INTERP_REPS = {"pdv", "physchem", "mordred", "maccs"}   # statically nameable cores

def _core(rep):
    for pre in ("drug_", "reactant_", "product_", "diff_"):
        if rep.startswith(pre):
            return rep[len(pre):]
    return rep

def chem_table(rep):
    """('name', [names]) or ('smarts', {idx:(smarts,count)}) or None, cached by core."""
    core = _core(rep)
    if core in _NAMECACHE:
        return _NAMECACHE[core]
    out = None
    try:
        if core in ("pdv", "physchem"):
            from kirby.representations.molecular import DEFAULT_DESCRIPTOR_LIST
            out = ("name", list(DEFAULT_DESCRIPTOR_LIST))
        elif core == "mordred":
            from mordred import Calculator, descriptors
            out = ("name", [str(d) for d in Calculator(descriptors, ignore_3D=True).descriptors])
        elif core == "maccs":
            from rdkit.Chem import MACCSkeys
            out = ("smarts", dict(MACCSkeys.smartsPatts))
    except Exception as e:
        print(f"    (name table for {core} unavailable: {e})")
        out = None
    _NAMECACHE[core] = out
    return out

def chem_meaning(rep, local_idx, interp):
    """Best chemical meaning for one feature. Returns (source, text)."""
    tbl = chem_table(rep)
    if tbl:
        kind, data = tbl
        if kind == "name" and 0 <= local_idx < len(data):
            return "descriptor", data[local_idx]
        if kind == "smarts" and local_idx in data:
            return "maccs_smarts", data[local_idx][0]
    # fall back to explain_hybrid's own interpretation (ECFP SMARTS, named reps, etc.)
    if isinstance(interp, dict):
        t = interp.get("type")
        if interp.get("smarts"):
            return t or "smarts", interp["smarts"]
        if interp.get("name"):
            return t or "named", interp["name"]
        if t and t not in ("embedding_dim",):
            return t, interp.get("description", "")
    return "opaque", ""

def interpretable(rep):
    core = _core(rep)
    if core in _INTERP_REPS:
        return "named"
    if "ecfp" in core or "morgan" in core:
        return "smarts(via SMILES)"
    return "opaque"


# ---------------------------------------------------------------- best hybrid + refit
def pick_best_hybrid(search):
    """(model, budget, row, blueprint) of the single best hybrid, direction-aware."""
    lib = bool(getattr(search, "lower_is_better", False))
    cands = []
    for m, bd in (getattr(search, "best_by_model_budget", None) or {}).items():
        for b, r in (bd or {}).items():
            s = r.get("score_mean") if isinstance(r, dict) else None
            if isinstance(s, (int, float)) and math.isfinite(s):
                cands.append((m, b, r))
    if not cands:
        return None
    m, b, row = max(cands, key=lambda t: (-t[2]["score_mean"] if lib else t[2]["score_mean"]))
    return m, b, row, row.get("blueprint")

def refit_indices(reps, labels, bp, aug_for_search, is_clf):
    """FittedResult with concrete per-rep selected_indices (cache has none)."""
    from kirby.find_best_hybrid import _compute_fitted_result
    kw = dict(weight_method="lightgbm_gain",
              augmentations=aug_for_search,
              augmentation_strategy=getattr(bp, "augmentation_strategy", "uncapped"),
              lower_is_better=bool(getattr(bp, "lower_is_better", False)),
              is_clf=is_clf)
    try:
        return _compute_fitted_result(reps, labels, bp, **kw)
    except TypeError:                       # tolerate signature drift
        try:
            return _compute_fitted_result(reps, labels, bp, weight_method="lightgbm_gain")
        except TypeError:
            return _compute_fitted_result(reps, labels, bp)


# ---------------------------------------------------------------- per-dataset
def trace_dataset(ds, aug_condition="uncapped", speed="full", top=25):
    from experiments._shared import load_data, get_precomputed_search
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from kirby.importance.hybrid import explain_hybrid

    cfg = DATASET_REPS.get(ds)
    if not cfg:
        print(f"  {ds}: not in DATASET_REPS — skip"); return None
    dom = get_domain(ds)
    gen_names = cfg.get("general", []); aug_names = cfg.get("augmentation", [])
    needed = sorted(set(gen_names + aug_names))
    try:
        data = load_data(ds, rep_names=needed)
    except Exception as e:
        print(f"  {ds}: load failed — {e}"); return None
    reps_all = data["reps"]; labels = data["labels"]; groups = data.get("groups")
    smiles = data.get("smiles") or data.get("drug_smiles")   # molecular / DTA only
    is_clf = data.get("task_type") == "classification"
    gen = {k: reps_all[k] for k in gen_names if k in reps_all}
    aug = {k: reps_all[k] for k in aug_names if k in reps_all}
    if len(gen) < 2:
        print(f"  {ds}: <2 general reps loaded — skip"); return None
    aug_for_search = aug if (aug_condition != "none" and aug) else None
    try:
        search = get_precomputed_search(
            reps=gen, labels=labels, groups=groups, aug_condition=aug_condition,
            augmentations=aug_for_search, speed=speed, domain=dom,
            dataset_name=ds, tier="general")
    except Exception as e:
        print(f"  {ds}: search load failed — {e}"); return None

    best = pick_best_hybrid(search)
    if not best:
        print(f"  {ds}: no hybrid in search — skip"); return None
    model, budget, row, bp = best
    if bp is None:
        print(f"  {ds}: best row has no blueprint — skip"); return None

    # refit on full data to get concrete selected column indices per rep
    all_reps = dict(gen); all_reps.update(aug)
    try:
        fitted = refit_indices(all_reps, labels, bp, aug_for_search, is_clf)
    except Exception as e:
        print(f"  {ds}: refit failed — {e}"); return None
    sel = getattr(fitted, "selected_indices", {}) or {}
    weights = getattr(fitted, "feature_weights", None) or {}
    feature_info = {rn: {"selected_indices": idx,
                         "importance_scores": weights.get(rn),
                         "n_features": len(idx)} for rn, idx in sel.items()}

    try:
        explanation = explain_hybrid(feature_info, smiles_list=(list(smiles) if smiles else None),
                                     top_k=10 ** 9)
        fmap = explanation.get("feature_map", [])
    except Exception as e:
        print(f"  {ds}: explain_hybrid failed — {e}"); fmap = []

    # ---- assemble the fully-traced feature rows
    rows = []
    for e in fmap:
        rep = e.get("rep_name"); li = e.get("local_idx")
        src, txt = chem_meaning(rep, li, e.get("interpretation", {}))
        rows.append({"hybrid_idx": e.get("hybrid_idx"), "rep_name": rep,
                     "local_idx": li, "importance": e.get("importance"),
                     "chem_source": src, "chem_meaning": txt})
    lib = bool(getattr(bp, "lower_is_better", False))
    rows.sort(key=lambda r: (r["importance"] if isinstance(r["importance"], (int, float)) else -1),
              reverse=True)

    per_rep = {rn: len(idx) for rn, idx in sel.items()}
    total = sum(per_rep.values()) or 1
    named = sum(1 for r in rows if r["chem_source"] not in ("opaque", ""))

    # ---- console
    print("\n" + "=" * 88)
    print(f"[{ds}]  domain={dom}  best hybrid: model={model} budget={budget} "
          f"score={row.get('score_mean'):.4f} ({'lower' if lib else 'higher'}=better)")
    print("=" * 88)
    print(f"  blueprint: allocation={getattr(bp,'allocation_method',None)}  "
          f"importance={getattr(bp,'importance_method',None)}  "
          f"evaluator={getattr(bp,'internal_evaluator',None)}  "
          f"budget={getattr(bp,'budget',None)}  aug={getattr(bp,'augmentation_strategy',None)}")
    print(f"  {len(per_rep)} reps contribute {total} features "
          f"({named}/{len(rows)} features chemically named):")
    for rn, n in sorted(per_rep.items(), key=lambda kv: -kv[1]):
        print(f"    {rn:<22}{n:>5} feats  {100*n/total:5.1f}%   [{interpretable(rn)}]")
    print(f"  top {min(top,len(rows))} features by importance:")
    print(f"    {'imp':>8}  {'rep':<20}{'col':>6}  {'source':<14}meaning")
    for r in rows[:top]:
        imp = f"{r['importance']:.4f}" if isinstance(r["importance"], (int, float)) else "--"
        print(f"    {imp:>8}  {str(r['rep_name'])[:19]:<20}{str(r['local_idx']):>6}  "
              f"{r['chem_source']:<14}{str(r['chem_meaning'])[:44]}")

    # ---- files
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / f"{ds}_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["hybrid_idx", "rep_name", "local_idx",
                                           "importance", "chem_source", "chem_meaning"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["hybrid_idx"] if isinstance(x["hybrid_idx"], int) else 0)):
            w.writerow(r)
    blueprint_out = {
        "dataset": ds, "domain": dom, "model": model, "budget": budget,
        "score_mean": row.get("score_mean"), "lower_is_better": lib,
        "blueprint": bp.to_dict() if hasattr(bp, "to_dict") else str(bp),
        "per_rep_feature_counts": per_rep, "n_features_total": total,
        "n_features_named": named, "n_features": len(rows),
        "feature_map": rows,
    }
    with open(OUTDIR / f"{ds}_blueprint.json", "w") as fh:
        json.dump(blueprint_out, fh, indent=2, default=str)
    print(f"  wrote {OUTDIR/f'{ds}_features.csv'} and {ds}_blueprint.json")
    return blueprint_out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="datasets (default: all DATASET_REPS)")
    ap.add_argument("--domain", default=None, help="only datasets in this domain")
    ap.add_argument("--aug", default="uncapped", choices=["none", "uncapped", "capped"])
    ap.add_argument("--speed", default="full", choices=["fast", "full"])
    ap.add_argument("--top", type=int, default=25, help="top-K features to print")
    args = ap.parse_args()

    from experiments.complexity.complexity_helpers import DATASET_REPS
    if args.datasets:
        datasets = args.datasets
    elif args.domain:
        datasets = sorted(k for k in DATASET_REPS if get_domain(k) == args.domain)
    else:
        datasets = sorted(DATASET_REPS.keys())

    print(f"Tracing {len(datasets)} datasets (aug={args.aug}, speed={args.speed}).")
    print("Chemical meaning is only recoverable for molecular reps pdv/mordred/maccs "
          "(names) and ecfp4 (SMARTS via SMILES); avalon / hashed FPs / learned "
          "embeddings are opaque per-dimension.\n")
    ok = 0
    for i, ds in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {ds} ...")
        try:
            if trace_dataset(ds, aug_condition=args.aug, speed=args.speed, top=args.top):
                ok += 1
        except Exception as e:
            print(f"  {ds}: unexpected error — {type(e).__name__}: {e}")
    print(f"\nDone: {ok}/{len(datasets)} traced. Per-dataset CSV+JSON under {OUTDIR}")


if __name__ == "__main__":
    main()
