#!/usr/bin/env python3
"""Trace the BEST hybrid's features back to base reps -> chemical meaning.

SERVER-SIDE (needs the data + precompute cache, like general_hybrid_vs_base.py).

For each dataset it:
  1. loads the precomputed search and picks the single best hybrid (best model x
     budget, direction-aware);
  2. re-fits that blueprint on full data to recover the CONCRETE per-rep selected
     column indices (the cache stores only the recipe, not the indices);
  3. maps every hybrid feature -> (base rep, within-rep original column) and, where
     the rep is interpretable, to an ACTUAL name / SMARTS. Column alignment for each
     rep below was adversarially verified against its factory's emit order:
       MOLECULAR: pdv/pdv_binary/pdv_rank -> RDKit physchem descriptor name;
        mordred -> mordred descriptor name; maccs -> MACCS key SMARTS;
        ecfp4/morgan -> substructure SMARTS (from example SMILES);
        pharmacophore3d/usr/usrcat -> 3D-descriptor component name;
        coulomb_matrix -> eigenvalue ordinal (positional, not chemical).
       REACTION: reactant_/product_/diff_ + mordred/maccs/ecfp4 reuse the molecular
        tables; reactant_pdv/product_pdv/diff_pdv use rdkit Descriptors._descList (~208,
        NOT the 200-name molecular list — distinct table).
       ANTIBODY: mutation_features(48), paratope(8), esm2_llr/esmc_llr/antiberty_llr
        (sum/mean/min_llr), esm2_per_region -> chain/IMGT-region block label;
        aac/onehot/blosum/structural/antifold are named by explain_hybrid (deferred).
       OPAQUE per-dimension: avalon, atom_pair/top_torsion/rdkfp, drfp/map4/rxnfp,
        graph_kernel, and all learned embeddings (ESM/AntiBERTy/IgBERT/...).
     Matching is EXACT core-equality (no substring) after stripping drug_/reactant_/
     product_/diff_; a width guard skips a table when the rep's actual width disagrees.

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


# ---------------------------------------------------------------- feature interpretation
# Static column->meaning tables, VERIFIED column-aligned to each factory's emit order
# (rep-interpretability-map workflow: survey -> adversarial verify -> synthesize). Reps
# NOT tabled here are either already named by explain_hybrid (we fall back to its
# interpretation) or genuinely opaque per-dimension. Exact core-equality is used (never
# substring) to avoid collisions (usr/usrcat, pdv/reactant_pdv/pdv_binary, esm2/esm2_llr);
# a width guard means a version drift in a descriptor set skips rather than mislabels.
_NAMECACHE = {}
_STRIP = ("drug_", "reactant_", "product_", "diff_")
_REACTION_PDV = {"reactant_pdv", "product_pdv", "diff_pdv"}  # rdkit _descList, NOT DEFAULT list

def _core(rep):
    for pre in _STRIP:
        if rep.startswith(pre):
            return rep[len(pre):]
    return rep

def _pharm3d_names():
    fam = ["Donor", "Acceptor", "Aromatic", "Hydrophobe", "PosIonizable",
           "NegIonizable", "LumpedHydrophobe"]
    names = [f"count_{f}" for f in fam]
    pairs = [(fam[i], fam[j]) for i in range(7) for j in range(i + 1, 7)]   # i<j nested
    names += [f"mean_dist_{a}_{b}" for a, b in pairs]
    names += [f"std_dist_{a}_{b}" for a, b in pairs]
    return names                                                            # 49

def chem_table(rep):
    """(kind, data, expected_width, strict) or None. kind='name'|'smarts'.
    strict=True => trust only when the rep's ACTUAL width equals expected_width
    (version-dependent descriptor sets: mordred, reaction _descList)."""
    key = rep if rep in _REACTION_PDV else _core(rep)
    if key in _NAMECACHE:
        return _NAMECACHE[key]
    out = None
    try:
        if rep in _REACTION_PDV:                     # reaction pdv = rdkit Descriptors._descList (~208)
            from rdkit.Chem import Descriptors
            names = [d[0] for d in Descriptors._descList]
            out = ("name", names, len(names), True)
        elif key in ("pdv", "physchem", "pdv_binary", "pdv_rank"):
            from kirby.representations.molecular import DEFAULT_DESCRIPTOR_LIST
            names = list(DEFAULT_DESCRIPTOR_LIST)
            out = ("name", names, len(names), False)          # 200, stable repo constant
        elif key == "mordred":
            from mordred import Calculator, descriptors
            names = [str(d) for d in Calculator(descriptors, ignore_3D=True).descriptors]
            out = ("name", names, len(names), True)           # ~1613, version-dependent
        elif key == "maccs":
            from rdkit.Chem import MACCSkeys
            out = ("smarts", dict(MACCSkeys.smartsPatts), 167, False)   # keys 1..166, bit0 unused
        elif key == "pharmacophore3d":
            out = ("name", _pharm3d_names(), 49, False)
        elif key == "usrcat":                        # MUST be tested before 'usr'
            subs = ["all", "hydrophobic", "aromatic", "acceptor", "donor"]
            refs = ["ctd", "cst", "fct", "ftf"]; moms = ["mean", "std", "cbrtskew"]
            out = ("name", [f"{s}_{r}_{m}" for s in subs for r in refs for m in moms], 60, False)
        elif key == "usr":
            refs = ["ctd", "cst", "fct", "ftf"]; moms = ["mean", "std", "skew"]
            out = ("name", [f"{r}_{m}" for r in refs for m in moms], 12, False)
        elif key == "coulomb_matrix":                # ordinal only — no per-atom chemistry
            out = ("name", [f"coulomb_eig_{i}" for i in range(50)], 50, False)
        elif key == "mutation_features":             # antibody
            AA = "ACDEFGHIKLMNPQRSTVWY"
            names = [f"wt_{a}" for a in AA] + [f"mut_{a}" for a in AA] + [
                "blosum62_sub", "hydrophobicity_delta", "mw_delta", "charge_delta",
                "norm_position", "is_heavy", "n_mutations", "is_silent"]
            out = ("name", names, 48, False)
        elif key in ("esm2_llr", "esmc_llr", "antiberty_llr"):
            out = ("name", ["sum_llr", "mean_llr", "min_llr"], 3, False)
        elif key == "paratope":
            out = ("name", ["n_paratope_residues", "fraction_paratope",
                            "paratope_net_charge_H", "paratope_mean_hydrophobicity_H",
                            "paratope_aromatic_fraction_H", "paratope_frac_H1",
                            "paratope_frac_H2", "paratope_frac_H3"], 8, False)
    except Exception as e:
        print(f"    (name table for {key} unavailable: {e})")
        out = None
    _NAMECACHE[key] = out
    return out

# esm2_per_region: block-level (chain, IMGT region) labels; within-block dim stays opaque.
_REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
def _region_block(rep, local_idx, rep_width):
    if _core(rep) != "esm2_per_region" or not rep_width or rep_width % 1280 or local_idx < 0:
        return None
    block = local_idx // 1280                        # H blocks 0-6, then L blocks 7-13
    chain = "H" if block < 7 else "L"
    return f"{chain}/{_REGIONS[block % 7]} esm2 latent {local_idx % 1280}"

def chem_meaning(rep, local_idx, interp, rep_width=None):
    """Best meaning for one feature -> (source, text). Prefers verified static tables
    (with width guard), then explain_hybrid's own interpretation (ECFP SMARTS, registry
    named reps), else opaque."""
    if not isinstance(local_idx, int):
        local_idx = -1
    rb = _region_block(rep, local_idx, rep_width)
    if rb is not None:
        return "esm_region_block", rb
    tbl = chem_table(rep)
    if tbl:
        kind, data, expw, strict = tbl
        if kind == "name":
            width_ok = (rep_width == expw) if rep_width is not None else (not strict)
            if width_ok and 0 <= local_idx < len(data):
                return "descriptor", data[local_idx]
        elif kind == "smarts":
            if rep_width in (None, 166, 167) and local_idx in data:
                return "maccs_smarts", data[local_idx][0]
    if isinstance(interp, dict):        # explain_hybrid fallback
        t = interp.get("type")
        if interp.get("smarts"):
            return t or "smarts", interp["smarts"]
        if interp.get("name"):
            return t or "named", interp["name"]
        if t and t not in ("embedding_dim",):
            return t, interp.get("description", "")
    return "opaque", ""

# reps explain_hybrid already names correctly (we defer to its interpretation, add nothing)
_DEFER_NAMED = {"aac", "onehot", "blosum", "structural", "antifold", "protein_aac",
                "protein_descriptors", "protein_kmer", "concat_ecfp4", "hallmark",
                "progeny", "kegg", "reactome", "oncogenic", "dorothea", "immune",
                "gene_stats", "chromosome_arm", "combined_pathway"}
def interpretable(rep):
    if rep in _REACTION_PDV or chem_table(rep):
        return "named"
    core = _core(rep)
    if "ecfp" in core or "morgan" in core:
        return "smarts(via SMILES)"
    if "_pca" in core:
        return "PC-index"
    if core == "esm2_per_region":
        return "region-block"
    if core in _DEFER_NAMED:
        return "named(explain)"
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

    # ---- assemble the fully-traced feature rows (rep_width feeds the alignment guard)
    def _w(rn):
        a = all_reps.get(rn)
        return int(a.shape[1]) if hasattr(a, "shape") and getattr(a, "ndim", 0) == 2 else None
    rows = []
    for e in fmap:
        rep = e.get("rep_name"); li = e.get("local_idx")
        src, txt = chem_meaning(rep, li, e.get("interpretation", {}), _w(rep))
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
    print("Interpretable: molecular pdv/mordred/maccs/pharmacophore3d/usr/usrcat (names), "
          "ecfp4 (SMARTS via SMILES); reaction reactant_/product_/diff_ variants; antibody "
          "mutation_features/paratope/*_llr/esm2_per_region + aac/onehot/blosum/structural/"
          "antifold. Hashed FPs, avalon, and learned embeddings stay opaque per-dimension.\n")
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
