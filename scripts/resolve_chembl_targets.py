#!/usr/bin/env python3
"""
Resolve ChEMBL target IDs for cross-species ADMET (T-Mol.2b).

For each (gene_symbol, organism) pair across {KCNH2, PTGS2, DRD2} ×
{human, dog, cat, cow, horse}, query the ChEMBL targets API to find
the species-specific target_chembl_id, then count activities with
IC50 or Ki measurements in nM.

Outputs:
  1. Console summary table.
  2. JSON dump at docs/cross_species_targets_resolved.json (machine-readable).
  3. Markdown snippet ready to paste into docs/cross_species_admet_design.md §5.

User runs this. Many vet target/activity combos will return zero — that's
the expected reality check per design doc §7. Sparse rows (<20) get
flagged "data sparse, skip"; rows between 20-50 get "drop to 3-fold CV".

Usage:
    python scripts/resolve_chembl_targets.py
    python scripts/resolve_chembl_targets.py --genes KCNH2 --species dog cat
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from kirby.utils.species import SPECIES_REGISTRY, chembl_organism, taxonomy_id


# Genes to resolve. Synonyms are extras the ChEMBL API may use as
# component_synonym for non-human orthologues; we union all matches.
GENES: Dict[str, List[str]] = {
    # NOTE: do NOT add bare 'ERG' — matches ERG transcription factor (oncogene).
    'KCNH2': ['KCNH2', 'HERG', 'H-ERG', 'LQT2', 'KIAA1623'],
    'PTGS2': ['PTGS2', 'COX-2', 'COX2', 'PHS-2', 'PGHS-2', 'cyclooxygenase-2'],
    'DRD2':  ['DRD2', 'D2R', 'D2DR', 'dopamine receptor D2'],
    # T-Mol.3 — canine CYP isoforms (dog-specific names) + human orthologue
    # synonyms so the human row in the resolved JSON anchors the comparison.
    'CYP1A2':  ['CYP1A2', 'cytochrome P450 1A2'],
    'CYP2B11': ['CYP2B11', 'cytochrome P450 2B11', 'CYP2B6'],
    'CYP2C21': ['CYP2C21', 'cytochrome P450 2C21', 'CYP2C9'],
    'CYP2D15': ['CYP2D15', 'cytochrome P450 2D15', 'CYP2D6'],
    'CYP3A12': ['CYP3A12', 'cytochrome P450 3A12', 'CYP3A4'],
    # T-Mol.3 — P-glycoprotein / multidrug-resistance transporter.
    'ABCB1':   ['ABCB1', 'MDR1', 'P-glycoprotein', 'P-gp', 'Pgp', 'PGY1'],
}

# Human anchors from existing loaders — used as a sanity check that
# resolution logic matches the hand-coded IDs in herg.py / cox2.py / drd2.py.
HUMAN_ANCHORS = {
    'KCNH2': 'CHEMBL240',
    'PTGS2': 'CHEMBL230',
    'DRD2':  'CHEMBL217',
    'CYP1A2':  'CHEMBL3356',
    'CYP2B11': 'CHEMBL3268',  # CYP2B6 (closest human orthologue)
    'CYP2C21': 'CHEMBL3397',  # CYP2C9
    'CYP2D15': 'CHEMBL289',   # CYP2D6
    'CYP3A12': 'CHEMBL340',   # CYP3A4
    'ABCB1':   'CHEMBL4302',
}

DEFAULT_SPECIES = ['human', 'dog', 'cat', 'cow', 'horse']
DEFAULT_OUT = REPO_ROOT / 'docs' / 'cross_species_targets_resolved.json'

# Activity thresholds (see design doc §13 decision 4)
SPARSE_THRESHOLD = 20   # below = skip
CV_THRESHOLD = 50       # below = drop to 3-fold


def _client():
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "ChEMBL client required: pip install chembl_webresource_client"
        )
    return new_client


def resolve_targets(gene: str, synonyms: List[str], tax_id: int
                    ) -> List[Dict]:
    """Find ChEMBL target(s) matching any synonym in a given organism.

    Filters by NCBI tax_id (rock-solid integer) rather than the Latin
    binomial — ChEMBL's organism string convention drifts between releases
    (e.g. `Canis familiaris` vs `Canis lupus familiaris` for tax_id 9615).

    Returns list of dicts with target_chembl_id, pref_name, target_type,
    organism (as reported), and which synonym matched. May be empty.
    """
    client = _client()
    target_client = client.target

    seen_ids = set()
    hits: List[Dict] = []

    for syn in synonyms:
        try:
            results = target_client.filter(
                target_synonym__iexact=syn,
                tax_id=tax_id,
            ).only([
                'target_chembl_id', 'pref_name', 'target_type', 'organism',
            ])
            # Force evaluation here so HTTP errors surface inside the try.
            results = list(results)
        except Exception as e:
            print(f"      [warn] synonym '{syn}' query failed: {e}")
            continue

        for r in results:
            tid = r.get('target_chembl_id')
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            hits.append({
                'target_chembl_id': tid,
                'pref_name': r.get('pref_name'),
                'target_type': r.get('target_type'),
                'organism': r.get('organism'),
                'matched_synonym': syn,
            })

    # Prefer SINGLE PROTEIN targets when multiple types come back.
    hits.sort(key=lambda r: (r['target_type'] != 'SINGLE PROTEIN', r['target_chembl_id']))
    return hits


def count_activities(target_chembl_id: str) -> Dict[str, int]:
    """Count IC50 + Ki rows in nM for a target. Returns dict per measure."""
    client = _client()
    activity_client = client.activity

    counts: Dict[str, int] = {}
    for stype in ('IC50', 'Ki'):
        try:
            q = activity_client.filter(
                target_chembl_id=target_chembl_id,
                standard_type=stype,
                standard_units='nM',
                standard_relation='=',
            ).only(['molecule_chembl_id'])
            # chembl_webresource_client QuerySets expose __len__ but not .count();
            # len() makes a single 'limit=1' HTTP call and reads the total header.
            counts[stype] = len(q)
        except Exception as e:
            print(f"      [warn] activity count {stype} failed: {e}")
            counts[stype] = -1
    return counts


def flag(n: int) -> str:
    if n < 0:
        return 'error'
    if n == 0:
        return 'none'
    if n < SPARSE_THRESHOLD:
        return 'skip'
    if n < CV_THRESHOLD:
        return 'cv3'
    return 'ok'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--genes', nargs='+', default=list(GENES.keys()),
                    help='Subset of genes to resolve (default: all).')
    ap.add_argument('--species', nargs='+', default=DEFAULT_SPECIES,
                    help='Subset of species (default: human dog cat cow horse).')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help=f'JSON output path (default: {DEFAULT_OUT}).')
    args = ap.parse_args()

    unknown_g = [g for g in args.genes if g not in GENES]
    unknown_s = [s for s in args.species if s not in SPECIES_REGISTRY]
    if unknown_g:
        raise SystemExit(f"Unknown genes: {unknown_g}. Known: {sorted(GENES)}")
    if unknown_s:
        raise SystemExit(f"Unknown species: {unknown_s}. Known: {sorted(SPECIES_REGISTRY)}")

    results: Dict[str, Dict[str, dict]] = {}

    for gene in args.genes:
        synonyms = GENES[gene]
        results[gene] = {}
        print(f"\n=== {gene} (synonyms: {synonyms}) ===")
        for species in args.species:
            latin = chembl_organism(species)
            tid = taxonomy_id(species)
            print(f"  [{species:5s}] tax_id={tid} ('{latin}')")
            hits = resolve_targets(gene, synonyms, tid)

            if not hits:
                print("    -> no targets found")
                results[gene][species] = {
                    'organism': latin,
                    'tax_id': tid,
                    'targets': [],
                    'best': None,
                    'activities': None,
                    'flag': 'none',
                }
                continue

            print(f"    -> {len(hits)} target(s):")
            for h in hits:
                print(f"       {h['target_chembl_id']:12s} "
                      f"{h['target_type']:20s} {h['pref_name']}")

            best = hits[0]  # SINGLE PROTEIN preferred by sort
            counts = count_activities(best['target_chembl_id'])
            total = sum(max(0, v) for v in counts.values())
            flag_str = flag(total)
            print(f"    -> best={best['target_chembl_id']}  "
                  f"IC50={counts.get('IC50', '?')}  "
                  f"Ki={counts.get('Ki', '?')}  "
                  f"total={total}  [{flag_str}]")

            # Cross-check human anchor
            if species == 'human':
                expected = HUMAN_ANCHORS.get(gene)
                match = '✓' if best['target_chembl_id'] == expected else '✗ MISMATCH'
                print(f"    -> human anchor check: expected {expected} {match}")

            results[gene][species] = {
                'organism': latin,
                'tax_id': tid,
                'targets': hits,
                'best': best['target_chembl_id'],
                'activities': counts,
                'total': total,
                'flag': flag_str,
            }

    # Write JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")

    # Print markdown snippet for design doc §5
    print("\n" + "=" * 60)
    print("Markdown snippet for docs/cross_species_admet_design.md §5:")
    print("=" * 60)
    for gene in args.genes:
        print(f"\n### {gene}")
        print()
        print("| organism | ChEMBL ID | IC50 | Ki  | total | status |")
        print("|---|---|---|---|---|---|")
        for species in args.species:
            r = results[gene].get(species, {})
            tid = r.get('best') or '—'
            acts = r.get('activities') or {}
            ic50 = acts.get('IC50', '—')
            ki = acts.get('Ki', '—')
            total = r.get('total', '—')
            status = r.get('flag', '—')
            print(f"| {species} | {tid} | {ic50} | {ki} | {total} | {status} |")

    # Summary
    print("\n" + "=" * 60)
    print("Summary (combos with enough data for an experiment):")
    print("=" * 60)
    n_ok = n_cv3 = n_skip = n_none = 0
    for gene in args.genes:
        for species in args.species:
            f = results[gene].get(species, {}).get('flag')
            if f == 'ok':
                n_ok += 1
            elif f == 'cv3':
                n_cv3 += 1
            elif f == 'skip':
                n_skip += 1
            else:
                n_none += 1
    print(f"  ok (>={CV_THRESHOLD}):           {n_ok}")
    print(f"  cv3 ({SPARSE_THRESHOLD}-{CV_THRESHOLD - 1}):           {n_cv3}")
    print(f"  skip (<{SPARSE_THRESHOLD}):            {n_skip}")
    print(f"  none/error:           {n_none}")


if __name__ == '__main__':
    main()
