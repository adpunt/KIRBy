#!/usr/bin/env python3
"""Look up ChEMBL target IDs for vet eval targets that aren't in the discovery JSON.

Tries multiple matching strategies in succession and saves whatever it finds to
`docs/cross_species_vet_target_ids.json`. Designed to be re-run safely — picks
up already-resolved entries from the JSON and skips them.

Targets to resolve:
  - Cow PTGS2 (n≈28 in T-Mol.2b strict count)
  - Dog PTGS2 / Dog COX-2 (n≈110 in T-Mol.2b tax_id count)

Strategies (tried in order):
  1. target_components__component_synonym = 'PTGS2'   AND organism = <latin>
  2. pref_name__iexact contains COX-2 synonyms       AND organism = <latin>
  3. pref_name__icontains 'cyclooxygenase'           AND organism = <latin>
  4. target_components__accession = <known UniProt>  AND organism = <latin>
  5. target_components__component_synonym = 'COX2'   AND organism = <latin>

Each hit is dumped with the strategy that found it. Activity count for the
resolved target is also fetched so we know whether the data actually exists.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from kirby.utils.species import chembl_organism

OUT_JSON = REPO_ROOT / 'docs' / 'cross_species_vet_target_ids.json'


# UniProt accessions for PTGS2 (cyclooxygenase-2) per species
UNIPROT_PTGS2 = {
    'cow':   'O62768',   # PTGS2 Bos taurus
    'dog':   'O62758',   # PTGS2 Canis familiaris
}

COX2_SYNONYMS = ['COX-2', 'COX2', 'Cyclooxygenase-2', 'Prostaglandin G/H synthase 2']

# Alternative organism strings ChEMBL uses for the same species
ORGANISM_ALIASES = {
    'cow':   ['Bos taurus'],
    'dog':   ['Canis familiaris', 'Canis lupus familiaris', 'Canis lupus'],
    'cat':   ['Felis catus'],
    'horse': ['Equus caballus'],
}

TARGETS_TO_RESOLVE = [
    {'species': 'cow', 'gene': 'PTGS2'},
    {'species': 'dog', 'gene': 'PTGS2'},
]


def _client():
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise SystemExit(
            "chembl_webresource_client not installed: "
            "pip install chembl_webresource_client"
        )
    return new_client


def _try(label: str, query_fn) -> List[dict]:
    """Run a strategy and report what came back."""
    try:
        hits = list(query_fn())
        if hits:
            print(f"    [{label}] hit: {len(hits)} target(s)")
        else:
            print(f"    [{label}] no hits")
        return hits
    except Exception as e:
        print(f"    [{label}] error: {e}")
        return []


def _build_strategies(target_client, gene: str, latin: str, species: str) -> List[tuple]:
    """Strategies for a single (organism string, gene). Filter-name aware:
    the ChEMBL REST API has changed some path names — only the ones below
    are accepted by the live endpoint as of 2026-05.
    """
    return [
        ('pref_name_iexact_cyclooxygenase2',
         lambda: target_client.filter(
             pref_name__iexact='Cyclooxygenase-2',
             organism=latin,
         ).only(['target_chembl_id', 'pref_name', 'organism'])[:5]),
        ('pref_name_icontains_cyclooxygenase',
         lambda: target_client.filter(
             pref_name__icontains='cyclooxygenase',
             organism=latin,
         ).only(['target_chembl_id', 'pref_name', 'organism'])[:10]),
        ('pref_name_icontains_prostaglandin',
         lambda: target_client.filter(
             pref_name__icontains='prostaglandin',
             organism=latin,
         ).only(['target_chembl_id', 'pref_name', 'organism'])[:10]),
        ('uniprot_accession',
         lambda: target_client.filter(
             target_components__accession=UNIPROT_PTGS2.get(species, ''),
             organism=latin,
         ).only(['target_chembl_id', 'pref_name', 'organism'])[:5])
            if species in UNIPROT_PTGS2 else (lambda: []),
    ]


def lookup(species: str, gene: str) -> Optional[Dict]:
    client = _client()
    target_client = client.target
    activity_client = client.activity
    latin_default = chembl_organism(species)
    print(f"  -> species={species} ({latin_default}), gene={gene}")

    all_candidates: List[Dict] = []
    for latin in ORGANISM_ALIASES.get(species, [latin_default]):
        print(f"  organism string: {latin!r}")
        for label, fn in _build_strategies(target_client, gene, latin, species):
            hits = _try(f"{latin}|{label}", fn)
            for h in hits:
                tid = h.get('target_chembl_id')
                if tid and not any(c['target_chembl_id'] == tid for c in all_candidates):
                    all_candidates.append({
                        'target_chembl_id': tid,
                        'pref_name': h.get('pref_name'),
                        'organism': h.get('organism'),
                        'matched_via': f"{latin}|{label}",
                    })

    # Last-ditch fallback: scan UniProt-anchored ChEMBL targets (no organism
    # filter) — relies on UNIPROT_PTGS2 being correct per species.
    if not all_candidates and species in UNIPROT_PTGS2:
        label = f'uniprot_only:{UNIPROT_PTGS2[species]}'
        hits = _try(label, lambda: target_client.filter(
            target_components__accession=UNIPROT_PTGS2[species],
        ).only(['target_chembl_id', 'pref_name', 'organism'])[:5])
        for h in hits:
            tid = h.get('target_chembl_id')
            if tid:
                all_candidates.append({
                    'target_chembl_id': tid,
                    'pref_name': h.get('pref_name'),
                    'organism': h.get('organism'),
                    'matched_via': label,
                })

    if not all_candidates:
        return {'species': species, 'gene': gene, 'organism': latin,
                'candidates': [], 'note': 'no strategy matched'}

    # For each candidate, count IC50/Ki activities so we know which one has data
    print(f"    counting activities for {len(all_candidates)} candidate(s)...")
    for cand in all_candidates:
        tid = cand['target_chembl_id']
        try:
            activities = activity_client.filter(
                target_chembl_id=tid,
                standard_type__in=['IC50', 'Ki'],
                standard_units='nM',
                standard_relation='=',
            ).only(['target_chembl_id'])
            cand['activity_count_strict'] = sum(1 for _ in activities)
        except Exception as e:
            cand['activity_count_strict'] = None
            cand['activity_count_error'] = str(e)[:120]

    return {
        'species': species, 'gene': gene, 'organism': latin,
        'candidates': all_candidates,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true',
                    help='Re-resolve even if entry already exists in JSON.')
    args = ap.parse_args()

    existing: Dict[str, Dict] = {}
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text())

    for entry in TARGETS_TO_RESOLVE:
        key = f"{entry['species']}_{entry['gene']}"
        if key in existing and not args.force and existing[key].get('candidates'):
            print(f"  [{key}] already resolved, skipping (use --force to redo)")
            continue
        print(f"\n=== Resolving {key} ===")
        result = lookup(entry['species'], entry['gene'])
        existing[key] = result

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(existing, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print("\nSummary:")
    for key, res in existing.items():
        cands = res.get('candidates', [])
        if not cands:
            print(f"  {key}: NO MATCH ({res.get('note', '')})")
            continue
        for c in cands:
            n = c.get('activity_count_strict', '?')
            print(f"  {key}: {c['target_chembl_id']} "
                  f"({c.get('pref_name', '?')}, n={n}, via {c['matched_via']})")


if __name__ == '__main__':
    main()
