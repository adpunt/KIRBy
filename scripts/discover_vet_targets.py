#!/usr/bin/env python3
"""
Discover ChEMBL targets with the most data for non-human species.

Flips T-Mol.2b: instead of asking "do our human ADMET targets survive
in vet ChEMBL?" this asks "what targets DO have vet ChEMBL data?".

For each non-human species (dog/cat/cow/horse by default):
  1. Stream all activities for that organism.
  2. Accumulate per-target activity counts (strict and optionally relaxed).
  3. Fetch pref_name + target_type for the top N targets.
  4. Check for a human counterpart by exact pref_name match.

Strict filter (matches T-Mol.2b):
    standard_type ∈ {IC50, Ki}, standard_units='nM', standard_relation='='

Relaxed filter (--relax):
    standard_type ∈ {IC50, Ki, EC50, Kd, AC50, Potency},
    standard_units ∈ {nM, uM, µM},  any relation in {'=', '~'}.

Output:
  - Console: per-species top-N table with human counterpart flag.
  - JSON: docs/cross_species_target_discovery.json
  - Markdown snippet for design doc.

Usage:
    python scripts/discover_vet_targets.py
    python scripts/discover_vet_targets.py --species dog --top 30
    python scripts/discover_vet_targets.py --relax
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from kirby.utils.species import SPECIES_REGISTRY, chembl_organism


DEFAULT_SPECIES = ['dog', 'cat', 'cow', 'horse']
DEFAULT_TOP = 25
DEFAULT_OUT = REPO_ROOT / 'docs' / 'cross_species_target_discovery.json'

STRICT_TYPES = ['IC50', 'Ki']
RELAX_TYPES = ['IC50', 'Ki', 'EC50', 'Kd', 'AC50', 'Potency']
RELAX_UNITS = ['nM', 'uM', 'µM']


def _client():
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "ChEMBL client required: pip install chembl_webresource_client"
        )
    return new_client


def stream_target_counts(organism_latin: str, relax: bool = False) -> Dict[str, int]:
    """Iterate all activities for a given organism, return per-target counts."""
    client = _client()
    activity_client = client.activity

    if relax:
        # chembl_webresource_client supports __in for list filters
        filter_kwargs = dict(
            target_organism=organism_latin,
            standard_type__in=RELAX_TYPES,
            standard_units__in=RELAX_UNITS,
        )
    else:
        filter_kwargs = dict(
            target_organism=organism_latin,
            standard_type__in=STRICT_TYPES,
            standard_units='nM',
            standard_relation='=',
        )

    activities = activity_client.filter(**filter_kwargs).only(['target_chembl_id'])

    target_counts: Dict[str, int] = {}
    n_total = 0
    for act in activities:
        tid = act.get('target_chembl_id')
        if tid:
            target_counts[tid] = target_counts.get(tid, 0) + 1
        n_total += 1
        if n_total % 2000 == 0:
            print(f"      …processed {n_total} activities, "
                  f"{len(target_counts)} unique targets so far", flush=True)
    print(f"      total: {n_total} activities, {len(target_counts)} unique targets")
    return target_counts


def fetch_target_meta(target_chembl_id: str) -> Dict[str, Optional[str]]:
    """Pull pref_name + target_type for a target_chembl_id."""
    client = _client()
    try:
        t = client.target.get(target_chembl_id)
        return {
            'pref_name': t.get('pref_name'),
            'target_type': t.get('target_type'),
        }
    except Exception as e:
        return {'pref_name': None, 'target_type': None, 'error': str(e)[:120]}


def has_human_counterpart(pref_name: Optional[str]) -> Optional[str]:
    """Look for a human target with the same pref_name. Returns its ChEMBL ID or None."""
    if not pref_name:
        return None
    client = _client()
    try:
        results = list(client.target.filter(
            pref_name__iexact=pref_name,
            tax_id=9606,
        ).only(['target_chembl_id'])[:1])
        if results:
            return results[0].get('target_chembl_id')
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--species', nargs='+', default=DEFAULT_SPECIES,
                    help='Species to discover (default: dog cat cow horse).')
    ap.add_argument('--top', type=int, default=DEFAULT_TOP,
                    help=f'Top N targets per species (default: {DEFAULT_TOP}).')
    ap.add_argument('--relax', action='store_true',
                    help='Relax filters: include EC50/Kd/AC50/Potency, uM units.')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help=f'JSON output path (default: {DEFAULT_OUT}).')
    args = ap.parse_args()

    unknown = [s for s in args.species if s not in SPECIES_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown species: {unknown}. "
                         f"Known: {sorted(SPECIES_REGISTRY)}")

    mode = 'relaxed' if args.relax else 'strict'
    print(f"Discovery mode: {mode}")
    print(f"Top N per species: {args.top}\n")

    results: Dict[str, dict] = {}

    for species in args.species:
        latin = chembl_organism(species)
        print(f"=== {species} (tax_id={SPECIES_REGISTRY[species]['taxonomy_id']}, "
              f"organism='{latin}') ===")
        print(f"  Streaming activities ({mode})...")
        target_counts = stream_target_counts(latin, relax=args.relax)

        if not target_counts:
            print(f"  No activities found for organism='{latin}'.")
            results[species] = {
                'organism': latin,
                'mode': mode,
                'n_targets_total': 0,
                'top': [],
            }
            continue

        top = sorted(target_counts.items(), key=lambda kv: -kv[1])[:args.top]
        print(f"  Top {len(top)} targets. Fetching pref_name + human counterparts...")

        enriched = []
        for tid, count in top:
            meta = fetch_target_meta(tid)
            human_id = has_human_counterpart(meta.get('pref_name'))
            enriched.append({
                'target_chembl_id': tid,
                'activity_count': count,
                'pref_name': meta.get('pref_name'),
                'target_type': meta.get('target_type'),
                'human_counterpart': human_id,
            })

        results[species] = {
            'organism': latin,
            'mode': mode,
            'n_targets_total': len(target_counts),
            'n_activities_total': sum(target_counts.values()),
            'top': enriched,
        }

        # Console table
        print(f"\n  Top {len(enriched)} targets for {species}:")
        print(f"  {'ChEMBL ID':14s} {'count':>6s} {'type':18s} "
              f"{'human ortholog':14s} pref_name")
        print(f"  {'-'*14} {'-'*6} {'-'*18} {'-'*14} {'-'*40}")
        for r in enriched:
            human = r['human_counterpart'] or '—'
            t_type = (r['target_type'] or '')[:18]
            pref = (r['pref_name'] or '')[:60]
            print(f"  {r['target_chembl_id']:14s} {r['activity_count']:>6d} "
                  f"{t_type:18s} {human:14s} {pref}")
        print()

    # Write JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")

    # Markdown snippet
    print("\n" + "=" * 60)
    print("Markdown snippet (top 10 per species with human counterpart):")
    print("=" * 60)
    for species in args.species:
        r = results.get(species, {})
        top = r.get('top', [])
        if not top:
            print(f"\n### {species} — no data")
            continue
        print(f"\n### {species} (n_activities={r.get('n_activities_total')}, "
              f"n_targets={r.get('n_targets_total')})")
        print()
        print("| rank | ChEMBL ID | count | type | human ortholog | pref_name |")
        print("|---|---|---|---|---|---|")
        for i, t in enumerate(top[:10], 1):
            human = t['human_counterpart'] or '—'
            print(f"| {i} | {t['target_chembl_id']} | {t['activity_count']} | "
                  f"{t['target_type'] or '—'} | {human} | {t['pref_name'] or '—'} |")


if __name__ == '__main__':
    main()
