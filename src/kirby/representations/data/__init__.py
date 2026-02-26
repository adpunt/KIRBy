"""
Local gene set data loader for TCGA pathway representations.

Loads pre-fetched gene sets from JSON files, cached in memory after first access.
Sources:
  - Enrichr API: Hallmark, KEGG, Reactome, Oncogenic
  - OmniPath API: PROGENy (weighted), DoRothEA (weighted regulons)
  - Enrichr Chromosome_Location: gene-to-arm map (collapsed to arm level)
"""

import json
from pathlib import Path
from typing import Dict, List

_DATA_DIR = Path(__file__).parent
_CACHE: Dict[str, object] = {}


def _load_json(name: str) -> dict:
    """Load a JSON file from the data directory, with in-memory caching."""
    if name not in _CACHE:
        path = _DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Gene set file not found: {path}")
        with open(path) as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


def hallmark() -> Dict[str, List[str]]:
    """MSigDB Hallmark 2020 gene sets (50 pathways)."""
    return _load_json('hallmark_2020.json')


def kegg() -> Dict[str, List[str]]:
    """KEGG 2021 Human gene sets (~320 pathways)."""
    return _load_json('kegg_2021.json')


def reactome() -> Dict[str, List[str]]:
    """Reactome 2022 gene sets (~1818 pathways)."""
    return _load_json('reactome_2022.json')


def oncogenic() -> Dict[str, List[str]]:
    """MSigDB C6 Oncogenic Signatures (189 gene sets)."""
    return _load_json('oncogenic_signatures.json')


def progeny_weights() -> Dict[str, Dict[str, float]]:
    """PROGENy pathway weights: {pathway: {gene: weight}}. 14 pathways."""
    return _load_json('progeny_weights.json')


def dorothea_regulons() -> Dict[str, Dict[str, float]]:
    """DoRothEA TF regulons: {tf: {target: weight}}. ~367 TFs."""
    return _load_json('dorothea_regulons.json')


def gene_chr_arm_map() -> Dict[str, str]:
    """Gene symbol to chromosome arm map: {gene: 'chr1p', ...}. ~30K genes."""
    return _load_json('gene_chr_arm_map.json')
