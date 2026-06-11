"""Cross-species antibody-ness loader.

Generalises `humanness.py` to a per-species one-vs-rest native-repertoire
classifier. v1 species:

  - dog       (positive class for canineness)
  - cow       (positive class for bovineness)
  - human, mouse, rhesus, camelid   (OAS-resourced background)

cat and horse are reserved in SPECIES_VOCAB_NESS for v2 promotion but
rejected at v1 with a clear "deferred to v2" ValueError so adding them
later does not break cache keys.

Loader returns the same dict shape as humanness.py / trastuzumab.py
(heavy_seqs / light_seqs / labels / task_type / n_classes /
is_intra_family / property_name / mutations / wildtype_heavy /
wildtype_light) plus per-species provenance and an ANARCI-derived
`groups` array for StratifiedGroupKFold by V-gene family.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Order is pinned for cache stability. cat and horse stay at their indices
# (forward-compat) but are rejected at v1.
SPECIES_VOCAB_NESS: List[str] = [
    'human', 'mouse', 'rhesus', 'camelid',
    'dog', 'cat', 'cow', 'horse',
]

V1_SUPPORTED_SPECIES: List[str] = [
    'human', 'mouse', 'rhesus', 'camelid', 'dog', 'cow',
]

# OAS species that the existing humanness.py path already pulls
_OAS_SPECIES = {'human', 'mouse', 'rhesus', 'camelid'}

# Per-species default cap. DoggifAI is over-resourced and the cow SRA
# assembly is under-resourced; both get capped at the same ceiling so the
# pool is balanced.
DEFAULT_MAX_SAMPLES_PER_SPECIES = 1500

# Where the manually-staged DoggifAI / bovine-SRA outputs live.
_DEFAULT_CACHE_ROOTS = [
    Path('/Volumes/seagate/kirby/cross_species'),
    Path('/data/stat-ecr/scat9264/kirby_data'),
]

# ANARCI numbering cache. The loader writes one file per
# (sha1(sequence), scheme) so per-row numbering can be reused across runs.
_ANARCI_CACHE_SUBDIR = 'anarci_cache'


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_species(species: str, *, kind: str) -> str:
    """Lowercase + validate against V1_SUPPORTED_SPECIES.

    kind is used to make the error message specific to the offending arg
    ('target_species' or 'source_species[i]').
    """
    name = species.strip().lower()
    if name in V1_SUPPORTED_SPECIES:
        return name
    if name in SPECIES_VOCAB_NESS:
        raise ValueError(
            f"{kind}={species!r} is reserved in SPECIES_VOCAB_NESS but "
            f"deferred to v2 (no public VH data available). v1 species: "
            f"{V1_SUPPORTED_SPECIES}."
        )
    raise ValueError(
        f"{kind}={species!r} is not in SPECIES_VOCAB_NESS. Known species: "
        f"{SPECIES_VOCAB_NESS}."
    )


def _resolve_cache_root(cache_dir: Optional[str]) -> Path:
    """Pick the cross-species data root. Env var > explicit kwarg > defaults."""
    env = os.environ.get('KIRBY_CROSS_SPECIES_DIR')
    if env:
        return Path(env)
    if cache_dir is not None:
        return Path(cache_dir)
    for root in _DEFAULT_CACHE_ROOTS:
        if root.parent.exists():
            return root
    # Last resort: local user cache. Make it on demand.
    fallback = Path.home() / '.cache' / 'kirby' / 'cross_species'
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ---------------------------------------------------------------------------
# Per-source helpers
# ---------------------------------------------------------------------------


def _load_oas_species(species: str) -> Tuple[List[str], List[str]]:
    """OAS-resourced species via humanness._download_and_extract_species.

    Returns (heavy_seqs, provenance) where provenance[i] == species.
    """
    from kirby.datasets.humanness import (
        _download_and_extract_species, _find_heavy_fasta, _parse_fasta_list,
    )

    root = _download_and_extract_species()
    fasta = _find_heavy_fasta(root, species)
    if fasta is None:
        return [], []
    seqs = _parse_fasta_list(fasta)
    return seqs, [species] * len(seqs)


def _load_canine(cache_root: Path) -> Tuple[List[str], List[str]]:
    """Read the manually-staged DoggifAI heavy chain FASTA, OR the SRA-
    assembled canine_vh.fasta produced by submit_canine_sra_assembly.sh.

    Resolution order (first hit wins):
      1. <cache_root>/canine_antibodies/heavy.fasta — DoggifAI (full ~430K).
      2. <cache_root>/canine_antibodies/heavy*.fasta — DoggifAI rename variants.
      3. <cache_root>/canine_antibodies/canine_vh.fasta — SRA-assembled
         from NCBI BioProject PRJNA790470 (Friedrichs 2022, ~43 dogs).
         This is the zero-auth preliminary path used while a DoggifAI
         Zenodo access request is pending.

    Raises with both the DoggifAI Zenodo URL and the SLURM-script path
    when none of the above is found.
    """
    from kirby.datasets.humanness import _parse_fasta_list

    dog_dir = cache_root / 'canine_antibodies'
    target: Optional[Path] = None
    chosen_source: str = ''

    # 1. DoggifAI canonical filename
    cand = dog_dir / 'heavy.fasta'
    if cand.exists() and cand.stat().st_size > 0:
        target, chosen_source = cand, 'doggifai'

    # 2. DoggifAI rename variants
    if target is None:
        variants = (list(dog_dir.glob('heavy*.fasta'))
                    + list(dog_dir.glob('*heavy*.fa'))
                    + list(dog_dir.glob('*heavy*.fasta')))
        variants = [v for v in variants if v.is_file() and v.stat().st_size > 0
                    and v.name != 'canine_vh.fasta']
        if variants:
            target, chosen_source = variants[0], 'doggifai'

    # 3. SRA-assembled fallback
    if target is None:
        cand = dog_dir / 'canine_vh.fasta'
        if cand.exists() and cand.stat().st_size > 0:
            target, chosen_source = cand, 'sra_PRJNA790470'

    if target is None:
        raise RuntimeError(
            f"Canine VH source not found at {dog_dir}. Either: (a) "
            f"request access at https://zenodo.org/records/15125375 + "
            f"unpack DoggifAI to {dog_dir}/heavy.fasta, or (b) run "
            f"tests/slurm_scripts/submit_canine_sra_assembly.sh on ARC "
            f"and rsync canine_vh.fasta back."
        )

    print(f"  [_load_canine] source={chosen_source} file={target.name}")
    seqs = _parse_fasta_list(target)
    return seqs, ['dog'] * len(seqs)


# Backwards-compat alias: the function was previously named `_load_doggifai`.
# Tests written against that name still work.
_load_doggifai = _load_canine


def _load_bovine_sra_assembly(cache_root: Path) -> Tuple[List[str], List[str]]:
    """Read the assembled bovine_vh.fasta.

    Raises with the SLURM-script path if the file is missing.
    """
    from kirby.datasets.humanness import _parse_fasta_list

    bov_dir = cache_root / 'bovine_antibodies'
    target = bov_dir / 'bovine_vh.fasta'
    if not target.exists():
        raise RuntimeError(
            f"Bovine VH source not found at {target}. "
            f"Run tests/slurm_scripts/submit_bovine_assembly.sh on ARC, "
            f"then rsync the output directory."
        )
    seqs = _parse_fasta_list(target)
    return seqs, ['cow'] * len(seqs)


# ---------------------------------------------------------------------------
# Canonical-sequence dedup
# ---------------------------------------------------------------------------


def _canonical_seq(seq: str) -> str:
    """Strip IMGT gaps, whitespace, uppercase. Same rule as the IgBLAST parser."""
    return (seq or '').replace('.', '').replace('-', '').strip().upper()


def _sha1_hex(seq: str) -> str:
    return hashlib.sha1(seq.encode('ascii', errors='ignore')).hexdigest()


def _dedup_preserve_first(
    seqs: List[str], provenance: List[str],
    exclude_seq_sha1: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Dedup by canonical sequence (first occurrence wins).

    Also drops any sequence whose canonical-SHA1 is in `exclude_seq_sha1` —
    used to keep therapeutic-eval sequences out of the training pool.
    """
    out_seqs: List[str] = []
    out_prov: List[str] = []
    seen: Set[str] = set()
    excl = exclude_seq_sha1 or set()
    for s, p in zip(seqs, provenance):
        cs = _canonical_seq(s)
        if not cs:
            continue
        if _sha1_hex(cs) in excl:
            continue
        if cs in seen:
            continue
        seen.add(cs)
        out_seqs.append(cs)
        out_prov.append(p)
    return out_seqs, out_prov


# ---------------------------------------------------------------------------
# Per-species cap (balanced stratified subsample)
# ---------------------------------------------------------------------------


def _balanced_subsample(
    seqs: List[str], provenance: List[str], max_per_species: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Cap each provenance bucket at max_per_species; preserve overall order."""
    rng = np.random.RandomState(seed)
    by_species: Dict[str, List[int]] = {}
    for i, p in enumerate(provenance):
        by_species.setdefault(p, []).append(i)

    keep: Set[int] = set()
    for species, idxs in by_species.items():
        if len(idxs) <= max_per_species:
            keep.update(idxs)
        else:
            chosen = rng.choice(idxs, max_per_species, replace=False)
            keep.update(int(x) for x in chosen)

    order = sorted(keep)
    return [seqs[i] for i in order], [provenance[i] for i in order]


# ---------------------------------------------------------------------------
# ANARCI V-gene grouping
# ---------------------------------------------------------------------------


_VGENE_FAMILY_RE = re.compile(r'^(IGHV\d+(?:S\d+)?)')


def _vgene_family(call: str) -> str:
    """Reduce a full V-gene call (e.g. 'IGHV3-23*01') to its family ('IGHV3').

    Falls back to the raw string when the regex doesn't match (rare, but
    keeps the group key usable as a fold key — different unknown calls
    still get different groups).
    """
    if not call:
        return 'UNKNOWN'
    m = _VGENE_FAMILY_RE.match(call.strip().upper())
    return m.group(1) if m else call.strip().upper()


def _anarci_cache_path(seq: str, scheme: str, cache_dir: Path) -> Path:
    return cache_dir / f'{_sha1_hex(seq)}__{scheme}.txt'


def _assign_vgene_groups(
    seqs: List[str], scheme: str, cache_root: Path,
) -> np.ndarray:
    """Number each VH with ANARCI, return V-gene family per sequence.

    Importable lazily so the loader can be imported on environments
    without ANARCI; the import error surfaces only when grouping is
    actually requested.
    """
    try:
        from anarci import number  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ANARCI is required for V-gene grouping in cross_species_ness. "
            "Install: `conda install -c bioconda hmmer && "
            "pip install kirby[antibody_ness]` (which pulls anarci>=2.5). "
            "See SETUP.md X-Ab section."
        ) from exc

    anarci_dir = cache_root / _ANARCI_CACHE_SUBDIR
    anarci_dir.mkdir(parents=True, exist_ok=True)

    out: List[str] = []
    for seq in seqs:
        cache_file = _anarci_cache_path(seq, scheme, anarci_dir)
        if cache_file.exists():
            out.append(cache_file.read_text().strip())
            continue
        try:
            numbering, chain_type = number(seq, scheme=scheme)  # type: ignore[misc]
            # ANARCI's `number` returns (numbering, chain_type). When we need
            # the V-gene call directly, use `run_anarci`. Lazy path below:
            from anarci import run_anarci  # type: ignore
            _, _, hits, _, _ = run_anarci(
                [('seq', seq)], scheme=scheme, output=False,
            )
            v_call = ''
            if hits and hits[0]:
                v_call = (hits[0][0].get('germlines', {})
                          .get('v_gene', [''])[0]
                          .get('id', ''))
            family = _vgene_family(v_call)
        except Exception:
            family = 'UNKNOWN'
        cache_file.write_text(family)
        out.append(family)
    return np.asarray(out, dtype=object)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_cross_species_ness(
    target_species: str,
    mode: str = 'one_vs_rest',
    source_species: Optional[List[str]] = None,
    max_samples_per_species: int = DEFAULT_MAX_SAMPLES_PER_SPECIES,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    exclude_seq_sha1: Optional[Set[str]] = None,
    anarci_scheme: str = 'imgt',
    assign_groups: bool = True,
) -> Dict:
    """Cross-species antibody-ness loader."""
    if mode not in ('one_vs_rest', 'multinomial'):
        raise ValueError(
            f"mode must be 'one_vs_rest' or 'multinomial', got {mode!r}.")

    target = _validate_species(target_species, kind='target_species')
    if source_species is None:
        sources = list(V1_SUPPORTED_SPECIES)
    else:
        sources = [_validate_species(s, kind='source_species[]')
                   for s in source_species]
    if target not in sources:
        raise ValueError(
            f"target_species={target!r} must appear in source_species "
            f"{sources}.")

    cache_root = _resolve_cache_root(cache_dir)

    # Load every source.
    all_seqs: List[str] = []
    all_prov: List[str] = []
    per_species_raw: Dict[str, int] = {}
    for species in sources:
        if species in _OAS_SPECIES:
            s, p = _load_oas_species(species)
        elif species == 'dog':
            s, p = _load_canine(cache_root)
        elif species == 'cow':
            s, p = _load_bovine_sra_assembly(cache_root)
        else:
            # _validate_species already guards cat / horse; this branch is
            # unreachable but defensive.
            raise RuntimeError(f"No loader for species={species!r}")
        per_species_raw[species] = len(s)
        all_seqs.extend(s)
        all_prov.extend(p)

    # Canonical-sequence dedup (also drops therapeutic-eval rows).
    all_seqs, all_prov = _dedup_preserve_first(
        all_seqs, all_prov, exclude_seq_sha1=exclude_seq_sha1)

    # Per-species cap (balanced).
    all_seqs, all_prov = _balanced_subsample(
        all_seqs, all_prov, max_samples_per_species, seed)

    n = len(all_seqs)
    if n == 0:
        raise RuntimeError(
            "No sequences survived load + dedup + cap. Check the cache "
            "root and the per-species source files.")

    # Labels.
    if mode == 'one_vs_rest':
        labels = np.asarray(
            [1 if p == target else 0 for p in all_prov], dtype=np.int64)
        n_classes = 2
        property_name = (
            f"{target}-ness (one-vs-rest, source={sources})")
        ts_for_output: Optional[str] = target
    else:  # 'multinomial'
        labels = np.asarray(
            [SPECIES_VOCAB_NESS.index(p) for p in all_prov], dtype=np.int64)
        n_classes = len(set(labels.tolist()))
        property_name = (
            f"species-ness multinomial (vocab={SPECIES_VOCAB_NESS}, "
            f"source={sources})")
        ts_for_output = None

    # ANARCI groups (lazy — heavy work only fires when requested).
    if assign_groups:
        groups = _assign_vgene_groups(all_seqs, anarci_scheme, cache_root)
    else:
        groups = np.array(all_prov, dtype=object)

    per_species_counts: Dict[str, int] = {}
    for p in all_prov:
        per_species_counts[p] = per_species_counts.get(p, 0) + 1

    return {
        'heavy_seqs': all_seqs,
        'light_seqs': [''] * n,
        'labels': labels,
        'task_type': 'classification',
        'n_classes': n_classes,
        'is_intra_family': False,
        'property_name': property_name,
        'mutations': [[] for _ in range(n)],
        'wildtype_heavy': None,
        'wildtype_light': None,
        'groups': groups,
        'source_species': sources,
        'per_species_counts': per_species_counts,
        'per_species_raw_counts': per_species_raw,
        'mode': mode,
        'target_species': ts_for_output,
        'anarci_scheme': anarci_scheme,
    }
