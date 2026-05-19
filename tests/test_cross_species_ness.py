"""Network-free tests for the T-Ab.3 cross-species antibody-ness loader.

Exercises every helper that does NOT require ANARCI / DoggifAI / SRA data:
species validation, canonical-sequence dedup, balanced sub-sampling, the
V-gene family regex, the exclude_seq_sha1 plumbing, and the end-to-end
`load_cross_species_ness` happy path with the per-source helpers
monkey-patched. ANARCI itself is mocked.
"""

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from kirby.datasets.cross_species_ness import (  # noqa: E402
    SPECIES_VOCAB_NESS,
    V1_SUPPORTED_SPECIES,
    DEFAULT_MAX_SAMPLES_PER_SPECIES,
    _validate_species,
    _canonical_seq,
    _sha1_hex,
    _dedup_preserve_first,
    _balanced_subsample,
    _vgene_family,
    _resolve_cache_root,
    load_cross_species_ness,
)


# ---------------------------------------------------------------------------
# Vocab + validator
# ---------------------------------------------------------------------------


def test_species_vocab_pinned_order():
    # Cache keys depend on this order; pin it explicitly.
    assert SPECIES_VOCAB_NESS == [
        'human', 'mouse', 'rhesus', 'camelid',
        'dog', 'cat', 'cow', 'horse',
    ]


def test_v1_supported_species_is_subset_in_vocab_order():
    assert V1_SUPPORTED_SPECIES == [
        'human', 'mouse', 'rhesus', 'camelid', 'dog', 'cow']
    # Every v1 species is in the broader vocab.
    assert all(s in SPECIES_VOCAB_NESS for s in V1_SUPPORTED_SPECIES)


def test_validator_accepts_v1_species_case_insensitive():
    assert _validate_species('DOG', kind='target_species') == 'dog'
    assert _validate_species(' Cow ', kind='target_species') == 'cow'


def test_validator_rejects_cat_and_horse_with_v2_message():
    for s in ('cat', 'horse', 'CAT'):
        with pytest.raises(ValueError, match='deferred to v2'):
            _validate_species(s, kind='target_species')


def test_validator_rejects_unknown_species_with_vocab_list():
    with pytest.raises(ValueError, match='not in SPECIES_VOCAB_NESS'):
        _validate_species('hamster', kind='target_species')


def test_validator_kind_appears_in_error():
    with pytest.raises(ValueError, match='source_species'):
        _validate_species('horse', kind='source_species[2]')


# ---------------------------------------------------------------------------
# Dedup + canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_seq_strips_imgt_gaps_and_uppercases():
    assert _canonical_seq('a.c-gT') == 'ACGT'
    assert _canonical_seq(' QVQL...VES ') == 'QVQLVES'
    assert _canonical_seq('') == ''
    assert _canonical_seq(None) == ''  # type: ignore[arg-type]


def test_dedup_preserve_first_drops_duplicates():
    seqs, prov = _dedup_preserve_first(
        ['ACGT', 'AC.GT', 'GGAA'], ['dog', 'dog', 'human'])
    assert seqs == ['ACGT', 'GGAA']
    assert prov == ['dog', 'human']


def test_dedup_respects_exclude_seq_sha1():
    excluded = {hashlib.sha1(b'ACGT').hexdigest()}
    seqs, prov = _dedup_preserve_first(
        ['ACGT', 'GGAA'], ['dog', 'human'], exclude_seq_sha1=excluded)
    assert seqs == ['GGAA']
    assert prov == ['human']


def test_dedup_drops_empty_or_gap_only():
    seqs, prov = _dedup_preserve_first(
        ['ACGT', '', '...', 'GGAA'], ['dog', 'dog', 'dog', 'cow'])
    assert seqs == ['ACGT', 'GGAA']
    assert prov == ['dog', 'cow']


# ---------------------------------------------------------------------------
# Balanced subsample
# ---------------------------------------------------------------------------


def test_balanced_subsample_caps_per_species():
    seqs = [f'SEQ_{i}' for i in range(50)]
    prov = (['dog'] * 30) + (['human'] * 20)
    sub_seqs, sub_prov = _balanced_subsample(seqs, prov, max_per_species=10,
                                             seed=0)
    counts: Dict[str, int] = {}
    for p in sub_prov:
        counts[p] = counts.get(p, 0) + 1
    assert counts == {'dog': 10, 'human': 10}
    assert len(sub_seqs) == 20
    # Order is preserved (sorted indices), so dog rows come before human.
    assert all(p == 'dog' for p in sub_prov[:10])
    assert all(p == 'human' for p in sub_prov[10:])


def test_balanced_subsample_keeps_all_when_under_cap():
    seqs = ['A', 'B', 'C', 'D']
    prov = ['dog', 'dog', 'human', 'human']
    out_s, out_p = _balanced_subsample(seqs, prov, max_per_species=100, seed=0)
    assert out_s == seqs and out_p == prov


def test_balanced_subsample_seed_reproducible():
    seqs = [f'SEQ_{i}' for i in range(40)]
    prov = ['dog'] * 40
    s1, _ = _balanced_subsample(seqs, prov, max_per_species=5, seed=42)
    s2, _ = _balanced_subsample(seqs, prov, max_per_species=5, seed=42)
    assert s1 == s2


# ---------------------------------------------------------------------------
# V-gene family regex
# ---------------------------------------------------------------------------


def test_vgene_family_strips_allele_and_position():
    assert _vgene_family('IGHV3-23*01') == 'IGHV3'
    assert _vgene_family('IGHV1') == 'IGHV1'
    assert _vgene_family('IGHV4-59*02') == 'IGHV4'


def test_vgene_family_keeps_bovine_S_variant():
    # Bovine germlines use the IGHV<n>S<m> notation; the family resolution
    # MUST preserve the S-segment because that's how bovine ultra-long
    # CDRH3 IGHV families are distinguished.
    assert _vgene_family('IGHV3S5*01') == 'IGHV3S5'
    assert _vgene_family('IGHV1S1') == 'IGHV1S1'


def test_vgene_family_fallback_for_unknown():
    assert _vgene_family('') == 'UNKNOWN'
    assert _vgene_family('garbage-call') == 'GARBAGE-CALL'


# ---------------------------------------------------------------------------
# Cache root resolution
# ---------------------------------------------------------------------------


def test_resolve_cache_root_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CROSS_SPECIES_DIR', str(tmp_path))
    assert _resolve_cache_root(None) == tmp_path
    # explicit cache_dir kwarg also ignored when env is set
    assert _resolve_cache_root('/other/path') == tmp_path


def test_resolve_cache_root_explicit_kwarg(tmp_path, monkeypatch):
    monkeypatch.delenv('KIRBY_CROSS_SPECIES_DIR', raising=False)
    assert _resolve_cache_root(str(tmp_path)) == tmp_path


# ---------------------------------------------------------------------------
# End-to-end with mocked sources + ANARCI
# ---------------------------------------------------------------------------


def _mock_loader_dog(_cache_root) -> Tuple[List[str], List[str]]:
    seqs = [f'QVQLVESGG{chr(65 + i)}STKLSCAASGFTFSDYAMSWVRQAP' for i in range(20)]
    return seqs, ['dog'] * len(seqs)


def _mock_loader_oas(species: str) -> Tuple[List[str], List[str]]:
    seqs = [f'EVQLLESGGGLVQPGG{species[:3].upper()}{i:04d}' for i in range(15)]
    return seqs, [species] * len(seqs)


def _mock_loader_cow(_cache_root) -> Tuple[List[str], List[str]]:
    # Make some sequences long (bovine ultra-long CDRH3 flavour).
    seqs = [f'QVQLVESGG{i:03d}LVQPGRSLRLSCVAS{"K" * (i % 5)}' for i in range(10)]
    return seqs, ['cow'] * len(seqs)


def _mock_anarci_groups(seqs: List[str], scheme: str, cache_root) -> np.ndarray:
    # Use a deterministic pseudo-V-gene: parity of seq length → family index.
    return np.asarray(
        [f'IGHV{(len(s) % 4) + 1}' for s in seqs], dtype=object)


def test_load_cross_species_ness_end_to_end(monkeypatch, tmp_path):
    import kirby.datasets.cross_species_ness as mod
    monkeypatch.setenv('KIRBY_CROSS_SPECIES_DIR', str(tmp_path))
    monkeypatch.setattr(mod, '_load_oas_species', _mock_loader_oas)
    monkeypatch.setattr(mod, '_load_canine', _mock_loader_dog)
    monkeypatch.setattr(mod, '_load_bovine_sra_assembly', _mock_loader_cow)
    monkeypatch.setattr(mod, '_assign_vgene_groups', _mock_anarci_groups)

    out = load_cross_species_ness(target_species='dog')
    assert out['mode'] == 'one_vs_rest'
    assert out['target_species'] == 'dog'
    assert out['task_type'] == 'classification'
    assert out['n_classes'] == 2
    assert set(out['per_species_counts'].keys()) <= set(V1_SUPPORTED_SPECIES)
    # Dog rows → label=1, everything else → 0.
    n_dog = out['per_species_counts'].get('dog', 0)
    assert int(np.sum(out['labels'] == 1)) == n_dog
    assert int(np.sum(out['labels'] == 0)) == (len(out['labels']) - n_dog)
    # groups vector aligned with labels.
    assert len(out['groups']) == len(out['labels'])
    # Heavy seqs are canonicalised (uppercased, no gaps).
    assert all(s.isupper() and '.' not in s for s in out['heavy_seqs'])
    # Light seqs are placeholder.
    assert all(s == '' for s in out['light_seqs'])


def test_load_cross_species_ness_multinomial(monkeypatch, tmp_path):
    import kirby.datasets.cross_species_ness as mod
    monkeypatch.setenv('KIRBY_CROSS_SPECIES_DIR', str(tmp_path))
    monkeypatch.setattr(mod, '_load_oas_species', _mock_loader_oas)
    monkeypatch.setattr(mod, '_load_canine', _mock_loader_dog)
    monkeypatch.setattr(mod, '_load_bovine_sra_assembly', _mock_loader_cow)
    monkeypatch.setattr(mod, '_assign_vgene_groups', _mock_anarci_groups)

    out = load_cross_species_ness(
        target_species='dog', mode='multinomial')
    assert out['mode'] == 'multinomial'
    assert out['target_species'] is None
    # Labels are SPECIES_VOCAB_NESS indices.
    expected_indices = {SPECIES_VOCAB_NESS.index(p)
                        for p in V1_SUPPORTED_SPECIES}
    assert set(int(x) for x in out['labels']) <= expected_indices


def test_load_cross_species_ness_excludes_seq_sha1(monkeypatch, tmp_path):
    import kirby.datasets.cross_species_ness as mod
    monkeypatch.setenv('KIRBY_CROSS_SPECIES_DIR', str(tmp_path))
    monkeypatch.setattr(mod, '_load_oas_species', _mock_loader_oas)
    monkeypatch.setattr(mod, '_load_canine', _mock_loader_dog)
    monkeypatch.setattr(mod, '_load_bovine_sra_assembly', _mock_loader_cow)
    monkeypatch.setattr(mod, '_assign_vgene_groups', _mock_anarci_groups)

    base = load_cross_species_ness(
        target_species='dog', source_species=['human', 'dog'])
    n_dog_before = base['per_species_counts'].get('dog', 0)

    # Grab one canonical dog seq, exclude it.
    one_dog = next(s for s, p in zip(base['heavy_seqs'],
                                     [SPECIES_VOCAB_NESS[i] for i in
                                      [SPECIES_VOCAB_NESS.index('dog')
                                       if l == 1 else 0
                                       for l in base['labels']]])
                   if p == 'dog')
    excl = {_sha1_hex(one_dog)}
    after = load_cross_species_ness(
        target_species='dog', source_species=['human', 'dog'],
        exclude_seq_sha1=excl)
    assert after['per_species_counts'].get('dog', 0) == n_dog_before - 1


def test_load_cross_species_ness_rejects_target_not_in_source(monkeypatch,
                                                              tmp_path):
    import kirby.datasets.cross_species_ness as mod
    monkeypatch.setenv('KIRBY_CROSS_SPECIES_DIR', str(tmp_path))
    monkeypatch.setattr(mod, '_load_oas_species', _mock_loader_oas)
    monkeypatch.setattr(mod, '_load_canine', _mock_loader_dog)
    monkeypatch.setattr(mod, '_load_bovine_sra_assembly', _mock_loader_cow)
    monkeypatch.setattr(mod, '_assign_vgene_groups', _mock_anarci_groups)
    with pytest.raises(ValueError, match='must appear in source_species'):
        load_cross_species_ness(
            target_species='dog', source_species=['human', 'mouse'])


def test_load_cross_species_ness_rejects_cat_target():
    # Lives before any data is touched, so no mocks needed.
    with pytest.raises(ValueError, match='deferred to v2'):
        load_cross_species_ness(target_species='cat')


def test_load_cross_species_ness_rejects_bad_mode():
    with pytest.raises(ValueError, match="'one_vs_rest' or 'multinomial'"):
        load_cross_species_ness(target_species='dog', mode='garbage')


def test_load_canine_prefers_doggifai_when_both_present(tmp_path):
    from kirby.datasets.cross_species_ness import _load_canine

    dog_dir = tmp_path / 'canine_antibodies'
    dog_dir.mkdir(parents=True)
    # DoggifAI canonical filename
    (dog_dir / 'heavy.fasta').write_text(
        '>doggifai_1\nQVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMS\n')
    # SRA-assembled fallback also present
    (dog_dir / 'canine_vh.fasta').write_text(
        '>sra_1\nEVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMS\n'
        '>sra_2\nQVQLVESGGGLVQPGGSLRLSCVASGFSFSAYAMS\n')

    seqs, prov = _load_canine(tmp_path)
    # DoggifAI wins — 1 sequence, not 2.
    assert len(seqs) == 1
    assert prov == ['dog']
    # Content matches DoggifAI input.
    assert 'QVQLVES' in seqs[0]


def test_load_canine_falls_back_to_sra_when_only_sra_present(tmp_path, capsys):
    from kirby.datasets.cross_species_ness import _load_canine

    dog_dir = tmp_path / 'canine_antibodies'
    dog_dir.mkdir(parents=True)
    (dog_dir / 'canine_vh.fasta').write_text(
        '>sra_1\nEVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMS\n'
        '>sra_2\nQVQLVESGGGLVQPGGSLRLSCVASGFSFSAYAMS\n')

    seqs, prov = _load_canine(tmp_path)
    assert len(seqs) == 2
    assert prov == ['dog'] * 2
    out = capsys.readouterr().out
    assert 'source=sra_PRJNA790470' in out


def test_load_canine_doggifai_rename_variants_match(tmp_path):
    from kirby.datasets.cross_species_ness import _load_canine

    dog_dir = tmp_path / 'canine_antibodies'
    dog_dir.mkdir(parents=True)
    # The user renamed the DoggifAI file.
    (dog_dir / 'heavy_chains_doggifai.fasta').write_text(
        '>variant_1\nQVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMS\n')

    seqs, prov = _load_canine(tmp_path)
    assert len(seqs) == 1
    assert prov == ['dog']


def test_load_canine_ignores_canine_vh_in_variant_glob(tmp_path):
    # Make sure the heavy*-glob does NOT pick up canine_vh.fasta when
    # DoggifAI is absent — the SRA fallback branch should handle that file.
    from kirby.datasets.cross_species_ness import _load_canine

    dog_dir = tmp_path / 'canine_antibodies'
    dog_dir.mkdir(parents=True)
    (dog_dir / 'canine_vh.fasta').write_text('>x\nQVQLVESGGGLV\n')
    # No heavy*.fasta at all -> SRA branch picks up canine_vh.fasta.

    seqs, prov = _load_canine(tmp_path)
    assert len(seqs) == 1
    assert prov == ['dog']


def test_load_canine_error_message_mentions_both_paths(tmp_path):
    from kirby.datasets.cross_species_ness import _load_canine

    dog_dir = tmp_path / 'canine_antibodies'
    dog_dir.mkdir(parents=True)  # empty dir

    with pytest.raises(RuntimeError) as exc_info:
        _load_canine(tmp_path)
    msg = str(exc_info.value)
    assert 'zenodo.org/records/15125375' in msg
    assert 'submit_canine_sra_assembly.sh' in msg
    assert 'data_inventory' in msg.lower()


def test_load_doggifai_alias_still_resolves(tmp_path):
    # Backwards compat: the symbol _load_doggifai must still work.
    from kirby.datasets.cross_species_ness import (
        _load_doggifai, _load_canine,
    )
    assert _load_doggifai is _load_canine


def test_load_cross_species_ness_no_groups_skips_anarci(monkeypatch, tmp_path):
    import kirby.datasets.cross_species_ness as mod
    monkeypatch.setenv('KIRBY_CROSS_SPECIES_DIR', str(tmp_path))
    monkeypatch.setattr(mod, '_load_oas_species', _mock_loader_oas)
    monkeypatch.setattr(mod, '_load_canine', _mock_loader_dog)
    monkeypatch.setattr(mod, '_load_bovine_sra_assembly', _mock_loader_cow)

    def boom(*args, **kwargs):
        raise AssertionError("ANARCI should not be called when assign_groups=False")
    monkeypatch.setattr(mod, '_assign_vgene_groups', boom)

    out = load_cross_species_ness(target_species='dog', assign_groups=False)
    # groups fallback = provenance per row.
    assert len(out['groups']) == len(out['labels'])
    assert set(out['groups'].tolist()) <= set(V1_SUPPORTED_SPECIES)
