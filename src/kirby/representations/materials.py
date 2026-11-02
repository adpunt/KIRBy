"""
Materials Representation Generators

Base representations for inorganic crystals and compositions (Matbench).
Each create_* function takes a list of pymatgen Composition or Structure
objects and returns a float32 matrix (n_compounds, n_features). With
return_names=True it returns (matrix, feature_names) instead.

Missing or infinite values are set to 0, as for the molecular descriptors.

COMPOSITION (every task; matminer):
- magpie:             ElementProperty, 'magpie' preset (element-property statistics)
- stoichiometry:      Stoichiometry (p-norms of the element fractions)
- valence_orbital:    ValenceOrbital (s/p/d/f valence electron counts and fractions)
- ion_property:       IonProperty (charge balance, ionic character)

STRUCTURE (structure tasks only; matminer and dscribe):
- sine_coulomb:       SineCoulombMatrix, sorted eigenvalues, zero-padded
- soap_avg:           dscribe SOAP averaged over sites
- site_stats_fingerprint: SiteStatsFingerprint, 'CrystalNNFingerprint_ops' preset
- density_features:   DensityFeatures (density, volume per atom, packing fraction)
- rdf:                RadialDistributionFunction
- xrd:                XRDPowderPattern
- global_symmetry:    GlobalSymmetryFeatures (numeric features only)

PRETRAINED (structure tasks only; frozen):
- chgnet:             CHGNet crystal feature vector (64)
- mace_mp:            MACE-MP-0 'medium' invariant descriptors, averaged over atoms

MEGNet / M3GNet are left out (author's decision, 2026-10-26). Two reasons:
- Leakage. MEGNet's published pretrained models include one trained to
  predict shear modulus from Materials Project data. matbench_log_gvrh is
  shear modulus from Materials Project, so that model has probably been
  trained on the log_gvrh test compounds. Unchecked: the overlap is inferred
  from where both datasets come from, not counted compound by compound.
- It does not install here. matgl 1.x needs DGL, whose macOS library does not
  load against torch 2.2.2 (missing symbol at::internal::invoke_parallel);
  matgl 2.x needs torch 2.4 or later (2026-10-23).
"""

import numpy as np

# Single process. matminer's worker pool made every featurizer slower on this
# machine (density_features: 0.002 s per compound alone, 0.33 s in the pool;
# measured 2026-10-23). Parallelism comes from building several (task,
# representation) pairs at once instead (tests/tools/matbench_build_reps.py).
MATMINER_N_JOBS = 1


def _clean(X):
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    return X.astype(np.float32)


def _featurize(featurizer, entries, n_jobs=MATMINER_N_JOBS):
    """Run a matminer featurizer row by row. A failed row is all NaN (then 0)."""
    featurizer.set_n_jobs(n_jobs)
    rows = featurizer.featurize_many(entries, ignore_errors=True, pbar=False)
    width = len(featurizer.feature_labels())
    out = np.full((len(entries), width), np.nan)
    for i, r in enumerate(rows):
        try:
            out[i] = np.asarray(r, dtype=float)
        except (TypeError, ValueError):
            pass
    return out


def _result(X, names, return_names):
    X = _clean(X)
    return (X, list(names)) if return_names else X


# =============================================================================
# COMPOSITION
# =============================================================================

def create_magpie(compositions, return_names=False):
    from matminer.featurizers.composition import ElementProperty
    f = ElementProperty.from_preset('magpie')
    return _result(_featurize(f, compositions), f.feature_labels(), return_names)


def create_stoichiometry(compositions, return_names=False):
    from matminer.featurizers.composition import Stoichiometry
    f = Stoichiometry()
    return _result(_featurize(f, compositions), f.feature_labels(), return_names)


def create_valence_orbital(compositions, return_names=False):
    from matminer.featurizers.composition import ValenceOrbital
    f = ValenceOrbital()
    return _result(_featurize(f, compositions), f.feature_labels(), return_names)


def create_ion_property(compositions, return_names=False):
    from matminer.featurizers.composition import IonProperty
    f = IonProperty()
    return _result(_featurize(f, compositions), f.feature_labels(), return_names)


# =============================================================================
# STRUCTURE
# =============================================================================

def create_sine_coulomb(structures, return_names=False):
    from matminer.featurizers.structure import SineCoulombMatrix
    f = SineCoulombMatrix(flatten=True)  # sorted eigenvalues, padded
    f.fit(structures)
    return _result(_featurize(f, structures), f.feature_labels(), return_names)


# SOAP settings: r_cut 5 A, n_max 6, l_max 4, Gaussian width 1 A, periodic,
# 'inner' averaging (average the density over sites, then expand), species =
# every element present in the dataset. Uncompressed SOAP grows with the square
# of the species count; dielectric and log_gvrh carry ~80 elements, which gives
# ~700,000 features per compound. 'mu1nu1' compression keeps one element index
# per power-spectrum term, so the width is (species x n_max^2 x (l_max+1)):
# 80 elements -> 14,400 features.
SOAP_SETTINGS = dict(r_cut=5.0, n_max=6, l_max=4, sigma=1.0,
                     periodic=True, average='inner',
                     compression={'mode': 'mu1nu1'})


def create_soap_avg(structures, return_names=False, n_jobs=MATMINER_N_JOBS):
    from dscribe.descriptors import SOAP
    from pymatgen.io.ase import AseAtomsAdaptor
    species = sorted({el.symbol for s in structures for el in s.composition})
    soap = SOAP(species=species, **SOAP_SETTINGS)
    atoms = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    X = soap.create(atoms, n_jobs=n_jobs)
    names = [f'soap_{i}' for i in range(X.shape[1])]
    return _result(X, names, return_names)


def create_site_stats_fingerprint(structures, return_names=False):
    from matminer.featurizers.structure import SiteStatsFingerprint
    f = SiteStatsFingerprint.from_preset('CrystalNNFingerprint_ops')
    return _result(_featurize(f, structures), f.feature_labels(), return_names)


def create_density_features(structures, return_names=False):
    from matminer.featurizers.structure import DensityFeatures
    f = DensityFeatures()
    return _result(_featurize(f, structures), f.feature_labels(), return_names)


def create_rdf(structures, return_names=False):
    from matminer.featurizers.structure import RadialDistributionFunction
    f = RadialDistributionFunction()
    return _result(_featurize(f, structures), f.feature_labels(), return_names)


def create_xrd(structures, return_names=False):
    from matminer.featurizers.structure import XRDPowderPattern
    f = XRDPowderPattern()
    return _result(_featurize(f, structures), f.feature_labels(), return_names)


def create_global_symmetry(structures, return_names=False):
    # 'crystal_system' is a string; its integer code crystal_system_int is kept.
    from matminer.featurizers.structure import GlobalSymmetryFeatures
    f = GlobalSymmetryFeatures(desired_features=[
        'spacegroup_num', 'crystal_system_int', 'is_centrosymmetric',
        'n_symmetry_ops'])
    return _result(_featurize(f, structures), f.feature_labels(), return_names)


# =============================================================================
# PRETRAINED
# =============================================================================

def create_chgnet(structures, return_names=False, batch_size=64):
    from chgnet.model import CHGNet
    model = CHGNet.load()
    X = np.zeros((len(structures), 64))
    for start in range(0, len(structures), batch_size):
        chunk = structures[start:start + batch_size]
        preds = model.predict_structure(chunk, return_crystal_feas=True,
                                        batch_size=batch_size)
        if isinstance(preds, dict):
            preds = [preds]
        for j, p in enumerate(preds):
            X[start + j] = p['crystal_fea']
    names = [f'chgnet_{i}' for i in range(X.shape[1])]
    return _result(X, names, return_names)


def create_mace_mp(structures, return_names=False):
    from mace.calculators import mace_mp
    from pymatgen.io.ase import AseAtomsAdaptor
    calc = mace_mp(model='medium', device='cpu', default_dtype='float64')
    rows = []
    for s in structures:
        atoms = AseAtomsAdaptor.get_atoms(s)
        d = calc.get_descriptors(atoms, invariants_only=True)
        rows.append(np.asarray(d).mean(axis=0))
    X = np.vstack(rows)
    names = [f'mace_mp_{i}' for i in range(X.shape[1])]
    return _result(X, names, return_names)


# =============================================================================
# REGISTRY
# =============================================================================

COMPOSITION_REPS = {
    'magpie': create_magpie,
    'stoichiometry': create_stoichiometry,
    'valence_orbital': create_valence_orbital,
    'ion_property': create_ion_property,
}

STRUCTURE_REPS = {
    'sine_coulomb': create_sine_coulomb,
    'soap_avg': create_soap_avg,
    'site_stats_fingerprint': create_site_stats_fingerprint,
    'density_features': create_density_features,
    'rdf': create_rdf,
    'xrd': create_xrd,
    'global_symmetry': create_global_symmetry,
    'chgnet': create_chgnet,
    'mace_mp': create_mace_mp,
}

MATERIALS_REPS = {**COMPOSITION_REPS, **STRUCTURE_REPS}
