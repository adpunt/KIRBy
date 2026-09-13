"""Close the noise-cleaning loop with the similarity-objective metric.

QUESTION (the user's): does a similarity metric CHOSEN for structural fidelity
clean noisy activity labels better than the count-fingerprint standard -- and
better than a metric chosen for activity coherence?

METHOD. For each dataset we take the recipe's candidate stacked similarity
spaces (count / +physchem_blacklisted / +shape / +fine2D, in equal and
coarse-down weightings -- the exact combos from creative_hybrids_v2), turn each
into a Euclidean distance matrix D, and rank them TWO ways:

  * by label-blind STRUCTURAL fidelity  = Spearman(D_pairs, MCS)   [no y]
  * by ACTIVITY read-across             = scaffold-grouped kNN rho [uses y]

We take the TOP-3 of EACH ranking and run every distance matrix through the
ChEMBL-calibrated synthetic-noise battery (simeval.noise_sweep_scores):

  detection AUROC  -- does a corrupted label disagree with its neighbours in
                      this metric's space?  (eval_noise_detect)
  recovery corr    -- does distance-weighted kNN pull the label back toward
                      truth, i.e. corr(applied correction, -injected eps)?
                      (eval_noise_recovery)

across sigma in {0.3,0.6,0.9} x rate in {0.10,0.20,0.30}, averaged over 3
injection seeds. The count-Tanimoto fingerprint is the baseline to beat.

MOLECULE-HELD-OUT SCORING (the fix this file exists for). Every fitted step --
the drop-one descriptor blacklist, the fine-2D column selection, and the
per-feature z-score mean/sd -- used to see all n molecules and then be graded on
all n molecules. winning_recipe_honest.py splits the MCS PAIRS 50/50, which is
not enough: both pair halves are drawn from the same molecules, so the selection
has still seen every molecule it is scored on. Here the MOLECULES are split
50/50 by whole Bemis-Murcko scaffold group (same convention as
_data._scaffold_subsample: sorted unique groups, fixed-seed permutation, whole
groups kept together), giving a FIT half and a HELD-OUT half:

  FIT half    the descriptor drop, the fine-2D selection and the z-score
              mean/sd are fitted here, and the top-3 rankings are taken here.
  HELD-OUT    structural fidelity, read-across rho and the whole noise
              detection/recovery battery are scored here, on the D submatrix
              cut to the held-out rows, over only those cached MCS pairs whose
              BOTH endpoints are held out.

Both arms are reported, and BOTH SCORE THE SAME MOLECULES. The `insample` arm
fits and scores on the held-out half (706 of herg_ki's 1415), so it is the
fit-equals-score number on exactly the molecules the `heldout` arm reports on.
It used to fit and score on all 1415, which made the difference between the two
arms a difference in molecule count as well as a difference in what the fitting
had seen: the count-Tanimoto baseline fits nothing and selects nothing, yet it
moved 0.0324 detection AUROC between the arms while the four fitted spaces
moved 0.0120 to 0.0177. On the same molecules that movement has nowhere to come
from, so whatever is left of it is what the fitting bought.

EVERY DATASET IS READ WHOLE (user decision, 2026-09-11). _data.load_prepared
cuts logd to a 1427-molecule scaffold subsample by default, to save time, and
time is not a reason to cut: run() loads with cap=None, so logd is scored on
all 5039 molecules. The cached MCS pairs are row numbers into the molecule list
they were drawn from, so the uncapped read needs the uncapped pair cache --
mcs_cache.load_pairs(..., whole=True), which refuses a file whose stored
molecule list disagrees with the one in hand. That cache is built: 8000 pairs
over all 5039 uncapped logd molecules, 2026-09-12, 1327 seconds. The three
capped caches were rebuilt the same day -- the files dated 3 September recorded
no molecule list at all, so the loader refused them and five scripts failed at
the cache read.

The 3D pharmacophore cache is the other thing indexed on a molecule list.
sim_logd_spatial.npz held 1427 molecules while the uncapped read needs 5039, and
load_pharm3d_full refuses to subset rather than quietly serving a smaller
candidate set, so the two candidate spaces carrying the shape block were not
built at all. It was rebuilt over all 5039 on 2026-09-12 -- 9881 seconds, no
failed conformers in any of the three descriptors -- and both reads are served
from it now. run() still prints and stores `shape_note` if a future molecule
list moves out from under it.

WHY THIS IS HONEST (not the n08/n10 circularity). The STRUCTURE ranking uses no
activity labels at all -- saboteur-drop and fine-2D selection are chosen against
MCS overlap, the stacking is label-free -- so using a structure-selected metric
to expose activity-noise is not the metric grading its own homework. The
ACTIVITY ranking is included precisely as the contrast that WOULD be circular if
it won; if structure-selected >= activity-selected on cleaning, the win is real.

LIGHT: distance matrices + kNN + Spearman only. No find_best_hybrid, no model
fit. Runs in the similarity study's self-contained row space, reusing the
precomputed MCS + 3D caches -- no server precompute.

  python similarity_noise_cleaning.py                       # herg_ki
  python similarity_noise_cleaning.py --dataset logd
  python similarity_noise_cleaning.py --all --out similarity_noise_cleaning.json
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import simeval
from _data import load_prepared
from _nanrank import rank_desc, is_defined
# one definition of the molecule split, shared with the other scripts that hold
# molecules out (feature_ablation, winning_recipe_*). ranker_experiment was
# deleted 2026-09-11 -- it chose and scored features on the same molecules.
from _heldout import (SPLIT_SEED, scaffold_halves as _scaffold_halves,
                      pairs_within as _pairs_within, zscore_fit as _zscore_fit)
# reuse the recipe machinery verbatim (module-level fns; importing runs no main)
from creative_hybrids_v2 import (
    desc_blacklist_greedy, select_fine2d, load_pharm3d_full,
    struct_fidelity, FINE_K,
)

TOP_K = 3
MIN_HELDOUT_PAIRS = 300   # below this the held-out structural fidelity is too
                          # thin to report as a bare number; say so instead


# The molecule split (_scaffold_halves), the pair cut (_pairs_within) and the
# fit-half z-score (_zscore_fit) now live in _heldout.py, unchanged, so
# feature_ablation.py and the two winning_recipe scripts cut their molecules
# exactly the same way. SPLIT_SEED comes from there too. ranker_experiment.py
# was on this list until it was deleted on 2026-09-11.


def _make_D_fit(blocks, fit_idx, weights=None):
    """make_D with the z-score statistics taken from the FIT rows only. The
    equal_block scaling (/sqrt(d_block)) and the optional per-block weight are
    identical to creative_hybrids_v2.make_D; only the mean/sd source differs.
    Returns the FULL (n, n) Euclidean distance matrix -- the caller cuts it."""
    parts = []
    for k, X in enumerate(blocks):
        Z = _zscore_fit(X, fit_idx)
        Z = Z / np.sqrt(Z.shape[1])
        w = 1.0 if weights is None else float(weights[k])
        parts.append(Z * w)
    return simeval.dist_euclidean(np.hstack(parts))


# ------------------------------------------------------------ candidate spaces
def _build_candidate_spaces(p, fit_idx, fit_pairs, score_idx, score_pairs):
    """Assemble the recipe's candidate similarity spaces for one dataset and
    return a list of dicts {name, D, struct, readacross, struct_fit,
    readacross_fit}. Pure geometry + label-blind selection; y is used ONLY to
    score read-across, never to build or select a space.

      fit_idx      molecule rows the descriptor drop, the fine-2D selection and
                   the z-score mean/sd are fitted on.
      fit_pairs    (I, J, mcs) re-indexed into fit_idx positions; the ranking
                   numbers (struct_fit / readacross_fit) are measured here.
      score_idx    molecule rows every REPORTED number is measured on. Each D is
                   cut to D[np.ix_(score_idx, score_idx)] before scoring.
      score_pairs  (I, J, mcs) re-indexed into score_idx positions.

    Passing fit_idx == score_idx makes a fit-equals-score arm; the in-sample arm
    passes the HELD-OUT rows for both, so it fits and scores on the same
    molecules the held-out arm is scored on."""
    n = p['n']
    coarse = p['ecfp4_count']
    desc = p['desc']
    X_fine_all = np.hstack([p['atompair_count'].astype(np.float64),
                            p['torsion'].astype(np.float64)])

    Ift, Jft, Mft = fit_pairs
    Isc, Jsc, Msc = score_pairs
    y_ft, g_ft = p['y'][fit_idx], p['groups'][fit_idx]
    y_sc, g_sc = p['y'][score_idx], p['groups'][score_idx]

    # label-blind block prep (same calls as creative_hybrids_v2.run) -- fitted on
    # the FIT rows only: desc[fit_idx] means the greedy's internal z-score sees
    # fit molecules only, and the pair indices are fit-local.
    _blk, trace, kept = desc_blacklist_greedy(desc[fit_idx], Ift, Jft, Mft)
    physchem_bl = desc[:, kept]
    fidx = select_fine2d(X_fine_all[fit_idx], Ift, Jft, Mft, FINE_K)
    fine = X_fine_all[:, fidx]
    pharm, n_built = load_pharm3d_full(dataset=p['name'], n=n)
    shape_ok = pharm is not None

    combos = [
        ('count',                          [coarse]),
        ('count+physchem_bl',              [coarse, physchem_bl]),
        ('count+shape',                    ([coarse, pharm] if shape_ok else None)),
        ('count+physchem_bl+shape+fine2D', ([coarse, physchem_bl, pharm, fine] if shape_ok else None)),
    ]

    spaces = []
    fit_cut = np.ix_(fit_idx, fit_idx)
    score_cut = np.ix_(score_idx, score_idx)
    same_half = np.array_equal(fit_idx, score_idx)

    def _add(name, D):
        Ds = D[score_cut]
        rho, _neg = simeval.eval_knn_readacross(Ds, y_sc, g_sc)
        # readacross / readacross_fit stay NaN when their Spearman is undefined.
        # The *_defined flags separate that from a measured rho of 0, and _topk
        # ranks through rank_desc so a NaN cannot occupy a top slot.
        rec = dict(name=name, D=Ds, struct=struct_fidelity(Ds, Isc, Jsc, Msc),
                   readacross=float(rho), readacross_defined=is_defined(rho))
        if same_half:
            rec['struct_fit'] = rec['struct']
            rec['readacross_fit'] = rec['readacross']
            rec['readacross_fit_defined'] = rec['readacross_defined']
        else:
            Df = D[fit_cut]
            rho_f, _negf = simeval.eval_knn_readacross(Df, y_ft, g_ft)
            rec['struct_fit'] = struct_fidelity(Df, Ift, Jft, Mft)
            rec['readacross_fit'] = float(rho_f)
            rec['readacross_fit_defined'] = is_defined(rho_f)
        spaces.append(rec)

    # THE baseline: count alone, Tanimoto (the field standard). Tanimoto is a
    # pure pairwise function of the raw counts -- no columns are selected and no
    # mean/sd is estimated -- so its held-out number is leak-free by
    # construction and is identical whether or not the molecules are split.
    _add('count_tanimoto[BASELINE]', simeval.dist_tanimoto(coarse))

    for cname, blocks in combos:
        if blocks is None:
            continue
        _add(f'{cname} [equal]', _make_D_fit(blocks, fit_idx))
        if len(blocks) > 1:  # weighting is moot for a single block
            w = [0.5] + [1.0] * (len(blocks) - 1)
            _add(f'{cname} [coarse_down x0.5]', _make_D_fit(blocks, fit_idx, weights=w))

    return spaces, dict(shape_ok=bool(shape_ok), n_built=int(n_built),
                        n_kept=int(kept.sum()), fine_k=FINE_K,
                        desc_struct_fit=trace['base_struct'],
                        desc_struct_fit_after=trace['final_struct'],
                        n_fit_mols=int(len(fit_idx)), n_fit_pairs=int(len(Ift)),
                        n_score_mols=int(len(score_idx)),
                        n_score_pairs=int(len(Isc)))


def _topk(spaces, key, k=TOP_K, exclude=('count_tanimoto[BASELINE]',)):
    """Top-k spaces by `key` (descending), skipping the baseline and any
    single-block count-only space (those are the standard, not a hybrid)."""
    pool = [s for s in spaces if s['name'] not in exclude
            and not s['name'].startswith('count [')
            and not s['name'].startswith('count_only')]
    # -s[key] left a non-finite score wherever it started, so it could hold a top
    # slot. rank_desc sends non-finite scores to the end; reverse=True is stable,
    # so the finite ordering is otherwise unchanged.
    return sorted(pool, key=lambda s: rank_desc(s[key]), reverse=True)[:k]


def _sweep_row(space, y):
    """Run the ChEMBL-calibrated noise battery on one space's D and summarise.
    `y` must be the activity of the SAME rows D was cut to."""
    grid = simeval.noise_sweep_scores(space['D'], y)      # {'sig|rate':[det,rec]}
    dets = [v[0] for v in grid.values() if np.isfinite(v[0])]
    recs = [v[1] for v in grid.values() if np.isfinite(v[1])]
    return dict(name=space['name'], struct=space['struct'],
                readacross=space['readacross'],
                readacross_defined=space['readacross_defined'],
                struct_fit=space['struct_fit'],
                readacross_fit=space['readacross_fit'],
                readacross_fit_defined=space['readacross_fit_defined'],
                detect_mean=float(np.mean(dets)) if dets else float('nan'),
                recovery_mean=float(np.mean(recs)) if recs else float('nan'),
                grid=grid)


def _score_arm(p, fit_idx, fit_pairs, score_idx, score_pairs):
    """Build the candidate spaces for one arm, take the top-3 of each ranking
    (ranked on the FIT half), and run the noise battery on the baseline plus the
    union of both top-3 sets, scored on the SCORE half."""
    spaces, meta = _build_candidate_spaces(p, fit_idx, fit_pairs,
                                           score_idx, score_pairs)
    baseline = next(s for s in spaces if s['name'] == 'count_tanimoto[BASELINE]')
    top_struct = _topk(spaces, 'struct_fit')
    top_activity = _topk(spaces, 'readacross_fit')

    # score baseline + the union of both top-3 sets (dedup by name), once each
    to_score = {baseline['name']: baseline}
    for s in top_struct + top_activity:
        to_score.setdefault(s['name'], s)
    y_sc = p['y'][score_idx]
    rows = {name: _sweep_row(s, y_sc) for name, s in to_score.items()}

    base = rows[baseline['name']]
    # recovery_mean is NaN when the sweep produced no finite recovery score, and
    # max() returns NaN whenever a NaN is the running maximum. rank_desc places
    # non-finite recovery last so it cannot be picked as the best space.
    best_s = max(top_struct, key=lambda s: rank_desc(rows[s['name']]['recovery_mean']))
    br = rows[best_s['name']]
    verdict = (f"best structure-selected space '{best_s['name']}': "
               f"recovery {br['recovery_mean']:+.4f} vs baseline "
               f"{base['recovery_mean']:+.4f} ({br['recovery_mean']-base['recovery_mean']:+.4f}); "
               f"detect {br['detect_mean']:.4f} vs {base['detect_mean']:.4f} "
               f"({br['detect_mean']-base['detect_mean']:+.4f})")
    return dict(meta=meta, baseline_name=baseline['name'],
                top_struct=[s['name'] for s in top_struct],
                top_activity=[s['name'] for s in top_activity],
                rows=rows, verdict=verdict)


def _print_arm(title, arm):
    base = arm['rows'][arm['baseline_name']]
    print(f"  -- {title} --")
    print(f"  {'space':56s} {'struct':>7} {'read_rho':>9} "
          f"{'detectAUC':>10} {'recovery':>9} {'d_detect':>9} {'d_recov':>8}")

    def _line(r, tag=''):
        dd = r['detect_mean'] - base['detect_mean']
        dr = r['recovery_mean'] - base['recovery_mean']
        print(f"  {tag+r['name']:56s} {r['struct']:>7.3f} {r['readacross']:>9.3f} "
              f"{r['detect_mean']:>10.4f} {r['recovery_mean']:>9.4f} "
              f"{dd:>+9.4f} {dr:>+8.4f}")

    _line(base)
    print("  top-3 by STRUCTURAL fidelity (label-blind selection):")
    for name in arm['top_struct']:
        _line(arm['rows'][name], tag='[S] ')
    print("  top-3 by ACTIVITY read-across (contrast):")
    for name in arm['top_activity']:
        _line(arm['rows'][name], tag='[A] ')
    print(f"  VERDICT: {arm['verdict']}")


def run(dataset='herg_ki'):
    t0 = time.time()
    # cap=None: every dataset is read whole, so logd is scored on all 5039
    # molecules rather than the 1427-molecule subsample the default cap keeps.
    p = load_prepared(dataset, cap=None)
    n = p['n']
    # the pair positions belong to the uncapped molecule list; load_pairs
    # refuses a cache built from another one instead of silently pairing each
    # distance with the wrong molecule.
    from mcs_cache import load_pairs
    m = load_pairs(dataset, p['smiles'], whole=True)
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)
    pairs_id = str(m['smiles_id'])   # which molecule list these positions mean

    fit_idx, held_idx = _scaffold_halves(p['groups'])
    fit_pairs = _pairs_within(I, J, mcs, fit_idx, n)
    held_pairs = _pairs_within(I, J, mcs, held_idx, n)
    n_held_pairs = int(len(held_pairs[0]))

    n_fit_scaf = len(set(p['groups'][fit_idx].tolist()))
    n_held_scaf = len(set(p['groups'][held_idx].tolist()))
    print(f"\n=== {dataset}: noise-cleaning via similarity metric ===")
    print(f"  {p['size_label']}")
    print(f"  molecule split by scaffold group (seed {SPLIT_SEED}): "
          f"fit {len(fit_idx)} mols / {n_fit_scaf} scaffolds, "
          f"held-out {len(held_idx)} mols / {n_held_scaf} scaffolds")
    print(f"  MCS pairs {len(I)} total -> {len(fit_pairs[0])} both-endpoints-fit, "
          f"{n_held_pairs} both-endpoints-held-out "
          f"({len(I) - len(fit_pairs[0]) - n_held_pairs} straddling, dropped)")

    pairs_note = None
    if n_held_pairs < MIN_HELDOUT_PAIRS:
        pairs_note = (f'ONLY {n_held_pairs} MCS pairs have both endpoints in the '
                      f'held-out half (< {MIN_HELDOUT_PAIRS}); both arms score '
                      f'those molecules, so every structural fidelity below is '
                      f'measured on that many pairs')
        print(f"  WARNING: {pairs_note}")

    # HELD-OUT arm: everything fitted on the fit half, everything reported on the
    # held-out half. IN-SAMPLE arm: fitted AND reported on that same held-out
    # half, so the two arms score the same molecules and differ only in whether
    # the descriptor drop, the fine-2D selection and the z-score mean/sd had
    # already seen them.
    heldout = _score_arm(p, fit_idx, fit_pairs, held_idx, held_pairs)
    insample = _score_arm(p, held_idx, held_pairs, held_idx, held_pairs)
    n_scored = int(len(held_idx))

    shape_note = None
    if not heldout['meta']['shape_ok']:
        # sim_<ds>_spatial.npz covers n_built molecules and load_pharm3d_full
        # refuses to subset, so the two combinations carrying the 3D
        # pharmacophore block are not built at all. That is a smaller candidate
        # set, not a missing print: say so next to the numbers.
        shape_note = (f"the 3D pharmacophore block covers "
                      f"{heldout['meta']['n_built']} molecules and this run "
                      f"holds {n}, so 'count+shape' and "
                      f"'count+physchem_bl+shape+fine2D' are absent from both "
                      f"rankings; rebuild sim_{dataset}_spatial.npz at {n} "
                      f"molecules to score them")
        print(f"  WARNING: {shape_note}")
    print(f"  shape block: {'ok' if heldout['meta']['shape_ok'] else 'MISSING'}  "
          f"desc kept {heldout['meta']['n_kept']}/208 (fit half) vs "
          f"{insample['meta']['n_kept']}/208 (held-out half)")
    print(f"  both arms scored on the same {n_scored} molecules "
          f"({n_held_pairs} MCS pairs)")
    _print_arm('HELD-OUT molecules (fitted on the other half)', heldout)
    _print_arm(f'IN-SAMPLE, fit == score, same {n_scored} molecules', insample)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)

    return dict(dataset=dataset, n=int(n), n_total=int(p['n_total']),
                capped=bool(p['capped']), size_label=p['size_label'],
                n_scaffolds=int(p['n_scaffolds']),
                n_scored_mols=dict(heldout=n_scored, insample=n_scored),
                split_seed=SPLIT_SEED,
                n_fit_mols=int(len(fit_idx)), n_heldout_mols=int(len(held_idx)),
                n_fit_scaffolds=int(n_fit_scaf), n_heldout_scaffolds=int(n_held_scaf),
                n_pairs_total=int(len(I)), pairs_smiles_id=pairs_id,
                n_pairs_fit=int(len(fit_pairs[0])),
                n_pairs_heldout=n_held_pairs,
                heldout_pairs_note=pairs_note,
                shape_note=shape_note,
                heldout=dict(meta=heldout['meta'],
                             top_struct=heldout['top_struct'],
                             top_activity=heldout['top_activity'],
                             rows=heldout['rows'], verdict=heldout['verdict']),
                insample=dict(meta=insample['meta'],
                              top_struct=insample['top_struct'],
                              top_activity=insample['top_activity'],
                              rows=insample['rows'], verdict=insample['verdict']),
                verdict=heldout['verdict'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--all', action='store_true',
                    help='run herg_ki, logd, adme_solubility')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    datasets = ['herg_ki', 'logd', 'adme_solubility'] if args.all else [args.dataset]
    result = {}
    for ds in datasets:
        try:
            result[ds] = run(ds)
        except FileNotFoundError as e:
            print(f'[{ds}] SKIPPED (missing file): {e}', flush=True)
            result[ds] = {'dataset': ds, 'error': str(e)}
        except ValueError as e:
            # load_pairs refusing a cache drawn from other molecules. One
            # dataset's stale cache must not take the other two down with it,
            # and the reason has to survive into the merged file.
            print(f'[{ds}] SKIPPED (pair cache): {e}', flush=True)
            result[ds] = {'dataset': ds, 'error': str(e)}
    if args.out:
        json.dump(result, open(args.out, 'w'), indent=1, default=float)
        print(f'\nwrote {os.path.basename(args.out)}')


if __name__ == '__main__':
    main()
