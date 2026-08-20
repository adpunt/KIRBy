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
# reuse the recipe machinery verbatim (module-level fns; importing runs no main)
from creative_hybrids_v2 import (
    make_D, desc_blacklist_greedy, select_fine2d, load_pharm3d_full,
    struct_fidelity, FINE_K,
)

TOP_K = 3


def _build_candidate_spaces(p, I, J, mcs):
    """Assemble the recipe's candidate similarity spaces for one dataset and
    return a list of dicts {name, D, struct, readacross}. Pure geometry +
    label-blind selection; y is used ONLY to score read-across, never to build
    or select a space."""
    n = p['n']
    y, groups = p['y'], p['groups']
    coarse = p['ecfp4_count']
    desc = p['desc']
    X_fine_all = np.hstack([p['atompair_count'].astype(np.float64),
                            p['torsion'].astype(np.float64)])

    # label-blind block prep (identical to creative_hybrids_v2.run)
    _blk, _trace, kept = desc_blacklist_greedy(desc, I, J, mcs)
    physchem_bl = desc[:, kept]
    fidx = select_fine2d(X_fine_all, I, J, mcs, FINE_K)
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

    def _add(name, D):
        rho, _neg = simeval.eval_knn_readacross(D, y, groups)
        spaces.append(dict(name=name, D=D, struct=struct_fidelity(D, I, J, mcs),
                           readacross=float(rho)))

    # THE baseline: count alone, Tanimoto (the field standard)
    _add('count_tanimoto[BASELINE]', simeval.dist_tanimoto(coarse))

    for cname, blocks in combos:
        if blocks is None:
            continue
        _add(f'{cname} [equal]', make_D(blocks))
        if len(blocks) > 1:  # weighting is moot for a single block
            w = [0.5] + [1.0] * (len(blocks) - 1)
            _add(f'{cname} [coarse_down x0.5]', make_D(blocks, weights=w))

    return spaces, dict(shape_ok=bool(shape_ok), n_built=int(n_built),
                        n_kept=int(kept.sum()), fine_k=FINE_K)


def _topk(spaces, key, k=TOP_K, exclude=('count_tanimoto[BASELINE]',)):
    """Top-k spaces by `key` (descending), skipping the baseline and any
    single-block count-only space (those are the standard, not a hybrid)."""
    pool = [s for s in spaces if s['name'] not in exclude
            and not s['name'].startswith('count [')
            and not s['name'].startswith('count_only')]
    return sorted(pool, key=lambda s: -s[key])[:k]


def _sweep_row(space, y):
    """Run the ChEMBL-calibrated noise battery on one space's D and summarise."""
    grid = simeval.noise_sweep_scores(space['D'], y)      # {'sig|rate':[det,rec]}
    dets = [v[0] for v in grid.values() if np.isfinite(v[0])]
    recs = [v[1] for v in grid.values() if np.isfinite(v[1])]
    return dict(name=space['name'], struct=space['struct'],
                readacross=space['readacross'],
                detect_mean=float(np.mean(dets)) if dets else float('nan'),
                recovery_mean=float(np.mean(recs)) if recs else float('nan'),
                grid=grid)


def run(dataset='herg_ki'):
    t0 = time.time()
    p = load_prepared(dataset)
    m = np.load(os.path.join(HERE, f'mcs_pairs_{dataset}.npz'))
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)
    y = p['y']

    spaces, meta = _build_candidate_spaces(p, I, J, mcs)
    baseline = next(s for s in spaces if s['name'] == 'count_tanimoto[BASELINE]')

    top_struct = _topk(spaces, 'struct')
    top_activity = _topk(spaces, 'readacross')

    # score baseline + the union of both top-3 sets (dedup by name), once each
    to_score = {baseline['name']: baseline}
    for s in top_struct + top_activity:
        to_score.setdefault(s['name'], s)
    rows = {name: _sweep_row(s, y) for name, s in to_score.items()}

    base = rows[baseline['name']]
    print(f"\n=== {dataset}: noise-cleaning via similarity metric "
          f"(n={p['n']}, shape={'ok' if meta['shape_ok'] else 'MISSING'}) ===")
    print(f"  {'space':40s} {'struct':>7} {'read_rho':>9} "
          f"{'detectAUC':>10} {'recovery':>9} {'d_detect':>9} {'d_recov':>8}")

    def _line(r, tag=''):
        dd = r['detect_mean'] - base['detect_mean']
        dr = r['recovery_mean'] - base['recovery_mean']
        print(f"  {tag+r['name']:40s} {r['struct']:>7.3f} {r['readacross']:>9.3f} "
              f"{r['detect_mean']:>10.4f} {r['recovery_mean']:>9.4f} "
              f"{dd:>+9.4f} {dr:>+8.4f}")

    _line(base)
    print("  -- top-3 by STRUCTURAL fidelity (label-blind selection) --")
    for s in top_struct:
        _line(rows[s['name']], tag='[S] ')
    print("  -- top-3 by ACTIVITY read-across (contrast) --")
    for s in top_activity:
        _line(rows[s['name']], tag='[A] ')

    # verdict: best structure-selected metric vs baseline, on cleaning
    best_s = max(top_struct, key=lambda s: rows[s['name']]['recovery_mean'])
    br = rows[best_s['name']]
    verdict = (f"best structure-selected space '{best_s['name']}': "
               f"recovery {br['recovery_mean']:+.4f} vs baseline "
               f"{base['recovery_mean']:+.4f} ({br['recovery_mean']-base['recovery_mean']:+.4f}); "
               f"detect {br['detect_mean']:.4f} vs {base['detect_mean']:.4f} "
               f"({br['detect_mean']-base['detect_mean']:+.4f})")
    print(f"  VERDICT: {verdict}")
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)

    return dict(dataset=dataset, n=int(p['n']), meta=meta,
                top_struct=[s['name'] for s in top_struct],
                top_activity=[s['name'] for s in top_activity],
                rows=rows, verdict=verdict)


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
    if args.out:
        json.dump(result, open(args.out, 'w'), indent=1, default=float)
        print(f'\nwrote {os.path.basename(args.out)}')


if __name__ == '__main__':
    main()
