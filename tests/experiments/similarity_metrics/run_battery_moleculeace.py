"""SERVER STEP 3 -- the 30-target similarity battery with paired statistics.

Every metric -- classical (representation x coefficient), transformer embedding
(ChemBERTa/MolFormer), and the triplet metric-learner -- is scored the SAME
honest way: 5-fold scaffold-grouped CV, and the battery is computed on the
HELD-OUT fold only. The triplet encoder is trained on the train fold's ECFP4 and
applied to the held-out fold (no leakage). Classical/transformer metrics simply
build their distance matrix on the held-out fold. This makes all arms comparable
and kills same-CV selection optimism (honest-comparison directive).

Aggregation across the 30 targets: per-target rank on each evaluation, mean rank,
paired Wilcoxon of every metric vs ECFP4:Tanimoto on kNN read-across, and SRD.

Output: moleculeace_rows.json (per target/fold/metric) + moleculeace_summary.json.
"""
import os, sys, json, glob, time, warnings
import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simeval as S
from learned_embeddings import train_triplet

DATA = os.path.join(os.environ.get('KIRBY_CACHE_DIR', '.'), 'similarity_metrics', 'sim_data')
N_MAX = 2000
N_FOLDS = 5
BIT_REPS = ['ecfp4', 'ecfp6', 'fcfp4', 'atompair', 'torsion', 'rdkit', 'maccs']
COUNT_REPS = ['ecfp4_count', 'atompair_count']
CONT_REPS = ['desc']
REF = 'ecfp4:tanimoto'


def subsample(y, groups, n_max, seed=0):
    if len(y) <= n_max:
        return np.arange(len(y))
    rs = np.random.RandomState(seed); keep, tot = [], 0
    for g in rs.permutation(np.unique(groups)):
        idx = np.where(groups == g)[0]
        if tot + len(idx) > n_max:
            continue
        keep += idx.tolist(); tot += len(idx)
        if tot >= n_max:
            break
    return np.sort(np.asarray(keep))


def z(X):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    return (X - mu) / sd


def metrics_on_fold(d, idx, te_local, tani_te, emb):
    """Return {metric_name: distance_matrix} for the held-out fold `te_local`
    (indices into the subsample). `emb` maps embedding-name -> (n_sub, d)."""
    out = {}
    for rep in BIT_REPS:
        X = d[rep][idx][te_local].astype(np.float64)
        for cn, cf in S.BIT_COEFFS.items():
            out[f'{rep}:{cn}'] = cf(X)
    for rep in COUNT_REPS:
        X = d[rep][idx][te_local].astype(np.float64)
        for cn, cf in S.COUNT_COEFFS.items():
            out[f'{rep}:{cn}'] = cf(X)
    for rep in CONT_REPS:
        X = z(d[rep][idx][te_local].astype(np.float64))
        for cn, cf in S.CONT_COEFFS.items():
            out[f'{rep}:{cn}'] = cf(X)
    for name, E in emb.items():           # transformer embeddings
        Xe = z(E[te_local])
        out[f'{name}:cosine'] = S.dist_cosine(Xe)
        out[f'{name}:euclidean'] = S.dist_euclidean(Xe)
    return out


def run_target(npz):
    name = os.path.basename(npz)[4:-4]
    d = np.load(npz, allow_pickle=True)
    y, groups, tani = d['y'], d['groups'], d['tanimoto'].astype(np.float64)
    idx = subsample(y, groups, N_MAX)
    y, groups, tani = y[idx], groups[idx], tani[np.ix_(idx, idx)]

    # load cached transformer embeddings if present (subsampled)
    emb = {}
    for which in ('chemberta', 'molformer'):
        p = os.path.join(DATA, f'emb_{name}_{which}.npy')
        if os.path.exists(p):
            emb[which] = np.load(p)[idx]

    base = d['ecfp4'][idx].astype(np.float64)     # triplet encoder input
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(y, y, groups)):
        tani_te = tani[np.ix_(te, te)]
        Dmap = metrics_on_fold(d, idx, te, tani_te, emb)
        # triplet encoders: train on train fold, transform held-out fold
        for mode in ('structural', 'activity'):
            try:
                f = train_triplet(z(base[tr]), y[tr], groups[tr], mode=mode, seed=fold)
                Emb = f(z(base[te]))
                Dmap[f'triplet_{mode}:euclidean'] = S.dist_euclidean(Emb)
            except Exception as e:
                print(f'    triplet {mode} failed: {type(e).__name__}: {e}')
        for m, D in Dmap.items():
            r = S.run_battery(D, y[te], groups[te], tani_te, seed=fold)
            # sigma x rate noise sweep (rankings can invert across regimes)
            r['noise_sweep'] = S.noise_sweep_scores(D, y[te])
            r.update(target=name, fold=fold, metric=m)
            rows.append(r)
        print(f'  {name} fold{fold}: {len(Dmap)} metrics', flush=True)
    return rows


def aggregate(rows):
    tgs = sorted(set(r['target'] for r in rows))
    mets = sorted(set(r['metric'] for r in rows))
    evals = S.EVALS
    # mean over folds -> per (target, metric, eval)
    cell = {}
    for r in rows:
        for ev in evals:
            cell.setdefault((r['target'], r['metric'], ev), []).append(r[ev])
    mean = {k: float(np.nanmean(v)) for k, v in cell.items()}
    # mean rank per metric per eval across targets
    meanrank = {}
    for ev in evals:
        acc = {m: [] for m in mets}
        for t in tgs:
            vals = np.array([mean.get((t, m, ev), np.nan) for m in mets])
            rk = rankdata(-np.where(np.isnan(vals), -np.inf, vals))
            for m, k in zip(mets, rk):
                acc[m].append(k)
        meanrank[ev] = {m: float(np.mean(v)) for m, v in acc.items()}
    # paired Wilcoxon vs REF on knn_rho across targets
    wilc = {}
    ref_by_t = {t: mean.get((t, REF, 'knn_rho'), np.nan) for t in tgs}
    for m in mets:
        if m == REF:
            continue
        pairs = [(mean.get((t, m, 'knn_rho'), np.nan), ref_by_t[t]) for t in tgs]
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        if len(pairs) >= 6:
            a = [x for x, _ in pairs]; b = [x for _, x in pairs]
            try:
                stat, p = wilcoxon(a, b)
                wilc[m] = dict(n=len(pairs), median_delta=float(np.median(np.array(a) - np.array(b))), p=float(p))
            except Exception:
                pass
    # ---- noise sweep: per (sigma|rate), ECFP4 rank + paired Wilcoxon across targets ----
    # collect per-target mean (over folds) detect/recovery for each metric & setting
    settings = sorted({k for r in rows for k in r.get('noise_sweep', {})})
    sweep_cell = {}   # (target, metric, setting, which) -> list over folds
    for r in rows:
        for st, (dv, rv) in r.get('noise_sweep', {}).items():
            sweep_cell.setdefault((r['target'], r['metric'], st, 'd'), []).append(dv)
            sweep_cell.setdefault((r['target'], r['metric'], st, 'r'), []).append(rv)
    sweep_mean = {k: float(np.nanmean(v)) for k, v in sweep_cell.items()}
    noise_sweep = {}
    for st in settings:
        for which, lab in (('d', 'detect'), ('r', 'recovery')):
            # ECFP4 mean rank across targets
            ranks, deltas = [], []
            for t in tgs:
                vals = np.array([sweep_mean.get((t, m, st, which), np.nan) for m in mets])
                rk = rankdata(-np.where(np.isnan(vals), -np.inf, vals))
                ranks.append(float(rk[mets.index(REF)]))
            # best non-ECFP4 metric this setting by mean-over-targets score
            avg = {m: np.nanmean([sweep_mean.get((t, m, st, which), np.nan) for t in tgs]) for m in mets}
            best = max(avg, key=lambda m: (avg[m] if np.isfinite(avg[m]) else -9))
            # paired Wilcoxon best-vs-ECFP4 across targets
            pa = [(sweep_mean.get((t, best, st, which), np.nan),
                   sweep_mean.get((t, REF, st, which), np.nan)) for t in tgs]
            pa = [(a, b) for a, b in pa if np.isfinite(a) and np.isfinite(b)]
            p = None
            if len(pa) >= 6 and best != REF:
                try:
                    p = float(wilcoxon([a for a, _ in pa], [b for _, b in pa]).pvalue)
                except Exception:
                    p = None
            noise_sweep[f'{st}|{lab}'] = dict(
                ecfp4_mean_rank=float(np.mean(ranks)), best=best,
                best_mean=float(avg[best]), ecfp4_mean=float(avg.get(REF, np.nan)),
                wilcoxon_best_vs_ecfp4_p=p)

    return dict(n_targets=len(tgs), targets=tgs, n_metrics=len(mets),
                meanrank_knn=meanrank['knn_rho'], meanrank_bedroc=meanrank['bedroc'],
                wilcoxon_vs_ecfp4_tanimoto_on_knn=wilc,
                noise_sweep=noise_sweep)


def main():
    npzs = sorted(glob.glob(os.path.join(DATA, 'sim_*.npz')))
    print(f'{len(npzs)} targets in {DATA}')
    rows = []
    for p in npzs:
        t = time.time()
        try:
            rows.extend(run_target(p))
            print(f'{os.path.basename(p)} done in {time.time()-t:.0f}s', flush=True)
        except Exception as e:
            print(f'FAIL {p}: {type(e).__name__}: {e}')
    json.dump(rows, open(os.path.join(DATA, 'moleculeace_rows.json'), 'w'))
    summ = aggregate(rows)
    json.dump(summ, open(os.path.join(DATA, 'moleculeace_summary.json'), 'w'), indent=1)
    print('\n=== mean rank on kNN read-across (1=best), top 12 of',
          summ['n_metrics'], 'over', summ['n_targets'], 'targets ===')
    for m, mr in sorted(summ['meanrank_knn'].items(), key=lambda kv: kv[1])[:12]:
        w = summ['wilcoxon_vs_ecfp4_tanimoto_on_knn'].get(m, {})
        tag = f"  Wilcoxon vs ECFP4:Tan Δ={w.get('median_delta',0):+.3f} p={w.get('p',1):.3g}" if w else ''
        print(f'  {m:26s} meanrank={mr:5.2f}{tag}')
    print(f"\n  ECFP4:tanimoto meanrank={summ['meanrank_knn'][REF]:.2f}")
    print('\n=== NOISE SWEEP: ECFP4:Tanimoto rank + best metric per regime ===')
    print(f'  {"sigma|rate|task":22s} {"ECFP4 rank":>10} {"best":>22} {"p(best>ECFP4)":>14}')
    for st in sorted(summ['noise_sweep']):
        v = summ['noise_sweep'][st]
        pv = f"{v['wilcoxon_best_vs_ecfp4_p']:.3g}" if v['wilcoxon_best_vs_ecfp4_p'] is not None else 'n/a'
        print(f"  {st:22s} {v['ecfp4_mean_rank']:>10.1f} {v['best']:>22} {pv:>14}")


if __name__ == '__main__':
    main()
