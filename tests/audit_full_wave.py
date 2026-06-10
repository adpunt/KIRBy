#!/usr/bin/env python3
"""Definitive per-job audit across the 2026-05-08 wave.

For every COMPLETED job since 2026-05-08 (cx_*, sim_*, AND exp_%j):
  - Reads the slurm log honestly:
      clean_finish      = "ALL DONE" and no bus error
      finished_then_bus = "ALL DONE" present but bus error afterwards
      bus_mid_run       = bus error, no "ALL DONE"
      oom / python_error / unknown / no_log otherwise
  - Counts expected JSONs with mtime >= job start (so a stale Apr-29 file
    or a duplicate-job overwrite from another wave can't credit this job).
  - Emits a verdict: OK / PARTIAL / EMPTY.

The (category, dataset) for exp_%j jobs is recovered from the log banner
'=== <category> experiments: <dataset> ===' that submit_dataset.sh prints.

Run on the server:
    cd /data/stat-ecr/scat9264/KIRBy
    python tests/audit_full_wave.py [--since 2026-05-08]

Writes a sortable TSV alongside the printed table:
    tests/results/_audit_full_wave.tsv
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

EXPECTED = {
    # M1: c12 (modality_ablation) MUST be listed so it is counted. It only
    # applies to tcga_* datasets; the per-job filter below drops it for the
    # rest. Omitting it here made the `e == 'c12'` filter dead, so c12 was
    # NEVER counted (even for tcga_*), producing a persistent sub-full count.
    'complexity': ['c01', 'c02', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10',
                   'c11', 'c12', 'c13', 'c15', 'c16', 'c17', 'c18', 'c19', 'c21',
                   'c22', 'c23', 'c24'],
    'similarity': ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
                   's10', 's11', 's12', 's13'],
    'noise': ['n01', 'n02', 'n03', 'n04', 'n05', 'n07', 'n08', 'n09',
              'n10', 'n11', 'n12', 'n13'],
    'federated': ['fed01', 'fed02', 'fed03', 'fed04', 'fed05'],
}

DIR_MAP = {
    'complexity': {
        'c01': 'c01_rep_characterization',
        'c02': 'c02_complexity_matching_grid',
        'c04': 'c04_importance_as_metric',
        'c05': 'c05_learning_curves',
        'c06': 'c06_within_family_gradient',
        'c07': 'c07_data_complexity_partitions',
        'c08': 'c08_calibration',
        'c09': 'c09_formal_analysis',
        'c10': 'c10_noise_robustness',
        'c11': 'c11_ensemble_vs_hybrid',
        'c12': 'c12_modality_ablation',
        'c13': 'c13_forward_compatibility',
        'c15': 'c15_ml_theory_depth',
        'c16': 'c16_feature_selection_as_regularization',
        'c17': 'c17_information_theoretic_decomposition',
        'c18': 'c18_diversity_vs_depth',
        'c19': 'c19_importance_weighted_training',
        'c21': 'c21_single_rep_augmentation',
        'c22': 'c22_explainability',
        'c23': 'c23_evaluator_model_matching',
        'c24': 'c24_blueprint_transfer',
    },
    'similarity': {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's07': 's07_noise_bridge',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    },
    'noise': {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n03': 'n03_sas_landscape',
        'n04': 'n04_gp_loo_detection',
        'n05': 'n05_noise_injection_benchmark',
        'n07': 'n07_remediation',
        'n08': 'n08_estimation_survey',
        'n09': 'n09_lnl_survey',
        'n10': 'n10_hybrid_neighborhood_lnl',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
    },
    'federated': {
        'fed01': 'fed01_noise_heterogeneity',
        'fed02': 'fed02_rep_diversity_silos',
        'fed03': 'fed03_synthetic_private',
        'fed04': 'fed04_cluster_aware_importance',
        'fed05': 'fed05_antibody_federated',
    },
}

PREFIX2CAT = {'cx': 'complexity', 'sim': 'similarity',
              'nz': 'noise', 'fed': 'federated'}


def classify_log(log_path: Path):
    """Return (status, n_done_from_banner, progress_marker, position_dict).

    position_dict has numeric snapshots useful for ETA: {
        'fold_pos': int, 'fold_total': int,
        'config_pos': int, 'config_total': int,
        'last_exp_done': str,   # e.g. 'c15'
        'saves': int,           # count of "[saved ...]" lines in tail
        'imports_secs': float,  # time for 'Importance pre-computed in X.Ys' if found
        'alloc_secs': float,    # time for 'Allocation scores pre-computed in X.Ys' if found
    }
    """
    if not log_path.exists():
        return 'no_log', None, '', {}
    try:
        txt = log_path.read_text(errors='ignore')
    except Exception:
        return 'unreadable_log', None, '', {}
    has_done = 'ALL DONE' in txt
    has_bus = 'Bus error' in txt
    has_oom = bool(re.search(r'OOMKilled|MemoryError|Out of memory', txt, re.IGNORECASE))
    has_trace = 'Traceback (most recent call last)' in txt
    m = re.search(r'ALL DONE:\s+\S+\s+[—-]\s+(\d+)\s+experiments', txt)
    n_done = int(m.group(1)) if m else None

    # Progress marker: the last meaningful stage line in the log.
    # Useful for RUNNING jobs that aren't yet COMPLETED.
    progress = ''
    # Iterate over last ~200 lines, looking for the latest progress signal
    tail_lines = txt.splitlines()[-200:]
    # Look for, in order of specificity:
    # - "ALL DONE" — finished
    # - " done in " — per-experiment done line (e.g. "  c05 done in 1234.5s")
    # - "Loading data" / "Precomputing hybrid searches" / "Importance pre-computed"
    # - "fold 0: NNN/MMMM" — adaptive pruning progress
    # - "[saved tests/results/.../xxx.json]"
    saved_files = []
    last_exp_done = ''
    last_fold = ''
    last_stage = ''
    last_config = ''
    fold_pos = fold_total = 0
    config_pos = config_total = 0
    # Scan the ENTIRE log for timing markers, not just the tail, so we can
    # extract intra-job stage durations like "Importance pre-computed in 387.2s"
    imports_secs = alloc_secs = 0.0
    for line in txt.splitlines():
        s = line.strip()
        m_imp = re.search(r'Importance pre-computed in\s+([\d.]+)s', s)
        if m_imp:
            imports_secs = float(m_imp.group(1))
        m_alloc = re.search(r'Allocation scores pre-computed in\s+([\d.]+)s', s)
        if m_alloc:
            alloc_secs = float(m_alloc.group(1))
    for line in tail_lines:
        s = line.strip()
        if s.startswith('[saved '):
            saved_files.append(s)
        m2 = re.match(r'^(c\d{2}|s\d{2}|n\d{2}|fed\d{2}) done in', s)
        if m2:
            last_exp_done = m2.group(1)
        m3 = re.search(r'fold \d+:\s*(\d+)/(\d+)', s)
        if m3:
            last_fold = m3.group(0)
            fold_pos, fold_total = int(m3.group(1)), int(m3.group(2))
        # Second precompute stage: per-config eval lines like
        # "[2435/2980] mlp/entropy_2.0/treeshap/b=100/ev=mlp: score=..."
        m_cfg = re.match(r'^\[(\d+)/(\d+)\]\s+\w+/', s)
        if m_cfg:
            last_config = f'config {m_cfg.group(1)}/{m_cfg.group(2)}'
            config_pos, config_total = int(m_cfg.group(1)), int(m_cfg.group(2))
        if 'Precomputing hybrid searches' in s:
            last_stage = 'precompute'
        if 'Pre-computing importance' in s:
            last_stage = 'precompute_importance'
        if 'Allocation scores pre-computed' in s:
            last_stage = 'alloc_done'
        if 'Adaptive pruning' in s:
            last_stage = 'pruning'
        if 'Loading' in s and 'reps' in s:
            last_stage = 'loading'
    bits = []
    if last_exp_done:
        bits.append(f'last_done={last_exp_done}')
    if last_fold:
        bits.append(last_fold)
    if last_config:
        bits.append(last_config)
    if last_stage:
        bits.append(last_stage)
    if saved_files:
        bits.append(f'+{len(saved_files)}saves')
    progress = ' | '.join(bits)

    positions = {
        'fold_pos': fold_pos, 'fold_total': fold_total,
        'config_pos': config_pos, 'config_total': config_total,
        'last_exp_done': last_exp_done, 'saves': len(saved_files),
        'imports_secs': imports_secs, 'alloc_secs': alloc_secs,
    }

    if has_done and not has_bus:
        return 'clean_finish', n_done, progress, positions
    if has_done and has_bus:
        return 'finished_then_bus', n_done, progress, positions
    if has_bus:
        return 'bus_mid_run', n_done, progress, positions
    if has_oom:
        return 'oom', n_done, progress, positions
    if has_trace:
        return 'python_error', n_done, progress, positions
    if not progress:
        return 'unknown', n_done, progress, positions
    return 'in_progress', n_done, progress, positions


def get_job_meta(jobid: str):
    out = subprocess.check_output(
        ['sacct', '-j', str(jobid),
         '--format=JobName%40,State,Start,End,Elapsed,ExitCode',
         '-X', '-P', '-n'],
        text=True).strip().split('\n')[0].split('|')
    return dict(zip(['name', 'state', 'start', 'end', 'elapsed', 'exit'], out))


def parse_exp_banner(log_path: Path):
    """For exp_%j jobs, recover (category, dataset) from the run banner."""
    if not log_path.exists():
        return None, None
    try:
        head = log_path.read_text(errors='ignore')[:8000]
    except Exception:
        return None, None
    m = re.search(r'===\s+(\w+)\s+experiments:\s+(\S+)\s+===', head)
    if m:
        cat = m.group(1).lower()
        if cat in EXPECTED:
            return cat, m.group(2)
    return None, None


SLURM_WALL_HOURS = 7 * 24  # submit_dataset.sh uses --time=7-00:00:00


def _h(seconds: float) -> str:
    """Format seconds as 'Xd Yh' or 'Yh Zm'."""
    if seconds < 0:
        return f'-{_h(-seconds)}'
    h = seconds / 3600
    if h >= 24:
        d = int(h // 24); rh = int(h - d * 24)
        return f'{d}d{rh}h'
    if h >= 1:
        m = int((h - int(h)) * 60)
        return f'{int(h)}h{m}m'
    return f'{int(seconds / 60)}m'


def estimate_eta(elapsed_secs: float, positions: dict, log_status: str,
                 n_expected: int):
    """Estimate time to per-experiment loop start and overall wall risk.

    Returns dict with:
        stage:          best guess at current stage
        precompute_eta: seconds to finish precompute (None if can't estimate)
        wall_eta:       seconds until SLURM wall (7d limit)
        verdict:        ON_TRACK / SLOW / WILL_HIT_WALL / DONE_PRECOMPUTE / N/A
        note:           short text reason
    """
    wall_eta = SLURM_WALL_HOURS * 3600 - elapsed_secs
    out = {'stage': '?', 'precompute_eta': None,
           'wall_eta': wall_eta, 'verdict': 'N/A', 'note': ''}

    # Already done with precompute (experiments are saving)
    if positions.get('saves', 0) > 0 or positions.get('last_exp_done'):
        out['stage'] = 'experiments'
        # Estimate from saves so far if any
        saves = positions['saves'] or 0
        if saves > 0:
            per_exp = elapsed_secs / saves   # rough — includes precompute time
            remaining_exp = max(0, n_expected - saves)
            remaining_secs = remaining_exp * per_exp * 0.5  # post-precompute is faster
            out['precompute_eta'] = 0
            if remaining_secs > wall_eta:
                out['verdict'] = 'WILL_HIT_WALL'
                out['note'] = f'{saves} saved, ~{_h(per_exp/2)}/exp expected, {remaining_exp} left'
            else:
                out['verdict'] = 'ON_TRACK'
                out['note'] = f'{saves}/{n_expected} saved, ~{_h(remaining_secs)} left'
        else:
            out['verdict'] = 'ON_TRACK'
            out['note'] = 'in experiments loop'
        return out

    # In second-stage config eval
    config_pos = positions.get('config_pos', 0)
    config_total = positions.get('config_total', 0)
    fold_pos = positions.get('fold_pos', 0)
    fold_total = positions.get('fold_total', 0)

    if config_pos > 0 and config_total > 0:
        out['stage'] = f'config_eval {config_pos}/{config_total}'
        frac = config_pos / config_total
        if frac > 0.02:  # need at least 2% to extrapolate
            # ASSUMPTION: this config phase took roughly all elapsed_secs minus
            # importance+alloc precompute. That's an upper bound.
            precomp_overhead = (positions.get('imports_secs', 0)
                                + positions.get('alloc_secs', 0))
            config_elapsed = max(elapsed_secs - precomp_overhead, elapsed_secs * 0.7)
            rate = config_pos / config_elapsed   # configs/sec
            remaining_configs = config_total - config_pos
            remaining_secs = remaining_configs / rate if rate > 0 else float('inf')
            # Then we'll need experiment time — estimate from precompute cost
            # (per_exp ≈ 1-3h based on observed completed jobs)
            est_exp_secs = n_expected * 1800  # 30 min/exp lower bound
            total_remaining = remaining_secs + est_exp_secs
            out['precompute_eta'] = int(remaining_secs)
            if total_remaining > wall_eta:
                margin = wall_eta - total_remaining
                out['verdict'] = 'WILL_HIT_WALL'
                out['note'] = (f'rate={rate*3600:.1f} cfg/h, '
                               f'~{_h(remaining_secs)} cfg + '
                               f'~{_h(est_exp_secs)} exps; over wall by {_h(-margin)}')
            elif total_remaining > wall_eta * 0.7:
                out['verdict'] = 'SLOW'
                out['note'] = (f'~{_h(remaining_secs)} cfg + '
                               f'~{_h(est_exp_secs)} exps; '
                               f'tight (wall left: {_h(wall_eta)})')
            else:
                out['verdict'] = 'ON_TRACK'
                out['note'] = f'~{_h(remaining_secs)} cfg + ~{_h(est_exp_secs)} exps'
        return out

    if fold_pos > 0 and fold_total > 0:
        out['stage'] = f'fold0_pruning {fold_pos}/{fold_total}'
        frac = fold_pos / fold_total
        if frac > 0.02:
            rate = fold_pos / elapsed_secs  # configs/sec for fold-0 sweep
            remaining_in_fold = fold_total - fold_pos
            secs_to_finish_fold = remaining_in_fold / rate
            # After fold 0 there's still config-eval phase (~similar size)
            # plus per-experiment loop. Total: ~3x fold-0 time as rough upper bound.
            total_remaining = secs_to_finish_fold * 3
            out['precompute_eta'] = int(secs_to_finish_fold)
            if total_remaining > wall_eta:
                margin = wall_eta - total_remaining
                out['verdict'] = 'WILL_HIT_WALL'
                out['note'] = (f'fold-0 rate={rate*3600:.0f} cfg/h, '
                               f'fold-0 ETA {_h(secs_to_finish_fold)}, '
                               f'projected total {_h(total_remaining)}; '
                               f'over wall by {_h(-margin)}')
            elif total_remaining > wall_eta * 0.7:
                out['verdict'] = 'SLOW'
                out['note'] = (f'fold-0 ETA {_h(secs_to_finish_fold)}, '
                               f'projected total {_h(total_remaining)}, '
                               f'wall left {_h(wall_eta)}')
            else:
                out['verdict'] = 'ON_TRACK'
                out['note'] = f'fold-0 ETA {_h(secs_to_finish_fold)}'
        return out

    # No quantitative signal — can only report stage
    out['stage'] = 'precompute (no quant signal)'
    out['note'] = 'no fold/config position; cannot estimate'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-05-08',
                    help='sacct --starttime (default: 2026-05-08)')
    ap.add_argument('--include-running', action='store_true',
                    help='Also include RUNNING jobs in the audit (cannot have OK yet)')
    ap.add_argument('--eta', action='store_true',
                    help='For RUNNING jobs, estimate time to finish based on '
                         'progress rate; flag WILL_HIT_WALL / SLOW / ON_TRACK')
    args = ap.parse_args()

    # Pull all jobs matching prefixes since SINCE
    states_filter = 'COMPLETED'
    if args.include_running:
        states_filter = 'COMPLETED,RUNNING'
    user = os.environ.get('USER', '')
    cmd = ['sacct', '-u', user, '-S', args.since,
           '--format=JobID,JobName%25,State', '-X', '-P', '-n']
    raw = subprocess.check_output(cmd, text=True)
    jobs = []
    for line in raw.strip().split('\n'):
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 3:
            continue
        jid, name, state = parts[0], parts[1], parts[2]
        if state not in states_filter.split(','):
            continue
        cat = ds = None
        m = re.match(r'^(cx|sim|nz|fed)_(.+)$', name)
        if m:
            cat = PREFIX2CAT[m.group(1)]
            ds = m.group(2)
        elif name == 'exp_%j':
            cat, ds = parse_exp_banner(ROOT / f'slurm-{jid}.out')
        if cat is None or ds is None:
            continue
        if cat not in EXPECTED:
            continue
        jobs.append((jid, name, state, cat, ds))

    rows = []
    print(f'{"JOBID":>10}  {"WAVE":4}  {"CAT":4}  {"DATASET":35}  '
          f'{"STATE":10}  {"LOG_STATUS":18}  {"FRESH":>5}  {"ERR":>3}  '
          f'{"MISS":>4}  {"EXP":>3}  {"ELAPSED":>11}  verdict | progress')
    print('-' * 180)
    summary = {}
    for jid, name, state, cat, ds in sorted(jobs, key=lambda x: int(x[0])):
        meta = get_job_meta(jid)
        try:
            start_ts = datetime.fromisoformat(meta['start']).timestamp()
        except Exception:
            start_ts = 0
        log_status, n_done, progress, positions = classify_log(
            ROOT / f'slurm-{jid}.out')
        expected = [e for e in EXPECTED[cat]
                    if not (cat == 'complexity' and e == 'c12'
                            and not ds.startswith('tcga_'))]
        fresh_ids, errored_ids, stale_ids, missing_ids = [], [], [], []
        for exp_id in expected:
            d = DIR_MAP[cat][exp_id]
            p = ROOT / 'tests' / 'results' / cat / d / f'{ds}.json'
            if not p.exists():
                missing_ids.append(exp_id)
                continue
            if p.stat().st_mtime < start_ts:
                stale_ids.append(exp_id)
                continue
            try:
                obj = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(obj, dict) and 'error' in obj:
                errored_ids.append(exp_id)
            else:
                fresh_ids.append(exp_id)
        fresh, errored, stale, missing = (len(fresh_ids), len(errored_ids),
                                          len(stale_ids), len(missing_ids))
        total = len(expected)
        if state == 'RUNNING':
            verdict = 'RUNNING'
        elif fresh == total:
            verdict = 'OK'
        elif fresh == 0:
            verdict = 'EMPTY'
        else:
            verdict = 'PARTIAL'
        # Wave: 1 = 12103xxx, 2 = 12111xxx, 3 = 12115xxx, else *
        wave = '1' if jid.startswith('12103') else \
               '2' if jid.startswith('12111') else \
               '3' if jid.startswith('12115') else '?'
        summary[verdict] = summary.get(verdict, 0) + 1
        elapsed = meta.get('elapsed', '')
        # Build detail string showing exactly which experiments errored/missing
        detail = []
        if errored_ids:
            detail.append(f'errored={",".join(errored_ids)}')
        if missing_ids and verdict != 'EMPTY':
            detail.append(f'missing={",".join(missing_ids)}')
        suffix = (' || ' + ' '.join(detail)) if detail else ''
        prog_suffix = f' || prog: {progress}' if progress else ''
        print(f'{jid:>10}  W{wave:3}  {cat[:4]:4}  {ds:35}  '
              f'{state:10}  {log_status:18}  {fresh:>5}  {errored:>3}  '
              f'{missing:>4}  {total:>3}  {elapsed:>11}  {verdict}'
              f'{prog_suffix}{suffix}')
        # ── ETA analysis (only meaningful for RUNNING jobs) ──
        eta = {'verdict': 'N/A', 'stage': '', 'note': '',
               'precompute_eta': None, 'wall_eta': None}
        if state == 'RUNNING':
            try:
                start_dt = datetime.fromisoformat(meta['start'])
                elapsed_secs = (datetime.now() - start_dt).total_seconds()
            except Exception:
                elapsed_secs = 0
            eta = estimate_eta(elapsed_secs, positions, log_status, total)

        rows.append({
            'jobid': jid, 'wave': wave, 'jobname': name, 'state': state,
            'category': cat, 'dataset': ds, 'log_status': log_status,
            'progress': progress, 'errored_ids': ','.join(errored_ids),
            'missing_ids': ','.join(missing_ids),
            'n_done_banner': n_done, 'fresh_files': fresh,
            'errored_files': errored, 'stale_files': stale,
            'missing_files': missing, 'expected_files': total,
            'elapsed': elapsed, 'verdict': verdict,
            'eta_verdict': eta['verdict'], 'eta_stage': eta['stage'],
            'eta_note': eta['note'],
            'eta_precompute_secs': eta['precompute_eta'],
            'eta_wall_secs': eta['wall_eta'],
        })
    print()
    print('Verdict counts:', summary)

    # Write TSV
    out_tsv = ROOT / 'tests' / 'results' / '_audit_full_wave.tsv'
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cols = ['jobid', 'wave', 'jobname', 'state', 'category', 'dataset',
            'log_status', 'progress', 'errored_ids', 'missing_ids',
            'n_done_banner', 'fresh_files', 'errored_files',
            'stale_files', 'missing_files', 'expected_files', 'elapsed',
            'verdict', 'eta_verdict', 'eta_stage', 'eta_note',
            'eta_precompute_secs', 'eta_wall_secs']
    with out_tsv.open('w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')
    print(f'\nTSV written to: {out_tsv}')

    # Per (category, dataset): which datasets have NO OK across any job?
    by_ds = {}
    for r in rows:
        if r['verdict'] == 'RUNNING':
            continue
        key = (r['category'], r['dataset'])
        by_ds.setdefault(key, []).append(r)
    no_ok = []
    for key, jr in sorted(by_ds.items()):
        if not any(r['verdict'] == 'OK' for r in jr):
            best = max((r['fresh_files'] for r in jr), default=0)
            no_ok.append((key, best, jr[0]['expected_files']))
    if no_ok:
        print('\n=== Datasets with NO fully-OK completion across all waves ===')
        for (cat, ds), best, total in no_ok:
            print(f'  {cat}/{ds}: best fresh = {best}/{total}')

    # ── ETA report ──
    if args.eta:
        running_rows = [r for r in rows if r['state'] == 'RUNNING']
        print(f'\n=== ETA analysis ({len(running_rows)} RUNNING jobs) ===')
        print('Wall = 7 days. Estimate is rough (assumes current rate continues, '
              'adds margin for per-experiment loop).')
        print()
        by_verdict = {}
        for r in running_rows:
            by_verdict.setdefault(r['eta_verdict'], []).append(r)
        for verdict in ['WILL_HIT_WALL', 'SLOW', 'ON_TRACK', 'N/A']:
            rows_v = by_verdict.get(verdict, [])
            if not rows_v: continue
            print(f'─── {verdict} ({len(rows_v)}) ───')
            for r in sorted(rows_v, key=lambda x: (x['category'], x['dataset'])):
                wall_secs = r['eta_wall_secs'] or 0
                print(f"  {r['jobid']:>10}  W{r['wave']}  {r['category'][:4]:4}  "
                      f"{r['dataset']:35}  wall_left={_h(wall_secs):>7}  "
                      f"{r['eta_stage']:30}  {r['eta_note']}")
            print()


if __name__ == '__main__':
    main()
