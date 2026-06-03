#!/bin/bash
# Monitor SLURM jobs grouped by domain, with progress + current step.
# Run on ARC: bash scripts/monitor_slurm.sh
# Or for a saveable snapshot:  bash scripts/monitor_slurm.sh > /tmp/snap_$(date +%H%M).txt

set -u

# --- Domain mapping --------------------------------------------------------
# Match the dataset name (the part after the cx_/sim_/etc. prefix).
classify() {
    local d="$1"
    case "$d" in
        qm9|logd|caco2|herg|herg_ki|adme_hlm|adme_rlm|adme_mdr1|adme_solubility|esol)
            echo "MOLECULAR" ;;
        kiba|bindingdb_kd|davis|metz)
            echo "DTA" ;;
        ddg_combined|skempi|graphinity_*|flab_thermo_tresanco)
            echo "DDG" ;;
        bh_doyle_plates|reaction_*|suzuki_*|buchwald_*)
            echo "REACTION" ;;
        tcga_*)
            echo "TCGA" ;;
        flab_*|abagym_*|abcov_*|trastuzumab|humanness_*|ginkgo_*)
            echo "ANTIBODY" ;;
        fed_*|federated_*)
            echo "FEDERATED" ;;
        *)
            echo "OTHER" ;;
    esac
}

DOMAIN_ORDER=(MOLECULAR DTA DDG REACTION TCGA ANTIBODY FEDERATED OTHER)

# --- Collect data ----------------------------------------------------------
USER_NAME="${USER}"
TMPFILE=$(mktemp)
ALERTFILE=$(mktemp)
trap 'rm -f $TMPFILE $ALERTFILE' EXIT

# Persisted state for cross-snapshot "stuck" detection. Saved as
# tests/.snapshot_state/jobid.txt with the last-seen (timestamp, pct, fbh_cfg).
# Next snapshot reads, compares, flags "stuck if the inner counter hasn't
# moved across two consecutive snapshots more than 20 minutes apart".
STATE_DIR="tests/.snapshot_state"
mkdir -p "$STATE_DIR" 2>/dev/null
NOW_EPOCH=$(date +%s)

squeue -u "$USER_NAME" -h -o "%i|%j|%T|%M|%L|%R|%S" | while IFS='|' read -r jid name state runtime left node startt; do
    [ "$state" = "RUNNING" ] || continue

    # Strip prefix to get bare dataset name (cx_, sim_, etc.); keep prefix
    # separately so the by-dataset pivot at the bottom can pair cx with sim.
    prefix=$(echo "$name" | sed -nE 's/^(cx|sim|cmp|noi|fed|opig|frm)_.*/\1/p')
    [ -z "$prefix" ] && prefix="other"
    dataset=$(echo "$name" | sed -E 's/^(cx|sim|cmp|noi|fed|opig|frm)_//')
    domain=$(classify "$dataset")

    # Map prefix → tests/results/<category> subdir for disk-based progress.
    case "$prefix" in
        cx)   cat_dir="complexity" ;;
        sim)  cat_dir="similarity" ;;
        noi)  cat_dir="noise" ;;
        fed)  cat_dir="federated" ;;
        opig) cat_dir="opig" ;;
        *)    cat_dir="" ;;
    esac

    # Disk-based experiment progress: count ${dataset}.json files written
    # AFTER the current job's start time. JSON saves are sync'd writes, so
    # this signal bypasses Python stdout buffering. The -newermt filter is
    # intentional: it shows progress made BY THIS RUNNING JOB ONLY (not
    # cumulative progress for the dataset across all waves). The "~" suffix
    # and the legend at the bottom flag this so the count isn't misread.
    disk_done=0
    disk_curr=""
    if [ -n "$cat_dir" ] && [ -d "tests/results/${cat_dir}" ] && [ -n "$startt" ] && [ "$startt" != "N/A" ]; then
        result_files=$(find "tests/results/${cat_dir}" -mindepth 2 -maxdepth 2 \
            -name "${dataset}.json" -type f -newermt "$startt" 2>/dev/null)
        if [ -n "$result_files" ]; then
            disk_done=$(echo "$result_files" | wc -l | tr -d ' ')
            # Latest by mtime → which experiment most recently saved.
            latest=$(echo "$result_files" | xargs -I{} stat -c "%Y {}" {} 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
            if [ -n "$latest" ]; then
                disk_curr=$(basename "$(dirname "$latest")" | grep -oE "^[cs][0-9]+" | tr '[:lower:]' '[:upper:]')
            fi
        fi
    fi

    log="slurm-${jid}.out"
    if [ ! -f "$log" ]; then
        pct="(no log)"
        step="-"
        size="0"
    else
        # Catch [N/M] anywhere on a line, not just at start. Precompute
        # prints "  [N/M] model/...", experiment phase prints things like
        # "[1/3] c01_baselines" without the leading two-space indent.
        pct=$(grep -oE "\[[0-9]+/[0-9]+\]" "$log" 2>/dev/null | tail -1)
        pct="${pct:-???}"
        # Experiment-phase markers in the log (when buffer has flushed):
        #   start of each sub-experiment:  "^EXPERIMENT C05: TITLE"
        #   end of each sub-experiment:    "^  c05 done in 84.2s"
        exp_curr=$(grep -oE "^EXPERIMENT [CS][0-9]+" "$log" 2>/dev/null \
            | tail -1 | awk '{print $2}')
        exp_done=$(grep -cE "^  [cs][0-9]+ done in [0-9.]+s" "$log" 2>/dev/null)
        # Precedence: log-based current marker (most accurate) > disk-based
        # progress (unaffected by stdout buffering) > legacy [exp] fallback.
        if [ -n "$exp_curr" ]; then
            pct="[${exp_curr} ${exp_done}done]"
        elif [ -n "$disk_curr" ]; then
            # "~" suffix: from disk; actual current is at-or-past this experiment.
            pct="[${disk_curr}~ ${disk_done}done]"
        elif [ "$pct" = "???" ] && grep -qE "^=== .* experiments:|^--- Category:" "$log" 2>/dev/null; then
            pct="[exp]"
        fi
        step=$(grep -oE "[a-z_0-9.]+/[a-z_0-9.]+/[a-z_]+/b=[0-9]+/ev=[a-z]+" "$log" 2>/dev/null | tail -1)
        step="${step:--}"
        size=$(stat -c %s "$log" 2>/dev/null || echo 0)

        # --- Health metrics ----------------------------------------------
        # log_age_sec: seconds since last write. >600 = potentially wedged.
        log_age_sec=$(( NOW_EPOCH - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
        # Cache-rejection counters. drift_n grows when stored rep_hashes don't
        # match in-memory; stale_n is the broader "cache rejected" counter.
        stale_n=$(grep -cE "stale\b|will recompute" "$log" 2>/dev/null)
        drift_n=$(grep -c "meta drift" "$log" 2>/dev/null)
        # JSON output count from THIS job's run.
        # json_1h / json_6h: per-dataset (counts overwrites/new files in
        # tests/results/<cat>/*/<dataset>.json). If both are 0 while
        # drift_n keeps climbing, the job is spinning on cache rejections.
        if [ -n "$cat_dir" ] && [ -d "tests/results/${cat_dir}" ]; then
            json_1h=$(find "tests/results/${cat_dir}" -mindepth 2 -maxdepth 2 \
                -name "${dataset}.json" -newermt '1 hour ago' 2>/dev/null | wc -l | tr -d ' ')
            json_6h=$(find "tests/results/${cat_dir}" -mindepth 2 -maxdepth 2 \
                -name "${dataset}.json" -newermt '6 hours ago' 2>/dev/null | wc -l | tr -d ' ')
        else
            json_1h=0
            json_6h=0
        fi

        # --- Cross-snapshot stuck detection -------------------------------
        # State file: $STATE_DIR/$jid.txt holds "<epoch>|<pct>|<step>" of the
        # last snapshot. If the inner counter (pct) hasn't moved since the
        # previous snapshot more than 20 minutes ago, flag as stuck.
        stuck_min=0
        prev="$STATE_DIR/$jid.txt"
        if [ -f "$prev" ]; then
            IFS='|' read -r prev_epoch prev_pct prev_step < "$prev"
            elapsed_since_prev=$(( NOW_EPOCH - prev_epoch ))
            if [ "$prev_pct" = "$pct" ] && [ "$elapsed_since_prev" -gt 1200 ]; then
                stuck_min=$(( elapsed_since_prev / 60 ))
            fi
        fi
        printf "%s|%s|%s\n" "$NOW_EPOCH" "$pct" "$step" > "$prev"

        # --- Alert classification ----------------------------------------
        alert=""
        if [ "$log_age_sec" -gt 600 ]; then
            alert="WEDGED ${log_age_sec}s_idle"
        elif [ "$stuck_min" -gt 0 ]; then
            alert="STUCK ${stuck_min}min_no_progress on ${pct}"
        elif [ "$drift_n" -gt 5 ] && [ "$json_6h" -eq 0 ]; then
            alert="SPINNING ${drift_n}_cache_drifts no_json_in_6h"
        fi
        if [ -n "$alert" ]; then
            printf "%-10s  %-30s  %s\n" "$jid" "$name" "$alert" >> "$ALERTFILE"
        fi
    fi

    # Compute % done if pct is parseable
    if [[ "$pct" =~ \[([0-9]+)/([0-9]+)\] ]]; then
        n="${BASH_REMATCH[1]}"
        m="${BASH_REMATCH[2]}"
        if [ "$m" -gt 0 ]; then
            pcent=$(awk -v n="$n" -v m="$m" 'BEGIN{printf "%.1f", 100*n/m}')
            pct_disp="${pct} (${pcent}%)"
        else
            pct_disp="$pct"
        fi
    else
        pct_disp="$pct"
    fi

    # Human-readable size
    if [ "$size" -ge 1048576 ]; then
        size_h=$(awk -v s="$size" 'BEGIN{printf "%.1fM", s/1048576}')
    elif [ "$size" -ge 1024 ]; then
        size_h=$(awk -v s="$size" 'BEGIN{printf "%.0fK", s/1024}')
    else
        size_h="${size}B"
    fi

    printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n" \
        "$domain" "$jid" "$dataset" "$runtime" "$left" "$pct_disp" "$size_h" "$step" "$prefix" \
        "$log_age_sec" "$stale_n" "$drift_n" "$json_1h" "$json_6h" \
        >> "$TMPFILE"
done

# --- Print grouped by domain ----------------------------------------------
TOTAL=$(wc -l < "$TMPFILE")
ALERT_COUNT=$(wc -l < "$ALERTFILE")
echo "==========================================================================="
echo " SLURM MONITOR  ($(date '+%Y-%m-%d %H:%M'))   running jobs: $TOTAL"
if [ "$ALERT_COUNT" -gt 0 ]; then
    echo "  ⚠  ALERTS: $ALERT_COUNT job(s) need attention — see below"
else
    echo "  ✓ no alerts (all jobs healthy)"
fi
echo "==========================================================================="

# --- Health alerts (only printed if any) ---------------------------------
if [ "$ALERT_COUNT" -gt 0 ]; then
    echo ""
    echo "--- HEALTH ALERTS ($ALERT_COUNT) ---"
    echo "  WEDGED   = log file untouched >10min (job may have hung/crashed silently)"
    echo "  STUCK    = inner progress counter hasn't moved across 2+ snapshots (>20min)"
    echo "  SPINNING = cache rejections accumulating with no JSON saves in 6h"
    echo ""
    printf "%-10s  %-30s  %s\n" "JOBID" "JOB_NAME" "WHY"
    cat "$ALERTFILE" | sort
fi

for domain in "${DOMAIN_ORDER[@]}"; do
    count=$(grep -c "^${domain}|" "$TMPFILE" || true)
    [ "$count" -eq 0 ] && continue

    echo ""
    echo "--- $domain ($count jobs) ---"
    printf "%-10s  %-28s  %-9s  %-20s  %-5s  %-5s  %-5s  %-5s  %-6s  %s\n" \
        "JOBID" "DATASET" "RUNTIME" "PROGRESS" "j1h" "j6h" "drift" "stale" "idle_s" "CURRENT STEP"
    # Columns: $2 jid, $3 dataset, $4 runtime, $6 pct_disp, $13 j1h, $14 j6h,
    # $12 drift, $11 stale, $10 idle_sec, $8 step
    grep "^${domain}|" "$TMPFILE" | sort -t'|' -k3,3 | \
        awk -F'|' '{ printf "%-10s  %-28s  %-9s  %-20s  %-5s  %-5s  %-5s  %-5s  %-6s  %s\n", \
                    $2, $3, $4, $6, $13, $14, $12, $11, $10, $8 }'
done

# --- Per-dataset cx vs sim pivot -------------------------------------------
# For each dataset, show cx and sim side-by-side with progress + elapsed +
# linear-rate ETA. When more than one job exists for a (dataset,prefix) pair
# (e.g. old wave + backfill resubmit), pick the highest job ID (most recent).
echo ""
echo "--- BY DATASET (cx vs sim: progress, elapsed, eta-remaining) ---"
printf "%-32s  %-30s  %-30s\n" "DATASET" "CX" "SIM"
awk -F'|' '
function hours(rt,    np,a,t,b,dd,nc) {
    # squeue %M format depends on duration:
    #   SS          (< 1 min)
    #   MM:SS       (< 1 hour)
    #   HH:MM:SS    (< 1 day)
    #   D-HH:MM:SS  (>= 1 day)
    np=split(rt, a, "-");
    if (np==2) { dd=a[1]; t=a[2]; } else { dd=0; t=a[1]; }
    nc=split(t, b, ":");
    if (nc==1)       return dd*24 + b[1]/3600;             # SS
    else if (nc==2)  return dd*24 + b[1]/60 + b[2]/3600;   # MM:SS
    else             return dd*24 + b[1] + b[2]/60 + b[3]/3600;  # HH:MM:SS
}
function fmt_elapsed(eh) {
    if (eh < 1) return sprintf("%3dm", int(eh*60+0.5));
    else        return sprintf("%3.0fh", eh);
}
function summarize(prog, rt,    nm, parts, n, m, eh, rem, ename, dcnt, tilde) {
    if (prog=="") return "-";
    eh=hours(rt);
    if (prog ~ /\[exp\]/) return sprintf(" [exp]  %s", fmt_elapsed(eh));
    # Experiment-phase: "[C05 4done]" (from log) or "[C05~ 4done]" (from disk)
    if (prog ~ /^\[[CS][0-9]+~? [0-9]+done\]$/) {
        tilde = (prog ~ /~/) ? "~" : ""
        ename=prog; sub(/^\[/, "", ename); sub(/~? .*/, "", ename);
        dcnt=prog;  sub(/^\[[CS][0-9]+~? /, "", dcnt); sub(/done\].*/, "", dcnt);
        return sprintf("%s%s %sdone %s", ename, tilde, dcnt, fmt_elapsed(eh));
    }
    nm=prog; sub(/^\[/, "", nm); sub(/\].*/, "", nm);
    if (split(nm, parts, "/") == 2) {
        n=parts[1]+0; m=parts[2]+0;
        if (n>0 && m>0 && n==m) return sprintf("%5.1f%% %s wrap-up", 100.0, fmt_elapsed(eh));
        if (n>0 && m>0) {
            rem = eh * (m/n - 1);
            if (rem >= 999.5) return sprintf("%5.1f%% %s +%4.0fh*", 100*n/m, fmt_elapsed(eh), rem);
            return sprintf("%5.1f%% %s +%4.0fh", 100*n/m, fmt_elapsed(eh), rem);
        }
    }
    return sprintf("    ? %s", fmt_elapsed(eh));
}
{
    pfx=$9; jid=$2+0; ds=$3; rt=$4; prog=$6;
    if (pfx=="cx"  && jid > cxj[ds]) { cxj[ds]=jid; cxp[ds]=prog; cxr[ds]=rt; ds_seen[ds]=1 }
    if (pfx=="sim" && jid > sj[ds])  { sj[ds]=jid;  sp[ds]=prog;  sr[ds]=rt;  ds_seen[ds]=1 }
}
END {
    for (d in ds_seen) {
        cx=summarize(cxp[d], cxr[d]);
        sm=summarize(sp[d], sr[d]);
        printf "%-32s  %-30s  %-30s\n", d, cx, sm;
    }
}' "$TMPFILE" | sort

echo ""
echo "  Reading:"
echo "    \"30.5% 43h +98h\"   = precompute at 30.5% after 43h elapsed, ~98h remaining."
echo "    \"+####h*\"          = ETA extrapolation exceeds 999h — will time out."
echo "    \"C05 4done 23h\"    = experiment phase, currently in C05, 4 cN done"
echo "                          (parsed from the slurm-<jobid>.out log)."
echo "    \"C05~ 4done 23h\"   = same, but parsed from JSON files written by"
echo "                          THIS JOB ONLY since it started (filtered by"
echo "                          squeue StartTime). Used when stdout is still"
echo "                          buffered. Does NOT include prior-wave results."
echo "  ETA assumes constant cell rate; late-grid cells are usually slower."

echo ""
echo "==========================================================================="
echo " To diff against an earlier snapshot:"
echo "   bash scripts/monitor_slurm.sh > /tmp/snap_now.txt"
echo "   diff /tmp/snap_earlier.txt /tmp/snap_now.txt"
echo "==========================================================================="
