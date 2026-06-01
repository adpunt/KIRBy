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
trap 'rm -f $TMPFILE' EXIT

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
    # AFTER job start in tests/results/${cat_dir}/<expN>/. Bypasses Python
    # stdout buffering — JSON writes are sync'd to disk regardless of
    # whether print() has flushed yet.
    disk_done=0
    disk_curr=""
    if [ -n "$cat_dir" ] && [ -d "tests/results/${cat_dir}" ] && [ -n "$startt" ] && [ "$startt" != "N/A" ]; then
        # `-newermt $startt` filters out stale JSONs from prior waves.
        # If the job has been running, only files written since are counted.
        result_files=$(find "tests/results/${cat_dir}" -mindepth 2 -maxdepth 2 \
            -name "${dataset}.json" -type f -newermt "$startt" 2>/dev/null)
        if [ -n "$result_files" ]; then
            disk_done=$(echo "$result_files" | wc -l | tr -d ' ')
            # Latest by mtime → which experiment most recently saved
            latest=$(echo "$result_files" | xargs -I{} stat -c "%Y {}" {} 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
            if [ -n "$latest" ]; then
                # Extract "c05" or "s08" from the directory name
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

    printf "%s|%s|%s|%s|%s|%s|%s|%s|%s\n" \
        "$domain" "$jid" "$dataset" "$runtime" "$left" "$pct_disp" "$size_h" "$step" "$prefix" \
        >> "$TMPFILE"
done

# --- Print grouped by domain ----------------------------------------------
TOTAL=$(wc -l < "$TMPFILE")
echo "==========================================================================="
echo " SLURM MONITOR  ($(date '+%Y-%m-%d %H:%M'))   running jobs: $TOTAL"
echo "==========================================================================="

for domain in "${DOMAIN_ORDER[@]}"; do
    count=$(grep -c "^${domain}|" "$TMPFILE" || true)
    [ "$count" -eq 0 ] && continue

    echo ""
    echo "--- $domain ($count jobs) ---"
    printf "%-10s  %-30s  %-9s  %-22s  %-7s  %s\n" \
        "JOBID" "DATASET" "RUNTIME" "PROGRESS" "LOG" "CURRENT STEP"
    grep "^${domain}|" "$TMPFILE" | sort -t'|' -k3,3 | \
        awk -F'|' '{ printf "%-10s  %-30s  %-9s  %-22s  %-7s  %s\n", $2, $3, $4, $6, $7, $8 }'
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
echo "  Reading: \"30.5% 43h +98h\" = at 30.5% after 43h elapsed, ~98h remaining."
echo "  \"+####h*\" means the * extrapolation exceeds 999h — will time out."
echo "  ETA assumes constant cell rate; late-grid cells are usually slower."

echo ""
echo "==========================================================================="
echo " To diff against an earlier snapshot:"
echo "   bash scripts/monitor_slurm.sh > /tmp/snap_now.txt"
echo "   diff /tmp/snap_earlier.txt /tmp/snap_now.txt"
echo "==========================================================================="
