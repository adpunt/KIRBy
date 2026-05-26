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

squeue -u "$USER_NAME" -h -o "%i|%j|%T|%M|%L|%R" | while IFS='|' read -r jid name state runtime left node; do
    [ "$state" = "RUNNING" ] || continue

    # Strip prefix to get bare dataset name (cx_, sim_, etc.)
    dataset=$(echo "$name" | sed -E 's/^(cx|sim|cmp|noi|fed|opig|frm)_//')
    domain=$(classify "$dataset")

    log="slurm-${jid}.out"
    if [ ! -f "$log" ]; then
        pct="(no log)"
        step="-"
        size="0"
    else
        pct=$(grep -oE "^  \[[0-9]+/[0-9]+\]" "$log" 2>/dev/null | tail -1)
        pct="${pct:-???}"
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

    printf "%s|%s|%s|%s|%s|%s|%s|%s\n" \
        "$domain" "$jid" "$dataset" "$runtime" "$left" "$pct_disp" "$size_h" "$step" \
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

echo ""
echo "==========================================================================="
echo " To diff against an earlier snapshot:"
echo "   bash scripts/monitor_slurm.sh > /tmp/snap_now.txt"
echo "   diff /tmp/snap_earlier.txt /tmp/snap_now.txt"
echo "==========================================================================="
