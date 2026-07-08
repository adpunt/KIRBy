#!/bin/bash
# Preview/cancel down to the reduced keep-set (2026-07-08).
#
# KEEPS:
#   * complexity / similarity / noise / simhyb jobs for the KEEP_DS datasets
#   * ALL opig_* jobs           (own datasets — keep regardless)
#   * ALL federated jobs        (keep regardless of dataset)
#   * ALL blueprint-trace jobs  (bp_trace — keep)
# CANCELS only:
#   * complexity / similarity / noise / simhyb jobs whose dataset is NOT in KEEP_DS
#
# Resolves each exp_<id> job's real (category, dataset) from its Command via
# scontrol (the name doesn't carry it). simhyb_/opig_ jobs are read from the name.
#
# DRY RUN by default — prints the KEEP/CANCEL classification and cancels NOTHING.
# To actually cancel the CANCEL set:  bash cancel_to_keepset.sh --go
set -uo pipefail
U=scat9264

# ── keep-set (20) ────────────────────────────────────────────────────────────
KEEP_DS=(
  # molecules
  adme_mdr1 adme_hlm caco2 herg_ki qm9
  # antibodies
  flab_expression_koenig ddg_combined abagym_EGFR abagym_HER2 abcov_kd
  ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho
  ginkgo_acsins ginkgo_colloidal ginkgo_purity ginkgo_aggregation
  # other domains
  nt_promoter_tata gb_human_enhancers_cohn
)
POTENTIAL_DS=(caco2 herg_ki qm9)          # flagged "potentially" — kept, highlighted
KEEP_CATS="complexity similarity noise"    # simhyb handled by name

GO=0; [[ "${1:-}" == "--go" ]] && GO=1

in_list() { local x=$1; shift; local e; for e in "$@"; do [[ "$e" == "$x" ]] && return 0; done; return 1; }

keep=(); cancel=(); unknown=()

while read -r jid name; do
  [[ -z "${jid:-}" ]] && continue
  if [[ "$name" == opig* ]]; then
    keep+=("$jid|$name|OPIG (kept)"); continue
  fi
  if [[ "$name" == bp_trace* || "$name" == *blueprint_trace* ]]; then
    keep+=("$jid|$name|blueprint trace (kept)"); continue
  fi
  if [[ "$name" == simhyb_* ]]; then
    ds="${name#simhyb_}"
    if in_list "$ds" "${KEEP_DS[@]}"; then
      p=""; in_list "$ds" "${POTENTIAL_DS[@]}" && p=" *potential*"
      keep+=("$jid|$name|simhyb $ds$p")
    else cancel+=("$jid|$name|simhyb $ds (dropped)"); fi
    continue
  fi
  # exp_<id> (or anything else): resolve category+dataset from the Command args
  cmd=$(scontrol show job "$jid" 2>/dev/null | grep -oP 'Command=\K\S.*' | head -1)
  read -r _script c ds _rest <<< "$cmd"
  if [[ -z "${ds:-}" || -z "${c:-}" ]]; then
    unknown+=("$jid|$name|UNPARSED: ${cmd:-<none>}"); continue
  fi
  if [[ "$c" == "federated" ]]; then
    keep+=("$jid|$name|federated $ds (kept)")
  elif in_list "$ds" "${KEEP_DS[@]}" && in_list "$c" $KEEP_CATS; then
    p=""; in_list "$ds" "${POTENTIAL_DS[@]}" && p=" *potential*"
    keep+=("$jid|$name|$c $ds$p")
  else
    cancel+=("$jid|$name|$c $ds (dropped)")
  fi
done < <(squeue -h -u "$U" -o "%i %j")

pr() { local arr=("$@"); printf '  %-12s %-16s %s\n' "JOBID" "NAME" "WHAT";
       for r in "${arr[@]}"; do IFS='|' read -r a b c <<< "$r"; printf '  %-12s %-16s %s\n' "$a" "$b" "$c"; done; }

echo "=================================================================="
echo "CANCEL-TO-KEEPSET  ($([[ $GO == 1 ]] && echo '*** WILL CANCEL ***' || echo 'DRY RUN — nothing cancelled'))"
echo "=================================================================="
echo; echo "### KEEP (${#keep[@]}) — left running:"; [[ ${#keep[@]} -gt 0 ]] && pr "${keep[@]}"
echo; echo "### CANCEL (${#cancel[@]}) — will be scancel'd with --go:"; [[ ${#cancel[@]} -gt 0 ]] && pr "${cancel[@]}"
if [[ ${#unknown[@]} -gt 0 ]]; then
  echo; echo "### UNKNOWN (${#unknown[@]}) — could NOT resolve; NOT cancelled, review manually:"; pr "${unknown[@]}"
fi
echo

if [[ $GO == 1 ]]; then
  ids=(); for r in "${cancel[@]}"; do ids+=("${r%%|*}"); done
  if [[ ${#ids[@]} -gt 0 ]]; then echo "Cancelling ${#ids[@]} jobs..."; scancel "${ids[@]}" && echo "done."; else echo "Nothing to cancel."; fi
else
  echo "DRY RUN. Re-run with --go to cancel the ${#cancel[@]} CANCEL jobs. UNKNOWN jobs are never auto-cancelled."
fi
