#!/bin/bash
# READ-ONLY. Says whether `git pull` here is safe, and what it would change.
# Fetches (which only writes to .git) and reports. Pulls nothing, deletes
# nothing, changes no file in the working tree.
#
#   bash tests/server_pull_check.sh                      # checks the current branch
#   bash tests/server_pull_check.sh similarity-metrics-study
#
# WHY THIS EXISTS
# ---------------
# Two things make a careless pull expensive in this repo.
#
# 1. tests/results/ is TRACKED — 193 files — and so are the slurm-*.out logs.
#    Experiment output lives in the repo. So `git reset --hard`, `git checkout .`
#    and `git clean -fd` do not "tidy the checkout": they delete results and job
#    logs that no cache holds a copy of. Never run those here.
#
# 2. The work is on a branch. where_to_submit.sh documents
#    `git pull --ff-only origin main`, and a pull of main gets none of it while
#    reporting success.
#
# A pull only refuses, or overwrites, when an INCOMING commit touches a file
# that is locally modified. That is the one thing worth checking, and it is what
# the last section below does.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"

echo "===== pull check — $(date) ====="
echo
echo "----- where you are -----"
echo "  branch : $(git rev-parse --abbrev-ref HEAD)"
echo "  commit : $(git log -1 --format='%h %ad %s' --date=short)"
echo "  asking about origin/${BRANCH}"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "${BRANCH}" ]; then
    echo "  !! you are on a different branch from the one you are pulling."
    echo "     Switching branches with modified files is its own hazard —"
    echo "     read the 'locally modified' section below first."
fi

echo
echo "----- fetching (writes only to .git) -----"
git fetch origin "${BRANCH}" 2>&1 | sed 's/^/  /'
if ! git rev-parse --verify -q "origin/${BRANCH}" >/dev/null; then
    echo "  !! origin/${BRANCH} does not exist. Has it been pushed?"
    exit 1
fi

echo
echo "----- what is coming -----"
_n=$(git rev-list --count "HEAD..origin/${BRANCH}" 2>/dev/null || echo 0)
echo "  ${_n} commit(s) ahead of you"
git log --oneline "HEAD..origin/${BRANCH}" 2>/dev/null | sed 's/^/    /' | head -20
_behind=$(git rev-list --count "origin/${BRANCH}..HEAD" 2>/dev/null || echo 0)
if [ "${_behind}" != "0" ]; then
    echo "  you also have ${_behind} local commit(s) that are NOT on origin:"
    git log --oneline "origin/${BRANCH}..HEAD" 2>/dev/null | sed 's/^/    /' | head -10
    if [ "${_n}" != "0" ]; then
        echo "  !! you are BOTH ahead and behind, so --ff-only will refuse."
        echo "     That is correct. Work out what the local commits are before"
        echo "     going further; do not force anything."
    else
        echo "  (nothing to pull — you are simply ahead. On the laptop that"
        echo "   means push. On the server it means someone committed here.)"
    fi
fi

echo
echo "----- locally modified (a pull will not touch these unless listed twice below) -----"
git status --porcelain | grep -v '^??' > /tmp/_pullchk_dirty.txt
_d=$(wc -l < /tmp/_pullchk_dirty.txt | tr -d ' ')
echo "  ${_d} tracked file(s) modified here"
if [ "${_d}" != "0" ]; then
    echo "  by area:"
    awk '{print $2}' /tmp/_pullchk_dirty.txt \
      | sed -E 's|^(tests/results)/.*|\1/ (EXPERIMENT OUTPUT — never discard)|; s|^(slurm-).*|slurm-*.out (JOB LOGS — never discard)|; s|^(src/kirby)/.*|\1/|; s|^(tests/experiments)/.*|\1/|; s|/[^/]*$||' \
      | sort | uniq -c | sed 's/^/    /'
fi

echo
echo "----- THE ONE THING THAT MATTERS: overlap -----"
# The merge base, not HEAD. `git diff HEAD..origin/B` reports the difference in
# BOTH directions, so a local commit origin has not seen shows up as "incoming"
# and the count is nonsense. Only the commits you are actually behind by can
# change a file under you.
_base=$(git merge-base HEAD "origin/${BRANCH}" 2>/dev/null)
git diff --name-only "${_base}..origin/${BRANCH}" 2>/dev/null | sort > /tmp/_pullchk_incoming.txt
awk '{print $2}' /tmp/_pullchk_dirty.txt | sort > /tmp/_pullchk_dirtynames.txt
comm -12 /tmp/_pullchk_incoming.txt /tmp/_pullchk_dirtynames.txt > /tmp/_pullchk_overlap.txt
_o=$(wc -l < /tmp/_pullchk_overlap.txt | tr -d ' ')
echo "  incoming commits change $(wc -l < /tmp/_pullchk_incoming.txt | tr -d ' ') file(s)"
if [ "${_o}" = "0" ]; then
    echo "  0 of them are modified here."
    echo
    echo "  SAFE. Your local edits and everything under tests/results/ are"
    echo "  untouched by this pull. Run:"
    echo
    echo "      git pull --ff-only origin ${BRANCH}"
else
    echo "  !! ${_o} file(s) are BOTH modified here and changed by the incoming"
    echo "     commits. A pull will refuse rather than clobber them, which is"
    echo "     correct. Decide what the local edits are before going further:"
    sed 's/^/       /' /tmp/_pullchk_overlap.txt
    echo
    echo "     If they are yours and wanted:   git stash push -- <those files>"
    echo "     then pull, then  git stash pop  and resolve."
    echo "     If they are accidental:         git checkout -- <that ONE file>"
    echo "     Never a bare 'git checkout .' or 'reset --hard' — see the header."
fi
rm -f /tmp/_pullchk_dirty.txt /tmp/_pullchk_incoming.txt \
      /tmp/_pullchk_dirtynames.txt /tmp/_pullchk_overlap.txt
echo
echo "===== end ====="
