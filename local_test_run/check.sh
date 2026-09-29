#!/bin/bash
# Run once inside the srun shell, after sourcing env.sh. Builds the run's own cache folder
# and stops if kirby would be imported from anywhere but this folder.
set -u
REAL=/data/stat-ecr/scat9264/.cache/kirby
mkdir -p "$KIRBY_CACHE_DIR" "$RUN/results" "$RUN/logs" "$RUN/state"
# Folders this run writes (same as the local run's own folders); everything else links to the real cache.
OWN="baselines fold_scores folds ft_embeddings noise_cache result_db searches"
for d in $OWN; do mkdir -p "$KIRBY_CACHE_DIR/$d"; done
for p in "$REAL"/*; do n=$(basename "$p")
  case " $OWN " in *" $n "*) continue;; esac
  [ -e "$KIRBY_CACHE_DIR/$n" ] || ln -s "$p" "$KIRBY_CACHE_DIR/$n"
done
echo "cache: $KIRBY_CACHE_DIR ($(ls "$KIRBY_CACHE_DIR" | wc -l) entries)"
where=$(python -c "import kirby,os;print(os.path.dirname(kirby.__file__))")
echo "kirby imported from: $where"
case "$where" in /data/stat-ecr/scat9264/KIRBy_localtest/*) echo "OK";;
  *) echo "STOP: kirby is not this folder's copy"; exit 1;; esac
echo "branch: $(git -C /data/stat-ecr/scat9264/KIRBy_localtest rev-parse --abbrev-ref HEAD) $(git -C /data/stat-ecr/scat9264/KIRBy_localtest log -1 --format=%h)"
