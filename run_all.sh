#!/usr/bin/env bash
# run_all.sh — four session-level seeds and three image-level, resumable
#
#   bash run_all.sh dry     # 2 epochs per stage, just to time the run
#   bash run_all.sh         # the real thing: 4 session-level seeds, 3 image-level
#
# Each (regime, seed) writes to its own checkpoint and results directory, so
# an interrupted run resumes by simply re-running this script: load_or_train()
# skips any model whose checkpoint already exists. Do not pass --retrain.
#
# Run it under nohup so a dropped SSH session does not kill it:
#   nohup bash run_all.sh > run.log 2>&1 &
#   tail -f run.log

set -u

ROOT=${DURIAN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CODE=$ROOT/github
SEEDS="${SEEDS:-42 1 2 3}"           # both regimes, as reported in the paper

SESSION_SPLIT=$ROOT/clean_split          # capture-session partition
IMAGE_SPLIT=$ROOT/image_split            # image-level control
VIETNAM=$ROOT/vietnam
SESSIONS=$CODE/sessions.csv

DRY=""
TAG=""
if [ "${1:-}" = "dry" ]; then
    DRY="--dry_run"
    TAG="_dry"
    SEEDS="42"
    echo "DRY RUN: 2 epochs per stage, one seed. Timing only — discard these numbers."
fi

cd "$CODE" || { echo "No $CODE"; exit 1; }

for f in "$SESSIONS" "$SESSION_SPLIT" "$VIETNAM"; do
    [ -e "$f" ] || { echo "Missing: $f"; exit 1; }
done
if [ ! -d "$IMAGE_SPLIT" ]; then
    echo "Missing $IMAGE_SPLIT — build it first:"
    echo "  python make_image_split.py --src $ROOT/Classication_model_split --dst $IMAGE_SPLIT"
    exit 1
fi

start=$(date +%s)

run_one () {
    local mode=$1 split=$2 seed=$3
    local ck="$ROOT/ckpt${TAG}/${mode}_s${seed}"
    local sv="$ROOT/results${TAG}/${mode}_s${seed}"
    mkdir -p "$ck" "$sv"

    if [ -f "$sv/DONE" ]; then
        echo "[skip] $mode seed=$seed already finished"
        return 0
    fi

    echo ""
    echo "=================================================================="
    echo " $mode  seed=$seed   $(date '+%H:%M:%S')"
    echo "=================================================================="
    python -u train.py \
        --split_dir     "$split" \
        --malaysia_data "$split" \
        --vietnam_data  "$VIETNAM" \
        --sessions      "$SESSIONS" \
        --cv_mode       "$mode" \
        --ckpt_dir      "$ck" \
        --save_dir      "$sv" \
        --seed          "$seed" \
        --no_latency    $DRY
    local rc=$?
    if [ $rc -eq 0 ]; then
        touch "$sv/DONE"
    else
        echo "!! $mode seed=$seed exited with $rc — rerun this script to retry"
    fi
    return $rc
}

# Session-level first: these are the numbers the paper will report.
for s in $SEEDS; do
    run_one group "$SESSION_SPLIT" "$s"
done

# Image-level control: the comparison that quantifies the leakage.
for s in $SEEDS; do
    run_one image "$IMAGE_SPLIT" "$s"
done

end=$(date +%s)
mins=$(( (end - start) / 60 ))
echo ""
echo "=================================================================="
echo " finished in ${mins} min"
echo "=================================================================="
if [ -n "$DRY" ]; then
    echo "That was one seed at 2 epochs. The real run is 4 seeds x 2 regimes"
    echo "at 15+15 epochs. Scale accordingly before committing."
else
    echo "Now aggregate:"
    echo "  python aggregate.py --root $ROOT/results --out $ROOT/summary"
fi
