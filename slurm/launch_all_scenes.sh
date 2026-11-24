#!/usr/bin/env bash
# Every removal benchmark scene through slurm/gw_scene.sh, driven by the index that
# gaussian_wrapping/prepare_removal_scenes.py writes (scene dir, masks, GT, per-dataset train.py args).
#
# Usage (from repo root):
#   bash slurm/launch_all_scenes.sh [name ...]      dry run: prints the sbatch lines, submits nothing
#   SUBMIT=1 bash slurm/launch_all_scenes.sh        submit them
#   bash slurm/launch_all_scenes.sh spinnerf_1 usid_carton          only those scenes
#   DATASETS="3dovs lerf" bash slurm/launch_all_scenes.sh           only those families
# Knobs: MESH (0 skips the multi-label mesh, the slowest step — the default here, this survey does not need it),
#   MAX (cap how many are submitted), FORCE=1 (re-submit scenes whose finetuned model already exists),
#   INDEX (default InpaintingPaper/data/removal_scenes/index.json), plus everything gw_scene.sh and
#   slurm/common.sh read (METHOD, REMOVE, ITER, FT_ITER, ...).
# A scene whose output/$METHOD/<name>/beags model at $FT_ITER already exists is skipped unless FORCE=1.
set -euo pipefail
source slurm/common.sh
INDEX=${INDEX:-/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/index.json}
[ -f "$INDEX" ] || { echo "no index $INDEX: run gaussian_wrapping/prepare_removal_scenes.py --dataset all" >&2; exit 2; }
export MESH=${MESH:-0}
mkdir -p logs

# name<TAB>scene_dir<TAB>masks<TAB>gt<TAB>args, filtered by the names / DATASETS asked for
rows=$($PY - "$INDEX" "${DATASETS:-}" "$@" <<'EOF'
import json, sys
index = json.load(open(sys.argv[1]))
families = sys.argv[2].split()
names = sys.argv[3:]
for name in sorted(index):
    e = index[name]
    if names and name not in names:
        continue
    if families and name.split("_", 1)[0] not in families:
        continue
    print("\t".join([name, e["scene_dir"], e["masks"], e["gt"], " ".join(e["args"])]))
EOF
)
[ -n "$rows" ] || { echo "no scene matched" >&2; exit 2; }

n=0; skipped=0
while IFS=$'\t' read -r name scene masks gt args; do
    [ -n "$name" ] || continue
    model=$ROOT/$(model_dir "$name" beags)
    if [ "${FORCE:-0}" != 1 ] && has_iter "$model" "$FT_ITER"; then
        skipped=$((skipped + 1)); echo "# skip $name (has $FT_ITER)"; continue
    fi
    if [ -n "${MAX:-}" ] && [ "$n" -ge "$MAX" ]; then echo "# stop at MAX=$MAX"; break; fi
    n=$((n + 1))
    # shellcheck disable=SC2086
    if [ "${SUBMIT:-0}" = 1 ]; then
        MESH=$MESH sbatch -J "gw_$name" slurm/gw_scene.sh "$name" "$scene" "$masks" "$gt" $args
    else
        echo "MESH=$MESH sbatch -J gw_$name slurm/gw_scene.sh $name $scene $masks $gt $args"
    fi
done <<< "$rows"

echo
echo "$n scene(s) $([ "${SUBMIT:-0}" = 1 ] && echo submitted || echo "to submit (SUBMIT=1 to run)"), $skipped skipped"
