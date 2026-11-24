#!/bin/bash
#SBATCH --job-name=gw_scene
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/%x_%j.out
# Full BEA-GS-on-GW pipeline on one scene, every step skipped when its output exists:
#   train 30k (--rasterizer ours --no-exposure_compensation --appearance bilagrid --hollow) -> export (+ gsplat check)
#   -> beags prepare / eval / finetune / eval -> beags export split per label + per-label renders -> hollow metrics (train, beags) -> mesh (pivot + multi-label PAM,
#   hole of object $REMOVE) -> render.py + metrics.py (train $ITER, beags $FT_ITER).
# Usage (from repo root):
#   sbatch -J gw_<name> slurm/gw_scene.sh <name> <scene_dir> <sam2_masks_dir> <gt_masks_dir> [extra train.py args]
#   <name> e.g. mip360_room; knobs (env): METHOD (gw_beags), REMOVE (1: object label for hole mesh / metrics),
#   FT_ARGS (extra beags_finetune.py args), MESH_ARGS (extra multilabel_mesh_extraction.py args), slurm/common.sh.
#   MESH=0 skips the multi-label mesh / hole PLY (the slowest step, ~11 min on kitchen); everything else still runs.
#   OCC_SOURCE=mesh: L_occ / L_piv against the multi-label mesh of the prepared $ITER model (gw_mesh.sh -> mesh_train,
#   its tets.npz) instead of the labelled-point voxels (default points). Like MASK_SOURCE=seed2gs, it shares the train
#   model and writes everything after it to output/$OCC_METHOD/<name>/... (default ${METHOD}_meshocc) through a linked
#   train/, so the voxel run's outputs are never touched.
#   MASK_SOURCE=seed2gs SEED2GS_PROMPT="<text>": masks from Seed2GS on the $ITER train model instead of <sam2_masks_dir>
#   (<sam2_masks_dir> is then unused). Training is shared with the provided-mask run (output/$METHOD/<name>/
#   train, trained here if missing); seed2gs.py -> output/seed2gs/<name>/<prompt slug> (+ seed2gs_eval.json vs
#   <gt_masks_dir> and, when the provided run exists, vs its labels.pt); BEA-GS and everything after it go to
#   output/$SEED2GS_METHOD/<name>/... (default ${METHOD}_seed2gs), whose train/ is a linked view of the shared model
#   (symlinked PLY + cfg_args, own labels.pt / beags/), so provided-mask outputs are never touched. REMOVE stays 1.
#   All prompts of a scene share output/$SEED2GS_METHOD/<name>: another prompt reruns prepare and everything after it
#   there (set SEED2GS_METHOD per prompt, e.g. gw_beags_seed2gs_<slug>, to keep each prompt's results).
# Outputs (flat): output/$METHOD/<name>/{train,export,beags,export_beags,mesh}; per-label views:
#   export_beags/label_<k>.ply (3DGS PLY, Gaussians of label k), mesh/label_<k>.ply (label-k region of the mesh),
#   renders_labels_<FT_ITER>/{full,label_<k>,grid}/*.png (8 views; grid = GT | full | label_0 | label_1 ...).
# Skips: train, export, prepare (labels.pt + labeled_points.npz exist, beags/mask_source.txt names this masks dir
#   (written by each prepare; a provided run without it, from before the stamp, is kept) and, for seed2gs, they are
#   newer than its seed2gs.json), finetune, split export and pivot mesh exist. When prepare runs, finetune, split
#   export and pivot mesh rerun.
set -euo pipefail
NAME=$1; SRC=$2; MASKS=$3; GT=$4; shift 4
source slurm/common.sh
REMOVE=${REMOVE:-1}
OCC_SOURCE=${OCC_SOURCE:-points}
T=$(model_dir "$NAME" train)                           # the $ITER GW model, shared by both mask sources
set -x
# shellcheck disable=SC2086
has_iter "$T" "$ITER" || \
  $PY gaussian_wrapping/train.py -s "$SRC" -m "$T" --eval --iterations "$ITER" \
      --test_iterations "$ITER" --save_iterations "$ITER" $GW_TRAIN_ARGS $ROBUST_ARGS $HOLLOW_ARGS "$@"
[ -f "output/$METHOD/$NAME/export/point_cloud.ply" ] || \
  $PY gaussian_wrapping/export_3dgs_ply.py -m "$T" --iteration "$ITER" --check --check_stride 10 --gs_python "$GS_RENDER_PY"
# the shared train model's own evals: once (a seed2gs run does not rewrite what the provided run wrote)
TRAIN_EVAL=1
{ [ "$MASK_SOURCE" = seed2gs ] || [ "$OCC_SOURCE" = mesh ]; } && [ -f "$T/results.json" ] && TRAIN_EVAL=0
[ "$TRAIN_EVAL" = 0 ] || bash slurm/gw_hollow_metrics.sh "$T" "$ITER"

M=$T
STAMP=""                                               # prepare reruns when its labels are older than this file
case "$MASK_SOURCE" in
  provided) ;;
  seed2gs)
    if [ -z "$SEED2GS_PROMPT" ] || [[ "$SEED2GS_PROMPT" == *"|"* ]]; then
      echo "MASK_SOURCE=seed2gs needs one SEED2GS_PROMPT (no '|')" >&2; exit 1
    fi
    S2=output/seed2gs/$NAME/$(slug "$SEED2GS_PROMPT")
    PROVIDED_LABELS=$T/point_cloud/iteration_$ITER/labels.pt   # from a provided-mask run, when there was one
    if ! { [ -f "$S2/labels.pt" ] && [ -d "$S2/masks" ] && grep -sqF "\"model_path\": \"$T\"" "$S2/seed2gs.json" \
           && grep -sqF "\"iteration\": $ITER," "$S2/seed2gs.json"; }; then
      # shellcheck disable=SC2086
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        $PY gaussian_wrapping/seed2gs.py -m "$T" --iteration "$ITER" --prompt "$SEED2GS_PROMPT" --out "$S2" $SEED2GS_ARGS
    fi
    $PY gaussian_wrapping/seed2gs_eval.py --seed "$S2" --gt_masks "$GT" --labels "$PROVIDED_LABELS"
    MASKS=$S2/masks
    STAMP=$S2/seed2gs.json
    METHOD=$SEED2GS_METHOD                             # model_dir now points at output/$SEED2GS_METHOD
    M=$(model_dir "$NAME" train)
    link_model "$T" "$M" "$ITER"
    ;;
  *) echo "MASK_SOURCE must be provided or seed2gs, not '$MASK_SOURCE'" >&2; exit 1 ;;
esac
if [ "$OCC_SOURCE" = mesh ]; then
  METHOD=${OCC_METHOD:-${METHOD}_meshocc}
  M=$(model_dir "$NAME" train)
  link_model "$T" "$M" "$ITER"
elif [ "$OCC_SOURCE" != points ]; then
  echo "OCC_SOURCE must be points or mesh, not '$OCC_SOURCE'" >&2; exit 1
fi
B=$(model_dir "$NAME" beags)

SRC_ID=$(realpath -m "$MASKS")                         # which masks the labels were prepared from
SRC_FILE=$M/beags/mask_source.txt
SAME_SRC=0
if [ -f "$SRC_FILE" ]; then
  [ "$(cat "$SRC_FILE")" = "$SRC_ID" ] && SAME_SRC=1
elif [ "$MASK_SOURCE" = provided ]; then
  SAME_SRC=1                                           # provided run prepared before the stamp existed: keep it
fi
FRESH=0
if ! { [ -f "$M/point_cloud/iteration_$ITER/labels.pt" ] && [ -f "$M/beags/labeled_points.npz" ] \
       && [ "$SAME_SRC" = 1 ] && { [ -z "$STAMP" ] || [ "$M/beags/labeled_points.npz" -nt "$STAMP" ]; }; }; then
  $PY gaussian_wrapping/beags_prepare.py -m "$M" --iteration "$ITER" --masks "$MASKS"
  mkdir -p "$M/beags" && echo "$SRC_ID" > "$SRC_FILE"
  FRESH=1                                              # new labels: everything built on them is stale
fi
$PY gaussian_wrapping/beags_eval.py -m "$M" --iteration "$ITER" --gt_masks "$GT" --masks "$MASKS"
OCC_ARGS=""
if [ "$OCC_SOURCE" = mesh ]; then                      # mesh of the prepared model, before the finetune
  MT=output/$METHOD/$NAME/mesh_train
  if [ "$FRESH" = 1 ] || [ ! -f "$MT/tets.npz" ]; then
    rm -f "$M/mesh_ours_2pivots.ply" "$MT"/dist_*.npz
    MESH_OUT=$MT SAVE_TETS=1 bash slurm/gw_mesh.sh "$M" "$ITER"
  fi
  OCC_ARGS="--occ_source mesh --mesh_dir $MT"
fi
if [ "$FRESH" = 1 ] || ! has_iter "$B" "$FT_ITER"; then
  # shellcheck disable=SC2086
  $PY gaussian_wrapping/beags_finetune.py -m "$M" --iteration "$ITER" --out "$B" $OCC_ARGS ${FT_ARGS:-}
  rm -f "$B/mesh_ours_2pivots.ply"                     # gw_mesh.sh skips the pivot mesh when it exists
fi
$PY gaussian_wrapping/beags_eval.py -m "$B" --iteration "$FT_ITER" --gt_masks "$GT" --masks "$MASKS" --render
if [ "$FRESH" = 1 ] || [ ! -f "output/$METHOD/$NAME/export_beags/label_0.ply" ]; then
  $PY gaussian_wrapping/export_3dgs_ply.py -m "$B" --iteration "$FT_ITER" --split_labels \
      --out "output/$METHOD/$NAME/export_beags/point_cloud.ply"
fi
$PY gaussian_wrapping/beags_render_labels.py -m "$B" --iteration "$FT_ITER" --n_views 8

bash slurm/gw_hollow_metrics.sh "$B" "$FT_ITER" --label "$REMOVE" --masks "$MASKS"
if [ "${MESH:-1}" = 1 ]; then
  # shellcheck disable=SC2086
  bash slurm/gw_mesh.sh "$B" "$FT_ITER" --remove "$REMOVE" ${MESH_ARGS:-}
fi

EVAL=("$B")
if [ "$TRAIN_EVAL" = 1 ]; then
  $PY gaussian_wrapping/render.py -m "$T" --iteration "$ITER" --rasterizer ours --skip_train
  EVAL=("$T" "$B")
fi
$PY gaussian_wrapping/render.py -m "$B" --iteration "$FT_ITER" --rasterizer ours --skip_train
$PY gaussian_wrapping/metrics.py -m "${EVAL[@]}"
