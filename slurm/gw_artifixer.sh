#!/bin/bash
#SBATCH --job-name=gw_afx
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/%x_%j.out
# ArtiFixer diffusion (submodules/artifixer_gs) on a BEA-GS GW model, object k = label <label>:
#   normal | bighull                                   removal + inpainting
#   vanilla_object | vanilla_vidsplat | vanilla_travel  no removal, ArtiFixer3D+ on a novel camera path
# ArtiFixer3D trains with gaussian_wrapping/train.py --rasterizer ours (--lpips_images: generated frames LPIPS-only;
# AF3D_TRAIN_ARGS adds the bilateral grid and the hollowing prunes), then export_3dgs_ply.py bakes it.
# Usage (from repo root):
#   sbatch -J afx_<variant>_<name> slurm/gw_artifixer.sh <name> <label> <variant> [2D masks dir]
#   <name> as in gw_scene.sh (model output/$METHOD/<name>/beags at $FT_ITER); the removal variants need the masks
#   (BEA-GS (C, H, W) masks: channel <label>; or 2D id maps). Knobs (env):
#   FALLBACK_MASKS masks for the photos the masks dir lacks (BEA-GS consensus masks cover training views only:
#                  give the SAM2 masks; removal needs a mask per frame)
#   HOLE_MODE      dilate | alpha | mesh | gavis (default mesh when output/$METHOD/<name>/mesh/hole_<label>.ply exists
#                  (slurm/gw_mesh.sh --remove <label>), else alpha). gavis: the dilated-mask pixels whose per-view
#                  unseen map (GAVIS visibility, beags/unseen_map.py -> <O>/unseen, UNSEEN_DIR) is above UNSEEN_THRESH
#   UNSEEN_MAPS    auto | on | off (default auto: rendered with HOLE_MODE=gavis); KAPPA (1) its vMF concentration
#   UNSEEN_THRESH  0.5 (HOLE_MODE=gavis, run.sh)
#   CONFIDENCE_MAPS  auto | on | off (default auto: per-view confidence maps <O>/confidence/<stem>.png, exported as
#                  CONFIDENCE_DIR, when the model was trained with --confidence; skipped with the reason otherwise)
#   CONF_THRESH    unset = off; e.g. 0.5: HOLE_MODE alpha | mesh also keep the dilated-mask pixels with confidence
#                  below it (run.sh, needs the maps)
#   CAPTION, METRIC_SCALE, AF_ROOT, STAGES, AF3D_STEPS, NUM_VIEWS, SEEDS, ... as BEA-GS_FSplat slurm/beags_artifixer.sh
#   AF3D_TRAIN_ARGS  train.py flags of the ArtiFixer3D distillation (default: GW recipe + bilagrid + hollow)
# Output (flat): output/artifixer/<name>/<variant>/{env.sh,inputs.json,object_mask.npy,export/,[confidence/,][unseen/,]<name>/}.
set -euo pipefail
NAME=$1; LABEL=$2; VARIANT=$3; MASKS=${4:-}
source slurm/common.sh
ENVS=/leonardo_scratch/fast/IscrC_EditGS/envs
MODEL=$ROOT/$(model_dir "$NAME" beags)
SCENE=$(basename "$NAME")
O=$ROOT/output/artifixer/$SCENE/$VARIANT
HOLE_MESH=$ROOT/output/$METHOD/$NAME/mesh/hole_$LABEL.ply
if [ -z "${HOLE_MODE:-}" ]; then HOLE_MODE=alpha; [ ! -f "$HOLE_MESH" ] || HOLE_MODE=mesh; fi
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
CKPTS=$AF/checkpoints                                  # shared with the ArtiFixer repo
AF_ROOT=${AF_ROOT:-$AF/output/recon/$SCENE}
case $VARIANT in
  normal|bighull) MODE=removal ;;
  vanilla_object|vanilla_vidsplat|vanilla_travel) MODE=vanilla ;;
  *) echo "unknown variant '$VARIANT' (normal, bighull, vanilla_object, vanilla_vidsplat, vanilla_travel)" >&2; exit 2 ;;
esac
[ "$MODE" = vanilla ] || [ -n "$MASKS" ] || { echo "$VARIANT needs the 2D masks dir (4th argument)" >&2; exit 2; }
if [ "$MODE" = removal ]; then export SEEDS=${SEEDS:-auto} OUTSIDE=${OUTSIDE:-photo}; fi
AF3D_TRAIN_ARGS=${AF3D_TRAIN_ARGS-"$GW_TRAIN_ARGS $ROBUST_ARGS $HOLLOW_ARGS"}
AF3D_TRAIN_ARGS=${AF3D_TRAIN_ARGS//--rasterizer ours/}  # TRAIN_CMD already passes it

set -x
$PY gaussian_wrapping/beags_artifixer.py -m "$MODEL" --iteration "$FT_ITER" --label "$LABEL" --variant "$VARIANT" \
  --out "$O" --hole_mode "$HOLE_MODE" --train_args "$AF3D_TRAIN_ARGS" --confidence_maps "${CONFIDENCE_MAPS:-auto}" \
  --unseen_maps "${UNSEEN_MAPS:-auto}" --kappa "${KAPPA:-1}" \
  ${MASKS:+--masks "$MASKS"} ${FALLBACK_MASKS:+--fallback_masks "$FALLBACK_MASKS"} ${METRIC_SCALE:+--metric_scale "$METRIC_SCALE"} \
  $([ ! -f "$HOLE_MESH" ] || echo --hole_mesh "$HOLE_MESH") \
  $([ -d "$AF_ROOT" ] && echo --artifixer_root "$AF_ROOT")
set -a; source "$O/env.sh"; set +a                     # PLY, OUT=$O/<scene>, TRAIN_CMD, AF3D_EXPORT, MESH_PY, ...
if [ -z "${CAPTION:-}" ]; then
  [ -n "${CAPTION_H5:-}" ] || { set +x; echo "no caption for $SCENE: set CAPTION (no caption.h5 under $AF_ROOT)" >&2; exit 2; }
  mkdir -p "$OUT/captions/$SCENE"
  [ -f "$OUT/captions/$SCENE/caption.h5" ] || cp -L "$CAPTION_H5" "$OUT/captions/$SCENE/caption.h5"
fi

# Three envs: artifixer (diffusion, prepare, handoff), gsplat (PLY renders, as a subprocess), gaussian_wrapping
# (ArtiFixer3D = this repo's train.py, and the hole mesh rasterization: the only env with nvdiffrast).
export PY=$ENVS/artifixer/bin/python GS_RENDER_PY=$ENVS/gsplat/bin/python
export MODEL_ID=$CKPTS/Wan2.1-T2V-1.3B-Diffusers CHECKPOINT_PT=$CKPTS/artifixer/artifixer-1.3b.pt
export MOGE_MODEL_PATH=$CKPTS/moge-2-vitl-normal/model.pt
# /leonardo_scratch/fast/IscrC_EditGS is at its quota: temp files and writable caches go to work.
export TMPDIR=/leonardo_work/IscrC_EditGS/tmp_amazzucc/afx_job_${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor TRITON_CACHE_DIR=$TMPDIR/triton
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export PYTHONPATH=
cd submodules/artifixer_gs   # keeps this repo's scene/ and utils/ off sys.path
bash scripts/run.sh
