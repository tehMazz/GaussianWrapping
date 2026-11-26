#!/bin/bash
#SBATCH --job-name=gw_today
#SBATCH --account=IscrC_P2R-GS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/%x_%j.out
# "GW of today" on one scene, from scratch: upstream GW "ours" recipe, NO bilateral grid, NO hollowing, plus today's
# fixes: full-bracket median depth in both rasterizer packages (installed, see below) and --camera_zfar 1e4.
#   sbatch [--qos=boost_qos_dbg --time=00:30:00] slurm/gw_today.sh <scene> [smoke]
#   scene: spinnerf_7 | mip360_kitchen
# Steps: train.py 30k -> render.py test + metrics.py -> GW mesh (pivot_based_mesh_extraction, zfar 1e4) -> stills
# (render_depth_views, render_mesh_views: >= 5 views) -> camera-path frames (rgb / depth / normal / mesh) for the videos.
# smoke: 1000 its at -r 8, multiview + depth-normal from 300, 10 path frames, into output/gw_today_smoke/<scene>.
# Outputs: output/gw_today/<scene>/{train, video, depth_views, mesh_views}
# Variants: METHOD=<name> (output/<name>/<scene>, smoke: output/<name>_smoke/<scene>) and TRAIN_EXTRA="<train.py flags>",
#   e.g. METHOD=gw_mono_nd TRAIN_EXTRA="--mono_prior_dir output/mono_prior/spinnerf_7/moge2 --mono_prior_config normal_depth"
set -euo pipefail
S=${1:?scene}; MODE=${2:-full}
# Rasterizers: the installed ones, which carry the full-bracket depth once slurm/install_full_bracket.sh has run
# (check: sbatch slurm/verify_full_bracket.sh). RASTER_PATH=<beags dir>:<ours dir> overrides with test builds.
export RASTER_PATH=${RASTER_PATH:-}
METHOD=${METHOD:-gw_today}         # before common.sh, which otherwise sets METHOD=gw_beags
source slurm/common.sh
PATHJ=output/compare/$S/path.json
if [ "$MODE" = smoke ]; then O=output/${METHOD}_smoke/$S; else O=output/$METHOD/$S; fi
M=$O/train
case $S in
  spinnerf_7)     SRC=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/scenes/SPIn-NeRF/7; IMG="" ;;
  mip360_kitchen) SRC=/leonardo_work/IscrC_EditGS/datasets/mip360/kitchen; IMG="-i images_4" ;;
  *) echo "unknown scene $S" >&2; exit 2 ;;
esac
ITERS=30000; FRAMES=0; EXTRA=""
if [ "$MODE" = smoke ]; then ITERS=1000; FRAMES=10; EXTRA="-r 8 --regularization_from_iter 300 --multiview_config smoke"; fi
$PY -c "import diff_gaussian_rasterization_ours as a, diff_gaussian_rasterization_ours_beags as b; print('rasterizers', a.__file__, b.__file__)"
set -x
if [ ! -f "$M/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
  # shellcheck disable=SC2086
  $PY gaussian_wrapping/train.py -s "$SRC" -m "$M" $IMG --eval --iterations $ITERS --test_iterations $ITERS \
      --save_iterations $ITERS --rasterizer ours --feature_dc_lr 0.0013 --feature_rest_lr 0.00011 --data_device cpu \
      --N_max_gaussians 6000000 --no-exposure_compensation --camera_zfar 10000 $EXTRA ${TRAIN_EXTRA:-}
fi
$PY gaussian_wrapping/render.py -m "$M" --iteration $ITERS --rasterizer ours --skip_train
$PY gaussian_wrapping/metrics.py -m "$M" || echo "metrics FAILED"
# shellcheck disable=SC2086
[ -f "$M/mesh_ours_2pivots.ply" ] || $PY gaussian_wrapping/pivot_based_mesh_extraction.py -m "$M" --iteration $ITERS \
    $GW_EXTRACT_ARGS --camera_zfar 10000
$PY gaussian_wrapping/render_depth_views.py -m "$M" --iteration $ITERS --n_views 6 --out "$O/depth_views"
$PY gaussian_wrapping/render_mesh_views.py -m "$M" --iteration $ITERS --mesh "$M/mesh_ours_2pivots.ply" --n_views 6 \
    --out "$O/mesh_views"
$PY gaussian_wrapping/render_path_gw.py -m "$M" --iteration $ITERS --path "$PATHJ" --out "$O/video" \
    --mesh "$M/mesh_ours_2pivots.ply" --max_frames $FRAMES
set +x
echo "$METHOD $S $MODE OK [$(date +%T)]"
