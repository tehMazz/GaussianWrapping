#!/bin/bash
#SBATCH --job-name=gw_mesh_views
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# GW mesh of a trained model, rendered next to the photos (render_mesh_views.py: far plane fixed vs the stock 100).
# 1. pivot_based_mesh_extraction.py (GW's mesh, GW_EXTRACT_ARGS) on a linked copy of the model (source only read),
# 2. render it, 3. multi-label PAM (multilabel_mesh_extraction.py, needs the model's labels.pt) and render that.
# 0. render_depth_views.py (photo | depth | normal | normal from depth) on $DEPTH_MODELS, stock and relwin rasterizer.
# All mesh steps pass --camera_zfar $ZFAR (default 1e4): with the stock 100 the mesh colouring crashes on SPIn-NeRF.
# Usage (from repo root): NAME=spinnerf_7 sbatch slurm/gw_mesh_views.sh
# Outputs (flat): output/gwmesh/<NAME>/train (linked model + mesh_ours_2pivots.ply), output/gwmesh/<NAME>/mesh_views,
#   output/gwmesh/<NAME>/mesh (PAM: mesh_labels.ply, label_<k>.ply), output/gwmesh/<NAME>/pam_views
set -euo pipefail
source slurm/common.sh
NAME=${NAME:-spinnerf_7}
SRC=output/gw_beags/$NAME/train
O=output/gwmesh/$NAME
M=$O/train
ZFAR=${ZFAR:-10000}
RELWIN=${RELWIN:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/raster_variants/beags_relwin}
DEPTH_MODELS=${DEPTH_MODELS:-"spinnerf_7 mip360_kitchen llff_fern"}
mkdir -p "$O"
for S in $DEPTH_MODELS; do
  # "stock" = the installed rasterizer: after slurm/install_full_bracket.sh that is the full-bracket build, so these
  # renders no longer show the stock holes (the 2026-11-25 depth_views_stock were made before the install).
  $PY gaussian_wrapping/render_depth_views.py -m output/gw_beags/$S/train --iteration "$ITER" \
      --out output/gwmesh/$S/depth_views_stock || echo "depth views $S stock FAILED"
  (export PYTHONPATH=$RELWIN:$PYTHONPATH; $PY gaussian_wrapping/render_depth_views.py -m output/gw_beags/$S/train \
      --iteration "$ITER" --out output/gwmesh/$S/depth_views_relwin) || echo "depth views $S relwin FAILED"
done
[ -e "$M/cfg_args" ] || link_model "$SRC" "$M" "$ITER"
mkdir -p "$M/beags"
cp -n "$SRC/point_cloud/iteration_$ITER/labels.pt" "$M/point_cloud/iteration_$ITER/labels.pt"
cp -n "$SRC/beags/labeled_points.npz" "$M/beags/labeled_points.npz"
set -x
# shellcheck disable=SC2086
[ -f "$M/mesh_ours_2pivots.ply" ] || $PY gaussian_wrapping/pivot_based_mesh_extraction.py -m "$M" --iteration "$ITER" $GW_EXTRACT_ARGS --camera_zfar "$ZFAR"
$PY gaussian_wrapping/render_mesh_views.py -m "$M" --iteration "$ITER" --mesh "$M/mesh_ours_2pivots.ply" --out "$O/mesh_views"
K=output/gw_beags/mip360_kitchen/beags          # kitchen's existing GW mesh (33k), for comparison
[ -f "$K/mesh_ours_2pivots.ply" ] && $PY gaussian_wrapping/render_mesh_views.py -m "$K" --iteration 33000 \
    --mesh "$K/mesh_ours_2pivots.ply" --n_views 6 --out output/gwmesh/mip360_kitchen/mesh_views || echo "kitchen mesh FAILED"
$PY gaussian_wrapping/multilabel_mesh_extraction.py -m "$M" --iteration "$ITER" --input_mesh "$M/mesh_ours_2pivots.ply" \
    --out_dir "$O/mesh" --camera_zfar "$ZFAR"
$PY gaussian_wrapping/render_mesh_views.py -m "$M" --iteration "$ITER" --mesh "$O/mesh/mesh_labels.ply" --out "$O/pam_views"
set +x
echo "mesh views OK"
