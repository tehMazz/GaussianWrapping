#!/bin/bash
#SBATCH --job-name=pgsr
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
# PGSR (repos/PGSR @ de24f1a, untouched) on one scene, settings matched to our GW runs, for the GW-vs-PGSR comparison.
#   sbatch [--qos=boost_qos_dbg --time=00:30:00] slurm/pgsr_scene.sh <scene> [smoke]
#   scene: spinnerf_7 | mip360_kitchen. Data: output/pgsr/_data/<scene> (symlinks, PGSR reads sparse/ not sparse/0).
# Steps: (build diff-plane-rasterization in place into output/pgsr/_build if missing) -> train.py 30k (PGSR's Mip-NeRF 360
# recipe: --densify_abs_grad_threshold 0.0002; -r 4 = the GW training size; spinnerf: multiview neighbour distances
# scaled like GW's relative ones) -> render_pgsr.py (PGSR render.py with float TSDF depth: test renders + TSDF mesh)
# -> GW metrics.py on the test renders -> camera-path frames (rgb / depth / normal) + mesh frames for the videos.
# The video uses the RAW TSDF mesh (tsdf_fusion.ply): GW's mesh is not cluster-filtered either, so neither is cleaned.
# kitchen max_depth 10 (PGSR README value; 6 would cut the room walls that GW's mesh keeps).
# smoke: debug-sized (1000 its at -r 8, multiview from 300, 10 path frames) into output/pgsr_smoke/<scene>.
# Outputs: output/pgsr/<scene>/train (model, test/, mesh/tsdf_fusion_post.ply, results.json), output/pgsr/<scene>/video
set -euo pipefail
S=${1:?scene}; MODE=${2:-full}
source slurm/common.sh
PG=/leonardo_work/IscrC_EditGS/repos/PGSR
BUILD=$ROOT/output/pgsr/_build/diff-plane-rasterization
DATA=$ROOT/output/pgsr/_data/$S
PATHJ=$ROOT/output/compare/$S/path.json
if [ "$MODE" = smoke ]; then O=$ROOT/output/pgsr_smoke/$S; else O=$ROOT/output/pgsr/$S; fi
M=$O/train
if ! ls "$BUILD"/diff_plane_rasterization/_C*.so > /dev/null 2>&1; then
  [ "$MODE" = smoke ] || { echo "diff-plane-rasterization not built: run the smoke first" >&2; exit 1; }
  mkdir -p "$(dirname "$BUILD")"; rm -rf "$BUILD"; cp -r "$PG/submodules/diff-plane-rasterization" "$BUILD"
  (cd "$BUILD" && $PY setup.py build_ext --inplace > ../build_plane.log 2>&1) || { tail -40 "$(dirname "$BUILD")/build_plane.log"; exit 1; }
fi
PGPATH=$ROOT/gaussian_wrapping/pgsr_tools/shim:$BUILD
# PGSR input: images + sparse/ (PGSR reads sparse/, not sparse/0) as symlinks to the scene used by GW
case $S in
  spinnerf_7)     SRC=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/scenes/SPIn-NeRF/7; IMG=images ;;
  mip360_kitchen) SRC=/leonardo_work/IscrC_EditGS/datasets/mip360/kitchen; IMG=images_4 ;;
  *) echo "unknown scene $S" >&2; exit 2 ;;
esac
if [ ! -d "$DATA/sparse" ]; then
  mkdir -p "$DATA/sparse"
  ln -sfn "$SRC/$IMG" "$DATA/$IMG"
  for f in "$SRC"/sparse/0/*; do
    case $f in *.ply) continue ;; esac            # PGSR writes its own points3D.ply next to points3D.bin/txt
    ln -sfn "$f" "$DATA/sparse/$(basename "$f")"
  done
fi
case $S in
  spinnerf_7)     RES="-r 4"; EXTRA="--multi_view_min_dis 0.018 --multi_view_max_dis 2.734"
                  MESH="--max_depth 200 --voxel_size 0.15 --num_cluster 50 --use_depth_filter" ;;
  mip360_kitchen) RES="-i images_4 -r 4"; EXTRA=""; MESH="--max_depth 10.0 --voxel_size 0.01 --num_cluster 50" ;;
  *) echo "unknown scene $S" >&2; exit 2 ;;
esac
ITERS=30000; FRAMES=0
if [ "$MODE" = smoke ]; then
  ITERS=1000; FRAMES=10; RES=${RES/-r 4/-r 8}
  EXTRA="$EXTRA --single_view_weight_from_iter 300 --multi_view_weight_from_iter 300 --densify_until_iter 800"
fi
set -x
if [ ! -f "$M/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
  # shellcheck disable=SC2086
  (cd "$PG" && PYTHONPATH=$PGPATH $PY train.py -s "$DATA" -m "$M" $RES --eval --data_device cpu \
      --densify_abs_grad_threshold 0.0002 --iterations $ITERS --test_iterations $ITERS --save_iterations $ITERS \
      --quiet $EXTRA)
fi
# shellcheck disable=SC2086
(cd "$PG" && PYTHONPATH=$PGPATH $PY "$ROOT/gaussian_wrapping/pgsr_tools/render_pgsr.py" -m "$M" $MESH)
$PY gaussian_wrapping/metrics.py -m "$M" || echo "metrics FAILED"
(cd "$PG" && PYTHONPATH=$PGPATH $PY "$ROOT/gaussian_wrapping/pgsr_tools/render_path_pgsr.py" \
    --ply "$M/point_cloud/iteration_$ITERS/point_cloud.ply" --path "$PATHJ" --out "$O/video" --max_frames $FRAMES)
$PY gaussian_wrapping/render_path_mesh.py --mesh "$M/mesh/tsdf_fusion.ply" --path "$PATHJ" --out "$O/video" \
    --max_frames $FRAMES
set +x
echo "pgsr $S $MODE OK [$(date +%T)]"
