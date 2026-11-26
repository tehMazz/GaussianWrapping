#!/bin/bash
#SBATCH --job-name=gw_today_smoke
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
# Smoke for the "GW of today" recipe: full-bracket median depth in BOTH rasterizer packages (test copies under $B,
# loaded through RASTER_PATH), --camera_zfar 1e4, no bilateral grid, no hollowing.
# 1. build $B/ours_full and $B/beags_full in place; 2. depth holes on the 30k spinnerf_7 model (stock vs full, depth.npz
# to compare valid pixels) + depth views on kitchen; 3. 1000-it training of spinnerf_7 at -r 4 with the full recipe and
# the multiview loss from 300 (configs/multiview/smoke.yaml) and depth-normal from 300, to prove forward+backward run.
# Usage (from repo root): sbatch slurm/gw_today_smoke.sh      Outputs: output/gw_today_smoke/...
set -euo pipefail
source slurm/common.sh
B=${B:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/raster_variants}
FULL=$B/beags_full:$B/ours_full
O=output/gw_today_smoke
mkdir -p "$O"
for V in ours_full beags_full; do
  echo "=== build $V [$(date +%T)]"
  (cd "$B/$V" && rm -rf build && $PY setup.py build_ext --inplace > "$ROOT/$O/build_$V.log" 2>&1) \
    || { echo "build $V FAILED"; tail -40 "$O/build_$V.log"; exit 1; }
done
SAM=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/masks/SPIn-NeRF/training/sam2/7/masks
set -x
(export PYTHONPATH=$FULL:$PYTHONPATH
 $PY -c "import diff_gaussian_rasterization_ours as a, diff_gaussian_rasterization_ours_beags as b; print('raster', a.__file__, b.__file__)"
 $PY gaussian_wrapping/beags_depth_holes.py -m output/gw_beags/spinnerf_7/train --iteration 30000 --masks "$SAM" \
     --n_views 6 --save_depth --out "$O/holes_full"
 $PY gaussian_wrapping/render_depth_views.py -m output/gw_beags/spinnerf_7/train --iteration 30000 --out "$O/depth_views_spinnerf_7_full"
 $PY gaussian_wrapping/render_depth_views.py -m output/gw_beags/mip360_kitchen/train --iteration 30000 --out "$O/depth_views_kitchen_full")
$PY gaussian_wrapping/beags_depth_holes.py -m output/gw_beags/spinnerf_7/train --iteration 30000 --masks "$SAM" \
    --n_views 6 --save_depth --out "$O/holes_stock"
rm -rf "$O/train_r4"
(export PYTHONPATH=$FULL:$PYTHONPATH
 $PY gaussian_wrapping/train.py -s /leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/scenes/SPIn-NeRF/7 \
     -m "$O/train_r4" -r 4 --eval --iterations 1000 --test_iterations 1000 --save_iterations 1000 \
     --rasterizer ours --feature_dc_lr 0.0013 --feature_rest_lr 0.00011 --data_device cpu --N_max_gaussians 6000000 \
     --no-exposure_compensation --camera_zfar 10000 --regularization_from_iter 300 --multiview_config smoke)
set +x
echo "gw_today smoke OK [$(date +%T)]"
