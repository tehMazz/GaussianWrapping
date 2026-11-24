#!/bin/bash
#SBATCH --job-name=gw_smoke
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
# Smoke test on the debug QoS: every pytest file (GPU tests included), then LLFF fern at -r 8:
#   300 its with GW defaults (exposure compensation, no new flag), 2000 its with everything on (bilateral grid,
#   confidence, variance losses, hollowing from 1000), export, BEA-GS prepare + 300-it finetune + eval, hollow metrics.
# Usage (from repo root): sbatch slurm/gw_smoke.sh      Outputs: output/smoke/fern/{gw_default,train,export,beags}.
# pytest is not in the env: PYTEST_LIB (a `uv pip install --target` dir of pytest, default below) goes on PYTHONPATH.
set -euo pipefail
source slurm/common.sh
PYTEST_LIB=${PYTEST_LIB:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/pylib_gw}
G=/leonardo_work/IscrC_EditGS/gabri/datasets
SRC=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/BEA-GS_FSplat/data/llff/fern
MASKS=$G/nerf_llff_data_original/masks/training/sam2/fern/masks
GT=$G/nerf_llff_data_original/masks/gt_masks/fern/masks
O=output/smoke/fern
rm -rf "$O"
set -x
TESTS=0                                                # a failing test does not stop the pipeline below
PYTHONPATH=$PYTEST_LIB $PY -m pytest -q -rs -p no:cacheprovider gaussian_wrapping/tests || TESTS=1
(cd submodules/artifixer_gs && PYTHONPATH=$PYTEST_LIB $PY -m pytest -q -rs -p no:cacheprovider tests/test_removal.py -k mesh_front_cuda) || TESTS=1

$PY gaussian_wrapping/train.py -s "$SRC" -m "$O/gw_default" -r 8 --eval --rasterizer ours --data_device cpu \
    --iterations 300 --test_iterations 300
$PY gaussian_wrapping/train.py -s "$SRC" -m "$O/train" -r 8 --eval --rasterizer ours --data_device cpu \
    --iterations 2000 --test_iterations 2000 --densify_until_iter 1000 --regularization_from_iter 1000 \
    --no-exposure_compensation --appearance bilagrid --confidence \
    --lambda_color_var 0.5 --lambda_normal_var 0.005 --var_from_iter 1500 \
    --hollow --hollow_from 1000 --prune_unseen_every 500 --carve_every 500 --free_grid_every 500 \
    --entropy_from 1500 --opacity_decay 0.995 --checkpoint_iterations 1500
cat "$O/train/gs_stats.json"
$PY gaussian_wrapping/export_3dgs_ply.py -m "$O/train" --iteration 2000
$PY gaussian_wrapping/beags_prepare.py -m "$O/train" --iteration 2000 --masks "$MASKS"
$PY gaussian_wrapping/beags_finetune.py -m "$O/train" --iteration 2000 --ft_iterations 300 --out "$O/beags"
$PY gaussian_wrapping/beags_eval.py -m "$O/beags" --iteration 2300 --gt_masks "$GT" --masks "$MASKS"
$PY gaussian_wrapping/hollow_metrics.py -m "$O/beags" --iteration 2300
set +x
[ "$TESTS" -eq 0 ] || { echo "pytest failed (see above)" >&2; exit 1; }
echo "smoke OK"
