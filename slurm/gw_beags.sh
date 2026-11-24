#!/bin/bash
#SBATCH --job-name=gw_beags
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
# BEA-GS on a trained GW model: prepare -> eval (labels before finetune) -> finetune -> eval (+ renders, metrics).
# Usage (from repo root):
#   sbatch -J gw_beags_<name> slurm/gw_beags.sh <name> <masks_dir> <gt_masks_dir> [extra beags_finetune.py args]
#   <masks_dir>: (C, H, W) <image_name>.npy masks (SAM2 benchmark masks, or seed2gs.py's <out>/masks).
# Input output/$METHOD/<name>/train (slurm/gw_train.sh); output output/$METHOD/<name>/beags at iteration $FT_ITER.
set -euo pipefail
NAME=$1; MASKS=$2; GT=$3; shift 3
source slurm/common.sh
M=$(model_dir "$NAME" train)
B=$(model_dir "$NAME" beags)
set -x
[ -f "$M/point_cloud/iteration_$ITER/labels.pt" ] && [ -f "$M/beags/labeled_points.npz" ] || \
  $PY gaussian_wrapping/beags_prepare.py -m "$M" --iteration "$ITER" --masks "$MASKS"
$PY gaussian_wrapping/beags_eval.py -m "$M" --iteration "$ITER" --gt_masks "$GT" --masks "$MASKS"
has_iter "$B" "$FT_ITER" || \
  $PY gaussian_wrapping/beags_finetune.py -m "$M" --iteration "$ITER" --out "$B" "$@"
$PY gaussian_wrapping/beags_eval.py -m "$B" --iteration "$FT_ITER" --gt_masks "$GT" --masks "$MASKS" --render
$PY gaussian_wrapping/render.py -m "$B" --iteration "$FT_ITER" --rasterizer ours --skip_train
$PY gaussian_wrapping/metrics.py -m "$B"
