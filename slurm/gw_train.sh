#!/bin/bash
#SBATCH --job-name=gw_train
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
# GW training (--rasterizer ours, bilateral grid, hollowing) + test renders and metrics.
# Usage (from repo root):
#   sbatch -J gw_train_<name> slurm/gw_train.sh <name> <scene_dir> [extra train.py args, e.g. -i images_4]
# Output: output/$METHOD/<name>/train (METHOD gw_beags; ROBUST_ARGS / HOLLOW_ARGS / GW_TRAIN_ARGS in slurm/common.sh,
# e.g. METHOD=gw HOLLOW_ARGS= for the plain GW arm). Skips training when iteration $ITER exists.
set -euo pipefail
NAME=$1; SRC=$2; shift 2
source slurm/common.sh
M=$(model_dir "$NAME" train)
set -x
# shellcheck disable=SC2086
has_iter "$M" "$ITER" || \
  $PY gaussian_wrapping/train.py -s "$SRC" -m "$M" --eval --iterations "$ITER" \
      --test_iterations "$ITER" --save_iterations "$ITER" $GW_TRAIN_ARGS $ROBUST_ARGS $HOLLOW_ARGS "$@"
$PY gaussian_wrapping/render.py -m "$M" --iteration "$ITER" --rasterizer ours --skip_train
$PY gaussian_wrapping/metrics.py -m "$M"
