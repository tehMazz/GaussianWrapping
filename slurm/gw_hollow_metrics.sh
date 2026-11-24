#!/bin/bash
#SBATCH --job-name=gw_hollow_metrics
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# Hollowing metrics (unseen / free-space mass, floaters, fog, hole alpha) of one model iteration.
# Usage (from repo root):
#   sbatch slurm/gw_hollow_metrics.sh <model_dir> <iteration> [--label k [--masks <dir>]] [extra hollow_metrics.py args]
#   --label k: hole metrics for object k (object mask = labels.pt == k, written next to the metrics json);
#   --masks <dir> adds hole tightness. Output: <model_dir>/hollow_metrics_<iteration>.json.
set -euo pipefail
M=$1; IT=$2; shift 2
LABEL=
if [ "${1:-}" = --label ]; then LABEL=$2; shift 2; fi
source slurm/common.sh
set -x
OBJ=()
if [ -n "$LABEL" ]; then
  OBJ=(--object_mask "$M/object_mask_${LABEL}_$IT.npy" --mask_channel "$LABEL")
  $PY -c "import sys, numpy as np, torch; np.save(sys.argv[2], torch.load(sys.argv[1], map_location='cpu').numpy() == int(sys.argv[3]))" \
    "$M/point_cloud/iteration_$IT/labels.pt" "$M/object_mask_${LABEL}_$IT.npy" "$LABEL"
fi
$PY gaussian_wrapping/hollow_metrics.py -m "$M" --iteration "$IT" "${OBJ[@]}" "$@"
