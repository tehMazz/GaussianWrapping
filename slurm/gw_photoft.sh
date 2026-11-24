#!/bin/bash
#SBATCH --job-name=gw_photoft
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
# HANDOFF 7.1 experiment 1: photo-only finetune of the 30k GW model (beags_finetune.py with SAG / occupancy / pivot
# weights 0: L1 + D-SSIM on the raw render, no grid; never-hit and low-opacity pruning as in the BEA-GS finetune),
# then test renders + metrics. ~30 dB means the BEA-GS finetune's +3.2 dB on kitchen is low-frequency colour re-fit.
# Usage (from repo root, debug QoS):
#   sbatch slurm/gw_photoft.sh [name, default mip360_kitchen] [extra beags_finetune.py args]
# Input output/$SRC_METHOD/<name>/train at $ITER (gw_beags; needs beags_prepare's labels.pt and beags/masks, as made by
# slurm/gw_beags.sh). Output output/$METHOD/<name>/ft at $FT_ITER (METHOD gw_photoft), results.json printed next to
# the input's.
set -euo pipefail
NAME=${1:-mip360_kitchen}; shift $(( $# > 0 ? 1 : 0 ))
METHOD=${METHOD:-gw_photoft}
source slurm/common.sh
SRC_METHOD=${SRC_METHOD:-gw_beags}
M=output/$SRC_METHOD/$NAME/train
O=$(model_dir "$NAME" ft)
set -x
has_iter "$O" "$FT_ITER" || \
  $PY gaussian_wrapping/beags_finetune.py -m "$M" --iteration "$ITER" --out "$O" \
      --lambda_sag 0 --lambda_occ 0 --lambda_piv 0 "$@"
$PY gaussian_wrapping/render.py -m "$O" --iteration "$FT_ITER" --rasterizer ours --skip_train
$PY gaussian_wrapping/metrics.py -m "$O"
set +x
echo "== $M (raw 30k)"; cat "$M/results.json"; echo
echo "== $O (photo-only finetune)"; cat "$O/results.json"; echo
