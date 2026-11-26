#!/bin/bash
#SBATCH --job-name=mono_prior
#SBATCH --account=IscrC_P2R-GS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# MoGe-2 depth + normal priors for one scene, plus their agreement with the COLMAP points (gaussian_wrapping/mono_prior_prepare.py).
#   sbatch slurm/mono_prior_prepare.sh <scene>     scene: spinnerf_7 | mip360_kitchen
# Output: output/mono_prior/<scene>/moge2/{depth,normal}/<image>.npy, stats.json, viz.jpg
set -euo pipefail
S=${1:?scene}
PY=/leonardo_scratch/fast/IscrC_EditGS/envs/artifixer/bin/python
export HF_HUB_OFFLINE=1
case $S in
  spinnerf_7)     SRC=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/scenes/SPIn-NeRF/7; ARGS="--width 1008" ;;
  mip360_kitchen) SRC=/leonardo_work/IscrC_EditGS/datasets/mip360/kitchen; ARGS="-i images_4" ;;
  *) echo "unknown scene $S" >&2; exit 2 ;;
esac
set -x
$PY gaussian_wrapping/mono_prior_prepare.py -s "$SRC" $ARGS --out output/mono_prior/$S/moge2
set +x
echo "mono_prior $S OK [$(date +%T)]"
