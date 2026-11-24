#!/bin/bash
#SBATCH --job-name=gw_smoke_meshocc
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
# GPU smoke of beags_finetune.py --occ_source mesh on the LLFF fern smoke model (output/smoke/fern/train, iteration
# 2000, prepared by slurm/gw_smoke.sh; nothing there is deleted). Every step runs even when an earlier one fails.
#   mesh      gw_mesh.sh on the prepared train model -> output/smoke/fern/mesh_train (tets.npz)
#   finetune  300 its, --occ_source mesh -> output/smoke/fern/beags_meshocc (the voxel run is output/smoke/fern/beags)
#   eval      beags_eval.py vs GT + beags_render_labels.py (8 views) -> output/smoke/fern/renders_labels_meshocc_2300
#             (the voxel run: renders_labels_2300)
# Usage (from repo root): sbatch slurm/gw_smoke_meshocc.sh [steps, default all]   FT_ARGS: extra finetune flags
set -uo pipefail
source slurm/common.sh
STEPS=${1:-"mesh finetune eval"}
G=/leonardo_work/IscrC_EditGS/gabri/datasets
MASKS=$G/nerf_llff_data_original/masks/training/sam2/fern/masks
GT=$G/nerf_llff_data_original/masks/gt_masks/fern/masks
F=output/smoke/fern; T=$F/train; MT=$F/mesh_train; O=$F/beags_meshocc
declare -A RC
step() { local name=$1; shift; [[ " $STEPS " == *" $name "* ]] || return 0
  echo; echo "===== $name ($(date +%T))"; local t0=$SECONDS
  ( set -ex; "$@" ); RC[$name]="rc=$? $((SECONDS - t0))s"; }

s_mesh() {
  rm -f "$MT"/dist_*.npz
  MESH_OUT=$MT SAVE_TETS=1 bash slurm/gw_mesh.sh "$T" 2000
  ls -la "$MT"
}
s_finetune() {
  rm -rf "$O"
  # shellcheck disable=SC2086
  $PY gaussian_wrapping/beags_finetune.py -m "$T" --iteration 2000 --ft_iterations 300 --out "$O" \
      --occ_source mesh --mesh_dir "$MT" ${FT_ARGS:-}
}
s_eval() {
  $PY gaussian_wrapping/beags_eval.py -m "$O" --iteration 2300 --gt_masks "$GT" --masks "$MASKS"
  $PY gaussian_wrapping/beags_render_labels.py -m "$O" --iteration 2300 --n_views 8 --out "$F/renders_labels_meshocc_2300"
}

step mesh s_mesh
step finetune s_finetune
step eval s_eval
echo; echo "===== summary"
for s in mesh finetune eval; do [ -n "${RC[$s]:-}" ] && echo "$s: ${RC[$s]}"; done
