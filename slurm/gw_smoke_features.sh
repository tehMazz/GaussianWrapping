#!/bin/bash
#SBATCH --job-name=gw_smoke_feat
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
# GPU smoke of the 2026-10-29 additions on the existing LLFF fern smoke models (output/smoke/fern/{train,beags}, from
# slurm/gw_smoke.sh; nothing there is deleted). Every step runs even when an earlier one fails; a summary closes the log.
#   grid      train.py 2000 its -r 8 with --bilagrid_identity_weight / _mean_weight / _until / --lambda_raw (+ confidence)
#   gsplat    export_3dgs_ply.py --check through the gsplat env (GS_RENDER_PY) -> fern_features/export/gsplat_check.json
#   conf      beags/confidence_map.py on the beags model (trained with --confidence)
#   mesh      slurm/gw_mesh.sh --remove 1: contact-patch hole -> output/smoke/fern/mesh (old one kept as mesh_prev/)
#   seed2gs   seed2gs.py "fern plant" + seed2gs_eval.py vs GT and the BEA-GS labels
#   artifixer slurm/gw_artifixer.sh STAGES=prepare+removal+split (no diffusion), HOLE_MODE=mesh, CONF_THRESH=0.5
# Usage (from repo root): sbatch slurm/gw_smoke_features.sh [steps, default all]   e.g. sbatch ... "gsplat seed2gs"
#   CHECK_ARGS: extra export_3dgs_ply.py flags of the gsplat step (e.g. --keep_check_views)
# Outputs: output/smoke/fern_features/{grid,export,confidence,seed2gs}, output/smoke/fern/mesh, output/artifixer/fern/normal.
# The GPU pytest run is separate: slurm/gw_tests.sh.
set -uo pipefail
source slurm/common.sh
STEPS=${1:-"grid gsplat conf mesh seed2gs artifixer"}
G=/leonardo_work/IscrC_EditGS/gabri/datasets/nerf_llff_data_original/masks
SRC=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/BEA-GS_FSplat/data/llff/fern
GT=$G/gt_masks/fern/masks
F=output/smoke/fern; O=output/smoke/fern_features
T=$F/train; B=$F/beags
mkdir -p "$O"
declare -A RC
step() { local name=$1; shift; [[ " $STEPS " == *" $name "* ]] || return 0
  echo; echo "===== $name ($(date +%T))"; local t0=$SECONDS
  ( set -ex; "$@" ); RC[$name]="rc=$? $((SECONDS - t0))s"; }

s_grid() {
  rm -rf "$O/grid"
  $PY gaussian_wrapping/train.py -s "$SRC" -m "$O/grid" -r 8 --eval --rasterizer ours --data_device cpu \
      --iterations 2000 --test_iterations 2000 --save_iterations 1000 2000 --densify_until_iter 1000 \
      --regularization_from_iter 1000 --no-exposure_compensation --appearance bilagrid --confidence \
      --bilagrid_identity_weight 10 --bilagrid_mean_weight 10 --bilagrid_until 1500 --lambda_raw 0.2
  ls "$O/grid"/bilagrid1000.pth "$O/grid"/bilagrid2000.pth
  cat "$O/grid/gs_stats.json"
}
s_gsplat() {
  $PY gaussian_wrapping/export_3dgs_ply.py -m "$T" --iteration 2000 --out "$O/export/point_cloud.ply" \
      --check --gs_python "$GS_RENDER_PY" ${CHECK_ARGS:-}
  cat "$O/export/gsplat_check.json" | head -40
}
s_conf() {
  $PY gaussian_wrapping/beags/confidence_map.py -m "$B" --iteration 2300 --out "$O/confidence"
  cat "$O/confidence/info.json"
}
s_mesh() {
  [ ! -d "$F/mesh" ] || [ -d "$F/mesh_prev" ] || cp -r "$F/mesh" "$F/mesh_prev"
  bash slurm/gw_mesh.sh "$B" 2300 --remove 1
  cat "$F/mesh/hole_1.json"
}
s_seed2gs() {
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $PY gaussian_wrapping/seed2gs.py -m "$B" --iteration 2300 --prompt "fern plant" --out "$O/seed2gs/fern_plant"
  $PY gaussian_wrapping/seed2gs_eval.py --seed "$O/seed2gs/fern_plant" --gt_masks "$GT" \
      --labels "$B/point_cloud/iteration_2300/labels.pt"
  $PY -c "import json; d=json.load(open('$O/seed2gs/fern_plant/seed2gs.json')); print('seconds', d['seconds']); print('sam3_ready', d.get('sam3_ready'))"
  cat "$O/seed2gs/fern_plant/seed2gs_eval.json" | head -40
}
s_artifixer() {
  METHOD=smoke FT_ITER=2300 FALLBACK_MASKS=$G/training/sam2/fern/masks HOLE_MODE=mesh CONF_THRESH=0.5 STAGES=prepare+removal+split METRIC_SCALE=1 \
    CAPTION="A fern plant in front of a wall, photographed outdoors." AF_ROOT=/nonexistent \
    bash slurm/gw_artifixer.sh fern 1 normal "$T/beags/masks"
  cat output/artifixer/fern/normal/inputs.json | head -60
}
for s in grid gsplat conf mesh seed2gs artifixer; do step "$s" "s_$s"; done

echo; echo "===== summary"
for s in grid gsplat conf mesh seed2gs artifixer; do [ -z "${RC[$s]:-}" ] || echo "$s: ${RC[$s]}"; done
