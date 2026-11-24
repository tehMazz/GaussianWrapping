#!/bin/bash
#SBATCH --job-name=gw_mesh
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/%x_%j.out
# Multi-label mesh + object holes of a labelled GW model: GW pivot mesh (candidate points) -> multi-label PAM.
# Usage (from repo root):
#   sbatch slurm/gw_mesh.sh <model_dir> <iteration> [multilabel_mesh_extraction.py args, e.g. --remove 1 --max_points 200000]
#   <model_dir> = output/<method>/<scene>/<mode>; labels from point_cloud/iteration_<it>/labels.pt.
#   Hole = label-k boundary + k<->0 contact patch dilated by h (--contact_h, default auto = BEA-GS voxel edge x 4.30).
# Output: <model_dir>/mesh_ours_2pivots.ply (GW) and, flat, output/<method>/<scene>/mesh/{mesh_labels,label_<k>,hole_<k>,
#   hole_<k>_closure}.ply + hole_<k>.json (h, face counts) + tets.npz (SAVE_TETS=1, default).
#   MESH_OUT overrides that dir (gw_scene.sh OCC_SOURCE=mesh: output/<method>/<scene>/mesh_train, the pre-finetune
#   mesh whose tets.npz beags_finetune.py --occ_source mesh reads).
# Other h on CPU (login node, no GPU, no sbatch), from tets.npz:
#   $PY gaussian_wrapping/multilabel_mesh_extraction.py --from_tets output/<method>/<scene>/mesh/tets.npz --remove 1 \
#       --contact_h <h> --out_dir output/<method>/<scene>/mesh_h<h>
set -euo pipefail
M=$1; IT=$2; shift 2
source slurm/common.sh
OUT=${MESH_OUT:-$(dirname "$M")/mesh}
SAVE_TETS=${SAVE_TETS:-1}                              # 1: --save_tets (labelled Delaunay, a few 100 MB)
TETS_ARGS=""; [ "$SAVE_TETS" = 1 ] && TETS_ARGS="--save_tets"
set -x
# shellcheck disable=SC2086
[ -f "$M/mesh_ours_2pivots.ply" ] || $PY gaussian_wrapping/pivot_based_mesh_extraction.py -m "$M" --iteration "$IT" $GW_EXTRACT_ARGS
# shellcheck disable=SC2086
$PY gaussian_wrapping/multilabel_mesh_extraction.py -m "$M" --iteration "$IT" --input_mesh "$M/mesh_ours_2pivots.ply" \
    --out_dir "$OUT" $TETS_ARGS "$@"
