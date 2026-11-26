#!/bin/bash
#SBATCH --job-name=gw_ftprobe
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
# Finetune probe: when and why the object erodes in beags_finetune.py, and whether B10 stops it.
# Re-runs the 3000-it BEA-GS finetune from an existing prepared train model (read only) once per arm, with
# beags/ft_probe.py snapshots every SNAP_EVERY steps, then beags_eval.py on the result.
#   base  the finetune as it ran in the 44-scene batch (the break)
#   b10   + --lambda_alpha_obj 2.5 (the fix)
#   occ0 / sag0 / piv0   one prior switched off (which term causes it)
# Usage (from repo root): sbatch slurm/gw_ftprobe.sh base b10        Outputs (flat):
#   output/ftprobe/<NAME>/<TAG><arm>        finetuned model (iteration FT_ITER) + beags/metrics_<FT_ITER>.json
#   output/ftprobe/<NAME>/snaps_<TAG><arm>  stats.jsonl, label_1/, full/
# Knobs: SRC_MODEL (prepared model to finetune from), TAG (output name prefix), SNAP_EVERY (250),
#        SNAP_DENSE_UNTIL (0; extra snapshots every SNAP_DENSE_EVERY=25 steps up to it).
# Figure: python gaussian_wrapping/beags_ftprobe_figure.py output/ftprobe/<NAME> (CPU, login node).
set -euo pipefail
source slurm/common.sh
NAME=${NAME:-spinnerf_7}
SRC_MODEL=${SRC_MODEL:-output/gw_beags/$NAME/train}
D=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/masks/SPIn-NeRF
GT=${GT:-$D/gt_masks/${NAME#spinnerf_}/masks}
SAM=${SAM:-$D/training/sam2/${NAME#spinnerf_}/masks}
SNAP_EVERY=${SNAP_EVERY:-250}
SNAP_DENSE_UNTIL=${SNAP_DENSE_UNTIL:-0}
SNAP_DENSE_EVERY=${SNAP_DENSE_EVERY:-25}
TAG=${TAG:-}
O=output/ftprobe/$NAME
[ $# -gt 0 ] || { echo "usage: sbatch slurm/gw_ftprobe.sh <arm> [<arm> ...]  (base b10 occ0 sag0 piv0)" >&2; exit 2; }
mkdir -p "$O"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for ARM in "$@"; do
  case $ARM in
    base) FLAGS="" ;;
    b10)  FLAGS="--lambda_alpha_obj 2.5" ;;
    occ0) FLAGS="--lambda_occ 0" ;;
    sag0) FLAGS="--lambda_sag 0" ;;
    piv0) FLAGS="--lambda_piv 0" ;;
    *) echo "unknown arm $ARM" >&2; exit 2 ;;
  esac
  echo "=== arm $ARM: $FLAGS  [$(date +%T)]"
  OUT=$O/$TAG$ARM SNAPS=$O/snaps_$TAG$ARM
  rm -rf "$OUT" "$SNAPS"
  set -x
  # shellcheck disable=SC2086
  $PY gaussian_wrapping/beags_finetune.py -m "$SRC_MODEL" --iteration "$ITER" --out "$OUT" $FLAGS \
      --snap_every "$SNAP_EVERY" --snap_dir "$SNAPS" --snap_gt_masks "$GT" --snap_sam_masks "$SAM" \
      --snap_dense_until "$SNAP_DENSE_UNTIL" --snap_dense_every "$SNAP_DENSE_EVERY"
  $PY gaussian_wrapping/beags_eval.py -m "$OUT" --iteration "$FT_ITER" --gt_masks "$GT" --masks "$SAM"
  set +x
done
echo "ftprobe OK [$(date +%T)]"
