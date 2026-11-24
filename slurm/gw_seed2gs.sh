#!/bin/bash
#SBATCH --job-name=gw_seed2gs
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
# Seed2GS: training-free text-prompt labels + BEA-GS masks on a trained GW model (SAM3 runs in its own env, SAM3_PYTHON),
# then, with GT set, seed2gs_eval.py (2D IoU / BIoU vs GT masks, per-Gaussian agreement vs a provided-mask labels.pt).
# Usage (from repo root):
#   [GT=<gt_masks_dir>] [LABELS=<labels.pt>] sbatch slurm/gw_seed2gs.sh <model_dir> <iteration> "<prompt>[|<prompt 2>...]" \
#       [extra seed2gs.py args, e.g. --ref_view 0]
#   Several '|'-separated prompts share one SAM3 load (e.g. LERF figurines, 7 objects:
#   "green apple|green toy chair|old camera|porcelain hand|red apple|red toy chair|rubber duck with red hat").
#   LABELS defaults to <model_dir>/point_cloud/iteration_<iteration>/labels.pt (a provided-mask beags_prepare.py of the
#   same model) when it exists. SEED2GS_ROOT (default output/seed2gs/<name>, <name> = the dir above <model_dir>, as in
#   output/<method>/<name>/train).
# Output: $SEED2GS_ROOT/<prompt slug>/{labels.pt,masks/,seed2gs.json[,seed2gs_eval.json]} (a prompt SAM3 does not
#   find: seed2gs_failed.json only), plus $SEED2GS_ROOT/seed2gs_eval.json (mIoU over prompts, failed ones as empty
#   predictions) with several prompts. The eval runs even when a prompt failed; the job then exits non-zero.
#   BEA-GS with these masks: MASK_SOURCE=seed2gs SEED2GS_PROMPT="<prompt>" slurm/gw_scene.sh ... (its own
#   output/$SEED2GS_METHOD dirs; not slurm/gw_beags.sh, which would write into the shared provided-mask train model).
set -euo pipefail
M=$1; IT=$2; PROMPTS=$3; shift 3
source slurm/common.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
ROOT_S2=${SEED2GS_ROOT:-output/seed2gs/$(basename "$(dirname "$(realpath "$M")")")}
LABELS=${LABELS:-$M/point_cloud/iteration_$IT/labels.pt}
IFS='|' read -r -a LIST <<< "$PROMPTS"
ARGS=(); DIRS=()
for p in "${LIST[@]}"; do ARGS+=(--prompt "$p"); DIRS+=("$ROOT_S2/$(slug "$p")"); done
set -x
rc=0
$PY gaussian_wrapping/seed2gs.py -m "$M" --iteration "$IT" "${ARGS[@]}" --out_root "$ROOT_S2" "$@" || rc=$?
if [ -n "${GT:-}" ]; then
  $PY gaussian_wrapping/seed2gs_eval.py --seed "${DIRS[@]}" --gt_masks "$GT" --labels "$LABELS"
fi
exit "$rc"
