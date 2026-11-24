#!/usr/bin/env bash
# One scene per dataset (GPU budget): Mip-NeRF 360 room (images_4), 3D-OVS bed, LERF figurines, LLFF fern, each through
# slurm/gw_scene.sh. Data, SAM2 and GT mask paths as BEA-GS_FSplat slurm/launch_benchmark.sh (its data/ dir).
# Usage (from repo root): bash slurm/launch_one_per_dataset.sh     (METHOD, REMOVE, ... pass through to gw_scene.sh)
#   MASK_SOURCE=seed2gs: Seed2GS masks from one text prompt per scene instead of the SAM2 masks (job names get
#   _seed2gs; outputs in output/${METHOD}_seed2gs, the train models are shared). Prompts (env): PROMPT_ROOM (footstool),
#   PROMPT_BED (red bag), PROMPT_FIGURINES (rubber duck with red hat), PROMPT_FERN (fern plant). The SAM2 label
#   REMOVE refers to may be another object than the prompt; seed2gs_eval.json matches the prompt to its GT object.
#   Both modes train the same output/$METHOD/<name>/train when it is missing: a job of the other mode for the same
#   scene that is still queued or running (squeue, same job name +/- _seed2gs) becomes an afterany dependency, so two
#   jobs never train into that dir at once (afterany: if it fails, this job trains the model itself).
set -euo pipefail
G=/leonardo_work/IscrC_EditGS/gabri/datasets
MIP=/leonardo_work/IscrC_EditGS/datasets/mip360
MM=$G/mipnerf360_hd_masks/masks
FS=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/BEA-GS_FSplat
mkdir -p logs
SUFFIX=; OTHER=_seed2gs
if [ "${MASK_SOURCE:-provided}" = seed2gs ]; then SUFFIX=_seed2gs; OTHER=; fi
submit() {  # submit JOB PROMPT gw_scene.sh-args...
  local job=$1 prompt=$2 dep ids; shift 2
  ids=$(squeue -h -u "$USER" -n "$job$OTHER" -o %i 2>/dev/null | paste -sd: - || true)  # same scene, other source
  dep=(); [ -n "$ids" ] && dep=(--dependency="afterany:$ids") && echo "$job$SUFFIX after job(s) $ids (shared train model)"
  SEED2GS_PROMPT=$prompt sbatch -J "$job$SUFFIX" "${dep[@]}" slurm/gw_scene.sh "$@"
}
submit gw_mip360_room "${PROMPT_ROOM:-footstool}" mip360_room "$MIP/room" "$MM/training/sam2/room/masks" "$MM/gt_masks/room/masks" -i images_4
submit gw_3dovs_bed "${PROMPT_BED:-red bag}" 3dovs_bed "$FS/data/3DOVS/bed" "$G/3DOVS/masks/training/sam2/bed/masks" "$G/3DOVS/masks/gt_masks/bed/masks"
submit gw_lerf_figurines "${PROMPT_FIGURINES:-rubber duck with red hat}" lerf_figurines "$FS/data/lerf/figurines" "$G/lerf/masks/training/sam2/figurines/masks" "$G/lerf/masks/gt_masks/figurines/masks"
submit gw_llff_fern "${PROMPT_FERN:-fern plant}" llff_fern "$FS/data/llff/fern" "$G/nerf_llff_data_original/masks/training/sam2/fern/masks" "$G/nerf_llff_data_original/masks/gt_masks/fern/masks"
