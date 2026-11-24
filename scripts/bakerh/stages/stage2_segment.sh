#!/usr/bin/env bash
# Stage 2: BEA-GS segmentation from the SAM3 masks: consensus masks + per-Gaussian labels (30k), 3000-it finetune
# (dilation weight 0.1) -> 33k, IoU / BIoU against the SAM3 masks at 30k and 33k. Estimate: 15-30 min per scene.
# Usage: bash scripts/bakerh/stages/stage2_segment.sh   [SCENES=..., VARIANT=...]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
run_stage 2_segment prepare,finetune,labels \
  "  REPORT.txt: 'BEA-GS vs SAM3 masks' IoU / BIoU at 30000 and 33000 (agreement with the input masks: BakerH has no GT)" \
  "  REPORT.txt: Gaussians per label after prepare and after the finetune" \
  "next: bash scripts/bakerh/stages/stage3_extract.sh   (optional: stage2b_seed2gs.sh)"
