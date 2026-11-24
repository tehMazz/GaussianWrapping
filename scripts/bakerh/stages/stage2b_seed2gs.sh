#!/usr/bin/env bash
# Stage 2b (optional): Seed2GS text-prompt segmentation on the 30k model, mask IoU against the SAM3 masks.
# Needs a SAM3 env on the VM: SAM3_PYTHON (+ SAM3_REPO, SAM3_CHECKPOINT). Prompts: common.sh seed_prompt
# (TA_BLUE_MOTOR "blue electric motor", Compressor "compressor"; SEED2GS_PROMPT overrides). Estimate: 2-5 min per scene.
# Usage: SAM3_PYTHON=/path/to/sam3/bin/python bash scripts/bakerh/stages/stage2b_seed2gs.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
[ -n "${SAM3_PYTHON:-}" ] && [ -x "$SAM3_PYTHON" ] || { echo "set SAM3_PYTHON to the Python of a SAM3 env (also SAM3_REPO, SAM3_CHECKPOINT)"; exit 2; }
export SAM3_PYTHON
run_stage 2b_seed2gs seed2gs \
  "  REPORT.txt: 'seed2gs' SAM3 score, object Gaussians, seconds, mask IoU vs the SAM3 masks" \
  "  output/seed2gs/<scene>/<prompt>/masks: the Seed2GS masks per view" \
  "next: bash scripts/bakerh/stages/stage3_extract.sh"
