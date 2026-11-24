#!/usr/bin/env bash
# Stage 5: ArtiFixer inputs + removal renders + the removal holes side by side, NO diffusion:
#   afinputs (with the GAVIS unseen maps, and the confidence maps when the model has confidence), ArtiFixer-GS
#   prepare,removal,split (default hole: mesh if stage 4 made hole_1.ply, else alpha), then the holes phase:
#   alpha | mesh | gavis 0.5 | gavis 0.3 | alpha + confidence 0.5 (VARIANT=conf only) -> hole_grid.jpg.
# Needs the ArtiFixer repo on the VM (AF_REPO: .venv + checkpoints). Compressor uses frames 49-148,284-304.
# Estimate: 20-60 min per scene (unseen maps over every camera; never measured at 1600 px).
# Usage: bash scripts/bakerh/stages/stage5_holes.sh   [SCENES=..., VARIANT=conf, UNSEEN_THRESHS="0.5 0.3"]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
export STAGES=prepare,removal,split AF_UNSEEN=on
run_stage 5_holes afinputs,artifixer,holes \
  "  output/artifixer/<scene>/$AF_DIR/hole_grid.jpg: photo | removal render | one column per hole mode | unseen map" \
  "  output/artifixer/<scene>/$AF_DIR/<scene>/removal/: background renders without the object (what gets inpainted)" \
  "  REPORT.txt: 'artifixer inputs' (object Gaussians, masks per photo, metric scale, caption)" \
  "next: bash scripts/bakerh/stages/stage6_artifixer.sh"
