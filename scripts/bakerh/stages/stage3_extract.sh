#!/usr/bin/env bash
# Stage 3: extraction. Label PLYs (label_0 = background after removal, label_1 = object), 8-view renders, the showcase
# grid, test-view metrics at 33k, hollow metrics. Estimate: 15-30 min per scene.
# Usage: bash scripts/bakerh/stages/stage3_extract.sh   [SCENES=..., VARIANT=...]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
run_stage 3_extract export_beags,renders,showcase,metrics,hollow \
  "  output/$METHOD/showcase/<scene>_grid_all.jpg: rows = photo | full | label_0 background | label_1 extracted object" \
  "  REPORT.txt: 'test views beags 33000' PSNR (should be ~3 dB over 30k), hollow metrics (hole_alpha, floaters)" \
  "  output/$METHOD/<scene>/export_beags/label_{0,1}.ply" \
  "next: bash scripts/bakerh/stages/stage4_mesh.sh   (optional: stage4b_meshocc.sh)"
