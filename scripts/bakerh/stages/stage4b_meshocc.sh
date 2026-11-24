#!/usr/bin/env bash
# Stage 4b (optional): BEA-GS finetune supervised by the model's own multi-label mesh (--occ_source mesh) instead of the
# labelled voxels: mesh of the prepared 30k model (tets.npz) -> finetune -> IoU vs SAM3 -> renders -> metrics.
# Compare with stage 2/3. Estimate: 1-3 h mesh + 15-30 min per scene. Output: output/<method>/<scene>/{mesh_train,beags_meshocc}.
# Usage: bash scripts/bakerh/stages/stage4b_meshocc.sh   [SCENES=..., VARIANT=...]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
run_stage 4b_meshocc mesh_train,finetune_meshocc,labels_meshocc,renders_meshocc,metrics \
  "  REPORT.txt: 'BEA-GS vs SAM3 masks beags_meshocc 33000' IoU / BIoU vs the 'beags 33000' line of stage 2" \
  "  output/$METHOD/showcase/<scene>_meshocc_grid_all.jpg next to <scene>_grid_all.jpg" \
  "next: bash scripts/bakerh/stages/stage5_holes.sh"
