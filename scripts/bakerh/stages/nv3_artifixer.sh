#!/usr/bin/env bash
# Novel views 3/3: ArtiFixer on the nv2 paths. For each path in TRAJS: ArtiFixer on the path renders (the photos are
# the anchors), handoff, ArtiFixer3D (GW train.py on photos + fixed frames, 30k), ArtiFixer3D+ (second ArtiFixer pass
# on the ArtiFixer3D renders), then a before/after grid. Finished ArtiFixer-GS stages are skipped.
# Estimate: 2-4 h per path and scene (never run end to end).
# Usage: bash scripts/bakerh/stages/nv3_artifixer.sh   [TRAJS="object", SCENES=..., CAPTION=...]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
unset STAGES
run_trajs nv3_artifixer afinputs,artifixer,trajgrid \
  "  output/$METHOD/showcase/<scene>_<AF_DIR>_grid.jpg: GW render | ArtiFixer | ArtiFixer3D render | ArtiFixer3D+" \
  "  output/artifixer/<scene>/<AF_DIR>/<scene>/af3d/export/point_cloud.ply: the fixed 3D scene" \
  "send back: REPORT.txt + the showcase grids"
