#!/usr/bin/env bash
# Stage 6: the ArtiFixer diffusion on the stage-5 hole: one rollout over the photo frames, propagation, ArtiFixer3D
# (GW train.py on the inpainted photos), ArtiFixer3D+ (ArtiFixer on the non-anchor photo frames). run.sh skips the
# stages stage 5 finished. Estimate: several hours per scene (never run end to end).
# Virtual views instead of removal: AF_VARIANT=vanilla_object | vanilla_vidsplat | vanilla_travel (novel camera path
# from artifixer_gs/trajectory.py, no removal; own dir output/artifixer/<scene>/<variant>).
# Usage: bash scripts/bakerh/stages/stage6_artifixer.sh   [SCENES=..., VARIANT=..., AF_VARIANT=..., CAPTION=...]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
unset STAGES
run_stage 6_artifixer afinputs,artifixer \
  "  REPORT.txt: 'ArtiFixer', 'ArtiFixer3D', 'ArtiFixer3D+' output dirs and the ArtiFixer3D gs_stats" \
  "  output/artifixer/<scene>/$AF_DIR/<scene>/artifixer/**/frames/batch_0000/pred: the inpainted frames" \
  "  output/artifixer/<scene>/$AF_DIR/<scene>/af3d/export/point_cloud.ply: the inpainted 3D scene" \
  "next: send REPORT.txt + the pred frames; VARIANT=conf / gridfix: stage7_variant.sh"
