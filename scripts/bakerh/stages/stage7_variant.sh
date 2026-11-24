#!/usr/bin/env bash
# Stage 7: a training variant through stages 1-3 in its own output dir, to compare with the baseline:
#   VARIANT=conf     train.py --confidence (per-Gaussian confidence; then stage5_holes.sh with VARIANT=conf adds the
#                    confidence hole)                                   -> output/gw_conf_beags_dw0.1
#   VARIANT=gridfix  bilateral-grid drift fixes (GRIDFIX_ARGS: identity + mean pulls 10, grid off after 25000,
#                    lambda_raw 0.2; untested at 30k)                    -> output/gw_gridfix_beags_dw0.1
# Estimate: as stages 1-3 (1.5-3 h per scene, Compressor longer).
# Usage: VARIANT=conf bash scripts/bakerh/stages/stage7_variant.sh
[ -n "${VARIANT:-}" ] || { echo "set VARIANT=conf or VARIANT=gridfix"; exit 2; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
run_stage 7_variant check,masks,train,export,prepare,finetune,labels,export_beags,renders,showcase,metrics,hollow \
  "  REPORT.txt: 'test views train 30000' PSNR vs the baseline stage 1 (gridfix should raise it)" \
  "  output/$METHOD/showcase/<scene>_grid_all.jpg vs output/gw_beags_dw0.1/showcase/<scene>_grid_all.jpg" \
  "next (conf): VARIANT=conf bash scripts/bakerh/stages/stage5_holes.sh"
