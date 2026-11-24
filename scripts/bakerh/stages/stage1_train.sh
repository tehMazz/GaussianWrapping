#!/usr/bin/env bash
# Stage 1: Gaussian Wrapping 30k with the bilateral grid + hollowing, PLY export (+ gsplat check), test-view metrics.
# Estimate (never measured at 1600 px): 1-2 h TA_BLUE_MOTOR, longer for Compressor (923 photos).
# Usage: bash scripts/bakerh/stages/stage1_train.sh   [SCENES=..., VARIANT=conf|gridfix]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
run_stage 1_train check,masks,train,export,metrics \
  "  REPORT.txt: 'test views train 30000' PSNR / SSIM / LPIPS (raw render: the grid lowers it, see HANDOFF.md 7.1)" \
  "  output/$METHOD/<scene>/train/test/ours_30000/renders: test renders next to gt/" \
  "next: bash scripts/bakerh/stages/stage2_segment.sh"
