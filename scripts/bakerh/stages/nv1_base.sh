#!/usr/bin/env bash
# Novel views 1/3: the base reconstruction. GW 30k with the bilateral grid + hollowing, export, BEA-GS labels (the
# object path orbits label 1) + 3000-it finetune (it also re-fits the colour the grid shifted: ~+3 dB on kitchen), test
# metrics. ArtiFixer starts from the 33k model. Estimate (never measured at 1600 px): 1.5-3 h per scene, Compressor longer.
# Usage: bash scripts/bakerh/stages/nv1_base.sh   [SCENES="TA_BLUE_MOTOR Compressor"]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
run_stage nv1_base check,masks,train,export,prepare,finetune,metrics \
  "  REPORT.txt: 'test views' PSNR at 30000 and 33000; Gaussians per label after the finetune" \
  "  output/$METHOD/<scene>/beags: the model ArtiFixer starts from" \
  "next: bash scripts/bakerh/stages/nv2_paths.sh"
