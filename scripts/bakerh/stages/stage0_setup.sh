#!/usr/bin/env bash
# Stage 0: one-time VM setup (venv, CUDA extensions, GPU tests, LPIPS, gsplat, ArtiFixer checks) + per-scene check
# (dataset, SAM3 masks). Estimate: 20-40 min the first time, seconds after. Usage: bash scripts/bakerh/stages/stage0_setup.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
SKIP_SETUP=
run_stage 0_setup check \
  "  output/bakerh_logs/setup/SETUP_REPORT.txt: last line SETUP OK (else: which step failed and why)" \
  "  REPORT.txt 'python' section: every extension 'ok' (tetranerf / nvdiffrast are needed by stages 4 and 5)" \
  "next: bash scripts/bakerh/stages/stage1_train.sh"
