#!/usr/bin/env bash
# Novel views 2/3: the candidate camera paths, no diffusion. For each path in TRAJS (object vidsplat travel):
# ArtiFixer-GS inputs from the 33k model, then its trajectory + prepare stages (path, collision checks, GW renders on
# the path, metric scale, caption). Compressor: at most 121 path frames (COMPRESSOR_TRAJ_FRAMES).
# Estimate: 10-30 min per path and scene.
# Usage: bash scripts/bakerh/stages/nv2_paths.sh   [TRAJS="object", SCENES=...]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
export STAGES=trajectory,prepare
run_trajs nv2_paths afinputs,artifixer,trajcands \
  "  output/artifixer/<scene>/<AF_DIR>/<scene>/trajectory_input.png: top view of the path over the walls" \
  "  output/artifixer/<scene>/<AF_DIR>/path_candidates.jpg: 16 GW renders along the path (holes / floaters = what ArtiFixer fixes)" \
  "  output/artifixer/<scene>/<AF_DIR>/<scene>/trajectory_input_info.json: frames, clearance, rejected cameras" \
  "next: bash scripts/bakerh/stages/nv3_artifixer.sh"
