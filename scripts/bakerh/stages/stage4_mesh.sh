#!/usr/bin/env bash
# Stage 4: GW pivot mesh of the 33k model + multi-label mesh + removal hole of label 1 (object faces + contact patch).
# Stage 5 uses hole_1.ply for the mesh hole (else alpha). Estimate: 1-3 h per scene (soft: a failure does not stop the run).
# Usage: bash scripts/bakerh/stages/stage4_mesh.sh   [SCENES=..., VARIANT=..., MESH_ARGS=...]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
run_stage 4_mesh mesh \
  "  REPORT.txt: 'mesh faces' (mesh_labels, label_0, label_1, hole_1, hole_1_closure)" \
  "  output/$METHOD/<scene>/mesh/{label_0,label_1,hole_1}.ply: open in MeshLab; hole_1.json = contact h + face counts" \
  "next: bash scripts/bakerh/stages/stage5_holes.sh"
