#!/bin/bash
#SBATCH --job-name=verify_fullbracket
#SBATCH --account=IscrC_P2R-GS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# After slurm/install_full_bracket.sh: the INSTALLED rasterizers (no RASTER_PATH) must give the full-bracket depth.
# Renders the old GW base of spinnerf_7 (stock rasterizer: 12-14 % depth-0 pixels per view, output/gwmesh/spinnerf_7/
# depth_views_stock) and checks depth0_share == 0 in every view; prints which .so files were imported.
set -euo pipefail
unset RASTER_PATH
source slurm/common.sh
$PY - <<'PY'
import hashlib, diff_gaussian_rasterization_ours as a, diff_gaussian_rasterization_ours_beags as b, os
for m in (a, b):
    so = [f for f in os.listdir(os.path.dirname(m.__file__)) if f.endswith(".so")][0]
    p = os.path.join(os.path.dirname(m.__file__), so)
    print("[verify]", m.__name__, p, hashlib.md5(open(p, "rb").read()).hexdigest())
PY
O=output/depthfix/spinnerf_7/depth_views_installed
$PY gaussian_wrapping/render_depth_views.py -m output/gw_beags/spinnerf_7/train --iteration 30000 --out $O
$PY - "$O/stats.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); s = [x["depth0_share"] for x in r]
print("[verify] depth0_share per view", [round(v, 5) for v in s])
sys.exit(0 if max(s) == 0 else 1)
PY
echo "verify_full_bracket OK [$(date +%T)]"
