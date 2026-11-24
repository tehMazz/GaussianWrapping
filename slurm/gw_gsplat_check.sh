#!/bin/bash
#SBATCH --job-name=gw_gsplat_check
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# gsplat-vs-native render equivalence of the exported 3DGS PLY (plan metric 6, target <= 0.5 dB): export_3dgs_ply.py
#   --check renders the train + test views with render_ours in the GW env, dumps them, and runs gaussian_wrapping/
#   gsplat_check.py in the gsplat env (GS_RENDER_PY) on the exported PLY. Also runs the CPU tests of the check.
# Usage (from repo root): sbatch slurm/gw_gsplat_check.sh [model dir] [iteration]
#   default model output/smoke/fern/train (slurm/gw_smoke.sh), iteration -1 (latest). Knobs (env):
#   GS_RENDER_PY  Python with torch + gsplat (default the gsplat env)
#   CHECK_ARGS    extra export_3dgs_ply.py flags (e.g. "--check_stride 4 --check_splits train --keep_check_views")
# Output (flat, next to the model dir): <model>/../export/{point_cloud.ply,export_info.json,gsplat_check.json}.
set -euo pipefail
MODEL=${1:-output/smoke/fern/train}; MODEL=${MODEL%/}; ITERATION=${2:--1}
source slurm/common.sh
PYTEST_LIB=${PYTEST_LIB:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/pylib_gw}
export GS_RENDER_PY=${GS_RENDER_PY:-/leonardo_scratch/fast/IscrC_EditGS/envs/gsplat/bin/python}
[ -d "$MODEL/point_cloud" ] || { echo "no trained model at $MODEL (run slurm/gw_smoke.sh first)" >&2; exit 2; }
set -x
PYTHONPATH=$PYTEST_LIB $PY -m pytest -q -rs -p no:cacheprovider gaussian_wrapping/tests/test_export_check.py
$GS_RENDER_PY -c "import gsplat, torch; print('gsplat', gsplat.__version__, 'cuda', torch.cuda.is_available())"
$PY gaussian_wrapping/export_3dgs_ply.py -m "$MODEL" --iteration "$ITERATION" --check --gs_python "$GS_RENDER_PY" ${CHECK_ARGS:-}
set +x
REPORT=$(dirname "$MODEL")/export/gsplat_check.json
$PY -c "import json, sys; r = json.load(open(sys.argv[1])); r.pop('views'); print(json.dumps(r, indent=1)); sys.exit(0 if r['pass'] is not False else 1)" "$REPORT"
