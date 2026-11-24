#!/bin/bash
#SBATCH --job-name=gw_tests
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
# Every pytest file on a GPU (the CUDA tests skip on the login node), without slurm/gw_smoke.sh's fern pipeline
# (which starts with rm -rf output/smoke/fern):
#   gaussian_wrapping/tests (GW env) and submodules/artifixer_gs/tests (artifixer env; the nvdiffrast mesh test in the GW env).
# Usage (from repo root): sbatch slurm/gw_tests.sh [pytest args, e.g. -k seed2gs]
set -uo pipefail
source slurm/common.sh
PYTEST_LIB=${PYTEST_LIB:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/pylib_gw}
AF_PY=${AF_PY:-/leonardo_scratch/fast/IscrC_EditGS/envs/artifixer/bin/python}
export GS_RENDER_PY
RC=0
set -x
PYTHONPATH=$PYTEST_LIB $PY -m pytest -q -rs -p no:cacheprovider gaussian_wrapping/tests "$@" || RC=1
cd submodules/artifixer_gs
PYTHONPATH= $AF_PY -m pytest -q -rs -p no:cacheprovider tests/test_removal.py tests/test_trajectory.py || RC=1
PYTHONPATH=$PYTEST_LIB $PY -m pytest -q -rs -p no:cacheprovider tests/test_removal.py -k mesh_front_cuda || RC=1
set +x
[ "$RC" -eq 0 ] && echo "tests OK" || { echo "tests FAILED (see above)" >&2; exit 1; }
