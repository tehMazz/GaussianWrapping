#!/bin/bash
#SBATCH --job-name=gw_gavis
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
# GAVIS unseen hole (HOLE_MODE=gavis) on an ArtiFixer removal that already ran prepare + removal (slurm/gw_artifixer.sh):
#   1. GPU tests: test_visibility.py, test_rasterizer_labels.py, test_render_ply_frames.py (TESTS=0 skips them)
#   2. beags/unseen_map.py of the model in <afx>/inputs.json (label, iteration) -> <afx>/unseen (+ vis/, avf.pt)
#   3. artifixer_gs.removal split with HOLE_MODE alpha and gavis (UNSEEN_THRESH list, default "0.5 0.3"), plus mesh when
#      <scene>/removal/mesh_front exists -> <afx>/<scene>/removal_split_<mode>; the existing removal_split is untouched
#   4. beags/hole_grid.py -> <afx>/hole_grid.jpg (photo | removal render | holes | unseen map, 8 frames)
# Usage (from repo root): sbatch slurm/gw_gavis_smoke.sh [<afx dir>, default output/artifixer/fern/normal]
set -uo pipefail
source slurm/common.sh
AFX=$(realpath "${1:-output/artifixer/fern/normal}")
PYTEST_LIB=${PYTEST_LIB:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/pylib_gw}
AF_PY=${AF_PY:-/leonardo_scratch/fast/IscrC_EditGS/envs/artifixer/bin/python}
read -r MODEL ITER LABEL COLMAP IMAGES < <($PY -c "
import json; d = json.load(open('$AFX/inputs.json')); print(d['model'], d['iteration'], d['label'], d['COLMAP'], d['IMAGES'])")
RC=0
set -x
if [ "${TESTS:-1}" = 1 ]; then  # before env.sh: its CENTER_PRINCIPAL_POINT etc. would leak into the tests
  PYTHONPATH=$PYTEST_LIB $PY -m pytest -q -rs -p no:cacheprovider gaussian_wrapping/tests/test_visibility.py \
    gaussian_wrapping/tests/test_rasterizer_labels.py gaussian_wrapping/tests/test_render_ply_frames.py || RC=1
fi
set -a; source "$AFX/env.sh"; set +a                      # OUT, MASK_DIR, MASK_LABEL
$PY gaussian_wrapping/beags/unseen_map.py -m "$MODEL" --iteration "$ITER" --remove "$LABEL" --source_path "$COLMAP" \
  --images "$IMAGES" --out "$AFX/unseen" --kappa "${KAPPA:-1}" --vis || exit 1
split() {  # <name> <hole_mode args...>
  local name=$1; shift
  (cd submodules/artifixer_gs && PYTHONPATH= $AF_PY -m artifixer_gs.removal split --scene_root "$OUT" \
    --removal_dir "$OUT/removal" --mask_dir "$MASK_DIR" ${MASK_LABEL:+--mask_label "$MASK_LABEL"} \
    --output_dir "$OUT/removal_split_$name" --hole_shape mask --dilate_px "${DILATE:-12}" --opacity hole0 "$@") || RC=1
  GRID+=(--split "$name=$OUT/removal_split_$name")
}
GRID=()
split alpha --hole_mode alpha
[ ! -d "$OUT/removal/mesh_front" ] || split mesh --hole_mode mesh
for t in ${UNSEEN_THRESH:-0.5 0.3}; do
  split "gavis$t" --hole_mode gavis --unseen_dir "$AFX/unseen" --unseen_threshold "$t"
done
$PY gaussian_wrapping/beags/hole_grid.py "${GRID[@]}" --unseen_dir "$AFX/unseen" --n 8 --out "$AFX/hole_grid.jpg" || RC=1
set +x
[ "$RC" -eq 0 ] && echo "gavis smoke OK" || { echo "gavis smoke FAILED (see above)" >&2; exit 1; }
