#!/bin/bash
#SBATCH --job-name=fb2_test
#SBATCH --account=IscrC_P2R-GS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
# Build the full-bracket v2 rasterizers ($F/{beags_fb2,ours_fb2}: copies of submodules/diff-gaussian-rasterization_ours*,
# whose sources carry the v2 patch) and check
# them with gaussian_wrapping/full_bracket_check.py against the installed stock build and the v1 test build:
# v2 must be bit-identical to stock wherever stock found a depth (render, sample_depth, integrate), fill stock's holes,
# and agree with v1 on the filled values. Models: old GW base spinnerf_7 (stock: 12-14 % holes), gw_today kitchen and
# spinnerf_7.  sbatch slurm/full_bracket_v2_test.sh   -> output/depthfix/fb2_check/*.npz + this log
set -uo pipefail
unset RASTER_PATH
source slurm/common.sh
F=${F:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/fb2}
V1=/leonardo_work/IscrC_EditGS/tmp_amazzucc/raster_variants
O=output/depthfix/fb2_check; mkdir -p $O
for v in beags_fb2 ours_fb2; do
  src=submodules/diff-gaussian-rasterization_${v%_fb2}; [ $v = beags_fb2 ] && src=submodules/diff-gaussian-rasterization_ours_beags
  if [ ! -d $F/$v ]; then
    mkdir -p $F && rsync -a --exclude build --exclude '*.egg-info' --exclude __pycache__ --exclude '*.so' $src/ $F/$v/
  fi
  if ! ls $F/$v/diff_gaussian_rasterization_*/_C*.so >/dev/null 2>&1; then
    (cd $F/$v && rm -rf build && $PY setup.py build_ext --inplace > $F/build_$v.log 2>&1) || { tail -30 $F/build_$v.log; exit 1; }
  fi
  echo "[fb2] built $v"
done
declare -A MODELS=([oldbase_spinnerf_7]=output/gw_beags/spinnerf_7/train [today_kitchen]=output/gw_today/mip360_kitchen/train
                   [today_spinnerf_7]=output/gw_today/spinnerf_7/train)
for name in "${!MODELS[@]}"; do
  M=${MODELS[$name]}
  PYTHONPATH=$PYTHONPATH $PY gaussian_wrapping/full_bracket_check.py dump -m $M --iteration 30000 --out $O/${name}_stock.npz
  PYTHONPATH=$V1/beags_full:$V1/ours_full:$PYTHONPATH $PY gaussian_wrapping/full_bracket_check.py dump -m $M --iteration 30000 --out $O/${name}_v1.npz
  PYTHONPATH=$F/beags_fb2:$F/ours_fb2:$PYTHONPATH $PY gaussian_wrapping/full_bracket_check.py dump -m $M --iteration 30000 --out $O/${name}_v2.npz
  echo "===== $name: v2 vs stock (ref v1)"
  $PY gaussian_wrapping/full_bracket_check.py compare $O/${name}_stock.npz $O/${name}_v2.npz --ref $O/${name}_v1.npz
  echo "===== $name: v1 vs stock"
  $PY gaussian_wrapping/full_bracket_check.py compare $O/${name}_stock.npz $O/${name}_v1.npz | tail -2
done
echo "fb2_test done [$(date +%T)]"
