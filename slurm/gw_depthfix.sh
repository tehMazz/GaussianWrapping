#!/bin/bash
#SBATCH --job-name=gw_depthfix
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
# Median-depth holes: test builds of the BEA-GS rasterizer with a wider depth-search window, on one scene.
#   sbatch slurm/gw_depthfix.sh build     build each variant in $VARIANTS (in place, in its own copy under $B), then
#                                         beags_depth_holes.py on the 30k model with the installed build and each variant
#   sbatch slurm/gw_depthfix.sh pipeline  with RASTER_PATH=$B/<VARIANT> (default beags_relwin): beags_prepare
#                                         (--consensus_fallback $FALLBACK, default zero = depth fix alone) into a linked
#                                         copy, eval at 30k, then slurm/gw_ftprobe.sh base b10 on it (TAG=<VARIANT>_)
# Variants (copies of submodules/diff-gaussian-rasterization_ours_beags, never the installed build):
#   beags_relwin  window = max(0.4, 5% of the first-pass guess)      beags_infer  config.h TRAINING 0 (window 10, 7 splits)
# Outputs: output/depthfix/<NAME>/holes_<variant>/ (holes.json, depth.npz, PNGs), output/depthfix/<NAME>/train_<variant>/
set -euo pipefail
STAGE=${1:?usage: sbatch slurm/gw_depthfix.sh build|pipeline}
source slurm/common.sh
NAME=${NAME:-spinnerf_7}
SRC=output/gw_beags/$NAME/train
D=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/masks/SPIn-NeRF
GT=$D/gt_masks/${NAME#spinnerf_}/masks
SAM=$D/training/sam2/${NAME#spinnerf_}/masks
B=${B:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/raster_variants}
VARIANTS=${VARIANTS:-beags_relwin beags_infer}
O=output/depthfix/$NAME
mkdir -p "$O"
holes() {  # holes TAG: depth-hole diagnostic with whatever rasterizer PYTHONPATH selects
  $PY gaussian_wrapping/beags_depth_holes.py -m "$SRC" --iteration "$ITER" --masks "$SAM" --n_views 6 \
      --save_depth --out "$O/holes_$1"
}
case $STAGE in
  build)
    for V in $VARIANTS; do
      echo "=== build $V [$(date +%T)]"
      (cd "$B/$V" && rm -rf build && $PY setup.py build_ext --inplace > "$ROOT/$O/build_$V.log" 2>&1) \
        || { echo "build $V FAILED, see $O/build_$V.log"; tail -30 "$O/build_$V.log"; continue; }
      echo "built $V [$(date +%T)]"
    done
    set -x
    holes stock
    for V in $VARIANTS; do
      (export PYTHONPATH=$B/$V:$PYTHONPATH; holes "$V") || echo "holes $V FAILED"
    done
    set +x
    ;;
  pipeline)
    VARIANT=${VARIANT:-beags_relwin}
    FALLBACK=${FALLBACK:-zero}
    export RASTER_PATH=$B/$VARIANT
    export PYTHONPATH=$RASTER_PATH:$PYTHONPATH
    M=$O/train_${VARIANT}_$FALLBACK
    rm -rf "$M"
    link_model "$SRC" "$M" "$ITER"
    set -x
    $PY -c "import diff_gaussian_rasterization_ours_beags as r; print('rasterizer', r.__file__)"
    $PY gaussian_wrapping/beags_prepare.py -m "$M" --iteration "$ITER" --masks "$SAM" --consensus_fallback "$FALLBACK"
    $PY gaussian_wrapping/beags_eval.py -m "$M" --iteration "$ITER" --gt_masks "$GT" --masks "$SAM"
    set +x
    SRC_MODEL=$M TAG=${VARIANT#beags_}_${FALLBACK}_ bash slurm/gw_ftprobe.sh base b10
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "depthfix $STAGE OK [$(date +%T)]"
