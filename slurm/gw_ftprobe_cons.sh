#!/bin/bash
#SBATCH --job-name=gw_ftprobe_cons
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
# Round 2 of the finetune probe: the mask-consensus fix (beags/consensus.py fallback="own").
# 1. consensus pytest, 2. depth-hole diagnostic on the 30k model, 3. beags_prepare with the fix into a linked copy
# of the 30k model (the original output/gw_beags/<NAME>/train is only read), 4. eval of the new labels,
# 5. slurm/gw_ftprobe.sh on it with TAG=cons_ (arms: $@, default base b10).
# Usage (from repo root): sbatch slurm/gw_ftprobe_cons.sh [arms]     Outputs: output/ftprobe/<NAME>/{train_cons,
#   depth_holes, cons_<arm>, snaps_cons_<arm>}
set -euo pipefail
source slurm/common.sh
NAME=${NAME:-spinnerf_7}
SRC=output/gw_beags/$NAME/train
D=/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/data/removal_scenes/masks/SPIn-NeRF
GT=$D/gt_masks/${NAME#spinnerf_}/masks
SAM=$D/training/sam2/${NAME#spinnerf_}/masks
PYTEST_LIB=${PYTEST_LIB:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/pylib_gw}
O=output/ftprobe/$NAME
M=$O/train_cons
rm -rf "$M" "$O/depth_holes"
link_model "$SRC" "$M" "$ITER"
set -x
PYTHONPATH=$PYTEST_LIB:$PYTHONPATH $PY -m pytest -q -rs -p no:cacheprovider gaussian_wrapping/tests/test_beags_utils.py -k consensus || true
$PY gaussian_wrapping/beags_depth_holes.py -m "$M" --iteration "$ITER" --masks "$SAM" --out "$O/depth_holes"
$PY gaussian_wrapping/beags_prepare.py -m "$M" --iteration "$ITER" --masks "$SAM" --consensus_fallback own
$PY gaussian_wrapping/beags_eval.py -m "$M" --iteration "$ITER" --gt_masks "$GT" --masks "$SAM"
set +x
[ $# -gt 0 ] || set -- base b10
SRC_MODEL=$M TAG=cons_ bash slurm/gw_ftprobe.sh "$@"
