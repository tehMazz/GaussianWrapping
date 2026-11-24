#!/bin/bash
#SBATCH --job-name=gw_depth_reveal
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
# Before / after depth study of an object removal (beags/depth_reveal.py): which revealed pixels did the training views see?
# Usage (from repo root): sbatch slurm/gw_depth_reveal.sh [<model> <iteration> <label> <out>] [extra depth_reveal.py args]
#   default: the kitchen BEA-GS model at 33000, label 1 -> output/depth_reveal/mip360_kitchen/label1
set -uo pipefail
source slurm/common.sh
MODEL=${1:-output/gw_beags/mip360_kitchen/beags}
ITER=${2:-33000}
LABEL=${3:-1}
OUT=${4:-output/depth_reveal/mip360_kitchen/label$LABEL}
shift $(( $# < 4 ? $# : 4 ))
set -x
$PY gaussian_wrapping/beags/depth_reveal.py -m "$MODEL" --iteration "$ITER" --remove "$LABEL" --out "$OUT" "$@"
