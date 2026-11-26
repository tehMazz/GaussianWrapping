#!/bin/bash
#SBATCH --job-name=compose
#SBATCH --account=IscrC_P2R-GS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# compose_videos.py on a compute node (CPU-heavy: several methods x 240+ frames exceeds the login node's 600 s CPU limit).
#   sbatch slurm/compose_videos.sh <out_dir> "<Name>=<frames_dir>" ["<Name>=<frames_dir>" ...]
set -euo pipefail
OUT=${1:?out}; shift
source slurm/common.sh
ARGS=(); for m in "$@"; do ARGS+=(--method "$m"); done
$PY gaussian_wrapping/compose_videos.py --out "$OUT" "${ARGS[@]}"
echo "compose OK [$(date +%T)]"
