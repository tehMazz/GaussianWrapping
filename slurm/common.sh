# Shared settings for slurm/gw_*.sh. Sourced from the repo root (sbatch copies the job script, so jobs are
# submitted from the repo root and source slurm/common.sh by that relative path). Every value is an env knob.
module load profile/deeplrn cuda/12.2 gcc/12.2.0       # gcc 12: libstdc++ with GLIBCXX_3.4.29 for simple_knn
ENV=${ENV:-/leonardo_scratch/fast/IscrC_EditGS/envs/gaussian_wrapping}
PY=${PY:-$ENV/bin/python}
export PATH=$ENV/bin:$PATH                             # ninja for torch JIT extensions (nvdiffrast)
export TORCH_CUDA_ARCH_LIST=8.0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONPATH=
# gs-atoms (ViewAdam, EpochPrune, decay_opacity_, host_state): pure PyTorch, imported as `atoms`.
GSATOMS=${GSATOMS:-/leonardo_work/IscrC_EditGS/repos/CVPR2027/gs-atoms}
export PYTHONPATH=$GSATOMS
ROOT=$(pwd)
METHOD=${METHOD:-gw_beags}                             # output/<METHOD>/<name>/<mode>
ITER=${ITER:-30000}                                    # GW training iterations
FT_ITER=${FT_ITER:-33000}                              # ITER + beags_finetune --ft_iterations (3000)
# GW's "ours" recipe (gaussian_wrapping/scripts/train_and_extract_gw_ours.py) without its exposure compensation,
# which the exported PLY cannot carry; robust/ and hollow/ on top.
GW_TRAIN_ARGS=${GW_TRAIN_ARGS-"--rasterizer ours --feature_dc_lr 0.0013 --feature_rest_lr 0.00011 --data_device cpu --N_max_gaussians 6000000"}
ROBUST_ARGS=${ROBUST_ARGS-"--no-exposure_compensation --appearance bilagrid"}
HOLLOW_ARGS=${HOLLOW_ARGS-"--hollow"}
# Pivot mesh for the multi-label PAM (train_and_extract_gw_ours.py's extraction, without --postprocess)
GW_EXTRACT_ARGS=${GW_EXTRACT_ARGS-"--sdf_mode ours --rasterizer ours --dtype int32 --isosurface_value 0.0 --n_binary_steps 10 --use_valid_mask --filter_large_edges --data_device cpu"}

export GS_RENDER_PY=${GS_RENDER_PY:-/leonardo_scratch/fast/IscrC_EditGS/envs/gsplat/bin/python}  # export_3dgs_ply.py --check
model_dir() { echo "output/$METHOD/$1/$2"; }        # model_dir NAME MODE
has_iter() { [ -f "$1/point_cloud/iteration_$2/point_cloud.ply" ]; }   # has_iter MODEL IT
# Masks for BEA-GS in slurm/gw_scene.sh: provided (the <sam2_masks_dir> argument) | seed2gs (SAM3 text prompt on the
# $ITER train model, seed2gs.py -> output/seed2gs/<name>/<prompt slug>). Seed2GS runs read the shared train model
# through a linked view in output/$SEED2GS_METHOD/<name>/train, so provided-mask outputs are never overwritten.
MASK_SOURCE=${MASK_SOURCE:-provided}
SEED2GS_PROMPT=${SEED2GS_PROMPT:-}                     # one prompt (gw_scene.sh); gw_seed2gs.sh takes "a|b|c"
# extra seed2gs.py args, word-split unquoted (no quotes inside), e.g. "--ref_view 0 --track_side 768"; SAM3 runner
# flags as one comma list: "--sam3_args=--no_fast_init,--no_prefetch" (or --sam3_args=--fake for a wiring test)
SEED2GS_ARGS=${SEED2GS_ARGS-}
SEED2GS_METHOD=${SEED2GS_METHOD:-${METHOD}_seed2gs}    # output/<SEED2GS_METHOD>/<name>/... with MASK_SOURCE=seed2gs

slug() {  # prompt -> dir name, as seg/masks.py prompt_slug
  local s; s=$(printf '%s' "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//' | tr '[:upper:]' '[:lower:]'); echo "${s:-object}"
}
link_model() {  # link_model SRC DST IT: DST loads SRC's iteration IT through symlinks, keeps its own labels.pt, beags/,
  local src dst=$2 it=$3 f   # renders and metrics (only cfg_args, cameras.json and the PLY dir's files are linked)
  src=$(realpath "$1")
  mkdir -p "$dst/point_cloud/iteration_$it"
  for f in cfg_args cameras.json; do
    if [ -e "$src/$f" ]; then ln -sfn "$src/$f" "$dst/$f"; fi
  done
  for f in "$src/point_cloud/iteration_$it"/*; do
    if [ "$(basename "$f")" != labels.pt ]; then ln -sfn "$f" "$dst/point_cloud/iteration_$it/$(basename "$f")"; fi
  done
}
