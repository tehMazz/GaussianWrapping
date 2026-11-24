# Shared settings for the BakerH VM scripts of BEA-GS on Gaussian Wrapping (setup.sh, run_scene.sh, run_all.sh,
# report.py). Sourced, not run. Every value is an env knob. VM paths as BEA-GS_FSplat scripts/bakerh/common.sh and
# ArtiFixer's scripts/bakerh/run_removal.sh.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=${GPU:-1}
SCENES=${SCENES:-"TA_BLUE_MOTOR PCV TA_TURBINE Compressor G35 COGEN1 COGEN2"}
DW=${DW:-0.1}                 # BEA-GS finetune --dilation_weight
LABEL=${LABEL:-1}             # the SAM3 masks hold one object: label 1
FBK=${FBK:-/workspace/amazzucchelli/fbk-3dworld}
SAM_MASKS=${SAM_MASKS:-$FBK/FlashSplat/sam3_single_object_anchor_tracking}   # <scene>/{full,0_400,400_800}/masks_npy
data_root() {  # BakerH_undis: <scene>/{images_1600,sparse}; this repo's datasets/, else the FastGS BEA-GS checkout's
  local d
  for d in "$ROOT/datasets/BakerH_undis" "$ROOT/../BEA-GS_FSplat/datasets/BakerH_undis" "$FBK/BEA-GS_FSplat/datasets/BakerH_undis"; do
    [ -d "$d" ] && { echo "$d"; return 0; }
  done
  echo "$ROOT/datasets/BakerH_undis"
}
DATA=${DATA:-$(data_root)}
IMAGES=${IMAGES:-images_1600}
OUTPUT=${OUTPUT:-$ROOT/output}
LOG_ROOT=$OUTPUT/bakerh_logs
METHOD=${METHOD:-gw_beags_dw$DW}                             # output/<METHOD>/<scene>/{masks,train,export,beags,...}
PY=${PY:-$ROOT/.venv/bin/python}                            # GW + BEA-GS venv (setup.sh builds it)
ITER=${ITER:-30000}
FT_ITER=${FT_ITER:-33000}                                   # ITER + beags_finetune --ft_iterations (3000)
# GW's "ours" recipe (gaussian_wrapping/scripts/train_and_extract_gw_ours.py) without exposure compensation, plus
# the bilateral grid and the hollowing stage (= slurm/common.sh GW_TRAIN_ARGS + ROBUST_ARGS + HOLLOW_ARGS)
GW_TRAIN_ARGS=${GW_TRAIN_ARGS-"--rasterizer ours --feature_dc_lr 0.0013 --feature_rest_lr 0.00011 --data_device cpu --N_max_gaussians 6000000 --no-exposure_compensation --appearance bilagrid --hollow"}
GW_EXTRACT_ARGS=${GW_EXTRACT_ARGS-"--sdf_mode ours --rasterizer ours --dtype int32 --isosurface_value 0.0 --n_binary_steps 10 --use_valid_mask --filter_large_edges --data_device cpu"}
FT_ARGS=${FT_ARGS:-}                                        # extra beags_finetune.py args
HM_ARGS=${HM_ARGS:-}                                        # extra hollow_metrics.py args (e.g. --floater_voxel 0.02)
hm_suffix() {  # file suffix of the hollow metrics under HM_ARGS: hollow_metrics_<it><suffix>.json ("" without HM_ARGS)
  [ -z "$HM_ARGS" ] || echo "_$(printf %s "$HM_ARGS" | tr -cs 'A-Za-z0-9.' _ | sed 's/^_*//; s/_*$//')"
}
N_VIEWS=${N_VIEWS:-8}                                       # beags_render_labels.py views

# ArtiFixer diffusion (phase artifixer): ArtiFixer-GS submodule run.sh, the ArtiFixer repo's .venv + checkpoints/
AFS=$ROOT/submodules/artifixer_gs
GS_RENDER_PY=${GS_RENDER_PY:-$PY}                           # Python with gsplat for ArtiFixer-GS's PLY renders
ARTIFIXER_SCENES=${ARTIFIXER_SCENES-"TA_BLUE_MOTOR PCV"}   # scenes whose default PHASES include artifixer
AF_VARIANT=${AF_VARIANT:-normal}                            # normal | bighull (removal) | vanilla_object|vidsplat|travel
AF_DIR=${AF_DIR:-$AF_VARIANT}                               # output/artifixer/<scene>/<AF_DIR> (stages/lib.sh: <variant>_<VARIANT>)
AF_UNSEEN=${AF_UNSEEN:-auto}                                # beags_artifixer.py --unseen_maps (GAVIS maps; holes phase: on)
AF_CONF=${AF_CONF:-auto}                                    # beags_artifixer.py --confidence_maps (needs train.py --confidence)
KAPPA=${KAPPA:-1}                                           # GAVIS vMF concentration
UNSEEN_THRESHS=${UNSEEN_THRESHS:-"0.5 0.3"}                 # holes phase: one gavis split per threshold
HOLE_CONF_THRESH=${HOLE_CONF_THRESH:-0.5}                   # holes phase: alpha + confidence split (when CONFIDENCE_DIR)
COMPRESSOR_FRAMES=${COMPRESSOR_FRAMES-49-148,284-304}      # all 923 Compressor photos ran ArtiFixer out of memory
COMPRESSOR_TRAJ_FRAMES=${COMPRESSOR_TRAJ_FRAMES-121}         # vanilla_* on Compressor: TRAJ_MAX_FRAMES (as many as the removal frames)
# ArtiFixer3D distillation: GW recipe without --rasterizer ours (TRAIN_CMD passes it)
AF3D_TRAIN_ARGS=${AF3D_TRAIN_ARGS-${GW_TRAIN_ARGS//--rasterizer ours/}}
EMPTY_CAPTION=${EMPTY_CAPTION:-"An empty floor without holes: one continuous, clean floor surface with nothing standing on it, in an industrial hall."}
scene_caption() {  # ArtiFixer's run_removal.sh scene captions (used when ArtiFixer's root has no caption.h5)
  case $1 in
    TA_BLUE_MOTOR) echo "An industrial hall with a large blue electric motor standing on the floor, surrounded by equipment." ;;
    PCV)           echo "An industrial hall with a large pressure control valve assembly, pipes and flanges standing on the floor." ;;
    TA_TURBINE)    echo "An industrial hall with a large turbine standing on the floor, surrounded by pipes and equipment." ;;
    Compressor)    echo "An industrial hall with a large compressor standing on the floor, surrounded by pipes and equipment." ;;  # ours: not in run_removal.sh
    G35)           echo "An industrial hall with large machinery standing on the floor, surrounded by pipes and equipment." ;;
    *)             echo "" ;;
  esac
}
# Seed2GS (phase seed2gs, opt-in: PHASES=...,seed2gs): SAM3 in its own env, text prompt per scene
SAM3_PYTHON=${SAM3_PYTHON:-}                                # required by the seed2gs phase (no VM default)
seed_prompt() {  # SEED2GS_PROMPT overrides
  if [ -n "${SEED2GS_PROMPT:-}" ]; then echo "$SEED2GS_PROMPT"; return; fi
  case $1 in
    TA_BLUE_MOTOR) echo "blue electric motor" ;;
    PCV)           echo "pressure control valve" ;;
    TA_TURBINE)    echo "turbine" ;;
    Compressor)    echo "compressor" ;;
    *)             echo "machine" ;;
  esac
}

model_dir() { echo "$OUTPUT/$METHOD/$1/$2"; }               # model_dir SCENE MODE
mask_src() {  # SAM3 masks of scene $1: full capture, else an available 400-frame chunk
  local d
  for d in "$SAM_MASKS/$1/full/masks_npy" "$SAM_MASKS/$1/0_400/masks_npy" "$SAM_MASKS/$1/400_800/masks_npy"; do
    [ -d "$d" ] && { echo "$d"; return 0; }
  done
  return 1
}
find_af_repo() {  # the ArtiFixer repo on the VM: its .venv runs the diffusion, checkpoints/ holds Wan2.1 + ArtiFixer
  local p
  if [ -n "${AF_REPO:-}" ]; then echo "$AF_REPO"; return 0; fi
  for p in "$FBK/ArtiFixer" "$FBK/artifixer" /workspace/artifixer /workspace/ArtiFixer "$ROOT/../ArtiFixer" \
           "$HOME/ArtiFixer" "$FBK"/*/; do
    p=${p%/}
    [ -x "$p/.venv/bin/python" ] && [ -d "$p/checkpoints" ] && { echo "$p"; return 0; }
  done
  return 1
}
af_scene_root() {  # af_scene_root SCENE AF_REPO: ArtiFixer's prepared root of the scene (metric scale, caption.h5)
  if [ -n "${AF_SCENE_ROOT:-}" ]; then echo "$AF_SCENE_ROOT"; return; fi
  case $1 in                                                 # as in ArtiFixer's scripts/bakerh/run_removal.sh
    Compressor) echo "$2/output/bakerh_undis_ds2_from1600/$1" ;;
    G35) echo "${G35_ROOT:-$2/output/bakerh_undis/$1}" ;;
    *) echo "$2/output/bakerh_undis/$1" ;;
  esac
}
activate() {  # repo venv + CUDA 12.1 toolchain (as BEA-GS_FSplat scripts/activate_env.sh), quietly
  [ -x "$PY" ] || { echo "no venv python at $PY (run scripts/bakerh/setup.sh)"; return 1; }
  local c
  for c in /usr/local/cuda-12.1 /usr/local/cuda; do [ -d "$c" ] && { export CUDA_HOME=${CUDA_HOME:-$c}; break; }; done
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
  export PATH=$(dirname "$PY"):$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
  if command -v gcc-11 > /dev/null && command -v g++-11 > /dev/null; then export CC=${CC:-gcc-11} CXX=${CXX:-g++-11}; fi
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0} PYTHONPATH=
  export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
