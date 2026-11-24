#!/usr/bin/env bash
# One-time VM setup + self-test for the BakerH runs of BEA-GS on Gaussian Wrapping (run_all.sh calls it first;
# also runnable alone). Needs PyPI and GitHub; never Hugging Face:
#   0. the working tree: nothing of the BEA-GS pipeline is committed on branch beags yet (gaussian_wrapping/{beags,
#      hollow,robust,seg}/, beags_*.py, seed2gs.py, hollow_metrics.py, export_3dgs_ply.py, the modified train.py /
#      arguments / dataset_readers.py, requirements_beags.txt, scripts/bakerh/, the ours_beags rasterizer): commit
#      and push it, or rsync the whole working tree to the VM; a plain clone of the branch has none of it
#   1. repo venv ($ROOT/.venv, Python 3.9, torch 2.3.1+cu121) + CUDA, torch sees the GPU
#   2. GW's PyPI requirements (requirements.txt without its local file:// lines) + requirements_beags.txt (pytest)
#   3. the CUDA extensions from submodules/: rasterizers (base, ms, ours, ours_beags = BEA-GS labels, sof),
#      simple-knn, fused-ssim; import test of all of them; 3b the BEA-GS rasterizer GPU tests;
#      3c (check) every gaussian_wrapping/tests file on the GPU
#   4. LPIPS weights for the training report and metrics.py (gaussian_wrapping/lpipsPyTorch/weights; symlinks into a
#      Leonardo checkout that dangle here are replaced)
#   5. (check) git submodules the mesh steps need (TNTUniScanEvals, nvdiffrast) + tetra_triangulation (cmake, CGAL,
#      GMP: `sudo apt install libcgal-dev libgmp-dev cmake` if the build fails); only the mesh phase needs it
#   6. per scene: dataset, SAM3 masks, existing models
#   7. (check) gsplat 1.5.3 in GS_RENDER_PY (default the repo venv; SETUP_GSPLAT=0 skips the install): the export's
#      gsplat-vs-native check and ArtiFixer-GS's PLY renders; one small CUDA render (JIT-builds its kernels)
#   8. (check) ArtiFixer-GS (submodules/artifixer_gs) with the GW changes: hole_mesh.py, run.sh HOLE_MODE / EPS2D
#   9. (check) the ArtiFixer repo on the VM: its .venv (diffusion) and checkpoints/ (Wan2.1, artifixer-1.3b, MoGe-2)
# Steps 7-9 matter only for the artifixer phase (ARTIFIXER_SCENES). Safe to rerun: finished steps are skipped.
# Writes output/bakerh_logs/setup/SETUP_REPORT.txt; the last line says SETUP OK or SETUP FAIL: <steps>
# (check steps only warn).
#
# Usage (from any directory):  bash scripts/bakerh/setup.sh
# Knobs (common.sh): GPU, PY, FBK, SAM_MASKS, DATA, SCENES, GS_RENDER_PY, AF_REPO, ARTIFIXER_PY, SETUP_GSPLAT (1).
set -uo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
mkdir -p "$LOG_ROOT/setup"
exec > >(tee "$LOG_ROOT/setup/SETUP_REPORT.txt") 2>&1
TEE_PID=$!
# bash does not wait for a process substitution: close our end and wait for tee, so the report is complete
# (last line SETUP OK / FAIL) when run_all.sh reads it
trap 'exec >&- 2>&-; wait "$TEE_PID" 2>/dev/null' EXIT
SETUP_GSPLAT=${SETUP_GSPLAT:-1}

failed=()
warned=()
step() {  # step NAME CMD...: run, print OK/FAIL, remember failures
  local name=$1; shift
  echo; echo "==================== $name ===================="
  if "$@"; then echo "---- $name: OK"; else echo "---- $name: FAIL"; failed+=("$name"); fi
}
check_step() {  # check_step NAME WHO CMD...: like step, but a failure is only a warning (WHO needs it)
  local name=$1 who=$2; shift 2
  echo; echo "==================== $name (check only) ===================="
  if "$@"; then echo "---- $name: OK"; else echo "---- $name: WARN (needed by $who only)"; warned+=("$name"); fi
}
echo "setup $(date -Is)   repo=$ROOT   git=$(git rev-parse --short HEAD 2>/dev/null)"
echo "GPU=$GPU PY=$PY GS_RENDER_PY=$GS_RENDER_PY FBK=$FBK SAM_MASKS=$SAM_MASKS DATA=$DATA"

tree_check() {  # the untracked / modified files of branch beags: a clone of the branch alone lacks them
  local f miss=()
  for f in gaussian_wrapping/beags gaussian_wrapping/hollow gaussian_wrapping/robust gaussian_wrapping/seg \
           gaussian_wrapping/beags_finetune.py gaussian_wrapping/beags_prepare.py gaussian_wrapping/beags_eval.py \
           gaussian_wrapping/beags_render_labels.py gaussian_wrapping/beags_artifixer.py gaussian_wrapping/seed2gs.py \
           gaussian_wrapping/hollow_metrics.py gaussian_wrapping/export_3dgs_ply.py \
           gaussian_wrapping/multilabel_mesh_extraction.py requirements_beags.txt \
           submodules/diff-gaussian-rasterization_ours_beags; do
    [ -e "$f" ] || miss+=("$f")
  done
  grep -q hollow gaussian_wrapping/train.py || miss+=("the modified gaussian_wrapping/train.py (hollowing, appearance)")
  [ ${#miss[@]} -eq 0 ] && { echo "ok"; return 0; }
  printf 'missing: %s\n' "${miss[@]}"
  echo "the BEA-GS code is not committed on branch beags: commit + push it, or rsync the whole working tree to the VM"
  return 1
}

uvpip() { uv pip install --python "$PY" "$@"; }

venv_gpu() {
  command -v uv > /dev/null || { echo "uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"; return 1; }
  if [ ! -x "$PY" ]; then
    [ "$PY" = "$ROOT/.venv/bin/python" ] || { echo "PY=$PY does not exist"; return 1; }
    uv venv --python 3.9 "$ROOT/.venv" || return 1
  fi
  "$PY" -c 'import torch' 2>/dev/null || \
    uvpip torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 || return 1
  activate || return 1
  echo "python=$PY nvcc=$(command -v nvcc) CUDA_HOME=$CUDA_HOME CC=${CC:-}"
  "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch sees no GPU"
free, total = torch.cuda.mem_get_info()
print("device", torch.cuda.get_device_name(0), f"free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
EOF
}

requirements() {
  activate || return 1
  "$PY" -c 'import plyfile, cv2, scipy, open3d, trimesh, yaml, pytest' 2>/dev/null && { echo "already installed"; return 0; }
  grep -vE '^(-e |.* @ file://|torch|torchvision|torchaudio|nvidia-|triton|pyg-lib|torch-)' requirements.txt \
    > "$LOG_ROOT/setup/requirements_pypi.txt"
  uvpip -r "$LOG_ROOT/setup/requirements_pypi.txt" -r requirements_beags.txt
}

extensions() {
  activate || return 1
  local test='
import torch
import diff_gaussian_rasterization, diff_gaussian_rasterization_ms, diff_gaussian_rasterization_ours  # noqa: F401
import diff_gaussian_rasterization_sof, simple_knn, fused_ssim  # noqa: F401
import diff_gaussian_rasterization_ours_beags as b
print("rasterizers, simple_knn, fused_ssim: ok;", b.__file__)'
  "$PY" -c "$test" 2>/dev/null && return 0
  local s
  for s in diff-gaussian-rasterization_ms diff-gaussian-rasterization diff-gaussian-rasterization_ours \
           diff-gaussian-rasterization_ours_beags diff-gaussian-rasterization_sof simple-knn fused-ssim; do
    [ -d "submodules/$s" ] || { echo "no submodules/$s (diff-gaussian-rasterization_ours_beags is not in git yet: rsync the whole working tree, step 0)"; return 1; }
    echo "building submodules/$s"
    NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -include cstdint" uvpip --no-build-isolation "submodules/$s" || return 1
  done
  "$PY" -c "$test"
}

labels_test() {  # the BEA-GS rasterizer: labelled render vs stock, votes / SAG / mask (GPU tests)
  activate || return 1
  "$PY" -m pytest -q -x -p no:cacheprovider gaussian_wrapping/tests/test_rasterizer_labels.py
}

all_tests() {  # every test file, GPU tests included (none of them has run on the VM GPU before)
  activate || return 1
  "$PY" -m pytest -q -rs -p no:cacheprovider gaussian_wrapping/tests
}

lpips_weights() {  # training_report + metrics.py (LPIPS vgg) load them from lpipsPyTorch/weights, never online
  local w=$ROOT/gaussian_wrapping/lpipsPyTorch/weights f
  mkdir -p "$w/v0.1"
  for f in "$w/v0.1/vgg.pth" "$w/vgg16-397923af.pth"; do  # symlinks into a Leonardo checkout dangle on the VM
    if [ -L "$f" ] && [ ! -e "$f" ]; then echo "replacing dangling symlink $f"; rm -f "$f"; fi
  done
  [ -s "$w/v0.1/vgg.pth" ] || curl -fL -o "$w/v0.1/vgg.pth" \
      https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth || return 1
  [ -s "$w/vgg16-397923af.pth" ] || curl -fL -o "$w/vgg16-397923af.pth" \
      https://download.pytorch.org/models/vgg16-397923af.pth || return 1
  ls -laL "$w/v0.1/vgg.pth" "$w/vgg16-397923af.pth"
}

mesh_deps() {
  activate || return 1
  git submodule update --init gaussian_wrapping/eval/TNTUniScanEvals submodules/nvdiffrast || return 1
  "$PY" -c 'import nvdiffrast.torch' 2>/dev/null || uvpip --no-build-isolation submodules/nvdiffrast || return 1
  if ! "$PY" -c 'import tetranerf.utils.extension' 2>/dev/null; then
    (cd submodules/tetra_triangulation && cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make && uvpip -e .) || return 1
  fi
  "$PY" -c 'import nvdiffrast.torch, tetranerf.utils.extension; print("nvdiffrast, tetra_triangulation: ok")'
}

gsplat_venv() {  # GS_RENDER_PY: gsplat for export_3dgs_ply.py --check and ArtiFixer-GS's PLY renders
  activate || return 1
  local test='
import gsplat, scipy, matplotlib, torch  # noqa: F401
from gsplat import rasterization
print("gsplat", gsplat.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
d = "cuda"
K = torch.tensor([[[16.0, 0, 16], [0, 16.0, 16], [0, 0, 1]]], device=d)
rgb, alpha, _ = rasterization(torch.tensor([[0.0, 0.0, 3.0]], device=d), torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=d),
                              torch.full((1, 3), 0.2, device=d), torch.ones(1, device=d), torch.ones(1, 3, device=d),
                              torch.eye(4, device=d)[None], K, 32, 32, rasterize_mode="classic")
assert alpha[0, 16, 16, 0] > 0.5, f"centre alpha {float(alpha[0, 16, 16, 0])}"
print("gsplat render ok: centre alpha", round(float(alpha[0, 16, 16, 0]), 3))'
  echo "GS_RENDER_PY=$GS_RENDER_PY"
  "$GS_RENDER_PY" -c "$test" && return 0
  [ "$SETUP_GSPLAT" = 1 ] || { echo "(SETUP_GSPLAT=1 installs gsplat==1.5.3 into $PY)"; return 1; }
  [ "$GS_RENDER_PY" = "$PY" ] || { echo "GS_RENDER_PY=$GS_RENDER_PY is not the repo venv: install gsplat there by hand"; return 1; }
  echo "installing into $PY: gsplat==1.5.3 matplotlib (CUDA kernels JIT-build on the first render)"
  uvpip "gsplat==1.5.3" matplotlib || return 1
  "$GS_RENDER_PY" -c "$test"
}

afs_submodule() {  # ArtiFixer-GS with the GW changes (local branch gw on Leonardo: push it, or copy the checkout)
  if [ ! -f "$AFS/scripts/run.sh" ]; then
    git submodule update --init submodules/artifixer_gs || {
      echo "could not fetch https://github.com/IVIazz97/Artifixers-3DGS.git: copy submodules/artifixer_gs by hand"; return 1; }
  fi
  git -C "$AFS" log --oneline -1
  local m missing=()
  for m in prepare render distill removal propagate trajectory caption hole_mesh; do
    [ -f "$AFS/artifixer_gs/$m.py" ] || missing+=("artifixer_gs/$m.py")
  done
  grep -q 'HOLE_MODE' "$AFS/scripts/run.sh" || missing+=("scripts/run.sh HOLE_MODE")
  grep -q 'EPS2D' "$AFS/scripts/run.sh" || missing+=("scripts/run.sh EPS2D")
  grep -q 'RENDER_CMD' "$AFS/artifixer_gs/render.py" || missing+=("artifixer_gs/render.py RENDER_CMD (GW renderer)")
  grep -q 'CENTER_PRINCIPAL_POINT' "$AFS/artifixer_gs/render.py" || missing+=("artifixer_gs/render.py CENTER_PRINCIPAL_POINT")
  [ ${#missing[@]} -eq 0 ] || { echo "ArtiFixer-GS lacks ${missing[*]}: it needs the gw branch (HANDOFF 9)"; return 1; }
}

artifixer_venv() {  # the ArtiFixer repo's .venv runs the diffusion (ArtiFixer-GS PY); checkpoints from its checkpoints/
  local af ok=0 f py
  af=$(find_af_repo) || { echo "no ArtiFixer repo with .venv/bin/python and checkpoints/ (set AF_REPO)"; return 1; }
  py=${ARTIFIXER_PY:-$af/.venv/bin/python}
  echo "AF_REPO=$af ARTIFIXER_PY=$py"
  for f in "${MODEL_ID:-$af/checkpoints/Wan2.1-T2V-1.3B-Diffusers}/model_index.json" \
           "${MODEL_ID:-$af/checkpoints/Wan2.1-T2V-1.3B-Diffusers}/text_encoder" \
           "${MODEL_ID:-$af/checkpoints/Wan2.1-T2V-1.3B-Diffusers}/transformer" \
           "${CHECKPOINT_PT:-$af/checkpoints/ArtiFixer/artifixer-1.3b.pt}"; do
    if [ -e "$f" ]; then echo "  ok       $f"; else echo "  MISSING  $f"; ok=1; fi
  done
  f=${MOGE_MODEL_PATH:-$af/checkpoints/moge-2-vitl-normal/model.pt}
  if [ -s "$f" ]; then echo "  ok       $f"; else echo "  absent   $f (only needed for a scene without a metric scale)"; fi
  # From the submodule: this repo's scene/ and utils/ must not shadow ArtiFixer-GS modules.
  (cd "$AFS" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$AFS" HF_HUB_OFFLINE=1 "$py" -) <<'EOF' || ok=1
import importlib.util, sys
import torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
bad = [n for n in ("diffusers", "transformers", "accelerate", "einops", "av", "cv2", "h5py", "scipy") if importlib.util.find_spec(n) is None]
print("gsplat in the diffusion venv:", importlib.util.find_spec("gsplat") is not None)
from model_eval import run_inference  # noqa: F401  (ArtiFixer-GS inference entry point)
print("model_eval.run_inference imports")
assert not bad, f"missing in the diffusion venv: {bad}"
EOF
  return $ok
}

scenes() {
  local s d m af
  af=$(find_af_repo 2>/dev/null || true)
  for s in $SCENES; do
    d=$DATA/$s
    printf '%-14s data %s (%s %s photos, sparse %s)\n' "$s" "$([ -d "$d" ] && echo ok || echo MISSING)" \
      "$IMAGES" "$(ls "$d/$IMAGES" 2>/dev/null | wc -l)" "$([ -d "$d/sparse" ] && echo ok || echo MISSING)"
    if m=$(mask_src "$s"); then echo "               masks $m ($(ls "$m" | grep -c '\.npy$') .npy)"
    else echo "               masks MISSING (no $SAM_MASKS/$s/{full,0_400,400_800}/masks_npy): this scene will fail"; fi
    m=$(model_dir "$s" train)
    echo "               GW $ITER: $([ -f "$m/point_cloud/iteration_$ITER/point_cloud.ply" ] && echo "ok $m" || echo "none at $m (will train)")"
    m=$(model_dir "$s" beags)
    echo "               BEA-GS $FT_ITER: $([ -f "$m/point_cloud/iteration_$FT_ITER/labels.pt" ] && echo "ok $m" || echo "none at $m (run_scene.sh)")"
    if [[ " $ARTIFIXER_SCENES " == *" $s "* ]]; then
      m=${af:+$(af_scene_root "$s" "$af")}
      echo "               ArtiFixer ($AF_VARIANT): scene root ${m:-<no ArtiFixer repo>}" \
           "$([ -n "$m" ] && [ -f "$m/metric_alignment/scale_info.txt" ] && echo '(metric scale ok)' || echo '(no metric scale: MoGe-2)')" \
           "$([ -n "$m" ] && [ -f "$m/split.json" ] && echo '(split.json: caption.h5)' || echo "(caption: '$(scene_caption "$s")')")"
    fi
  done
}

step "0 working tree (uncommitted BEA-GS code present)" tree_check
step "1 venv + GPU" venv_gpu
step "2 PyPI requirements + pytest" requirements
step "3 CUDA extensions (rasterizers with BEA-GS labels, simple-knn, fused-ssim)" extensions
step "3b BEA-GS rasterizer GPU tests" labels_test
step "4 LPIPS weights" lpips_weights
check_step "5 mesh dependencies (TNTUniScanEvals, nvdiffrast, tetra_triangulation)" "the mesh phase" mesh_deps
check_step "3c all unit tests on the GPU" "nothing (a report: send it back)" all_tests
echo; echo "==================== 6 scenes ===================="
scenes
check_step "7 gsplat (GS_RENDER_PY)" "the export check and the artifixer phase" gsplat_venv
if [ -n "$ARTIFIXER_SCENES" ]; then
  check_step "8 ArtiFixer-GS submodule (gw changes)" "the artifixer phase" afs_submodule
  check_step "9 ArtiFixer diffusion venv + checkpoints" "the artifixer phase" artifixer_venv
fi

echo
[ ${#warned[@]} -eq 0 ] || echo "WARN (check steps): ${warned[*]}"
if [ ${#failed[@]} -eq 0 ]; then
  echo "SETUP OK"
else
  echo "SETUP FAIL: ${failed[*]}"
  exit 1
fi
