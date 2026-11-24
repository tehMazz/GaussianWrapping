#!/usr/bin/env bash
# Every BakerH scene on the VM, one after another on GPU 1: setup.sh, then per scene run_scene.sh
# (BEA-GS on Gaussian Wrapping: GW 30k with bilateral grid + hollowing, export + gsplat check, BEA-GS prepare /
# finetune / eval against the SAM3 masks, label PLYs, extraction / background renders collected in
# output/<method>/showcase/, PSNR, hollow metrics; FULL=1 adds the multi-label mesh + hole, ArtiFixer inputs, and
# ArtiFixer removal + ArtiFixer3D(+) on ARTIFIXER_SCENES; run_scene.sh's header documents the phases).
#
# Usage (from any directory):
#   bash scripts/bakerh/run_all.sh        starts in the background, keeps running after you log out,
#                                         prints the commands to follow and stop it
#   bash scripts/bakerh/run_all.sh --fg   runs in this terminal instead
#
# Everything is logged under output/bakerh_logs/<timestamp>/ (also output/bakerh_logs/latest/):
#   all.log     full console output of setup and every scene and phase
#   REPORT.txt  environment (git + submodules, GPU, Python imports), setup result (SETUP OK / FAIL and why),
#               status + duration of each scene, per-phase status, durations and peak GPU memory (summary.log),
#               what each phase found (job_summary.txt), the scene's numbers (report.py: PSNR / SSIM / LPIPS,
#               gsplat check, BEA-GS IoU / BIoU vs the SAM3 masks, Gaussians per label, hollow metrics, mesh face
#               counts, label renders, Seed2GS, ArtiFixer / ArtiFixer3D / ArtiFixer3D+ outputs), the last 80 lines
#               of every failed phase (40 of a failed soft phase), and a one-line-per-scene table at the end.
#               Updated after each scene. If anything fails, send this file back.
#   <scene>/    per-phase logs NN_<phase>.log, summary.log, job_summary.txt, report.txt
# A failed scene is skipped and the next one runs. Rerunning run_all.sh redoes only what is missing
# (finished phases are skipped in seconds). SCENES="PCV Compressor" runs a subset; GPU (1), DW (0.1),
# FORCE=1, PHASES, ARTIFIXER_SCENES, AF_VARIANT, ... pass through to run_scene.sh. SKIP_SETUP=1 skips setup.sh.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/common.sh"
cd "$ROOT"
export GPU SCENES DW LABEL METHOD ARTIFIXER_SCENES AF_VARIANT AF_DIR FULL

LATEST=$LOG_ROOT/latest
if [ -z "${BAKERH_CHILD:-}" ] && [ -f "$LATEST/pgid" ] && kill -0 -- "-$(cat "$LATEST/pgid")" 2>/dev/null; then
  echo "a run is already going (log: $LATEST/all.log). Stop it first: kill -- -$(cat "$LATEST/pgid")"
  exit 1
fi

STAMP=${BAKERH_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$LOG_ROOT/$STAMP
mkdir -p "$RUN_DIR"
ln -sfn "$STAMP" "$LATEST"

if [ -z "${BAKERH_CHILD:-}" ] && [ "${1:-}" != "--fg" ]; then
  # Relaunch detached (own session, immune to hangup) with all output going to all.log.
  launcher=(nohup); command -v setsid > /dev/null && launcher=(setsid nohup)
  BAKERH_CHILD=1 BAKERH_STAMP=$STAMP "${launcher[@]}" bash "$HERE/run_all.sh" >> "$RUN_DIR/all.log" 2>&1 < /dev/null &
  sleep 3
  echo "Started in the background; it keeps running if you close this terminal."
  echo "  follow:  tail -f $LATEST/all.log"
  echo "  report:  $LATEST/REPORT.txt   (send this back if something fails)"
  echo "  stop:    kill -- -\$(cat $LATEST/pgid)"
  exit 0
fi
if [ -n "${BAKERH_CHILD:-}" ]; then
  ps -o pgid= -p $$ | tr -d ' ' > "$RUN_DIR/pgid"   # kill -- -<pgid> stops this run and its python jobs
else
  exec > >(tee -a "$RUN_DIR/all.log") 2>&1
fi
export BAKERH_RUN_DIR=$RUN_DIR
REPORT=$RUN_DIR/REPORT.txt
log() { echo "[$(date '+%F %T')] $*"; }
section() { printf '\n==================== %s ====================\n' "$*" >> "$REPORT"; }
report() {  # report ARGS...: report.py over this run's settings, into REPORT.txt
  timeout 600 "$PY" "$HERE/report.py" --output "$OUTPUT" --method "$METHOD" --iter "$ITER" --ft_iter "$FT_ITER" \
      --label "$LABEL" --af_variant "$AF_DIR" --hm_suffix "$(hm_suffix)" "$@" >> "$REPORT" 2>&1 || echo "(report.py $* failed)" >> "$REPORT"
}

log "run_all $STAMP   scenes: $SCENES   dw: $DW   gpu: $GPU"
log "report: $REPORT"
{
  echo "bakerh BEA-GS on Gaussian Wrapping report   started $(date -Is)   run $STAMP"
  echo "scenes: $SCENES"
  echo "method=$METHOD dw=$DW label=$LABEL gpu=$GPU images=$IMAGES iter=$ITER ft_iter=$FT_ITER repo=$ROOT"
  echo "data=$DATA masks=$SAM_MASKS py=$PY gs_render_py=$GS_RENDER_PY"
  echo "train args: $GW_TRAIN_ARGS"
  echo "finetune args: --dilation_weight $DW $FT_ARGS; hollow metrics args: $HM_ARGS; mesh args: ${MESH_ARGS:-}"
  echo "artifixer: scenes '$ARTIFIXER_SCENES' variant $AF_VARIANT repo $(find_af_repo || echo '<none found: set AF_REPO>')"
  echo "ArtiFixer3D train args: $AF3D_TRAIN_ARGS"
  echo "phases: ${PHASES:-<run_scene.sh default$([ -z "${FULL:-}" ] || echo ' + FULL')>}"
} > "$REPORT"

section "git"
{
  git log --oneline -3
  git submodule status 2>&1
  git status --short | head -40
  echo "artifixer_gs: $(git -C "$AFS" log --oneline -1 2>&1) ($(git -C "$AFS" status --porcelain 2>/dev/null | wc -l) changed)"
} >> "$REPORT" 2>&1

section "GPU"
nvidia-smi >> "$REPORT" 2>&1 || echo "nvidia-smi failed" >> "$REPORT"

section "setup"
if [ -n "${SKIP_SETUP:-}" ]; then
  echo "skipped (SKIP_SETUP=1)" >> "$REPORT"
else
  log "setup.sh"
  timeout 7200 bash "$HERE/setup.sh"
  code=$?
  # setup.sh waits for its tee on exit; a killed setup (timeout) may not: give tee a moment to flush
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    grep -q '^SETUP \(OK\|FAIL\)' "$LOG_ROOT/setup/SETUP_REPORT.txt" 2>/dev/null && break; sleep 0.5
  done
  cat "$LOG_ROOT/setup/SETUP_REPORT.txt" >> "$REPORT" 2>/dev/null
  if [ "$code" -ne 0 ]; then
    section "summary   finished $(date -Is)"
    echo "setup failed (exit $code): no scene was run. The setup section above says which step and why." >> "$REPORT"
    log "setup failed; report: $REPORT"
    exit 1
  fi
fi

section "python"
if [ -x "$PY" ]; then
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH= timeout 600 "$PY" - >> "$REPORT" 2>&1 <<'PYEOF'
import importlib.util, sys
print("python", sys.version.split()[0], sys.executable)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print("device", torch.cuda.get_device_name(0), f"free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
for name in ("diff_gaussian_rasterization_ours", "diff_gaussian_rasterization_ours_beags", "diff_gaussian_rasterization_ms",
             "diff_gaussian_rasterization_sof", "simple_knn", "fused_ssim", "nvdiffrast", "tetranerf", "gsplat", "plyfile",
             "cv2", "scipy", "open3d", "trimesh"):
    print(f"  {name:40s} {'ok' if importlib.util.find_spec(name) else 'MISSING'}")
PYEOF
else
  echo "no venv python at $PY (setup.sh builds it)" >> "$REPORT"
fi

results=()
for scene in $SCENES; do
  start=$(date +%s)
  log "scene $scene start"
  bash "$HERE/run_scene.sh" "$scene"
  code=$?
  took=$(($(date +%s) - start))
  logs=$RUN_DIR/$scene
  if [ "$code" -eq 0 ]; then status=OK; else status="FAILED(exit $code)"; fi
  [ ! -f "$logs/SOFT_FAILED" ] || status+=" soft-failed:$(sed 's/ log=.*//; s/phase=//' "$logs/SOFT_FAILED" | paste -sd, -)"
  results+=("$(printf '%-16s %-36s %dh%02dm' "$scene" "$status" $((took / 3600)) $((took % 3600 / 60)))")
  log "scene $scene $status after $((took / 60)) min"

  section "scene $scene: $status ($((took / 60)) min)"
  echo "--- per-phase status ($scene/summary.log)" >> "$REPORT"
  cat "$logs/summary.log" >> "$REPORT" 2>/dev/null
  if [ -s "$logs/job_summary.txt" ]; then
    echo "--- what each phase found ($scene/job_summary.txt)" >> "$REPORT"
    tail -60 "$logs/job_summary.txt" >> "$REPORT"
  fi
  echo "--- numbers (report.py)" >> "$REPORT"
  report --scene "$scene"
  if [ "$code" -ne 0 ] && [ -f "$logs/FAILED" ]; then
    failed_log=$(sed -n 's/.*log=//p' "$logs/FAILED")
    echo "--- last 80 lines of $failed_log" >> "$REPORT"
    tail -80 "$failed_log" >> "$REPORT" 2>&1
  elif [ "$code" -ne 0 ]; then
    echo "--- no phase log; last 60 lines of all.log" >> "$REPORT"
    tail -60 "$RUN_DIR/all.log" >> "$REPORT"
  fi
  if [ -f "$logs/SOFT_FAILED" ]; then
    while IFS= read -r line; do
      soft=${line%% *}; echo "--- soft phase ${soft#phase=} failed: last 40 lines of ${line##*log=}" >> "$REPORT"
      tail -40 "${line##*log=}" >> "$REPORT" 2>&1
    done < "$logs/SOFT_FAILED"
  fi
done

section "summary   finished $(date -Is)"
printf '%s\n' "${results[@]}" >> "$REPORT"
echo >> "$REPORT"
scene_args=()
for scene in $SCENES; do scene_args+=(--scene "$scene"); done
report "${scene_args[@]}" --table
echo "extraction / background renders: $OUTPUT/$METHOD/showcase/<scene>_grid_all.jpg (GT | full | label_0 | label_1)" >> "$REPORT"
log "run_all done"
printf '%s\n' "${results[@]}"
log "report: $REPORT"
