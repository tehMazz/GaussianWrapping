#!/usr/bin/env bash
# One BakerH scene on the VM, one GPU, one log per phase: BEA-GS (dilation weight 0.1) on Gaussian Wrapping, then
# ArtiFixer removal + inpainting on the result.
#
# Usage, from any directory:  bash scripts/bakerh/run_scene.sh SCENE
#   SCENE  TA_BLUE_MOTOR | PCV | TA_TURBINE | Compressor | G35 | COGEN1 | COGEN2
# Phases, in order (PHASES=a,b runs a subset; a finished phase is skipped in seconds, FORCE=1 redoes it):
#   check        dataset, SAM3 masks, venv, LPIPS weights present
#   masks        SAM3 id maps -> BEA-GS (2, H, W) masks                           output/<method>/<scene>/masks
#   train        GW 30k: --rasterizer ours, bilateral grid, hollowing (GW_TRAIN_ARGS)  output/<method>/<scene>/train
#   export       standard 3DGS PLY with the 3D filter baked; gsplat vs native check when GS_RENDER_PY imports
#                gsplat (export/gsplat_check.json)                                .../export
#   prepare      beags_prepare.py: consensus masks, per-Gaussian labels (30k)
#   finetune     beags_finetune.py --dilation_weight $DW (0.1) [FT_ARGS] -> 33000 .../beags
#   labels       beags_eval.py at 30000 and 33000 against the SAM3 masks (BakerH has no GT: agreement with the
#                input masks; <model>/beags/metrics_<it>.json)
#   export_beags 33000 PLY + label_<k>.ply per label (export_3dgs_ply.py --split_labels)   .../export_beags
#   renders      beags_render_labels.py, N_VIEWS (8) views: full, label_<k> alone, grids   .../renders_labels_33000
#   showcase     copies the renders to output/<method>/showcase/<scene>/ and <scene>_grid_all.jpg (all scenes side by
#                side; label_0 = background after removal, label_1 = the extracted object)
#   metrics      render.py + metrics.py: PSNR / SSIM / LPIPS of the test views at 30000 and 33000 (and beags_meshocc);
#                a model not trained yet is skipped
#   hollow       hollow_metrics.py at 30000 and at 33000 with object label $LABEL (hole_alpha, ...) [HM_ARGS]
#   --- FULL=1 only (not in the default phases) ---
#   mesh         GW pivot mesh + multi-label PAM, hole of label $LABEL; face counts   .../mesh   (soft: a failure
#                is reported and the scene goes on, with an alpha hole for ArtiFixer)
#   afinputs     beags_artifixer.py --variant $AF_VARIANT (normal) --hole_mode mesh|alpha: ArtiFixer-GS env.sh,
#                ArtiFixer3D trains with this repo's train.py (AF3D_TRAIN_ARGS)  output/artifixer/<scene>/<variant>
#   artifixer    ArtiFixer-GS scripts/run.sh (diffusion in the ArtiFixer repo's .venv): removal, split, single
#                rollout, propagation, ArtiFixer3D (GW train.py --lpips_images), ArtiFixer3D+; run.sh skips its
#                finished stages. With FULL=1, only for ARTIFIXER_SCENES (TA_BLUE_MOTOR PCV)
#   seed2gs      opt-in (not in the default PHASES; soft): Seed2GS text-prompt labels on the 30k model (SAM3_PYTHON,
#                SAM3_REPO, SAM3_CHECKPOINT; prompt SEED2GS_PROMPT or per scene) and their mask IoU vs the SAM3 masks
#                output/seed2gs/<scene>/<prompt>
#   --- opt-in, never in the default phases (scripts/bakerh/stages/ runs them) ---
#   holes        soft: removal holes side by side without diffusion (after artifixer with STAGES=prepare,removal,split):
#                alpha | mesh | gavis per UNSEEN_THRESHS (0.5 0.3; needs afinputs with AF_UNSEEN=on) | alpha +
#                confidence < HOLE_CONF_THRESH (needs a model trained with --confidence)  -> $AFO/hole_grid.jpg
#   trajcands    vanilla_* after artifixer with STAGES=trajectory,prepare: top view of the novel path + contact sheet
#                of the GW renders on it                                           $AFO/path_candidates.jpg
#   trajgrid     vanilla_* after the full artifixer phase: GW render | ArtiFixer | ArtiFixer3D render | ArtiFixer3D+
#                on N_VIEWS path frames   $AFO/novel_views_grid.jpg, showcase/<scene>_<AF_DIR>_grid.jpg
#   mesh_train   multi-label mesh of the prepared 30k model with tets.npz             .../mesh_train
#   finetune_meshocc  beags_finetune.py --occ_source mesh --mesh_dir mesh_train -> 33000  .../beags_meshocc
#   labels_meshocc    beags_eval.py of beags_meshocc against the SAM3 masks
#   renders_meshocc   label renders of beags_meshocc -> .../renders_labels_meshocc_33000, showcase/<scene>_meshocc_grid_all.jpg
#   summary      report.py: every number above into <logs>/report.txt (run_all.sh puts it in REPORT.txt)
# Outputs: output/<method>/<scene>/{masks,train,export,beags,export_beags,renders_labels_33000,mesh} (method
# gw_beags_dw0.1), output/artifixer/<scene>/<variant>, output/seed2gs/<scene>.
# Logs: $BAKERH_RUN_DIR/<scene>/ (run_all.sh) or output/bakerh_logs/<stamp>_<scene>/: 00_config.log,
# NN_<phase>.log, summary.log (start, end, duration, peak GPU memory of every phase), job_summary.txt
# (what each phase found), FAILED (the phase that stopped the scene), SOFT_FAILED (soft phases that failed).
# Knobs (common.sh): GPU (1), DW (0.1), LABEL (1), METHOD, GW_TRAIN_ARGS, GW_EXTRACT_ARGS, FT_ARGS, HM_ARGS,
# MESH_ARGS, N_VIEWS, GS_RENDER_PY, AF_VARIANT, ARTIFIXER_SCENES, AF_REPO, ARTIFIXER_PY, MODEL_ID, CHECKPOINT_PT,
# MOGE_MODEL_PATH, AF_SCENE_ROOT, CAPTION, METRIC_SCALE, NUM_VIEWS, AF3D_STEPS, AF3D_TRAIN_ARGS, SEEDS, FRAMES,
# SAM3_PYTHON, SEED2GS_PROMPT, FULL, FORCE, PHASES.
set -euo pipefail
S=${1:?usage: run_scene.sh SCENE   (TA_BLUE_MOTOR|PCV|TA_TURBINE|Compressor|G35|COGEN1|COGEN2)}
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
activate || { echo "activate failed"; exit 1; }
BASE=$OUTPUT/$METHOD/$S
M=$(model_dir "$S" train)
B=$(model_dir "$S" beags)
MASKS=$(model_dir "$S" masks)
MESH=$(model_dir "$S" mesh)
AFO=$OUTPUT/artifixer/$S/$AF_DIR
MT=$BASE/mesh_train                       # mesh of the prepared 30k model (tets.npz) for --occ_source mesh
BM=$BASE/beags_meshocc                     # BEA-GS finetune against that mesh
MASK_SRC=$(mask_src "$S" || echo "<no masks_npy under $SAM_MASKS/$S>")
# default: GW (bilateral grid + hollowing) -> BEA-GS -> extraction / background renders; FULL=1 adds the mesh and
# ArtiFixer phases (artifixer only on ARTIFIXER_SCENES)
DEFAULT_PHASES=check,masks,train,export,prepare,finetune,labels,export_beags,renders,showcase,metrics,hollow
if [ -n "${FULL:-}" ]; then
  DEFAULT_PHASES+=,mesh,afinputs
  [[ " $ARTIFIXER_SCENES " != *" $S "* ]] || DEFAULT_PHASES+=,artifixer
fi
PHASES=${PHASES:-$DEFAULT_PHASES,summary}
SOFT_PHASES=" mesh seed2gs holes "                 # a failure is recorded and the scene goes on

STAMP=$(date +%Y%m%d_%H%M%S)
LOGS=${BAKERH_RUN_DIR:+$BAKERH_RUN_DIR/$S}
LOGS=${LOGS:-$LOG_ROOT/${STAMP}_$S}
mkdir -p "$LOGS"
rm -f "$LOGS/FAILED" "$LOGS/SOFT_FAILED"
SUMMARY=$LOGS/summary.log
JS=$LOGS/job_summary.txt
N=0
note() { echo "[$(date '+%F %T')] $S $*" | tee -a "$SUMMARY"; }
js() { echo "$*" >> "$JS"; echo "[job summary] $*"; }
hms() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
done_or_force() { [ -z "${FORCE:-}" ] && [ -e "$1" ]; }   # done_or_force PATH: skip the phase?
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

# run_phase NAME: runs phase_NAME with its output in NN_NAME.log (and on screen), samples GPU memory every
# 5 s, records OK/FAILED + duration + peak memory in summary.log, stops the scene on failure (soft phases:
# records it in SOFT_FAILED and goes on).
run_phase() {
  local name=$1 log start status peak sampler
  N=$((N + 1))
  log=$LOGS/$(printf %02d "$N")_$name.log
  note "START $name   log: $log"
  start=$(date +%s)
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 5000 > "$log.gpumem" 2>/dev/null &
  sampler=$!
  set +e
  ( set -euo pipefail; "phase_$name" ) 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  peak=$(sort -n "$log.gpumem" 2>/dev/null | tail -1)
  if [ "$status" -eq 0 ]; then
    note "END   $name   OK   $(hms $(($(date +%s) - start)))   peak GPU mem ${peak:-?} MiB"
    return
  fi
  note "END   $name   FAILED (exit $status)   $(hms $(($(date +%s) - start)))   peak GPU mem ${peak:-?} MiB"
  if grep -q -i -E 'out of memory|OutOfMemoryError' "$log"; then
    note "      cause: CUDA out of memory (peak ${peak:-?} MiB on GPU $GPU)"
  fi
  echo "---- last 30 lines of $log"
  tail -30 "$log"
  if [[ "$SOFT_PHASES" == *" $name "* ]]; then
    echo "phase=$name log=$log" >> "$LOGS/SOFT_FAILED"
    note "      $name is a soft phase: the scene goes on"
    return
  fi
  echo "phase=$name log=$log" > "$LOGS/FAILED"
  exit "$status"
}

pyjson() {  # pyjson FILE EXPR: EXPR over the json dict d, printed
  "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); print($2)" "$1"
}
label_counts() {  # label_counts MODEL IT: Gaussians per label (-1 = never hit)
  "$PY" -c "import torch, sys; l = torch.load(sys.argv[1], map_location='cpu').long(); print(len(l), 'Gaussians, never hit', int((l < 0).sum()), ', per label', torch.bincount(l[l >= 0]).tolist())" \
      "$1/point_cloud/iteration_$2/labels.pt"
}
faces() {  # faces PLY: face count from the PLY header
  { head -c 4096 "$1" | grep -a -m1 '^element face' | awk '{print $3}'; } 2>/dev/null || true
}
gsplat_py() {  # a Python that imports gsplat: GS_RENDER_PY, else the given ones (empty if none)
  local p
  for p in "$GS_RENDER_PY" "$@"; do
    [ -n "$p" ] && [ -x "$p" ] && "$p" -c 'import gsplat' 2>/dev/null && { echo "$p"; return 0; }
  done
  return 0
}

phase_check() {
  local missing=() f
  for f in "$PY" "$DATA/$S/$IMAGES" "$DATA/$S/sparse" "$MASK_SRC" "$ROOT/gaussian_wrapping/lpipsPyTorch/weights/v0.1/vgg.pth" \
           "$ROOT/gaussian_wrapping/lpipsPyTorch/weights/vgg16-397923af.pth"; do
    [ -e "$f" ] || missing+=("$f")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing: %s\n' "${missing[@]}"
    echo "(venv / LPIPS weights: run scripts/bakerh/setup.sh; masks: $SAM_MASKS/$S/{full,0_400}/masks_npy)"
    return 1
  fi
  js "photos: $(ls "$DATA/$S/$IMAGES" | wc -l) in $DATA/$S/$IMAGES; masks: $MASK_SRC ($(ls "$MASK_SRC" | grep -c '\.npy$') .npy)"
  js "gsplat (export check, ArtiFixer renders): $(gsplat_py || true)"
}

phase_masks() {
  if done_or_force "$MASKS/.done"; then echo "found $MASKS"; return; fi
  mkdir -p "$MASKS"
  "$PY" scripts/bakerh/masks_to_beags.py --mask_dir "$MASK_SRC" --out "$MASKS" | tee "$MASKS.txt"
  js "masks: $(tail -1 "$MASKS.txt")"
  touch "$MASKS/.done"
}

phase_train() {
  local ply=$M/point_cloud/iteration_$ITER/point_cloud.ply
  if done_or_force "$ply"; then echo "found $ply"; return; fi
  # shellcheck disable=SC2086
  "$PY" gaussian_wrapping/train.py -s "$DATA/$S" -m "$M" --images "$IMAGES" --eval --iterations "$ITER" \
      --test_iterations "$ITER" --save_iterations "$ITER" $GW_TRAIN_ARGS
  [ -f "$ply" ] || { echo "train.py wrote no $ply"; return 1; }
  js "GW $ITER: $(tr -d '\n ' < "$M/gs_stats.json")"
}

phase_export() {
  local out=$BASE/export/point_cloud.ply check=() gs
  if done_or_force "$out"; then echo "found $out"; return; fi
  gs=$(gsplat_py "$PY")
  [ -z "$gs" ] || check=(--check --check_stride 10 --gs_python "$gs")
  "$PY" gaussian_wrapping/export_3dgs_ply.py -m "$M" --iteration "$ITER" --out "$out" "${check[@]}"
  js "export: $(tr -d '\n' < "$BASE/export/export_info.json")"
  if [ -f "$BASE/export/gsplat_check.json" ]; then
    js "gsplat check: $(pyjson "$BASE/export/gsplat_check.json" "', '.join(f'{k} {d[k]}' for k in ('n_views', 'psnr', 'alpha_mae', 'gt_drop', 'worst') if k in d)")"
  else
    js "gsplat check: none ($([ -n "$gs" ] && echo "no gsplat_check.json" || echo "no Python with gsplat: setup.sh step 7"))"
  fi
}

phase_prepare() {
  if done_or_force "$M/point_cloud/iteration_$ITER/labels.pt" && [ -f "$M/beags/labeled_points.npz" ]; then echo "found labels"; return; fi
  "$PY" gaussian_wrapping/beags_prepare.py -m "$M" --iteration "$ITER" --masks "$MASKS"
  js "prepare ($ITER): $(label_counts "$M" "$ITER")"
}

phase_finetune() {
  if done_or_force "$B/point_cloud/iteration_$FT_ITER/labels.pt"; then echo "found iteration $FT_ITER"; return; fi
  # shellcheck disable=SC2086
  "$PY" gaussian_wrapping/beags_finetune.py -m "$M" --iteration "$ITER" --dilation_weight "$DW" --out "$B" $FT_ARGS
  [ -f "$B/point_cloud/iteration_$FT_ITER/labels.pt" ] || { echo "no iteration_$FT_ITER/labels.pt"; return 1; }
  js "finetune dw $DW ($FT_ITER): $(label_counts "$B" "$FT_ITER")"
}

eval_labels() {  # eval_labels MODEL IT: beags_eval.py against the SAM3 masks (BakerH has no GT: agreement with the input masks)
  local m=$1 it=$2 out=$1/beags/metrics_$2.json
  done_or_force "$out" || "$PY" gaussian_wrapping/beags_eval.py -m "$m" --iteration "$it" --gt_masks "$MASKS" --masks "$MASKS"
  js "BEA-GS vs SAM3 masks $(basename "$m") $it: $(pyjson "$out" "'; '.join(s + ': ' + ', '.join(f'{k} {d[s][k]:.4f}' for k in ('IoU', 'BIoU', 'EIoU', 'BEIoU') if d[s].get(k) is not None) + f' ({len(d[s][\"per_view\"])} views)' for s in d)")"
}

phase_labels() {
  eval_labels "$M" "$ITER"
  eval_labels "$B" "$FT_ITER"
}

phase_export_beags() {
  local out=$BASE/export_beags/point_cloud.ply
  if done_or_force "$BASE/export_beags/label_0.ply"; then echo "found $BASE/export_beags"; return; fi
  "$PY" gaussian_wrapping/export_3dgs_ply.py -m "$B" --iteration "$FT_ITER" --split_labels --out "$out"
  js "export_beags: $(ls "$BASE/export_beags"/*.ply | xargs -n1 basename | tr '\n' ' ')"
}

render_labels() {  # render_labels MODEL OUT
  if done_or_force "$2/grid_all.jpg"; then echo "found $2"; return; fi
  "$PY" gaussian_wrapping/beags_render_labels.py -m "$1" --iteration "$FT_ITER" --n_views "$N_VIEWS" --out "$2"
  js "label renders: $2 ($(ls "$2/full" | wc -l) views; $(ls "$2" | tr '\n' ' '))"
}

phase_renders() {
  render_labels "$B" "$BASE/renders_labels_$FT_ITER"
}

phase_showcase() {  # the label renders of every scene in one place: GT | full | label_0 (background) | label_1 (extraction)
  local src=$BASE/renders_labels_$FT_ITER out=$OUTPUT/$METHOD/showcase/$S
  [ -f "$src/grid_all.jpg" ] || { echo "no $src/grid_all.jpg: run the renders phase first"; return 1; }
  mkdir -p "$out"
  cp -r "$src/grid" "$src/full" "$src"/label_* "$out"/
  cp "$src/grid_all.jpg" "$OUTPUT/$METHOD/showcase/${S}_grid_all.jpg"
  js "showcase: $OUTPUT/$METHOD/showcase/${S}_grid_all.jpg (rows: GT | full | label_0 background | label_1 extraction; $(ls "$out/grid" | wc -l) views in $out)"
}

phase_metrics() {
  local m it
  for m in "$M:$ITER" "$B:$FT_ITER" "$BM:$FT_ITER"; do
    it=${m##*:}; m=${m%:*}
    if [ ! -f "$m/point_cloud/iteration_$it/point_cloud.ply" ]; then
      [ "$m" = "$BM" ] || echo "no $m iteration $it yet: skipped"; continue
    fi
    if [ -z "${FORCE:-}" ] && grep -q "ours_$it" "$m/results.json" 2>/dev/null; then echo "found $m/results.json"; else
      "$PY" gaussian_wrapping/render.py -m "$m" --iteration "$it" --rasterizer ours --skip_train
      "$PY" gaussian_wrapping/metrics.py -m "$m"
    fi
    js "test views $(basename "$m") $it: $(pyjson "$m/results.json" "', '.join(f'{k} {v:.4f}' for k, v in d['ours_$it'].items())")"
  done
}

hollow_line() {  # hollow_line JSON: the headline hollow metrics (legacy hole keys read as the new ones)
  pyjson "$1" "', '.join(f'{k} {d.get(k, d.get(o))}' for k, o in (('n_gaussians', 0), ('unseen_share', 0), ('unseen_mass_share', 0), ('free_space_mass', 0), ('floaters', 0), ('floater_gaussians', 0), ('floater_voxel', 'voxel'), ('fog_share_005', 0), ('fog_share_010', 0), ('opacity_entropy', 0), ('hole_views', 0), ('hole_alpha', 'hole_emptiness'), ('hole_alpha_8x8', 'hole_emptiness_8x8'), ('hole_alpha_unseen', 'hole_unseen_alpha'), ('hole_tightness', 0)) if k in d or o in d)"
}

phase_hollow() {
  local hs out obj=$B/object_mask_${LABEL}_$FT_ITER.npy
  hs=$(hm_suffix); out=$M/hollow_metrics_$ITER$hs.json  # HM_ARGS: own file, so a voxel change never reuses old counts
  # shellcheck disable=SC2086
  done_or_force "$out" || "$PY" gaussian_wrapping/hollow_metrics.py -m "$M" --iteration "$ITER" --out "$out" $HM_ARGS
  js "hollow metrics train $ITER: $(hollow_line "$out")"
  out=$B/hollow_metrics_$FT_ITER$hs.json
  if ! done_or_force "$out"; then
    "$PY" -c "import sys, numpy as np, torch; np.save(sys.argv[2], torch.load(sys.argv[1], map_location='cpu').numpy() == int(sys.argv[3]))" \
      "$B/point_cloud/iteration_$FT_ITER/labels.pt" "$obj" "$LABEL"
    # shellcheck disable=SC2086
    "$PY" gaussian_wrapping/hollow_metrics.py -m "$B" --iteration "$FT_ITER" --object_mask "$obj" --masks "$MASKS" \
      --mask_channel "$LABEL" --out "$out" $HM_ARGS
  fi
  js "hollow metrics beags $FT_ITER (object $LABEL): $(hollow_line "$out")"
}

phase_mesh() {
  local f line=
  if ! done_or_force "$MESH/hole_$LABEL.json"; then   # .json: holes from before the contact patch have none
    # shellcheck disable=SC2086
    [ -f "$B/mesh_ours_2pivots.ply" ] || "$PY" gaussian_wrapping/pivot_based_mesh_extraction.py -m "$B" --iteration "$FT_ITER" $GW_EXTRACT_ARGS
    # shellcheck disable=SC2086
    "$PY" gaussian_wrapping/multilabel_mesh_extraction.py -m "$B" --iteration "$FT_ITER" --input_mesh "$B/mesh_ours_2pivots.ply" \
        --out_dir "$MESH" --remove "$LABEL" ${MESH_ARGS:-}
    [ -f "$MESH/hole_$LABEL.ply" ] || { echo "no $MESH/hole_$LABEL.ply"; return 1; }
  fi
  for f in "$B/mesh_ours_2pivots.ply" "$MESH"/*.ply; do
    [ -f "$f" ] && line+="$(basename "$f") $(faces "$f"), "
  done
  js "mesh faces: ${line%, }"
}

colmap_view() {  # ArtiFixer-GS reads sparse/0/*.bin: a view of the scene with that layout when it has sparse/*.bin
  local colmap=$DATA/$S
  if [ ! -f "$colmap/sparse/0/images.bin" ] && [ -f "$colmap/sparse/images.bin" ]; then
    mkdir -p "$AFO/colmap/sparse"
    ln -sfn "$DATA/$S/sparse" "$AFO/colmap/sparse/0"
    ln -sfn "$DATA/$S/$IMAGES" "$AFO/colmap/$IMAGES"
    colmap=$AFO/colmap
  fi
  echo "$colmap"
}

phase_afinputs() {
  local mode=alpha af sr args
  # an env.sh from before the GW renderer (no RENDER_CMD / CENTER_PRINCIPAL_POINT) is rewritten (the export is cached)
  if done_or_force "$AFO/env.sh" && grep -q '^export RENDER_CMD=' "$AFO/env.sh" \
     && grep -q '^export CENTER_PRINCIPAL_POINT=' "$AFO/env.sh" \
     && { [ "$AF_UNSEEN" != on ] || grep -q '^export UNSEEN_DIR=' "$AFO/env.sh"; }; then echo "found $AFO/env.sh"; return; fi
  [ ! -f "$MESH/hole_$LABEL.ply" ] || mode=mesh
  mkdir -p "$AFO"
  args=(-m "$B" --iteration "$FT_ITER" --label "$LABEL" --variant "$AF_VARIANT" --masks "$MASKS" --hole_mode "$mode"
        --out "$AFO" --scene "$S" --colmap "$(colmap_view)" --images "$IMAGES" --train_args "$AF3D_TRAIN_ARGS"
        --unseen_maps "$AF_UNSEEN" --confidence_maps "$AF_CONF" --kappa "$KAPPA")
  [ ! -f "$MESH/hole_$LABEL.ply" ] || args+=(--hole_mesh "$MESH/hole_$LABEL.ply")
  [ -z "${METRIC_SCALE:-}" ] || args+=(--metric_scale "$METRIC_SCALE")
  if af=$(find_af_repo); then
    sr=$(af_scene_root "$S" "$af")
    [ ! -d "$sr" ] || args+=(--artifixer_root "$sr")   # its metric scale and caption.h5
  fi
  "$PY" gaussian_wrapping/beags_artifixer.py "${args[@]}"
  js "artifixer inputs ($AF_VARIANT -> $AFO, $mode hole; unseen maps $AF_UNSEEN, confidence maps $AF_CONF): $(pyjson "$AFO/inputs.json" "f\"object {d['object_gaussians']} of {d['num_gaussians']} Gaussians; masks {(d.get('mask_stats') or {}).get('masks')} of {(d.get('mask_stats') or {}).get('photos')} photos, object in {(d.get('mask_stats') or {}).get('object_views')}; metric scale {d.get('metric_scale')} ({d.get('metric_scale_source')}); caption.h5 {d.get('CAPTION_H5', '-')}\"")"
}

newest_pred() {  # newest ArtiFixer prediction dir under $1 (empty if none)
  [ -d "$1" ] || return 0
  find "$1" -type d -path '*/frames/batch_0000/pred' -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
}

phase_artifixer() {
  local af apy gs caption root=$AFO/$S
  [ -f "$AFO/env.sh" ] || { echo "no $AFO/env.sh: run the afinputs phase first"; return 1; }
  af=$(find_af_repo) || { echo "no ArtiFixer repo with .venv/bin/python and checkpoints/ (set AF_REPO)"; return 1; }
  apy=${ARTIFIXER_PY:-$af/.venv/bin/python}
  gs=$(gsplat_py "$apy")
  # renders go through GW's render_ours (RENDER_CMD); gsplat only without it, or for a trajectory (TRAJ_MODE) path
  if ! grep -q '^export RENDER_CMD=' "$AFO/env.sh" || grep -q '^export TRAJ_MODE=' "$AFO/env.sh"; then
    [ -n "$gs" ] || { echo "no Python with gsplat (GS_RENDER_PY=$GS_RENDER_PY, $apy): SETUP_GSPLAT=1 bash scripts/bakerh/setup.sh"; return 1; }
  fi
  [ -f "$AFS/scripts/run.sh" ] && [ -f "$AFS/artifixer_gs/hole_mesh.py" ] \
    || { echo "$AFS lacks scripts/run.sh / artifixer_gs/hole_mesh.py: it needs ArtiFixer-GS's gw branch (setup.sh step 8)"; return 1; }
  (
    set -a; source "$AFO/env.sh"; set +a
    if [ -n "${CAPTION_H5:-}" ] && [ ! -f "$root/captions/$S/caption.h5" ]; then
      mkdir -p "$root/captions/$S"; cp -L "$CAPTION_H5" "$root/captions/$S/caption.h5"; echo "caption from $CAPTION_H5"
    fi
    # run.sh keeps captions/<id>/caption.h5 only when CAPTION is empty: ArtiFixer's caption.h5 first, the scene
    # caption only when there is none (as ArtiFixer's run_removal.sh); an explicit CAPTION always wins.
    caption=${CAPTION:-}
    if [ -z "$caption" ] && [ ! -f "$root/captions/$S/caption.h5" ]; then caption=$(scene_caption "$S"); fi
    if [ -z "$caption" ] && [ ! -f "$root/captions/$S/caption.h5" ]; then
      echo "no caption for $S: set CAPTION (no caption.h5 in ArtiFixer's root $(af_scene_root "$S" "$af"))"; exit 2
    fi
    if [ -n "$caption" ]; then export CAPTION=$caption; else unset CAPTION; fi
    if [ "${MODE:-}" = removal ]; then
      export SEEDS=${SEEDS:-auto} OUTSIDE=${OUTSIDE:-photo}
      if [ "$S" = Compressor ] && [ -n "$COMPRESSOR_FRAMES" ]; then export FRAMES=${FRAMES:-$COMPRESSOR_FRAMES}; fi
      [ "$AF_VARIANT" != bighull ] || export EMPTY_CAPTION
    fi
    if [ -n "${TRAJ_MODE:-}" ] && [ "$S" = Compressor ] && [ -n "$COMPRESSOR_TRAJ_FRAMES" ]; then
      export TRAJ_MAX_FRAMES=${TRAJ_MAX_FRAMES:-$COMPRESSOR_TRAJ_FRAMES}   # one path frame per photo = 923: too many
    fi
    [ -z "${TRAJ_MAX_FRAMES:-}" ] || export TRAJ_MAX_FRAMES
    export PY=$apy GS_RENDER_PY=${gs:-} MODEL_ID=${MODEL_ID:-$af/checkpoints/Wan2.1-T2V-1.3B-Diffusers}
    export CHECKPOINT_PT=${CHECKPOINT_PT:-$af/checkpoints/ArtiFixer/artifixer-1.3b.pt}
    export MOGE_MODEL_PATH=${MOGE_MODEL_PATH:-$af/checkpoints/moge-2-vitl-normal/model.pt}
    export NUM_VIEWS=${NUM_VIEWS:-6} AF3D_STEPS=${AF3D_STEPS:-30000} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=
    echo "ArtiFixer-GS: PY=$PY GS_RENDER_PY=$GS_RENDER_PY MODEL_ID=$MODEL_ID CHECKPOINT_PT=$CHECKPOINT_PT"
    echo "  OUT=$OUT MODE=${MODE:-} HOLE_MODE=${HOLE_MODE:-} FRAMES=${FRAMES:-all} SEEDS=${SEEDS:-} CAPTION=${CAPTION:-<caption.h5>}"
    cd "$AFS"   # keeps this repo's scene/ and utils/ off sys.path
    bash scripts/run.sh
  )
  js "ArtiFixer: $(newest_pred "$root/artifixer")"
  js "ArtiFixer3D: $root/af3d/export/point_cloud.ply; $(tr -d '\n ' < "$root/af3d/fastgs/gs_stats.json" 2>/dev/null || echo 'no gs_stats.json')"
  js "ArtiFixer3D+: $(newest_pred "$root/artifixer3d_plus")"
}

phase_seed2gs() {
  local prompt slug out
  [ -n "$SAM3_PYTHON" ] && [ -x "$SAM3_PYTHON" ] || { echo "seed2gs needs SAM3_PYTHON (a SAM3 env; also SAM3_REPO, SAM3_CHECKPOINT)"; return 1; }
  prompt=$(seed_prompt "$S")
  slug=$(echo "$prompt" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '_')
  out=$OUTPUT/seed2gs/$S/$slug
  done_or_force "$out/seed2gs.json" || \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" gaussian_wrapping/seed2gs.py -m "$M" --iteration "$ITER" \
      --prompt "$prompt" --out "$out" --sam3_python "$SAM3_PYTHON"
  "$PY" - "$out/masks" "$MASKS" "$out/iou_vs_sam3.json" <<'EOF'
import json, os, sys
import cv2
import numpy as np
pred_dir, ref_dir, out = sys.argv[1:]
ious = []
for f in sorted(os.listdir(pred_dir)):
    ref = os.path.join(ref_dir, f)
    if not f.endswith(".npy") or not os.path.exists(ref):
        continue
    p, r = np.load(os.path.join(pred_dir, f))[1].astype(np.float32) > 0.5, np.load(ref)[1] > 0
    if p.shape != r.shape:
        p = cv2.resize(p.astype(np.uint8), (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    union = (p | r).sum()
    if union:
        ious.append(float((p & r).sum() / union))
res = {"mean_iou": float(np.mean(ious)) if ious else None, "views": len(ious)}
json.dump(res, open(out, "w"), indent=1)
print(res)
EOF
  js "seed2gs '$prompt': $(pyjson "$out/seed2gs.json" "f\"SAM3 score {d['sam3_score']:.3f}, object {d['n_object']} of {d['n_gaussians']}, {d['seconds']['total']:.0f} s\""); mask IoU vs SAM3: $(tr -d '\n' < "$out/iou_vs_sam3.json")"
}

phase_holes() {  # removal holes side by side, no diffusion: alpha | mesh | gavis per UNSEEN_THRESHS | alpha + confidence
  local af apy
  [ -f "$AFO/env.sh" ] || { echo "no $AFO/env.sh: run the afinputs phase first"; return 1; }
  af=$(find_af_repo) || { echo "no ArtiFixer repo with .venv/bin/python (set AF_REPO)"; return 1; }
  apy=${ARTIFIXER_PY:-$af/.venv/bin/python}
  (
    set -a; source "$AFO/env.sh"; set +a                     # OUT, MASK_DIR, MASK_LABEL, UNSEEN_DIR, CONFIDENCE_DIR
    [ -d "$OUT/removal" ] || { echo "no $OUT/removal: run the artifixer phase with STAGES=prepare,removal,split first"; exit 1; }
    frames=; [ "$S" != Compressor ] || frames=${FRAMES:-$COMPRESSOR_FRAMES}
    grid=() rc=0
    split() {  # split NAME HOLE_MODE_ARGS...: one removal split into $OUT/removal_split_NAME
      local name=$1 dir=$OUT/removal_split_$1; shift
      if ! done_or_force "$dir/split.json"; then
        (cd "$AFS" && PYTHONPATH= "$apy" -m artifixer_gs.removal split --scene_root "$OUT" --removal_dir "$OUT/removal" \
          --mask_dir "$MASK_DIR" ${MASK_LABEL:+--mask_label "$MASK_LABEL"} --output_dir "$dir" --hole_shape mask \
          --dilate_px "${DILATE:-12}" --opacity hole0 --frames "$frames" "$@") || { echo "split $name failed"; rc=1; return; }
      fi
      grid+=(--split "$name=$dir")
    }
    split alpha --hole_mode alpha
    if [ -d "$OUT/removal/mesh_front" ]; then split mesh --hole_mode mesh; else echo "no $OUT/removal/mesh_front: no mesh split"; fi
    if [ -n "${UNSEEN_DIR:-}" ] && [ -d "$UNSEEN_DIR" ]; then
      for t in $UNSEEN_THRESHS; do split "gavis$t" --hole_mode gavis --unseen_dir "$UNSEEN_DIR" --unseen_threshold "$t"; done
    else
      echo "no UNSEEN_DIR in $AFO/env.sh: no gavis split (afinputs with AF_UNSEEN=on)"
    fi
    if [ -n "${CONFIDENCE_DIR:-}" ] && [ -d "$CONFIDENCE_DIR" ]; then
      split "conf$HOLE_CONF_THRESH" --hole_mode alpha --confidence_dir "$CONFIDENCE_DIR" --conf_threshold "$HOLE_CONF_THRESH"
    else
      echo "no CONFIDENCE_DIR: no confidence split (needs a model trained with --confidence: VARIANT=conf)"
    fi
    [ ${#grid[@]} -gt 0 ] || exit 1
    "$PY" gaussian_wrapping/beags/hole_grid.py "${grid[@]}" ${UNSEEN_DIR:+--unseen_dir "$UNSEEN_DIR"} --n 8 \
      --out "$AFO/hole_grid.jpg" || rc=1
    exit $rc
  )
  js "holes: $AFO/hole_grid.jpg (photo | removal render | one column per hole mode | unseen map; splits $(ls -d "$AFO/$S"/removal_split_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '))"
}

newest_dir() {  # newest_dir ROOT PATH_GLOB: newest directory under ROOT matching the find -path glob (empty if none)
  [ -d "$1" ] || return 0
  find "$1" -type d -path "$2" -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
}

phase_trajcands() {  # the novel camera path (vanilla_*) after STAGES=trajectory,prepare: top view + path renders
  local root=$AFO/$S renders
  [ -f "$root/trajectory_input.json" ] || { echo "no $root/trajectory_input.json: run the artifixer phase with STAGES=trajectory,prepare"; return 1; }
  renders=$(newest_dir "$root" '*/renders/trajectory/renders')
  [ -n "$renders" ] || { echo "no renders/trajectory/renders under $root (prepare stage)"; return 1; }
  "$PY" scripts/bakerh/traj_grid.py --col "GW render on the $AF_VARIANT path=$renders" --n 16 --wrap 4 --out "$AFO/path_candidates.jpg"
  js "path: $root/trajectory_input.png (top view) + $AFO/path_candidates.jpg ($(ls "$renders" | wc -l) frames in $renders); info: $(tr -d '\n' < "$root/trajectory_input_info.json" 2>/dev/null | head -c 400)"
}

phase_trajgrid() {  # before / after on the novel path: GW render | ArtiFixer | ArtiFixer3D render | ArtiFixer3D+
  local root=$AFO/$S af plus cols=()
  af=$(newest_pred "$root/artifixer"); plus=$(newest_pred "$root/artifixer3d_plus")
  [ -n "$af" ] || { echo "no ArtiFixer predictions under $root/artifixer: run the artifixer phase"; return 1; }
  cols=(--col "GW render (input)=$(dirname "$af")/rendered" --col "ArtiFixer=$af")
  [ -z "$plus" ] || cols+=(--col "ArtiFixer3D render=$(dirname "$plus")/rendered" --col "ArtiFixer3D+=$plus")
  "$PY" scripts/bakerh/traj_grid.py "${cols[@]}" --n "$N_VIEWS" --out "$AFO/novel_views_grid.jpg"
  mkdir -p "$OUTPUT/$METHOD/showcase"
  cp "$AFO/novel_views_grid.jpg" "$OUTPUT/$METHOD/showcase/${S}_${AF_DIR}_grid.jpg"
  js "novel views: $OUTPUT/$METHOD/showcase/${S}_${AF_DIR}_grid.jpg (GW render | ArtiFixer | ArtiFixer3D render | ArtiFixer3D+)"
}

phase_mesh_train() {  # multi-label mesh of the prepared 30k model, with tets.npz: the source of --occ_source mesh
  if done_or_force "$MT/tets.npz"; then echo "found $MT/tets.npz"; return; fi
  [ -f "$M/point_cloud/iteration_$ITER/labels.pt" ] || { echo "no labels at $ITER: run the prepare phase first"; return 1; }
  # shellcheck disable=SC2086
  [ -f "$M/mesh_ours_2pivots.ply" ] || "$PY" gaussian_wrapping/pivot_based_mesh_extraction.py -m "$M" --iteration "$ITER" $GW_EXTRACT_ARGS
  # shellcheck disable=SC2086
  "$PY" gaussian_wrapping/multilabel_mesh_extraction.py -m "$M" --iteration "$ITER" --input_mesh "$M/mesh_ours_2pivots.ply" \
      --out_dir "$MT" --save_tets ${MESH_ARGS:-}
  [ -f "$MT/tets.npz" ] || { echo "no $MT/tets.npz"; return 1; }
  js "mesh_train: $MT ($(du -h "$MT/tets.npz" | cut -f1) tets.npz)"
}

phase_finetune_meshocc() {  # the BEA-GS finetune with L_occ / L_piv against the mesh instead of the labelled voxels
  if done_or_force "$BM/point_cloud/iteration_$FT_ITER/labels.pt"; then echo "found $BM iteration $FT_ITER"; return; fi
  [ -f "$MT/tets.npz" ] || { echo "no $MT/tets.npz: run the mesh_train phase first"; return 1; }
  # shellcheck disable=SC2086
  "$PY" gaussian_wrapping/beags_finetune.py -m "$M" --iteration "$ITER" --dilation_weight "$DW" --out "$BM" \
      --occ_source mesh --mesh_dir "$MT" $FT_ARGS
  [ -f "$BM/point_cloud/iteration_$FT_ITER/labels.pt" ] || { echo "no $BM iteration_$FT_ITER/labels.pt"; return 1; }
  js "finetune meshocc dw $DW ($FT_ITER): $(label_counts "$BM" "$FT_ITER")"
}

phase_labels_meshocc() {
  eval_labels "$BM" "$FT_ITER"
  [ ! -f "$B/beags/metrics_$FT_ITER.json" ] || js "  (voxel finetune for comparison: the labels phase line above)"
}

phase_renders_meshocc() {
  local out=$BASE/renders_labels_meshocc_$FT_ITER
  render_labels "$BM" "$out"
  mkdir -p "$OUTPUT/$METHOD/showcase"
  cp "$out/grid_all.jpg" "$OUTPUT/$METHOD/showcase/${S}_meshocc_grid_all.jpg"
  js "showcase meshocc: $OUTPUT/$METHOD/showcase/${S}_meshocc_grid_all.jpg (compare with ${S}_grid_all.jpg)"
}

phase_summary() {
  "$PY" scripts/bakerh/report.py --output "$OUTPUT" --method "$METHOD" --scene "$S" --iter "$ITER" --ft_iter "$FT_ITER" \
      --label "$LABEL" --af_variant "$AF_DIR" --hm_suffix "$(hm_suffix)" | tee "$LOGS/report.txt"
}

{
  echo "scene=$S phases=$PHASES gpu=$GPU dw=$DW label=$LABEL method=$METHOD"
  echo "data=$DATA/$S images=$IMAGES masks=$MASK_SRC -> $MASKS"
  echo "train=$M beags=$B mesh=$MESH artifixer=$AFO"
  echo "train args: $GW_TRAIN_ARGS"
  echo "finetune args: --dilation_weight $DW $FT_ARGS; hollow metrics args: $HM_ARGS; mesh args: ${MESH_ARGS:-}"
  echo "ArtiFixer3D train args: $AF3D_TRAIN_ARGS"
  echo "python=$PY gs_render_py=$GS_RENDER_PY on PATH: python=$(command -v python) nvcc=$(command -v nvcc) CUDA_HOME=${CUDA_HOME:-}"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) changed files;" \
       "artifixer_gs $(git -C "$AFS" rev-parse --short HEAD 2>/dev/null) ($(git -C "$AFS" status --porcelain 2>/dev/null | wc -l) changed)"
  nvidia-smi -i "$GPU" 2>&1 | head -20 || true
} > "$LOGS/00_config.log"
note "RUN $STAMP   phases=$PHASES   config: $LOGS/00_config.log"

IFS=, read -ra phase_list <<< "$PHASES"
for p in "${phase_list[@]}"; do
  declare -F "phase_$p" > /dev/null || { echo "unknown phase '$p'" >&2; exit 2; }
done
for p in "${phase_list[@]}"; do
  run_phase "$p"
done
note "DONE  phases=$PHASES$([ ! -f "$LOGS/SOFT_FAILED" ] || echo "   soft failures: $(sed 's/ log=.*//; s/phase=//' "$LOGS/SOFT_FAILED" | tr '\n' ' ')")"
echo
echo "BEA-GS model: $B   renders: $BASE/renders_labels_$FT_ITER   mesh + hole: $MESH   ArtiFixer: $AFO"
