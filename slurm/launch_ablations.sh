#!/usr/bin/env bash
# Arms and ablations on ONE scene (plan "Arms and ablations", HANDOFF 7.4.5), through the slurm/gw_*.sh job scripts.
# Dry run by default: prints the sbatch lines (a runnable script: J<n>=$(sbatch --parsable ...), later jobs
# --dependency=afterok:$J<n>); SUBMIT=1 submits them. Each arm is its own METHOD, so outputs stay flat:
# output/<METHOD>/<name>/{train,beags,...}. A job whose key output exists is printed as "# skip" (FORCE=1 keeps it).
#
# Usage (from repo root):  bash slurm/launch_ablations.sh [group ...]      groups (default: all four)
#   arms        gw          GW recipe + ROBUST_ARGS, no hollowing (HOLLOW_ARGS=)       train -> beags -> hollow metrics
#               gw_hollow   = output/gw_beags/<name>/train (the full arm's train mode: same flags, no extra job)
#               gw_beags    full pipeline, slurm/gw_scene.sh (train, BEA-GS, renders, hollow metrics, mesh)
#               (fastgs_beags is the baseline in ../BEA-GS_FSplat, not launched here)
#   appearance  hollowing on, ROBUST_ARGS per arm                                       train -> hollow metrics
#               gw_app_none (--appearance none), gw_app_exposure (GW exposure compensation; the exported PLY and
#               the BEA-GS finetune do not carry it), gw_app_bilagrid_dec (bilagrid+decoupled), gw_app_bilagrid_dec_conf
#               (+ --confidence); bilagrid = the full arm (common.sh ROBUST_ARGS), no extra job
#   hollow      one hollowing step off at a time (on = common.sh HOLLOW_ARGS)          train -> beags -> hollow metrics
#               gw_noh1         H1 visibility ledger: --prune_unseen_every 0 --seen_tau 0 (finalize() too: seen < 0)
#               gw_noh2         H2 free-space prior and free grid: --lambda_fs 0
#               gw_noh2grid     H2 ray-carved free grid only: --lambda_free 0
#               gw_noh3         H3 entropy + final opacity prune: --lambda_entropy 0 --final_prune_opacity 0
#                               (opacity decay is already off: --opacity_decay 0)
#               gw_noh4         H4 see-through carving: --carve_every 0 --carve_threshold 1 (finalize() carves T > 1:
#                               never)
#               gw_nofinal      no final hollow prune at the last iteration: --no_hollow_finalize (HANDOFF 7.1 suspect)
#               gw_h5           H5 colour / normal variance losses are OFF in the full arm: this arm turns them ON
#                               (ROBUST_ARGS + --lambda_color_var 0.5 --lambda_normal_var 0.005)
#   piv         gw_beags_nopiv  BEA-GS finetune with --lambda_piv 0 on the full arm's train model: train/ is a
#                               symlink to output/gw_beags/<name>/train (gw_beags.sh reads output/$METHOD/<name>/train)
# Scene (Mip-NeRF 360, paths as slurm/launch_one_per_dataset.sh / HANDOFF 10): SCENE (kitchen), NAME (mip360_$SCENE),
#   TRAIN_EXTRA (-i images_4). Knobs: SUBMIT (0), FORCE, REMOVE (1: object label of the hole metrics), FULL_METHOD
#   (gw_beags), HM_ARGS (extra hollow_metrics.py args, e.g. "--floater_voxel 0.02" so the floater counts of
#   all arms share one world-aligned grid; every arm, the full one too, then writes and skips on
#   hollow_metrics_<it>_<HM_ARGS as a name>.json), FT_ARGS (extra beags_finetune.py args), ITER / FT_ITER
#   (30000 / 33000),
#   and everything in slurm/common.sh passes through to the jobs (sbatch exports the environment).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SUBMIT=${SUBMIT:-0}
SCENE=${SCENE:-kitchen}
NAME=${NAME:-mip360_$SCENE}
MIP=/leonardo_work/IscrC_EditGS/datasets/mip360
MM=/leonardo_work/IscrC_EditGS/gabri/datasets/mipnerf360_hd_masks/masks
SRC=${SRC:-$MIP/$SCENE}
MASKS=${MASKS:-$MM/training/sam2/$SCENE/masks}
GT=${GT:-$MM/gt_masks/$SCENE/masks}
read -ra TRAIN_EXTRA <<< "${TRAIN_EXTRA--i images_4}"
REMOVE=${REMOVE:-1}
FULL_METHOD=${FULL_METHOD:-gw_beags}
ITER=${ITER:-30000}
FT_ITER=${FT_ITER:-33000}
read -ra HM <<< "${HM_ARGS:-}"
# with HM_ARGS the hollow metrics go to their own file (--out), also the skip test: hollow_metrics_<it><HMS>.json
# (e.g. _floater_voxel_0.02), so a relaunch with another voxel never mixes grids nor overwrites the plain JSONs
HMS=; [ ${#HM[@]} -eq 0 ] || HMS=_$(printf %s "${HM_ARGS}" | tr -cs 'A-Za-z0-9.' _ | sed 's/^_*//; s/_*$//')
hm() { echo "$1/hollow_metrics_$2$HMS.json"; }  # hm MODEL IT: the hollow metrics file of this launch
read -ra FT <<< "${FT_ARGS:-}"
# common.sh defaults (not sourced here: it loads modules); a value set in the environment wins, as in the jobs.
ROBUST_DEFAULT=${ROBUST_ARGS-"--no-exposure_compensation --appearance bilagrid"}
HOLLOW_DEFAULT=${HOLLOW_ARGS-"--hollow"}
GROUPS_=("$@"); [ ${#GROUPS_[@]} -gt 0 ] || GROUPS_=(arms appearance hollow piv)
for g in "${GROUPS_[@]}"; do
  case $g in arms|appearance|hollow|piv) ;; *) echo "unknown group '$g' (arms, appearance, hollow, piv)" >&2; exit 2 ;; esac
done
for p in "$SRC" "$MASKS" "$GT"; do [ -e "$p" ] || echo "# WARNING: missing $p" >&2; done
mkdir -p logs

q() {  # q WORD...: shell-quoted words, plain when safe
  local w out=()
  for w in "$@"; do
    if [ -z "$w" ]; then out+=("''"); elif [[ "$w" =~ ^[A-Za-z0-9_./:=+,@%-]+$ ]]; then out+=("$w"); else out+=("'${w//\'/\'\\\'\'}'"); fi
  done
  echo "${out[*]}"
}
qenv() {  # qenv VAR=VALUE...: assignments with the value quoted (a quoted VAR= would not assign)
  local a out=()
  for a in "$@"; do out+=("${a%%=*}=$(q "${a#*=}")"); done
  echo "${out[*]}"
}
N=0
# job DONE_PATH DEPS JOBNAME ENV... -- SCRIPT ARGS...: prints (SUBMIT=1: also runs) one sbatch line and sets
# LAST to its J<n> variable name, or to "" when skipped. DEPS: space-separated J<n> names ("" = none).
job() {
  local done_path=$1 deps=$2 jname=$3 envs=() d dep=() line
  shift 3
  while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  if [ -z "${FORCE:-}" ] && [ -n "$done_path" ] && [ -e "$done_path" ]; then
    echo "# skip $jname (exists: $done_path)"; LAST=; return
  fi
  for d in $deps; do dep+=("\$$d"); done
  N=$((N + 1)); LAST=J$N
  line="J$N=\$(${envs[*]:+$(qenv "${envs[@]}") }sbatch --parsable -J $jname"
  [ ${#dep[@]} -eq 0 ] || line+=" --dependency=afterok:$(IFS=:; echo "${dep[*]}")"
  line+=" $(q "$@"))"
  echo "$line"
  if [ "$SUBMIT" = 1 ]; then eval "$line"; echo "#   -> job ${!LAST}"; fi
}

# arm METHOD ROBUST HOLLOW STAGES: one training arm; STAGES "train" (+ hollow metrics) or "beags" (+ BEA-GS,
# hollow metrics of both models).
arm() {
  local method=$1 robust=$2 hollow=$3 stages=$4 m b env jt jb
  m=output/$method/$NAME/train; b=output/$method/$NAME/beags
  env=(METHOD="$method" ROBUST_ARGS="$robust" HOLLOW_ARGS="$hollow")
  echo; echo "# $method: ROBUST_ARGS='$robust' HOLLOW_ARGS='$hollow'"
  job "$m/results.json" "" "ab_${method}_$NAME" "${env[@]}" -- slurm/gw_train.sh "$NAME" "$SRC" "${TRAIN_EXTRA[@]}"
  jt=$LAST
  job "$(hm "$m" "$ITER")" "$jt" "ab_hm_${method}_$NAME" "${env[@]}" -- \
      slurm/gw_hollow_metrics.sh "$m" "$ITER" "${HM[@]}" --out "$(hm "$m" "$ITER")"
  [ "$stages" = beags ] || return 0
  job "$b/results.json" "$jt" "ab_beags_${method}_$NAME" "${env[@]}" -- \
      slurm/gw_beags.sh "$NAME" "$MASKS" "$GT" "${FT[@]}"
  jb=$LAST
  job "$(hm "$b" "$FT_ITER")" "$jb" "ab_hmb_${method}_$NAME" "${env[@]}" -- \
      slurm/gw_hollow_metrics.sh "$b" "$FT_ITER" --label "$REMOVE" --masks "$MASKS" "${HM[@]}" --out "$(hm "$b" "$FT_ITER")"
}

echo "# scene $NAME: $SRC ${TRAIN_EXTRA[*]}; masks $MASKS; gt $GT; $([ "$SUBMIT" = 1 ] && echo SUBMIT || echo 'dry run (SUBMIT=1 submits)')"
echo "set -e"
FULL_JOB=
for g in "${GROUPS_[@]}"; do
  case $g in
    arms)
      arm gw "$ROBUST_DEFAULT" "" beags
      echo; echo "# gw_hollow = output/$FULL_METHOD/$NAME/train (same flags as the full arm's training: no extra job)"
      echo; echo "# $FULL_METHOD (full): slurm/gw_scene.sh"
      job "output/$FULL_METHOD/$NAME/mesh/hole_$REMOVE.json" "" "ab_${FULL_METHOD}_$NAME" METHOD="$FULL_METHOD" REMOVE="$REMOVE" -- \
          slurm/gw_scene.sh "$NAME" "$SRC" "$MASKS" "$GT" "${TRAIN_EXTRA[@]}"
      FULL_JOB=$LAST
      [ ${#HM[@]} -eq 0 ] || {  # gw_scene.sh runs hollow_metrics.py without HM_ARGS: once more with them, to the
        m=output/$FULL_METHOD/$NAME/train; b=output/$FULL_METHOD/$NAME/beags  # suffixed file (its JSONs stay as is)
        job "$(hm "$m" "$ITER")" "$FULL_JOB" "ab_hm_${FULL_METHOD}_$NAME" METHOD="$FULL_METHOD" -- \
            slurm/gw_hollow_metrics.sh "$m" "$ITER" "${HM[@]}" --out "$(hm "$m" "$ITER")"
        job "$(hm "$b" "$FT_ITER")" "$FULL_JOB" "ab_hmb_${FULL_METHOD}_$NAME" METHOD="$FULL_METHOD" -- \
            slurm/gw_hollow_metrics.sh "$b" "$FT_ITER" --label "$REMOVE" --masks "$MASKS" "${HM[@]}" --out "$(hm "$b" "$FT_ITER")"
      } ;;
    appearance)
      echo; echo "# appearance: bilagrid = the full arm ($FULL_METHOD, ROBUST_ARGS='$ROBUST_DEFAULT')"
      arm gw_app_none "--appearance none" "$HOLLOW_DEFAULT" train
      arm gw_app_exposure "--appearance exposure" "$HOLLOW_DEFAULT" train
      arm gw_app_bilagrid_dec "--appearance bilagrid+decoupled" "$HOLLOW_DEFAULT" train
      arm gw_app_bilagrid_dec_conf "--appearance bilagrid+decoupled --confidence" "$HOLLOW_DEFAULT" train ;;
    hollow)
      arm gw_noh1 "$ROBUST_DEFAULT" "$HOLLOW_DEFAULT --prune_unseen_every 0 --seen_tau 0" beags
      arm gw_noh2 "$ROBUST_DEFAULT" "$HOLLOW_DEFAULT --lambda_fs 0" beags
      arm gw_noh2grid "$ROBUST_DEFAULT" "$HOLLOW_DEFAULT --lambda_free 0" beags
      arm gw_noh3 "$ROBUST_DEFAULT" "$HOLLOW_DEFAULT --lambda_entropy 0 --final_prune_opacity 0" beags
      arm gw_noh4 "$ROBUST_DEFAULT" "$HOLLOW_DEFAULT --carve_every 0 --carve_threshold 1" beags
      arm gw_nofinal "$ROBUST_DEFAULT" "$HOLLOW_DEFAULT --no_hollow_finalize" beags
      arm gw_h5 "$ROBUST_DEFAULT --lambda_color_var 0.5 --lambda_normal_var 0.005" "$HOLLOW_DEFAULT" beags ;;
    piv)
      p=gw_beags_nopiv; b=output/$p/$NAME/beags
      echo; echo "# $p: beags_finetune.py --lambda_piv 0 on output/$FULL_METHOD/$NAME/train (symlinked as $p's train)"
      link="mkdir -p output/$p/$NAME && ln -sfn ../../$FULL_METHOD/$NAME/train output/$p/$NAME/train"
      echo "$link"
      [ "$SUBMIT" != 1 ] || eval "$link"
      # without the full job in this run, its train model must already exist (else gw_beags.sh fails at prepare)
      [ -n "$FULL_JOB" ] || [ -f "output/$FULL_METHOD/$NAME/train/point_cloud/iteration_$ITER/point_cloud.ply" ] \
        || echo "# WARNING: output/$FULL_METHOD/$NAME/train has no iteration $ITER yet: run the arms group first" >&2
      job "$b/results.json" "$FULL_JOB" "ab_beags_${p}_$NAME" METHOD="$p" -- \
          slurm/gw_beags.sh "$NAME" "$MASKS" "$GT" --lambda_piv 0 "${FT[@]}"
      job "$(hm "$b" "$FT_ITER")" "$LAST" "ab_hmb_${p}_$NAME" METHOD="$p" -- \
          slurm/gw_hollow_metrics.sh "$b" "$FT_ITER" --label "$REMOVE" --masks "$MASKS" "${HM[@]}" --out "$(hm "$b" "$FT_ITER")" ;;
  esac
done
echo
echo "# $N job(s)$([ "$SUBMIT" = 1 ] && echo ' submitted' || echo ' printed; SUBMIT=1 submits them'). Logs: logs/<job name>_<id>.out"
