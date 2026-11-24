# Shared code of the step-by-step stage scripts (sourced, not run). Each stage = run_all.sh with a fixed PHASES list
# (run_scene.sh phases + summary): detached, one REPORT.txt per stage in output/bakerh_logs/<stamp>_stage<N>[_<VARIANT>]/,
# plus CHECK.txt (what to look at when the stage is done). Finished phases are skipped, so a stage reruns in seconds
# up to where it stopped. Only one run at a time (run_all.sh refuses a second).
#
# Knobs (env):
#   SCENES   default "TA_BLUE_MOTOR Compressor"
#   VARIANT  unset = the baseline recipe (output/gw_beags_dw0.1); conf = train.py --confidence
#            (output/gw_conf_beags_dw0.1, enables the confidence hole); gridfix = bilateral-grid drift fixes
#            (GRIDFIX_ARGS, output/gw_gridfix_beags_dw0.1). A variant also gets its own ArtiFixer dir
#            output/artifixer/<scene>/<AF_VARIANT>_<VARIANT>.
#   GPU, DW, FORCE, AF_VARIANT, ... as in common.sh; --fg (first argument) runs in this terminal.
STAGES_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SCENES=${SCENES:-"TA_BLUE_MOTOR Compressor"}
GRIDFIX_ARGS=${GRIDFIX_ARGS:-"--bilagrid_identity_weight 10 --bilagrid_mean_weight 10 --bilagrid_until 25000 --lambda_raw 0.2"}
# shellcheck disable=SC1091
source "$STAGES_DIR/../common.sh"   # the baseline GW_TRAIN_ARGS, DW, LOG_ROOT

case ${VARIANT:-} in
  "") ;;
  conf)    export GW_TRAIN_ARGS="$GW_TRAIN_ARGS --confidence" ;;
  gridfix) export GW_TRAIN_ARGS="$GW_TRAIN_ARGS $GRIDFIX_ARGS" ;;
  *) echo "VARIANT must be empty, conf or gridfix (not '$VARIANT')" >&2; exit 2 ;;
esac
if [ -n "${VARIANT:-}" ]; then
  export METHOD=gw_${VARIANT}_beags_dw$DW
  export AF_DIR=${AF_VARIANT}_$VARIANT
fi

# run_stage NAME PHASES CHECK_LINE...: launch run_all.sh with PHASES=<PHASES>,summary; print and save CHECK.txt
run_stage() {
  local name=$1 phases=$2 stamp; shift 2
  stamp=$(date +%Y%m%d_%H%M%S)_stage$name${VARIANT:+_$VARIANT}
  export PHASES=$phases,summary SKIP_SETUP=${SKIP_SETUP-1} BAKERH_STAMP=$stamp
  local check=("stage $name   scenes: $SCENES   variant: ${VARIANT:-baseline}   method: ${METHOD:-gw_beags_dw$DW}"
               "phases: $PHASES" "" "check when it is done:" "  $LOG_ROOT/$stamp/REPORT.txt (status, time, peak GPU memory, numbers)" "$@")
  mkdir -p "$LOG_ROOT/$stamp"
  printf '%s\n' "${check[@]}" | tee "$LOG_ROOT/$stamp/CHECK.txt"
  echo
  bash "$STAGES_DIR/../run_all.sh" "$FG"
}
FG=; [ "${1:-}" != --fg ] || FG=--fg

# run_trajs NAME PHASES CHECK_LINE...: run_stage once per novel camera path in TRAJS (default "object vidsplat travel"),
# AF_VARIANT=vanilla_<t>, one after another in one detached chain (scene by scene inside each). Stop the whole chain:
# kill -- -$(cat output/bakerh_logs/latest/pgid). Each path gets its own <stamp>_stage<NAME>_<t>/REPORT.txt.
TRAJS=${TRAJS:-"object vidsplat travel"}
run_trajs() {
  local name=$1 phases=$2 t stamp chain; shift 2
  local latest=$LOG_ROOT/latest
  if [ -z "${BAKERH_CHAIN:-}" ]; then
    if [ -f "$latest/pgid" ] && kill -0 -- "-$(cat "$latest/pgid")" 2>/dev/null; then
      echo "a run is already going (log: $latest/all.log). Stop it first: kill -- -$(cat "$latest/pgid")"; exit 1
    fi
    chain=$LOG_ROOT/$(date +%Y%m%d_%H%M%S)_stage${name}_chain.log
    mkdir -p "$LOG_ROOT"
    if [ "$FG" != --fg ]; then
      launcher=(nohup); command -v setsid > /dev/null && launcher=(setsid nohup)
      BAKERH_CHAIN=1 "${launcher[@]}" bash "$0" > "$chain" 2>&1 < /dev/null &
      sleep 3
      echo "Started in the background: paths '$TRAJS' on '$SCENES', one after another."
      echo "  chain:   tail -f $chain"
      echo "  current: tail -f $latest/all.log"
      echo "  stop:    kill -- -\$(cat $latest/pgid)"
      exit 0
    fi
  fi
  for t in $TRAJS; do
    export AF_VARIANT=vanilla_$t AF_DIR=vanilla_$t${VARIANT:+_$VARIANT} PHASES=$phases,summary SKIP_SETUP=1
    stamp=$(date +%Y%m%d_%H%M%S)_stage${name}_$t${VARIANT:+_$VARIANT}
    mkdir -p "$LOG_ROOT/$stamp"
    printf '%s\n' "stage $name   path: $t ($AF_VARIANT)   scenes: $SCENES   method: ${METHOD:-gw_beags_dw$DW}" \
      "phases: $PHASES   ArtiFixer-GS stages: ${STAGES:-all}" "" "check when it is done:" \
      "  $LOG_ROOT/$stamp/REPORT.txt" "$@" | sed "s/<AF_DIR>/$AF_DIR/g" | tee "$LOG_ROOT/$stamp/CHECK.txt"
    echo "[$(date '+%F %T')] path $t: $LOG_ROOT/$stamp"
    BAKERH_CHILD=1 BAKERH_STAMP=$stamp bash "$STAGES_DIR/../run_all.sh" >> "$LOG_ROOT/$stamp/all.log" 2>&1
    echo "[$(date '+%F %T')] path $t done (exit $?): $LOG_ROOT/$stamp/REPORT.txt"
  done
}
