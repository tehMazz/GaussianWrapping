#!/bin/bash
# Install the full-bracket median-depth fix (v2, 2026-11-26; GLOSSARY "full-bracket depth") into the rasterizers GW imports:
#   diff_gaussian_rasterization_ours        -> env site-packages (stock GW renderer: integrate / sample_depth, meshing)
#   diff_gaussian_rasterization_ours_beags  -> editable install, .so inside submodules/diff-gaussian-rasterization_ours_beags
# v2 = the stock search first (bit-identical to stock wherever it finds the T = 0.5 crossing), then a second search over
# [0, t_up] only for the pixels / points it missed, with extra splits so the final bin is no wider than stock.
# The sources in submodules/ carry the patch (config.h, render_forward.cu, sample_forward.cu); the .so files come from
# the builds in $F (slurm/full_bracket_v2_test.sh builds and checks them from exactly those sources).
# Each .so is swapped by an atomic rename, so jobs that already imported the old one keep it. Backups next to it:
#   <so>.stock      the original stock build, written once and never overwritten
#   <so>.prev-<date-time>  whatever was installed before a later reinstall
# Rollback to stock:  cp -p <dir>/<so>.stock <dir>/.<so>.new && mv -f <dir>/.<so>.new <dir>/<so>
#   bash slurm/install_full_bracket.sh          (login node; no GPU needed; then sbatch slurm/verify_full_bracket.sh)
set -euo pipefail
F=${F:-/leonardo_work/IscrC_EditGS/tmp_amazzucc/fb2}
SP=/leonardo_scratch/fast/IscrC_EditGS/envs/gaussian_wrapping/lib/python3.9/site-packages
SO=_C.cpython-39-x86_64-linux-gnu.so
install_so() {  # install_so <src .so> <dst dir>
  local src=$1 dst=$2
  [ -f "$src" ] || { echo "missing $src (build it with slurm/full_bracket_v2_test.sh)" >&2; exit 1; }
  if cmp -s "$src" "$dst/$SO"; then echo "[full-bracket] $dst already has this build"; return; fi
  if [ ! -e "$dst/$SO.stock" ]; then
    cp -p "$dst/$SO" "$dst/$SO.stock"
  else
    cp -p "$dst/$SO" "$dst/$SO.prev-$(date +%Y%m%d-%H%M%S)"
  fi
  cp "$src" "$dst/.$SO.new"
  mv -f "$dst/.$SO.new" "$dst/$SO"
  echo "[full-bracket] installed $src -> $dst/$SO"
}
install_so "$F/ours_fb2/diff_gaussian_rasterization_ours/$SO" "$SP/diff_gaussian_rasterization_ours"
install_so "$F/beags_fb2/diff_gaussian_rasterization_ours_beags/$SO" \
  "submodules/diff-gaussian-rasterization_ours_beags/diff_gaussian_rasterization_ours_beags"
echo "[full-bracket] done $(date +%T)"
