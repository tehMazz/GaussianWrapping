# BEA-GS on Gaussian Wrapping

Project state, results, open problems and next steps: `HANDOFF.md`.

BEA-GS (SAG + labelled-voxel occupancy; BEA-GS_FSplat `BEAGS.md`) on the Gaussian Wrapping (GW) backbone, plus a
hollowing stage (no geometry where none is needed), robustness to non-clean data, a multi-label mesh with per-object
holes, training-free text-prompt masks (Seed2GS) and ArtiFixer-GS on the exported PLY. Every script runs from the repo
root as `python gaussian_wrapping/<script>.py`. With no new flag, `train.py` trains exactly as GW.

## Setup (Leonardo)
```bash
module load profile/deeplrn cuda/12.2 gcc/12.2.0           # gcc 12 libstdc++ (simple_knn, fused_ssim)
PY=/leonardo_scratch/fast/IscrC_EditGS/envs/gaussian_wrapping/bin/python
uv pip install --python $PY --no-build-isolation -e submodules/diff-gaussian-rasterization_ours_beags   # done
uv pip install --python $PY -r requirements_beags.txt      # pytest is not in the env yet; gw_smoke.sh uses PYTEST_LIB
git submodule update --init gaussian_wrapping/eval/TNTUniScanEvals   # mesh extraction imports it (done)
```
LPIPS weights (`gaussian_wrapping/lpipsPyTorch/weights`, used by the training report, `--lpips_images` and
`metrics.py`) are links to the original GW checkout `/leonardo_work/IscrC_EditGS/repos/GaussianWrapping`.
`.gitignore` ignores `*.sh` and `*.md` but re-includes `slurm/*.sh`, `scripts/bakerh/*.sh`, `GWBEAGS.md`, `HANDOFF.md`,
`CLAUDE.md` and `logs/.gitkeep` (the SLURM scripts write `logs/%x_%j.out`, so `logs/` must exist before `sbatch`).

## Pipeline
```bash
python gaussian_wrapping/train.py -s <scene> -m output/gw_beags/<name>/train --eval --rasterizer ours \
    --no-exposure_compensation --appearance bilagrid --hollow               # 30k, writes gs_stats.json
python gaussian_wrapping/export_3dgs_ply.py -m <train> --iteration 30000 [--check --gs_python $GS_RENDER_PY]
    # -> output/gw_beags/<name>/export; --check: native vs gsplat (gsplat_check.py in the gsplat env) -> gsplat_check.json
python gaussian_wrapping/beags_prepare.py  -m <train> --iteration 30000 --masks <sam2 masks>   # labels.pt (-1 = never hit)
python gaussian_wrapping/beags_finetune.py -m <train> --iteration 30000 --out output/gw_beags/<name>/beags [--hollow]
    # --occ_source mesh --mesh_dir <dir with tets.npz>: L_occ / L_piv against the multi-label mesh of the prepared
    # model (gw_mesh.sh MESH_OUT=output/gw_beags/<name>/mesh_train) instead of the labelled-point voxels;
    # --mesh_max_nodes (2e8 per label) / --mesh_cell / --mesh_band (2 cells) set the distance grids (beags/mesh_field.py)
python gaussian_wrapping/beags_eval.py     -m <beags> --iteration 33000 --gt_masks <gt> --masks <sam2 masks> [--render]
python gaussian_wrapping/export_3dgs_ply.py -m <beags> --iteration 33000 --split_labels \
    --out output/gw_beags/<name>/export_beags/point_cloud.ply                # + label_<k>.ply (Gaussians of label k)
python gaussian_wrapping/beags_render_labels.py -m <beags> --iteration 33000 [--n_views 8]
    # -> output/gw_beags/<name>/renders_labels_<it>/{full,label_<k>,grid}/<view>.png + grid_all.jpg
    #    (white bg; grid row = GT | full | label_0 | label_1 ...)
python gaussian_wrapping/hollow_metrics.py -m <beags> --iteration 33000 [--object_mask <npy> --masks <dir>] [--floater_voxel <abs>]
    # keys hole_alpha / hole_alpha_8x8 / hole_alpha_unseen (were hole_emptiness*; old JSONs are still read)
python gaussian_wrapping/pivot_based_mesh_extraction.py -m <beags> --iteration 33000 --sdf_mode ours --rasterizer ours
python gaussian_wrapping/multilabel_mesh_extraction.py -m <beags> --iteration 33000 --remove <k> --out_dir output/gw_beags/<name>/mesh
python gaussian_wrapping/seed2gs.py -m <train> --iteration 30000 --prompt "<text>" [--prompt ...]   # masks instead of SAM2
python gaussian_wrapping/seed2gs_eval.py --seed output/seed2gs/<scene>/<slug> --gt_masks <gt> [--labels <labels.pt>]
python gaussian_wrapping/beags/confidence_map.py -m <model> --iteration N --out <dir>   # models trained with --confidence
python gaussian_wrapping/beags/unseen_map.py -m <beags> --iteration N --remove <k> --out <dir> [--kappa 1] [--vis]   # GAVIS unseen maps
python gaussian_wrapping/beags_artifixer.py -m <beags> --label <k> --variant normal --masks <masks> --hole_mode mesh [--confidence_maps auto|on|off] \
    [--unseen_maps auto|on|off] [--kappa 1] [--fallback_masks <SAM2 masks>] [--renderer gw|gsplat]   # removal needs a mask per photo (consensus: training views only)
    # --renderer gw (default): ArtiFixer-GS renders every PLY with GW's render_ours (RENDER_CMD = render_ply_frames.py);
    # ArtiFixer only does the diffusion. CENTER_PRINCIPAL_POINT=1 (GW ignores COLMAP cx, cy)
    # auto: <out>/confidence/<stem>.png (+ CONFIDENCE_DIR in env.sh) when the PLY has confidence and CUDA is up;
    # non-pinhole COLMAP cameras (fern, bed) -> IGNORE_DISTORTION=1: ArtiFixer-GS renders them as pinhole, as GW trained them
```
Outputs are flat: `output/<method>/<name>/<mode>`, method in {gw, gw_hollow, gw_beags}, mode in {train, export, beags,
export_beags, renders_labels_<it>, mesh}; `mesh/` holds `mesh_labels.ply`, `label_<k>.ply` (every label) and, with
`--remove k`, `hole_<k>.ply` (label-k boundary + the k<->0 contact patch dilated by h) + `hole_<k>_closure.ply` (the
patch) + `hole_<k>.json` (h, face counts) [+ `tets.npz` with `--save_tets`, to redo holes on CPU with `--from_tets`]; `output/seed2gs/<scene>/<prompt>`, `output/artifixer/<scene>/<variant>`.

## SLURM (Leonardo, account IscrC_ESGS, from the repo root)
| script | what |
|---|---|
| `slurm/gw_smoke.sh` | debug QoS, 30 min: every pytest file (GPU included) + LLFF fern -r 8: 300 its GW defaults, 2000 its all flags, export, prepare, 300-it finetune, eval, hollow metrics -> `output/smoke/fern` |
| `slurm/gw_scene.sh <name> <scene> <masks> <gt> [train args]` | end to end: train 30k -> export -> BEA-GS -> per-label export + renders -> hollow metrics -> mesh + hole of `$REMOVE` -> render + metrics |
| `slurm/launch_one_per_dataset.sh` | `gw_scene.sh` on one scene per dataset: mip360 room (`-i images_4`), 3D-OVS bed, LERF figurines, LLFF fern |
| `slurm/gw_train.sh <name> <scene> [args]` | train + test renders + metrics (`output/$METHOD/<name>/train`) |
| `slurm/gw_beags.sh <name> <masks> <gt> [ft args]` | prepare, eval, finetune (`--out .../beags`), eval, metrics |
| `slurm/gw_hollow_metrics.sh <model> <it> [--label k] [args]` | `hollow_metrics.py` (object mask = `labels.pt == k`) |
| `slurm/gw_mesh.sh <model> <it> [--remove k ...]` | GW pivot mesh, then multi-label PAM into `output/<method>/<name>/mesh` |
| `slurm/gw_seed2gs.sh <model> <it> "<a\|b>" [args]` | Seed2GS labels + masks (several prompts share one SAM3 load); `GT=<dir>` adds `seed2gs_eval.py` |
| `slurm/gw_tests.sh [pytest args]` | debug QoS: every pytest file on a GPU (GW + ArtiFixer-GS), no fern pipeline, nothing deleted |
| `slurm/gw_smoke_features.sh [steps]` | debug QoS: the 2026-10-29 additions on the fern smoke models (grid flags, gsplat check, confidence maps, contact-patch mesh, Seed2GS + eval, ArtiFixer prepare/removal/split) |
| `slurm/gw_smoke_meshocc.sh [steps]` | debug QoS: fern train mesh -> `mesh_train`, 300-it finetune `--occ_source mesh` -> `beags_meshocc`, eval + label renders (`renders_labels_meshocc_2300`) |
| `slurm/gw_gavis_smoke.sh [<afx dir>]` | debug QoS: GPU tests (`TESTS=0` skips), unseen maps, alpha / mesh / gavis splits into `removal_split_<mode>`, `hole_grid.jpg`. Needs prepare + removal done (HANDOFF 5d) |
| `slurm/gw_gsplat_check.sh [model] [it]` | debug QoS: export + gsplat check of one model |
| `slurm/gw_photoft.sh [name] [ft args]` | debug QoS: photo-only finetune of the 30k model (HANDOFF 7.1 experiment 1) + render + metrics |
| `slurm/launch_ablations.sh [arms appearance hollow piv]` | prints the ablation submissions (dry run; `SUBMIT=1` submits): gw / gw_beags arms, appearance, H1-H5 + `gw_nofinal`, L_piv |
| `slurm/gw_artifixer.sh <name> <k> <variant> [masks]` | ArtiFixer-GS; ArtiFixer3D = GW `train.py` (`AF3D_TRAIN_ARGS`: recipe + bilagrid + hollow) |

Knobs in `slurm/common.sh`: `METHOD` (gw_beags), `ITER` / `FT_ITER`, `GW_TRAIN_ARGS` (GW's "ours" recipe:
`--rasterizer ours --feature_dc_lr 0.0013 --feature_rest_lr 0.00011 --data_device cpu --N_max_gaussians 6000000`),
`ROBUST_ARGS` (`--no-exposure_compensation --appearance bilagrid`), `HOLLOW_ARGS` (`--hollow`), `GW_EXTRACT_ARGS`,
`GS_RENDER_PY` (gsplat env, for `--check`), `MASK_SOURCE` (`provided` | `seed2gs`, with `SEED2GS_PROMPT`, `SEED2GS_ARGS`
(comma form for runner flags: `--sam3_args=--no_fast_init,--no_prefetch`), `SEED2GS_METHOD` (`${METHOD}_seed2gs`)).
With `MASK_SOURCE=seed2gs`, `gw_scene.sh` reuses the provided run's train model through a symlinked view in
`output/$SEED2GS_METHOD/<name>/train`; every prepare stamps `beags/mask_source.txt`, and prepare (then finetune, split
export, pivot mesh) reruns when the masks dir changes. `gw_mesh.sh`: `SAVE_TETS` (1), `MESH_OUT` (default
`output/<method>/<name>/mesh`). `gw_scene.sh`: `OCC_SOURCE` (`points` | `mesh`: mesh of the prepared model into
`mesh_train` before the finetune, rebuilt when prepare reruns; shares the train model through a linked `train/`, all else
in `output/$OCC_METHOD/<name>`, default `${METHOD}_meshocc`). `gw_artifixer.sh`: `FALLBACK_MASKS`, `CONFIDENCE_MAPS`
(auto), `CONF_THRESH` (unset = off: hole also takes pixels of confidence < thresh inside the dilated mask).
Plain GW arm: `METHOD=gw HOLLOW_ARGS= ROBUST_ARGS=--no-exposure_compensation`.

## BakerH on the VM
`bash scripts/bakerh/run_all.sh` (detaches; `--fg` stays in the terminal): `setup.sh` (venv `.venv`, Python 3.9,
torch 2.3.1+cu121, the CUDA extensions from `submodules/`, LPIPS weights, mesh deps), then per scene `run_scene.sh`:
SAM3 masks -> GW 30k (bilateral grid + hollowing) -> export -> prepare -> finetune (dilation weight 0.1) -> label
renders -> showcase (`output/gw_beags_dw0.1/showcase/<scene>_grid_all.jpg`: photo | full | label_0 | label_1) -> test
metrics -> hollow metrics; `FULL=1` adds mesh + hole -> ArtiFixer inputs -> ArtiFixer. Handoff for the VM: `HANDOFF_BAKERH.md`. Send back `output/bakerh_logs/latest/REPORT.txt`. Outputs: `output/gw_beags_dw0.1/<scene>/
{masks,train,export,beags,mesh}`, `output/artifixer/<scene>/normal`. Knobs in `scripts/bakerh/common.sh`.

## train.py flags (all off by default)
| flag | default | effect |
|---|---|---|
| `--appearance` | unset | `none`, `exposure`, `network`, `bilagrid`, `bilagrid+decoupled`. Unset: GW's `--exposure_compensation` (default on) / `--decoupled_appearance` decide; set: they must agree (`--appearance bilagrid` needs exposure off, which is implied unless `--exposure_compensation` is passed) |
| `bilagrid` | | per-view 16x16x8 bilateral grid on the training render before the loss (gsplat), TV weight `--bilagrid_tv_weight` 10, lr 2e-3 with warmup + decay, own Adam; `bilagrid<it>.pth` at every saved iteration |
| `--bilagrid_identity_weight`, `--bilagrid_mean_weight` | 0, 0 | pull each view's grid toward identity / the mean grid (mean squared; drift ~1e-3, so weights ~1-100) |
| `--bilagrid_until` | -1 | last iteration with the grid; after it the losses use the raw render and the grid stops |
| `--lambda_raw` | 0 | extra photometric loss on the raw render while a compensated render drives the loss |
| `--no_hollow_finalize` | off | skip the final hollow prune at the last iteration (no recovery iterations follow it) |
| `bilagrid+decoupled` | | L1 on the graded render, SSIM = luminance(graded) x contrast-structure(raw) (CoMe) |
| `--confidence` | off | per-Gaussian confidence c from `--conf_from_iter` 500: `c L - 0.075 log c`, densify thresholds / clamp(c, max 1) (CoMe) |
| `--lambda_color_var`, `--lambda_normal_var` | 0, 0 | CoMe variance losses from `--var_from_iter` 15000 (CoMe: 0.5, 0.005); 2 extra renders |
| `--ignore_masks <dir>` | | `<image_name>.png/.npy`, nonzero pixels left out of the photometric loss |
| `--hollow` | off | (`--opacity_decay`, CoMe's decay, is a separate knob, default 0 = off) `hollow.schedule.HollowSchedule` from `--hollow_from` 20000: free-space prior (`--lambda_fs` 1e-3, ray-carved grid every 5000), opacity entropy from 25000, unseen prune every 2000 (`--seen_tau` 1 px), see-through carving every 5000, final prune (unseen, see-through, opacity < 0.05) before the last save |
| `--labels <labels.pt>` | | per-Gaussian labels, carried through densification and pruning (needs a `--start_checkpoint` with the same count) |
| `--lpips_images <file>`, `--lambda_lpips` | , 0.1 | ArtiFixer3D distillation: listed frames get 0.1 LPIPS(vgg) only (no grid, no confidence, no variance), photos keep the normal loss |

Every run writes `<model>/gs_stats.json`: Gaussians, filtered-opacity shares < 0.05 / < 0.1, median, mean opacity
entropy, train seconds, LPIPS-only frames, appearance / confidence / hollow.

## Code
| where | what |
|---|---|
| `submodules/diff-gaussian-rasterization_ours_beags` | GW's rasterizer + labels: `sag` image, `votes` [P, L]; SAG alpha gradient only for mismatched Gaussians. No labels = stock |
| `gaussian_renderer/ours.py`, `beags_api.py` | `render_ours(..., mask, gaussian_labels, pixel_labels, num_labels)` (+ `alpha`, `sag`, `votes`); `render_labels`, `render_depth_alpha`, `contribution` |
| `scene/gaussian_model.py` | per-Gaussian extras (`register_extra`, `labels`) and confidence, kept through prune / densify / checkpoints; `labels.pt` next to the PLY |
| `export_3dgs_ply.py` | standard 3DGS PLY with the 3D filter baked; `--check` gsplat vs native |
| `robust/` | bilateral grid, photometric loss (decoupled / confidence / ignore masks), variance losses |
| `hollow/` | visibility ledger, free-space prior + free grid, opacity entropy / decay, see-through carving, schedule; `hollow_metrics.py` |
| `beags/`, `beags_{prepare,finetune,eval}.py` | BEA-GS port: filtered scale / opacity, chunked occupancy, pivot term, label -1 = never hit (pruned) |
| `beags/mesh_field.py` | `--occ_source mesh`: label tets of `tets.npz` rasterized on the GPU into inside masks, EDT distance grids cached as `dist_<k>.npz`, trilinear lookup; same `contains` / `distance` / `size` interface as `LabeledVoxels` |
| `multilabel_mesh_extraction.py`, `utils/multilabel_meshing.py` | multi-label PAM: one Delaunay, per-label gated occupancy, faces at label changes, `label_<k>.ply`, `hole_<k>.ply`; crop `--bounding_box_method gaussians` (default here), mask carving opt-in with `--label_masks` |
| `beags_render_labels.py` | per-label renders on `--n_views` views: full, each label alone, GT/full/labels grids, `grid_all.jpg` |
| `seed2gs.py`, `seg/` | SAM3 text prompt -> seed -> orbits -> SAM3 tracking -> FlashSplat votes -> `labels.pt` + masks |
| `beags_artifixer.py`, `submodules/artifixer_gs` | ArtiFixer-GS inputs (`env.sh`), `--eps2d 0`, alpha / mesh / gavis holes (`--hole_mode` default dilate; gavis = GAVIS unseen maps, HANDOFF 5d; `gw_artifixer.sh` uses mesh if `hole_<k>.ply` exists, else alpha; `run.sh` `DROP_POINTS=1`) |
| `tests/` (in `gaussian_wrapping/`) | rasterizer, extras, robust, hollow, BEA-GS utils, mesh, Seed2GS, ArtiFixer inputs; GPU tests skip without CUDA |
