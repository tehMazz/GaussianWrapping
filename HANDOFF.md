# Handoff: BEA-GS on Gaussian Wrapping (BEA-GS_GWrap)

State on 2026-10-29. This repo replaces BEA-GS_FSplat (FastGS backbone) as the place where coding continues.
`GWBEAGS.md` is the user guide (setup, pipeline, scripts, `train.py` flags). This file holds the project state: why the
repo exists, what was built, what was measured, what is broken or still open, and what to do next.
Every claim below was checked against the code, logs and outputs on 2026-10-29.

## 1. Goal
The best editable 3DGS representation for object removal and inpainting:
1. **No geometry where none is needed. This is the focus.** After an object is removed, the unseen space behind it must
   be empty, so the region to inpaint is obvious.
2. Given masks, clean per-Gaussian labels and clean object boundaries (BEA-GS).
3. Holes and artifacts that ArtiFixer-GS can see and fix.
4. Optionally, masks computed from a text prompt without training and without cameras (Seed2GS), instead of the provided
   SAM2 masks.

AuraFusion360 is dropped. BEA-GS_FSplat stays as the FastGS baseline (`../BEA-GS_FSplat`, see `BEAGS.md` there).

## 2. Decisions (taken with the user, 2026-10-27)
| decision | why |
|---|---|
| **Backbone: Gaussian Wrapping (GW), not CoMe** | GW composites colour exactly like 3DGS (EWA alpha x T). A baked PLY therefore renders in gsplat/ArtiFixer almost unchanged, and the BEA-GS votes/SAG CUDA patch ports directly. CoMe ray-evaluates kernels with a per-pixel sort (StopThePop), which gsplat cannot reproduce, so ArtiFixer's alpha would be wrong. Quality is about equal: on the same 7 Mip-NeRF 360 scenes (images auto-downscaled to 1600 px) GW 28.31 vs CoMe 28.08 PSNR; on kitchen CoMe is higher (29.56 vs 28.94) |
| Grafted from CoMe | per-Gaussian confidence, SSIM-decoupled appearance, colour/normal variance losses, opacity decay |
| **Unseen space = empty (global free-space prior)** | Reimplemented from scratch after "Optimize the Unseen" (Segre & Avidan, arXiv 2412.12772). wenlei's NeRF unseen-space prior was never found: STDLoc, SplatHLoc, ULF-Loc and RAP are clean upstream clones. The user approved the rewrite |
| **Masks from text: SAM3** | Seed2GS (arXiv 2608.11928, no public code), reimplemented with SAM3 + FlashSplat closed-form votes |
| **New repo forked from GW** | `origin` = `https://github.com/tehMazz/GaussianWrapping` (the user's fork of diego1401/GaussianWrapping), branch `beags` |
| Bilateral grid + robustness | For non-clean data (exposure changes, vignetting, transients), so they are not fitted with floaters or fog. **Note:** the grid as configured is behind most of the kitchen PSNR drop (section 7.1) |
| BEA-GS occupancy + mesh | Compatible, with changes: filtered scale/opacity, chunked occupancy, a pivot term, labels carried through densification, per-label gating at extraction (multi-label PAM) |

## 3. Rules the user set (apply in every session)
- **GPU budget:**
  - test on ONE scene per dataset (one Mip-NeRF 360, one 3D-OVS, one LERF, one LLFF);
  - smoke-test with short runs first;
  - ask before launching anything bigger;
  - use the debug QoS (`boost_qos_dbg`, 30 min) for smoke tests.
- **Flat outputs:** `output/<method>/<scene>/<mode>`, never nested inside another model dir. Some exceptions were
  inherited from BEA-GS_FSplat (section 5, "Output layout").
- **Show results as renders:** at least 5 views, with label 0 and label 1 each rendered alone (`beags_render_labels.py`),
  not only PLYs.
- **VM scripts** (BakerH VM, not Leonardo) copy the REPORT.txt pattern of ArtiFixer's `scripts/bakerh/run_all.sh`: a
  detached run, per-phase logs, and a REPORT.txt the user pastes back. VM paths are in
  `/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/scripts/bakerh/`.
- **Git:** commit or push only when asked. Nothing is committed yet (section 9).

## 4. Environment (Leonardo)
- **Repo and layout:**
  - repo: `/leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/BEA-GS_GWrap`;
  - Python code is in `gaussian_wrapping/`;
  - run everything from the repo root as `python gaussian_wrapping/<script>.py`.
- **Env:** `/leonardo_scratch/fast/IscrC_EditGS/envs/gaussian_wrapping` (uv venv, Python 3.9.25, torch 2.3.1+cu121).
  - Run `module load profile/deeplrn cuda/12.2 gcc/12.2.0` first, even on the login node. Without gcc 12 the CUDA
    extensions fail to import (`GLIBCXX_3.4.29 not found`). `cuda/12.2` sets `CUDA_HOME`, which rebuilds need.
  - `slurm/common.sh` does all of this. It also adds `$ENV/bin` to `PATH` (ninja, for nvdiffrast's JIT build), sets
    `PYTHONPATH=` and sets `TORCH_CUDA_ARCH_LIST=8.0`.
- **GPU:** the login node has none. Every GPU test and pipeline step goes through SLURM (account `IscrC_ESGS`,
  partition `boost_usr_prod`). Do not use the legacy upstream `gaussian_wrapping/slurm/*`.
- **Rasterizer extension:** `submodules/diff-gaussian-rasterization_ours_beags` is installed editable. To rebuild:
  `uv pip install --python $PY --no-build-isolation -e submodules/diff-gaussian-rasterization_ours_beags`.
  `gaussian_renderer/ours.py` tries to import it first, so plain GW training also goes through it (with labels off it
  matches stock).
- **pytest:** not in the env. Tests use `PYTEST_LIB=/leonardo_work/IscrC_EditGS/tmp_amazzucc/pylib_gw` on
  `PYTHONPATH` (see `slurm/gw_smoke.sh`).
- **LPIPS weights:** `gaussian_wrapping/lpipsPyTorch/weights/*.pth` are symlinks into the original GW checkout
  `/leonardo_work/IscrC_EditGS/repos/GaussianWrapping`. Do not delete that checkout. It also holds reference runs in
  `output/mip360/<scene>`, trained on `images` auto-downscaled to 1600 px width.
- **SAM3 (Seed2GS)** runs in another user's env, `/leonardo_work/IscrC_EditGS/gabri/venvs/tttlrm_env` (override with
  `SAM3_PYTHON`).
- **gsplat** is NOT importable in the GW env. The gsplat env is `/leonardo_scratch/fast/IscrC_EditGS/envs/gsplat`
  (gsplat 1.5.3; `GS_RENDER_PY`, exported by `slurm/common.sh`). `export_3dgs_ply.py --check` therefore runs in two
  processes: native renders are dumped, then `gaussian_wrapping/gsplat_check.py` renders the exported PLY in the gsplat
  env (section 5b).
- **Network and disk:**
  - compute nodes have no internet, so Seed2GS and ArtiFixer set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`;
  - `/leonardo_scratch/fast/IscrC_EditGS` is at its quota, so `gw_artifixer.sh` puts TMPDIR and caches in
    `/leonardo_work/IscrC_EditGS/tmp_amazzucc`.
- **Harmless warnings:** the pyg-lib / torch-sparse `GLIBC_2.29` warnings in the mesh step.
- **Data** (as in BEA-GS_FSplat `slurm/launch_benchmark.sh`, wired in `slurm/launch_one_per_dataset.sh`):
  - Mip-NeRF 360: `/leonardo_work/IscrC_EditGS/datasets/mip360/<scene>`, run with `-i images_4`;
  - masks: `/leonardo_work/IscrC_EditGS/gabri/datasets/<dataset>/masks/{training/sam2,gt_masks}/<scene>/masks`
    (Mip-NeRF 360 uses `mipnerf360_hd_masks/masks/...`);
  - 3D-OVS bed, LERF figurines and LLFF fern are under `../BEA-GS_FSplat/data/`.
- **Debug job scripts** used for the smoke stages: `/leonardo_work/IscrC_EditGS/tmp_amazzucc/gw_debug/`
  (`tail.sh`, `mesh.sh`, `renders.sh`, `debug_grads_depth.py`).

## 5. What was built (all uncommitted)
Pipeline, in order (usage in `GWBEAGS.md`, end to end in `slurm/gw_scene.sh`):

| step | code | notes |
|---|---|---|
| Train GW + robust + hollow | `train.py` (`--rasterizer ours --appearance bilagrid --hollow [--confidence ...]`) | Writes `gs_stats.json`. With no new flag it trains as GW, except the train/test split (`beags/splits.py`): images named `*test*` always go to test, and 3D-OVS uses fixed frames. The split is identical to GW's on Mip-NeRF 360 and LLFF |
| Votes / SAG in GW's rasterizer | `submodules/diff-gaussian-rasterization_ours_beags`, `gaussian_renderer/ours.py`, `beags_api.py` | SAG backward reaches only mismatched Gaussians. Labels off = stock |
| Robustness | `robust/`: bilateral grid, decoupled SSIM, confidence, variance losses, ignore masks | `--appearance bilagrid` replaces GW exposure compensation (the two are mutually exclusive) |
| Hollowing | `hollow/`, plus `hollow_metrics.py` | See the list below the table |
| Export | `export_3dgs_ply.py` | Bakes the 3D filter: scale' = log sqrt(s^2+f^2), opacity' = logit(sigmoid(o) sqrt(det1/det2)). `--split_labels` writes `label_<k>.ply` |
| BEA-GS | `beags/` (`common`, `consensus`, `labels`, `masks`, `metrics`, `occupancy`, `pivots`, `splits`), `beags_prepare.py`, `beags_finetune.py`, `beags_eval.py` | Consensus masks, label votes, SAG, labelled-voxel occupancy, pivot term. Label -1 = never hit (pruned). The finetune also prunes raw opacity < 0.05 every 100 its |
| Per-label renders | `beags_render_labels.py` | 8 views: `full/`, `label_<k>/`, `grid/` (GT, full, label_0, label_1 per row) plus a stacked `grid_all.jpg` |
| Multi-label mesh | `multilabel_mesh_extraction.py`, `utils/multilabel_meshing.py` | See the list below the table |
| Seed2GS masks | `seed2gs.py`, `seg/` (`sam3.py`, `sam3_runner.py`, `orbits.py`, `lift.py`, `votes.py`) | Text prompt -> seed lift -> flat + spiral orbits (`--n_views` 24 each) -> SAM3 tracking -> FlashSplat votes -> `labels.pt` + masks in `output/seed2gs/<scene>/<prompt>` |
| ArtiFixer-GS | `beags_artifixer.py`, `submodules/artifixer_gs` (local branch `gw`) | See the list below the table |
| Scripts | `slurm/*.sh`, `scripts/bakerh/` (VM: `run_all.sh`, `run_scene.sh`, `setup.sh`, `common.sh`, `masks_to_beags.py`) | Every knob is in `slurm/common.sh`. `sbatch` copies the job script at submission (section 10) |
| Tests | `gaussian_wrapping/tests/`: 9 files, 158 tests (2026-10-29) | GPU tests skip without CUDA |

Hollowing steps (`hollow/`):
- H1: visibility ledger.
- H2: free-space prior, plus a ray-carved free grid.
- H3: opacity entropy from `--entropy_from` 25000, and opacity decay. Decay is `--opacity_decay`, default 0 = **off**; it
  was on only in the fern smoke.
- H4: see-through carving.
- Schedule: losses from `--hollow_from` 20000. The unseen prune runs every 2000 its from 22k, carving every 5000 from 25k,
  and a final prune at 30k after the last optimizer step.

Multi-label mesh outputs:
- One Delaunay with per-label gated occupancy.
- `mesh_labels.ply`, plus `label_<k>.ply` for every label.
- With `--remove k`: `hole_<k>.ply` (the same faces as `label_<k>.ply`) and `hole_<k>_closure.ply` (the background faces
  that touched label k).
- `--bounding_box_method` defaults to `gaussians` here. GW's own `primal_adaptive_meshing_extraction.py` still defaults to
  `scene`, which comes out empty on forward-facing scenes.
- Per-label mask carving (`--label_masks`) is opt-in and was not used so far.

ArtiFixer-GS:
- `--eps2d 0` for GW.
- `scripts/run.sh` sets `DROP_POINTS=1` and `HOLE_MODE=dilate|alpha|mesh`.
- `beags_artifixer.py --hole_mode` defaults to `dilate`. `slurm/gw_artifixer.sh` picks `mesh` when `hole_<k>.ply`
  exists, else `alpha`.
- ArtiFixer3D distillation = GW `train.py --lpips_images`.

Changes to existing GW files:
- `train.py` and `scene/gaussian_model.py`: per-Gaussian extras, labels and confidence, carried through prune, densify and
  checkpoints.
- `gaussian_renderer/ours.py`: returns `alpha`, `sag` and `votes`.
- `scene/dataset_readers.py`:
  - `SIMPLE_RADIAL`, `RADIAL` and `OPENCV*` are read as pinhole and **their distortion coefficients are ignored**. Fern
    and bed have small k.
  - the split goes through `beags/splits.py`.
- `arguments/__init__.py`: `get_combined_args` keeps unset optional flags that are missing from `cfg_args`. Before this,
  5 scripts crashed with `Namespace has no attribute`.
- `primal_adaptive_meshing_extraction.py`: new `gaussians` crop mode; an empty crop now raises.
- `.gitignore`: re-includes `slurm/*.sh`, `scripts/bakerh/*.sh`, `GWBEAGS.md`, `HANDOFF.md`, `CLAUDE.md` and
  `logs/.gitkeep`. The rest of `logs/` is ignored. SLURM's `--output=logs/...` needs the directory to exist.
- `.gitmodules`: adds `submodules/artifixer_gs`. This one is already staged.

**Output layout** (`output/<method>/<scene>/<mode>`, with `<mode>` in `train`, `export`, `beags`, `export_beags`,
`renders_labels_<it>`, `mesh`). Exceptions inherited from BEA-GS_FSplat, which stay inside model dirs:
- the BEA-GS sidecar `<model>/beags/`: `labeled_points.npz`, consensus `masks/`, `metrics_<it>.json`, and
  `renders_<it>/{extracted_k,removal_k}` from `beags_eval --render`;
- `labels.pt` next to each PLY;
- `<model>/mesh_ours_2pivots.ply`, `hollow_metrics_<it>.json` and `object_mask_<k>_<it>.npy`;
- ArtiFixer3D trains into `output/artifixer/<scene>/<variant>/<scene>/af3d/fastgs` (named fastgs, but it is GW).

To re-prepare with other masks, delete `train/beags/labeled_points.npz`, because `gw_scene.sh` skips prepare when it
exists.

## 5b. Added on 2026-10-29 (smoke-tested on fern; gsplat check also on kitchen)
Smoke: `sbatch slurm/gw_tests.sh` (GPU pytest) and `sbatch slurm/gw_smoke_features.sh` (the features on the fern smoke
models), both debug QoS, neither deletes `output/smoke/fern`. Results (jobs 62723512, 62723514, 2026-10-29):
- **Tests on GPU:** 157 passed, 1 skipped (needs gsplat + CUDA in one env); ArtiFixer-GS 53 passed + the nvdiffrast mesh test.
- **grid** (2000 its, identity/mean 10, until 1500, lambda_raw 0.2, + confidence): ran, `bilagrid{1000,2000}.pth` saved,
  identity_rms 0.0034, view_spread_rms 0.0029, in-loop test PSNR 25.95 / 27.21 (`output/smoke/fern_features/grid`).
- **gsplat check FAILS** (open problem 7.5): drop vs photos 1.03 dB on fern, 1.63 dB on kitchen 30k, 1.39 dB on kitchen
  33k (target <= 0.5), with alpha MAE <= 0.00016 everywhere.
- **conf:** 20 maps, mean confidence 0.96, no pixel < 0.5 (2000-it model; the maps carry little signal yet).
- **mesh:** contact h = 0.404 (voxel 0.094 x 4.3): hole = 840k object faces + 39k patch faces (old closure: 159 faces).
  The previous fern mesh is kept in `output/smoke/fern/mesh_prev`.
- **seed2gs:** 130 s total (was 175): SAM3 import 81 s + build 11 s (overlapped with the scene load; 88 s waited),
  detect 32 s, tracking 4 s. IoU 0.710 / Gaussian IoU 0.832 as in 6. Still > 1 min: the SAM3 import from gabri's env.
- **artifixer** (prepare + removal + split, no diffusion): **passes** after two fixes (jobs 62728410, 62729701):
  - fern's `SIMPLE_RADIAL` camera (k1 0.0198) was refused by ArtiFixer-GS. `IGNORE_DISTORTION=1` (set by
    `beags_artifixer.py` for non-pinhole COLMAP cameras, as GW drops the distortion when training) now makes ArtiFixer-GS
    render and distil them as pinhole. Kitchen is PINHOLE.
  - `split` needs a mask for every frame, and the BEA-GS consensus masks cover training views only (fern 17 of 20,
    kitchen 244 of 279). `beags_artifixer.py --fallback_masks <SAM2 masks>` (`FALLBACK_MASKS` in `gw_artifixer.sh`)
    links the consensus mask where there is one, else the SAM2 one, into `<out>/masks`.
  - Result: 20/20 frames, mesh hole (879k faces rasterized), hole fraction 0.39; `CONF_THRESH=0.5` added nothing (no
    pixel below 0.5 on this short model). Outputs: `output/artifixer/fern/normal/fern/{removal,removal_split}`.

| item | code | what |
|---|---|---|
| Confidence artifact maps (7.4.5) | `beags/confidence_map.py`, `beags_artifixer.py --confidence_maps auto\|on\|off`, ArtiFixer-GS `removal.py split --confidence_dir --conf_threshold`, `run.sh` `CONF_THRESH` | One PNG per view (all cameras), `<out>/confidence/<stem>.png`, 255*min(exp(conf), 1), dark = low confidence; `CONFIDENCE_DIR` in `env.sh`. With `CONF_THRESH` set and `HOLE_MODE` alpha/mesh, the hole also takes low-confidence pixels inside the dilated mask. Needs a model trained with `--confidence` (off in `common.sh`; the fern smoke has it). Confidence survives `beags_finetune` (PLY property) |
| Contact-patch hole (7.4.5) | `utils/multilabel_meshing.py` (`contact_patch`, `removal_hole`), `multilabel_mesh_extraction.py --contact_h --contact_vicinity --contact_points --save_tets --from_tets` | `hole_<k>.ply` = label-k boundary + background faces within h of it (h = BEA-GS voxel edge x 4.30 by default; `--contact_h 0` = the old outputs). `hole_<k>_closure.ply` = the patch; `hole_<k>.json` = h + face counts/areas. `gw_mesh.sh` saves `tets.npz` (`SAVE_TETS=1`), so other h values rerun on the login node with `--from_tets` |
| gsplat check (7.4.1) | `export_3dgs_ply.py --check --gs_python --check_splits --max_drop --check_dir --keep_check_views`, `gsplat_check.py`, `slurm/gw_gsplat_check.sh` | Per view PSNR gsplat vs native, alpha MAE, PSNR of each vs the photo; `pass` = mean drop vs photo <= 0.5 dB. `gsplat_check.json` next to the export. A failed check is recorded in `export_info.json["check"]["error"]` and does not abort the export. `gw_scene.sh` now passes `--gs_python` |
| Seed2GS (7.4.2, 7.4.5) | `seed2gs.py` (repeatable `--prompt`, `--track_side` 1008, `--mask_split all`, `--sam3_args`), `seg/masks.py`, `seg/sam3*.py`, `seed2gs_eval.py`, `gw_scene.sh MASK_SOURCE=seed2gs` | SAM3 worker starts before torch is imported, checkpoint prefetch, no random init, several prompts per SAM3 load; per-op timings in `seed2gs.json`. Masks now also on test views. A prompt SAM3 does not find writes `seed2gs_failed.json`. `seed2gs_eval.json`: IoU/BIoU per GT view, mIoU (failed prompts = 0) and mIoU_found, Gaussian agreement with a provided `labels.pt`. The < 1 min target is unverified: ~110 s was SAM3 import + load |
| Grid drift fixes (7.1) | `robust/` (`bilagrid.py`, `losses.py`), `train.py`; `slurm/gw_photoft.sh` | Opt-in `--bilagrid_identity_weight`, `--bilagrid_mean_weight`, `--bilagrid_until`, `--lambda_raw`, `--bilagrid_tv_weight` (10). `bilagrid<it>.pth` at every save (7.1 item 4 done). `gs_stats.json["bilagrid"]` has identity_rms / view_spread_rms. Defaults unchanged |
| `--no_hollow_finalize` (7.1) | `hollow/schedule.py` | Skips the final prune at the last iteration; arm `gw_nofinal` in `launch_ablations.sh` |
| Hollow metrics (7.4.3, 7.4.7) | `hollow_metrics.py --floater_voxel` | Keys renamed to `hole_alpha`, `hole_alpha_8x8`, `hole_alpha_unseen`; old JSONs still read (`load_metrics`). Absolute world-aligned floater voxel, recorded in the JSON |
| Ablations (7.4.5) | `slurm/launch_ablations.sh [arms appearance hollow piv]` | Dry run prints the `sbatch` lines (with dependencies; existing outputs skipped); `SUBMIT=1` submits. H1 and H4 have no clean switch (flag combinations are used) |
| BakerH VM (7.4.5) | `scripts/bakerh/*` (+ new `report.py`) | Audited against the current CLIs: adds BEA-GS eval, per-label PLYs and renders, 30k hollow metrics, gsplat check, the ArtiFixer diffusion phase, `--train_args` / `--artifixer_root`, numbers in REPORT.txt, a working-tree check in `setup.sh` (nothing is committed, so rsync the tree). Dry-checked on the login node against the kitchen outputs; never run on the VM |

## 5c. ArtiFixer renders with GW's own rasterizer (2026-10-29)
ArtiFixer is used only for the diffusion: every render ArtiFixer-GS makes of a PLY (prepare, removal background / deleted
+ depth for the mesh hole, ArtiFixer3D+) now goes through GW's `render_ours` instead of gsplat.
- `submodules/artifixer_gs/artifixer_gs/render.py` `render_ply`: with `RENDER_CMD` set it runs
  `$RENDER_CMD --ply P --trajectory T --out_dir D [--depth]` in `RENDER_CWD` (ArtiFixer-GS on PYTHONPATH); same layout
  (renders on black, opacity, depth). `run.sh` adds it to the rerun keys of removal / plus (and prepare outside removal).
- `gaussian_wrapping/render_ply_frames.py`: loads the export_3dgs_ply.py PLY (no filter_3D: the baked, i.e. identical,
  model) into `GaussianModel` and renders `render_ours`. Depth = GW median z-depth, or the expected depth (view z as
  colour / alpha) where GW has no median (alpha in [0.5, 0.55) or no T = 0.5 crossing within SAMPLE_RANGE), 0 where
  alpha < 0.5. Empty PLY -> background frames.
- `CENTER_PRINCIPAL_POINT=1` (`artifixer_gs.render.trajectory_cameras`): principal point at the image centre for the
  renders and `hole_mesh`, as GW trains (it ignores COLMAP cx, cy). Kitchen and fern have cx = W/2 already.
- `beags_artifixer.py --renderer gw` (default) writes `RENDER_CMD`, `RENDER_CWD` (`--kernel_size 0.0`: train.py renders
  with 0) and always `CENTER_PRINCIPAL_POINT=1`; `--renderer gsplat` keeps ArtiFixer-GS's gsplat.
- `export_3dgs_ply.py --check` also renders the exported PLY with `render_ours` -> `ours_ply_check.json` (runs without
  gsplat). Kitchen 33k, 56 views: **115.6 dB vs native, alpha MAE 0** (gsplat: 37.1 dB).
- Fern (job 62733305): prepare + removal + split pass with GW renders at 4032x3024, 20/20 frames, hole fraction 0.375.
- Tests on GPU after 5c (jobs 62735838, 62737127): 168 passed, 1 skipped (GW, incl. native vs PLY -> transforms.json ->
  `render_ply_frames` within 1/255); ArtiFixer-GS 54 passed + the nvdiffrast mesh test. Fern rerun with the depth
  fallback + centred principal point (job 62735840): 20/20 frames, hole fraction 0.384.
- BakerH: `run_scene.sh` rewrites an `env.sh` without `RENDER_CMD`; gsplat is required only without it or for TRAJ_MODE;
  `setup.sh` checks ArtiFixer-GS has `RENDER_CMD` / `CENTER_PRINCIPAL_POINT`.
- Reviewed by two independent agents; their findings (depth zeros, principal point, stale BakerH env.sh, gsplat
  requirement, kernel_size, prepare rerun) are fixed.

## 5d. Mesh occupancy supervision (2026-11-02)
`beags_finetune.py --occ_source mesh --mesh_dir <dir>`: L_occ / L_piv use the model's own multi-label mesh instead of
the BEA-GS labelled-point voxels (`--occ_source points`, the default, is unchanged).
- Source: `tets.npz` of `multilabel_mesh_extraction.py --save_tets` on the prepared (pre-finetune) model, i.e. the
  labelled Delaunay whose label interfaces are `label_<k>.ply`. The PLYs themselves are not used: on fern they have ~20k
  open / non-manifold edges each, so their inside/outside is not well defined.
- `beags/mesh_field.py`: per label, its tets are rasterized on the GPU into an inside mask on its own grid (default
  cell (AABB volume / 2e8)^(1/3), `--mesh_max_nodes`; one shared cell would have made the kitchen object grid as coarse
  as its 18-unit background, 0.036), then a scipy EDT gives the distance to the region (0 inside), cached as
  `<mesh_dir>/dist_<k>.npz`. Lookups are trilinear `grid_sample`. `contains` = distance <= band (`--mesh_band`,
  default 2 cells of that label), pivot d = that distance (h = that band). Off the grid: label 0 = inside, objects = outside.
- Rejected: open3d signed distance to `label_<k>.ply` (0.02-0.1 M queries/s on the fern meshes, hours for a 512^3
  grid, and the sign needs watertight meshes); a per-iteration CPU query (~1-2 h finetune instead of 2 min).
- Pipeline: `gw_scene.sh OCC_SOURCE=mesh` (linked train model, output in `output/gw_beags_meshocc/<name>`, mesh of the
  prepared model -> `mesh_train`, +~12 min on kitchen);
  `gw_mesh.sh MESH_OUT`. Smoke: `slurm/gw_smoke_meshocc.sh` (fern).
- Tests: `tests/test_mesh_field.py` 9 passed on CPU (cube-in-box tets vs analytic distance, band, off-grid, cache,
  `LabeledVoxels.distance` == old rings); `test_beags_utils.py` 13 passed, 3 skipped (CUDA).
- Fern smoke with the first, shared-cell grids (job 63268702, 300 its from the 2000-it train model):
  - Time: mesh 325 s; grids built in 14.5 s (label 0, 529x254x440) + 5.5 s (label 1), cell 0.132, band 0.265 (the voxel
    run: voxel 0.404); finetune 65 it/s (voxel run 58).
  - Train-view IoU / BIoU / EIoU / BEIoU: 0.748 / 0.624 / 0.831 / 0.455; voxel run (job 62575946): 0.739 / 0.651 /
    0.830 / 0.526.
  - Renders: `output/smoke/fern/renders_labels_meshocc_2300/grid_all.jpg`, visually on par with
    `renders_labels_2300/fern_labels_8views.jpg`.
  - Fern has 1 GT view, so this only shows the path works. Kitchen (`OCC_SOURCE=mesh`, another `METHOD`) is next.

## 5d. GAVIS unseen hole, `HOLE_MODE=gavis` (2026-11-05): works, adds almost nothing over `alpha` so far
GAVIS (arXiv 2605.30342, github.com/xsj01/gavis) reimplemented for the removal hole. Opt-in; no default changed.
- **Score:** per pixel of the render without label k, U = sum w_i (1 - v_i(d)) + (1 - alpha). v_i is the per-Gaussian
  visibility field (9 SH coefficients, kappa 1), built on the full model from the training cameras.
  `HOLE_MODE=gavis` keeps the dilated-mask pixels with U > `UNSEEN_THRESH` (0.5) or deleted alpha > 0.05.
- **Not ported from the paper:** virtual particles (1 - alpha stands in for them) and the entropy score.
- **Code:** `beags/visibility.py`, `beags/unseen_map.py`, `beags/hole_grid.py`, `beags_artifixer.py --unseen_maps --kappa`,
  ArtiFixer-GS `removal.py split --unseen_dir --unseen_threshold`, `run.sh`, `slurm/gw_gavis_smoke.sh`, and an opt-in
  `render_ours(vis=True)` in the rasterizer: sum of T, pixel count, footprint area per Gaussian. The Python wrapper
  passes `vis` only when asked, and the extension accepts both call forms.
- **Tests (job 63573322, GPU):** GW 196 passed, 1 skipped; ArtiFixer-GS 56 passed, 1 skipped + the nvdiffrast mesh test.

Hole share of the image (fern: all 20 frames; kitchen: every 4th frame, 70 of 279, `images_4`). Mesh on kitchen was not
compared: `hole_1.ply` predates the contact patch and `mesh_front` was not rendered.
| scene | alpha | gavis 0.5 | gavis 0.3 | mesh |
|---|---|---|---|---|
| fern smoke (`output/artifixer/fern/normal/hole_grid.jpg`) | 36.1% | 36.2% | 36.8% | 38.4% |
| kitchen 33k (`output/artifixer/mip360_kitchen/normal/hole_grid.jpg`) | 13.43% | 13.45% | 13.48% | - |

Kitchen, measured on the same 70 frames:
- **The hole is the deleted silhouette in every mode.** Deleted alpha > 0.05 alone covers 13.426%; see-through background
  inside it adds 0.001%. So the smear (7.2) is already inside the `alpha` hole, and `gavis` does not change that.
- **U does not flag the smear.** Inside the silhouette, dark pixels (removal render luminance < 0.2, 30% of it) have
  median U 0.012 and 99th percentile 0.53; only 1.5% have U > 0.3. Only 0.5% of all silhouette pixels have U > 0.5, so
  dropping the `deleted` term would lose nearly the whole hole.
- **Likely cause, not proven:** the score saturates on dense captures. v is about 1 - exp(-Vtilde), and with 244 views the
  sum is large even for weakly seen Gaussians: the sphere-mean Vtilde of label 0 has median 23.7, and only 0.5% of the
  Gaussians are below 0.5. This agrees with 7.2, which found these Gaussians are seen. GAVIS was designed for active
  mapping with 10-80 views.
- **Untested idea:** threshold Vtilde(d) directly instead of the saturated v (a smear pixel at U 0.012 means v near 0.988,
  Vtilde about 4.4, against the median of 23.7), or raise kappa (needs SH degree > 2).

Not run: diffusion on a `gavis` hole, and the BakerH VM (`setup.sh` builds the rasterizer from the rsynced tree).

## 6. Results so far
GPU used in total: ~14 min of debug-QoS jobs (6 jobs) + 60 min for kitchen.

### Smoke: LLFF fern, `-r 8` (`output/smoke/fern`)
Produced by `slurm/gw_smoke.sh` job 62575681 (tests + training) plus the debug scripts in `tmp_amazzucc/gw_debug/`. The
smoke script itself has **never completed in one job**: 62575681 failed at export, before the `get_combined_args` fix.
- **Tests:** 104/104 passed in job 62575681 (2026-10-28 16:11), plus ArtiFixer `tests/test_removal.py -k mesh_front_cuda`.
  They have NOT been rerun since `get_combined_args`, the crop raise, `--split_labels`, `label_<k>.ply` and
  `beags_render_labels.py` changed. Kitchen ran those code paths end to end.
- **GW defaults, 300 its:** test PSNR 21.53 (train 21.96), opacity < 0.05 share 38%.
- **All flags, 2000 its:** test PSNR 25.24 (train 26.08), opacity < 0.05 share 0%. These come from train.py's in-loop
  eval; `render.py` was not run on fern.
- **BEA-GS eval** on the finetuned model (iteration 2300): IoU 0.739, BIoU 0.651, EIoU 0.830, BEIoU 0.526. This is on
  the only GT view, `IMG_4027`, which is a TRAINING view, so it is not a held-out number.
- **Seed2GS** ("fern plant"): SAM3 score 0.80, 48/48 orbit views tracked, 175 s, of which 158 s is the SAM3 subprocess
  (per-op timings, added later: ~110 s SAM3 import + model load, ~42 s first `detect` (Triton JIT), ~5 s tracking).
  `seed2gs_eval.py` on these masks (CPU, 2026-10-29): IoU 0.710, BIoU 0.634 on `IMG_4027` (a training view), and
  Gaussian IoU 0.832 against the BEA-GS labels; BEA-GS with provided masks scored IoU 0.739 on the same view.
  (`seed2gs/seed2gs.json`).
- **Renders:** `renders_labels_2300/grid/*.png`. The fern is extracted cleanly. In label 0 the region where the fern was
  is empty (white). A few leaf fragments remain in label 0 at the hole border.

### Full run: Mip-NeRF 360 kitchen, `images_4` (779x520), job 62590809, 59:40 on 1 GPU -> `output/gw_beags/mip360_kitchen`
| | train 30k (GW + bilagrid + hollow) | BEA-GS finetune 33k |
|---|---|---|
| test PSNR / SSIM / LPIPS (`render.py` + `metrics.py`, 35 views) | 26.82 / 0.922 / 0.082 | 29.99 / 0.937 / 0.083 |
| in-loop eval at 30k (`render_imp`, another rasterizer) | test 27.06, train 27.08 PSNR | |
| Gaussians | 929k (peak 1.2M at 22k; the hollow prunes removed ~490k) | 713k (label 0: 523k, label 1: 189k; ~217k removed by the opacity prune) |
| IoU / BIoU / EIoU / BEIoU (10 GT test views) | 0.974 / 0.892 / 0.946 / 0.805 (30k model with the prepare labels) | 0.976 / 0.881 / 0.971 / 0.893 |
| floaters, components / Gaussians (*) | 2095 / 41.5k | 1019 / 8.0k |
| opacity < 0.05 / < 0.1 share | 0.0% / 15.5% | 4.4% / 15.4% |

(*) The floater counts use AABB-relative voxels (0.166 vs 0.066), so the two columns are not directly comparable.
Recompute both with a fixed `--voxel` to compare them.

Hole metrics (`beags/hollow_metrics_33000.json`). All are over 244 training views, inside the rendered label-1
silhouette (alpha > 0.5):
- **alpha of everything except label 1:** 0.990 (0.994 with 8x8 pooling). This is expected, because the table under
  the bulldozer is seen. `hole_emptiness` is a misnomer: it is an alpha, so higher means fuller.
- **the same, rendering only unseen Gaussians** (total render weight < 1 px over the training set): **0.0002**, against
  a target of 0.
- **silhouette area / SAM2 mask area:** 0.986.

References at the same `images_4`, with the same split and byte-identical test GT:
- FastGS BEA-GS (`../BEA-GS_FSplat/output/beags/mip360/kitchen`): 32.44 PSNR at 30k, 32.15 at 33k.
- Its segmentation at 33k: 0.969 / 0.881 / 0.967 / 0.886, so BEA-GS on GW matches or beats FastGS on segmentation.
- Original GW at 1600 px width (not comparable): 28.94.

Renders, in `renders_labels_33000/`:
- `grid/*.png`, plus `kitchen_labels_8views.jpg`, which was stitched by hand. `beags_render_labels.py` now writes
  `grid_all.jpg` itself.
- Label 1: the bulldozer is extracted cleanly.
- Label 0 keeps the bucket teeth, plus a large dark smear the size of the bulldozer's footprint on the table, visible in
  all 8 views.
- Removal renders on the 10 GT test views already exist: `beags/beags/renders_33000/{extracted_1,removal_1}`.

Other outputs:
- `export/point_cloud.ply`: 30k model, colours drifted (section 7.1).
- `export_beags/{point_cloud,label_0,label_1}.ply`.
- `mesh/{mesh_labels,label_0,label_1,hole_1,hole_1_closure}.ply`. The closure has only 73 faces, because the label-1
  tets touch the background tets almost nowhere.
- `train/beags/{masks,labeled_points.npz}`: the consensus masks used by the finetune.

## 7. Open problems (most important first)
### 7.1 The bilateral grid causes most of the kitchen PSNR drop (measured on CPU from the saved test renders)
The 30k GW model scores 26.82, against 32.44 for FastGS. The BEA-GS finetune then adds +3.2 dB.

**Mechanism:**
- With `--appearance bilagrid`, L1 and SSIM are both computed on the *graded* render (`robust/losses.py:80-88`).
- Eval (`render.py`, the in-loop eval) and the finetune (`beags_finetune.py:126-127`, no grid) use the *raw* render.
- So the raw colours drift per view and spatially: the offsets reach 14/255, and 68% of the MSE is low-frequency.
- Train ≈ test in the in-loop eval (27.08 vs 27.06), so the raw render does not fit even the training views.

**Evidence:**
- A per-image 3x4 affine colour fit gives 26.82 -> 28.60 (global) and 32.16 with 8x8 tiles.
- The high-frequency error is the same before and after the finetune (7.04e-4 vs 7.08e-4).
- So the finetune's +3.2 dB is low-frequency colour re-fit, not BEA-GS and not geometry.

**Ruled out:**
- The train/test split: same `splits.py` as FastGS, the same 35 views, byte-identical GT.
- Hollowing as the cause of the 30k drop: it keeps 3.3x FastGS's Gaussian count, and the finetune gains 3.2 dB while it
  removes Gaussians.
- "Exposure compensation off" as a separate suspect: it is mutually exclusive with the grid.

**A residual GW gap remains:**
- Even after an 8x8-tile colour correction, GW is ~1.5 dB below FastGS (32.16 vs 33.69).
- Its high-frequency MSE is worse (7.04e-4 vs 5.08e-4), and this survives the finetune (29.99 vs 32.15).
- Candidates: GW's regularisers (multiview NCC is ~0.13 of the ~0.15 training loss; normal field; depth-normal) or the
  final hollow prune at 30k, which has no recovery iterations after it.

**Cheapest next experiments:**
1. Photo-only finetune from the 30k model, ~5 GPU-min on the debug QoS:
   ```bash
   beags_finetune.py -m output/gw_beags/mip360_kitchen/train --iteration 30000 \
       --out output/gw_photoft/mip360_kitchen/ft --lambda_sag 0 --lambda_occ 0 --lambda_piv 0
   ```
   Then run `render.py` + `metrics.py`. If it reaches ~30 dB, the +3.2 dB is colour re-fit.
2. Fix the drift in training rather than with a colour correction at eval. A global `color_correct` fit only reaches
   28.6. Options:
   - a stronger grid regulariser: TV weight > 10, or pulling each grid toward identity or the mean grid;
   - or grid-free final iterations.

   Keep reporting raw-render PSNR so the comparison with FastGS stays fair.
3. For the residual ~1.5 dB, run kitchen without the grid (`ROBUST_ARGS=--no-exposure_compensation`, which resolves to
   appearance `none`) and without hollowing (`HOLLOW_ARGS=`). Use `METHOD=gw_nogrid` / `gw_nohollow`, ~1 h each, and
   compare their high-frequency MSE with FastGS's 5.08e-4.
4. Add `--checkpoint_iterations 30000` so the grid is saved: `bilagrid<it>.pth` is written only with checkpoints.

Consequence: the 30k export (`export/point_cloud.ply`), which is ArtiFixer's input, carries the drifted colours. The 33k
export does not.

### 7.2 Removal residue in label 0 (kitchen)
The bucket teeth and a dark smear under the bulldozer stay in label 0. The unseen-only hole alpha is ~0, so these are
*seen* Gaussians: mislabelled boundary Gaussians (the teeth) and table-surface / mixed Gaussians under the object.
- Places to look: `beags/labels.py` voting, the SAG and occupancy weights, and `--dilation_weight` (default 1.0; the VM
  uses 0.1).
- Compare against FastGS BEA-GS kitchen removal renders.
- Compare runs with the `beags_render_labels.py` grids.

### 7.3 ArtiFixer was never run end to end on GW output
Only the mesh-hole unit test ran. The correct command:
```bash
AF_ROOT=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/recon/kitchen \
  sbatch -J afx_normal_mip360_kitchen slurm/gw_artifixer.sh mip360_kitchen 1 normal \
  output/gw_beags/mip360_kitchen/train/beags/masks
```
- The 4th argument (masks) is required for `normal`.
- Without `AF_ROOT`, set `CAPTION` (the caption is `ArtiFixer/output/recon/kitchen/captions/kitchen/caption.h5`).
- `HOLE_MODE` becomes `mesh`, because `mesh/hole_1.ply` exists.
- Add `FALLBACK_MASKS=/leonardo_work/IscrC_EditGS/gabri/datasets/mipnerf360_hd_masks/masks/training/sam2/kitchen/masks`:
  the consensus masks lack the 35 test views, and `split` needs every frame (5b).
- Fix 7.1 first, or the inputs carry the colour drift. Rebuild the mesh first for the contact-patch hole (5b).
- On fern, prepare + removal + split already pass (5b); the diffusion stages (infer onwards) have never run on GW output.

### 7.5 The exported PLY renders ~1.4-1.6 dB worse in gsplat than natively (no longer used by ArtiFixer)
**Bypassed since 2026-10-29:** ArtiFixer-GS renders through GW's own `render_ours` (`RENDER_CMD` = `render_ply_frames.py`,
section 5c), and the exported PLY rendered that way equals native (kitchen 33k: 115.6 dB, alpha MAE 0). gsplat is left
only for ArtiFixer's optional trajectory visibility checks (vanilla_vidsplat, `TRAJ_RENDER_CHECK`). The gap below
matters only for other 3DGS viewers or tools.
`export_3dgs_ply.py --check` (two processes, 5b), drop of PSNR vs the photos from native to gsplat:
| model | views | gsplat vs native | native / gsplat vs photos | drop | MSE gsplat vs native, centre -> corner rings |
|---|---|---|---|---|---|
| fern smoke 2000 its, -r 8 | 20 | 32.41 dB | 25.68 / 24.65 | 1.03 | 2.3e-5, 4.5e-4, 7.5e-4, 7.0e-4 |
| kitchen train 30k | 56 (stride 5) | 32.72 dB | 27.03 / 25.40 | 1.63 | 2.3e-3, 3.4e-3, 3.8e-3, 3.6e-3 |
| kitchen beags 33k | 56 | 37.06 dB | 30.56 / 29.17 | 1.39 | 4.2e-4, 5.9e-4, 7.1e-4, 7.5e-4 |

Alpha MAE is <= 0.00016, so coverage matches but colour does not. ArtiFixer (gsplat) therefore sees renders ~1.5 dB worse
than the model is.
- **Checked and equal to 3DGS / gsplat:**
  - the 3D-filter bake (`get_scaling_n_opacity_with_3D_filter`);
  - EWA cov2D, with kernel_size 0 and the 1.3 tan_fov clamp;
  - alpha = min(0.99, o exp(power)), the 1/255 and T < 1e-4 cut-offs;
  - SH evaluation (dir = pos - campos, standard constants);
  - SH bands 2-3 are zero on fern, so the SH degree is not the cause there.
- **Differences found:**
  - GW's tile sort key is the distance to the camera (`depths[idx] = norm3df(p_view)` in `render_forward.cu`
    preprocess), gsplat's is view z. The order changes colour but not alpha. That fits fern's 30x radial growth
    (keys agree at the image centre), but not kitchen's flat profile.
  - So at least one more difference is unexplained. Note that `render_imp` (in-loop eval) and `render_ours` (render.py)
    already disagree by 0.24 dB on kitchen (section 6).
- **Next:**
  1. Save the gsplat renders in `gsplat_check.py`, and look at where the error sits (edges, thin structures,
     view-dependent highlights).
  2. Add a compile-time or flag-switchable view-z sort key to `diff-gaussian-rasterization_ours_beags` and re-measure.
     Rebuild into a separate env or copy: the editable install is shared by every GW job.

### 7.4 Smaller items
1. **gsplat equivalence unmeasured.** The two-process check exists (5b) but has never run: `sbatch slurm/gw_gsplat_check.sh`
   or the `gsplat` step of `gw_smoke_features.sh`.
2. **Seed2GS:** 175 s on fern against a target of < 1 min. Most of it is SAM3 import + load (~110 s) and the first detect
   (~42 s Triton JIT, cached afterwards), not tracking. The speed-ups in 5b are unmeasured. It depends on gabri's env. It
   was never run on a full model; `seed2gs_eval.py` now gives the LERF figurines mIoU once it runs there.
3. **Floaters:** 1019 components after BEA-GS on kitchen; the voxel size is AABB-relative (see 6). Recompute both
   columns with `hollow_metrics.py --floater_voxel <abs>`.
4. **Not enabled in the full pipeline:**
   - GW non-maximal pruning (`--non_maximal_pruning_iterations`, yaml `use_pruning`);
   - `--confidence` (off in `slurm/common.sh`);
   - `--opacity_decay`;
   - per-label mask carving in the mesh.
5. **Implemented, not run on a GPU** (5b): the confidence artifact maps, `MASK_SOURCE=seed2gs`, the contact-patch hole,
   the ablation launcher (plain `gw` arm, appearance, H1-H5 + `gw_nofinal`, L_piv) and the BakerH pipeline (no
   REPORT.txt yet). **Still not done:** the GW reproduction check on room (30.85 PSNR, within 0.2 dB); the ablation
   runs themselves; clean on/off flags for H1 and H4.
6. **Tests:** CPU, login node, 2026-10-29: 134 passed, 24 skipped (CUDA) over 9 files, plus ArtiFixer-GS
   `test_removal.py` + `test_trajectory.py` 53 passed, 1 skipped. The login node kills a process after 600 s CPU, so run
   one pytest per file there. The GPU tests have not run since 2026-10-28:
   `sbatch slurm/gw_tests.sh`.
7. **`slurm/gw_mesh.sh`** skips the pivot mesh when it exists, even after a new finetune. `gw_scene.sh` deletes it when
   it re-finetunes; direct users of `gw_mesh.sh` must do the same.

## 8. Next steps (in order)
- 2026-11-24: step-by-step BakerH stages for TA_BLUE_MOTOR + Compressor in `scripts/bakerh/stages/` (`HANDOFF_BAKERH.md`
  4b): new `run_scene.sh` phases `holes` (alpha / mesh / gavis / confidence hole grid, no diffusion), `mesh_train`,
  `finetune_meshocc`, `labels_meshocc`, `renders_meshocc`; `VARIANT=conf|gridfix`; `AF_DIR`, `AF_UNSEEN`, `AF_CONF`.
  Only `bash -n` and a variant dry check on Leonardo; never run.
- 2026-11-05: GAVIS hole implemented and checked (5d): no gain over `alpha` on fern or kitchen. Keep `alpha` / `mesh` as
  the defaults unless the Vtilde threshold in 5d separates the smear.
- 2026-11-02: BakerH VM run of the 7 scenes set up (`HANDOFF_BAKERH.md`): GW 30k + bilateral grid + hollowing ->
  BEA-GS -> extraction / background renders in `output/gw_beags_dw0.1/showcase/`. `run_scene.sh` default phases now
  stop after `hollow` (new `showcase` phase); `FULL=1` adds mesh + ArtiFixer. `ModelParams.data_device` default is now
  `cpu` (was `cuda`); images are always loaded on the CPU.
0. Done 2026-10-29: smoke of 5b and 5c. ArtiFixer now renders with GW (5c), so 7.5 no longer blocks it.
1. Photo-only finetune on kitchen (7.1, ~5 GPU-min): `sbatch slurm/gw_photoft.sh`. Then try the new grid flags
   (`--bilagrid_identity_weight`, `--bilagrid_until`, `--lambda_raw`) and `gw_nofinal`. Ask the user before any job over
   debug size.
2. Once the backbone is within ~0.5 dB of FastGS on kitchen, run `bash slurm/launch_one_per_dataset.sh` (room, bed,
   figurines, fern; ~1 h each). Ask first.
3. ArtiFixer on kitchen (7.3), then look at the renders. Rebuild the kitchen mesh first (`slurm/gw_mesh.sh ... --remove 1`,
   ~12 min measured, ask) so `hole_1.ply` carries the contact patch; the existing one predates it.
4. Fix the removal residue (7.2), comparing with the label grids.
5. Seed2GS on LERF figurines vs the provided masks.

## 9. Git state
- **Repo:** branch `beags` at `6b5acc8` (= `origin/main` of the tehMazz fork; upstream diego1401's tip is `a064133`).
  Nothing is committed.
  - Staged: `.gitmodules` and the `submodules/artifixer_gs` gitlink (`8d4c5d2`).
  - Modified, not staged: the GW files in section 5.
  - Untracked: all new code, `slurm/`, `scripts/`, `logs/`, `requirements_beags.txt`,
    `submodules/diff-gaussian-rasterization_ours_beags/`, `GWBEAGS.md`, `HANDOFF.md`, `CLAUDE.md`.
- **`submodules/artifixer_gs`:**
  - checked out at `8d4c5d2` (= its `origin/main`) on a **local-only** branch `gw` (no upstream, not pushed);
  - uncommitted changes: modified `README.md`, `distill.py`, `prepare.py`, `render.py`,
    `data_processing/camera_trajectories.py`, `scripts/run.sh`; new `caption.py`, `hole_mesh.py`, `propagate.py`,
    `removal.py`, `trajectory.py`, tests;
  - commit there and `git push -u origin gw` before committing the pointer here, or the gitlink points at a commit nobody
    can fetch.
- **`submodules/diff-gaussian-rasterization_ours_beags`** (27 MB) is a plain copy that includes vendored
  `third_party/glm` with its docs, about 1500 files. Before `git add`, make glm a submodule (as in
  `diff-gaussian-rasterization_ours`), or commit only the rasterizer sources.
- **`gaussian_wrapping/eval/TNTUniScanEvals`:** submodule initialised (mesh extraction imports it).

## 10. How to run
```bash
cd /leonardo_work/IscrC_EditGS/repos/CVPR2027/InpaintingPaper/BEA-GS_GWrap
MIP=/leonardo_work/IscrC_EditGS/datasets/mip360; MM=/leonardo_work/IscrC_EditGS/gabri/datasets/mipnerf360_hd_masks/masks
sbatch -J gw_mip360_kitchen slurm/gw_scene.sh mip360_kitchen $MIP/kitchen $MM/training/sam2/kitchen/masks \
    $MM/gt_masks/kitchen/masks -i images_4     # full pipeline, ~1 h on kitchen (asks for 14 h)
```
`gw_scene.sh` skips these steps when their outputs exist: train, export, prepare, finetune, the split export and the pivot
mesh. The evals, label renders, hollow metrics, multi-label mesh, `render.py` and `metrics.py` always rerun.

`sbatch slurm/gw_smoke.sh` runs all tests plus a fern smoke on the debug QoS (30 min limit, ~8 min expected).
**Warning:** it starts with `rm -rf output/smoke/fern`, which deletes the fern Seed2GS, mesh, `export_beags` and render
outputs above and does not recreate them. Copy them first.

`sbatch` copies the job script at submission. Edits to `slurm/*.sh` therefore do not reach a queued job (resubmit it),
while edits to Python files do. Logs: `logs/<job name>_<job id>.out`.
